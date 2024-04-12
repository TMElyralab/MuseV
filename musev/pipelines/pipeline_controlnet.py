from __future__ import annotations

import inspect
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from einops import rearrange, repeat
import PIL.Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionSafetyChecker,
    EXAMPLE_DOC_STRING,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import (
    StableDiffusionControlNetImg2ImgPipeline as DiffusersStableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    BaseOutput,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models.attention import (
    BasicTransformerBlock as DiffusersBasicTransformerBlock,
)
from mmcm.vision.process.correct_color import (
    hist_match_color_video_batch,
    hist_match_video_bcthw,
)

from ..models.attention import BasicTransformerBlock
from ..models.unet_3d_condition import UNet3DConditionModel
from ..utils.noise_util import random_noise, video_fusion_noise
from ..data.data_util import (
    adaptive_instance_normalization,
    align_repeat_tensor_single_dim,
    batch_adain_conditioned_tensor,
    batch_concat_two_tensor_with_index,
    batch_index_select,
    fuse_part_tensor,
)
from ..utils.text_emb_util import encode_weighted_prompt
from ..utils.tensor_util import his_match
from ..utils.timesteps_util import generate_parameters_with_timesteps
from .context import get_context_scheduler, prepare_global_context

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    latents: Union[torch.Tensor, np.ndarray]
    videos_mid: Union[torch.Tensor, np.ndarray]
    down_block_res_samples: Tuple[torch.FloatTensor] = None
    mid_block_res_samples: torch.FloatTensor = None
    up_block_res_samples: torch.FloatTensor = None
    mid_video_latents: List[torch.FloatTensor] = None
    mid_video_noises: List[torch.FloatTensor] = None


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


def prepare_image(
    image,  # b c t h w
    batch_size,
    device,
    dtype,
    image_processor: Callable,
    num_images_per_prompt: int = 1,
    width=None,
    height=None,
):
    if isinstance(image, List) and isinstance(image[0], str):
        raise NotImplementedError
    if isinstance(image, List) and isinstance(image[0], np.ndarray):
        image = np.concatenate(image, axis=0)
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim == 5:
        image = rearrange(image, "b c t h w-> (b t) c h w")
    if height is None:
        height = image.shape[-2]
    if width is None:
        width = image.shape[-1]
    width, height = (x - x % image_processor.vae_scale_factor for x in (width, height))
    if height != image.shape[-2] or width != image.shape[-1]:
        image = torch.nn.functional.interpolate(
            image, size=(height, width), mode="bilinear"
        )
    image = image.to(dtype=torch.float32) / 255.0
    do_normalize = image_processor.config.do_normalize
    if image.min() < 0:
        warnings.warn(
            "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
            f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
            FutureWarning,
        )
        do_normalize = False

    if do_normalize:
        image = image_processor.normalize(image)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)
    return image


class MusevControlNetPipeline(
    DiffusersStableDiffusionControlNetImg2ImgPipeline, TextualInversionLoaderMixin
):
    """
    a union diffusers pipeline, support
    1. text2image model only, or text2video model, by setting skip_temporal_layer
    2. text2video, image2video, video2video;
    3. multi controlnet
    4. IPAdapter
    5. referencenet
    6. IPAdapterFaceID
    """

    _optional_components = [
        "safety_checker",
        "feature_extractor",
    ]
    print_idx = 0

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        controlnet: ControlNetModel
        | List[ControlNetModel]
        | Tuple[ControlNetModel]
        | MultiControlNetModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        # | MultiControlNetModel = None,
        # text_encoder: CLIPTextModel = None,
        # tokenizer: CLIPTokenizer = None,
        # safety_checker: StableDiffusionSafetyChecker = None,
        # feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = False,
        referencenet: nn.Module = None,
        vision_clip_extractor: nn.Module = None,
        ip_adapter_image_proj: nn.Module = None,
        face_emb_extractor: nn.Module = None,
        facein_image_proj: nn.Module = None,
        ip_adapter_face_emb_extractor: nn.Module = None,
        ip_adapter_face_image_proj: nn.Module = None,
        pose_guider: nn.Module = None,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        self.referencenet = referencenet

        # ip_adapter
        if isinstance(vision_clip_extractor, nn.Module):
            vision_clip_extractor.to(dtype=self.unet.dtype, device=self.unet.device)
        self.vision_clip_extractor = vision_clip_extractor
        if isinstance(ip_adapter_image_proj, nn.Module):
            ip_adapter_image_proj.to(dtype=self.unet.dtype, device=self.unet.device)
        self.ip_adapter_image_proj = ip_adapter_image_proj

        # facein
        if isinstance(face_emb_extractor, nn.Module):
            face_emb_extractor.to(dtype=self.unet.dtype, device=self.unet.device)
        self.face_emb_extractor = face_emb_extractor
        if isinstance(facein_image_proj, nn.Module):
            facein_image_proj.to(dtype=self.unet.dtype, device=self.unet.device)
        self.facein_image_proj = facein_image_proj

        # ip_adapter_face
        if isinstance(ip_adapter_face_emb_extractor, nn.Module):
            ip_adapter_face_emb_extractor.to(
                dtype=self.unet.dtype, device=self.unet.device
            )
        self.ip_adapter_face_emb_extractor = ip_adapter_face_emb_extractor
        if isinstance(ip_adapter_face_image_proj, nn.Module):
            ip_adapter_face_image_proj.to(
                dtype=self.unet.dtype, device=self.unet.device
            )
        self.ip_adapter_face_image_proj = ip_adapter_face_image_proj

        if isinstance(pose_guider, nn.Module):
            pose_guider.to(dtype=self.unet.dtype, device=self.unet.device)
        self.pose_guider = pose_guider

    def decode_latents(self, latents):
        batch_size = latents.shape[0]
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = super().decode_latents(latents=latents)
        video = rearrange(video, "(b f) h w c -> b c f h w", b=batch_size)
        return video

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        video_length: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        latents: torch.Tensor = None,
        w_ind_noise: float = 0.5,
        image: torch.Tensor = None,
        timestep: int = None,
        initial_common_latent: torch.Tensor = None,
        noise_type: str = "random",
        add_latents_noise: bool = False,
        need_img_based_video_noise: bool = False,
        condition_latents: torch.Tensor = None,
        img_weight=1e-3,
    ) -> torch.Tensor:
        """
        支持多种情况下的latens：
        img_based_latents: 当Image t=1，latents=None时，使用image赋值到shape，然后加噪；适用于text2video、middle2video。
        video_based_latents：image =shape或Latents!=None时，加噪，适用于video2video；
        noise_latents：当image 和latents都为None时，生成随机噪声，适用于text2video

        support multi latents condition:
        img_based_latents: when Image t=1, latents=None, use image to assign to shape, then add noise; suitable for text2video, middle2video.
        video_based_latents: image =shape or Latents!=None, add noise, suitable for video2video;
        noise_laten: when image and latents are both None, generate random noise, suitable for text2video

        Args:
            batch_size (int): _description_
            num_channels_latents (int): _description_
            video_length (int): _description_
            height (int): _description_
            width (int): _description_
            dtype (torch.dtype): _description_
            device (torch.device): _description_
            generator (torch.Generator): _description_
            latents (torch.Tensor, optional): _description_. Defaults to None.
            w_ind_noise (float, optional): _description_. Defaults to 0.5.
            image (torch.Tensor, optional): _description_. Defaults to None.
            timestep (int, optional): _description_. Defaults to None.
            initial_common_latent (torch.Tensor, optional): _description_. Defaults to None.
            noise_type (str, optional): _description_. Defaults to "random".
            add_latents_noise (bool, optional): _description_. Defaults to False.
            need_img_based_video_noise (bool, optional): _description_. Defaults to False.
            condition_latents (torch.Tensor, optional): _description_. Defaults to None.
            img_weight (_type_, optional): _description_. Defaults to 1e-3.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            torch.Tensor: latents
        """

        # ref https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py#L691
        # ref https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#L659
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None or (latents is not None and add_latents_noise):
            if noise_type == "random":
                noise = random_noise(
                    shape=shape, dtype=dtype, device=device, generator=generator
                )
            elif noise_type == "video_fusion":
                noise = video_fusion_noise(
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                    w_ind_noise=w_ind_noise,
                    initial_common_noise=initial_common_latent,
                )
            if (
                need_img_based_video_noise
                and condition_latents is not None
                and image is None
                and latents is None
            ):
                if self.print_idx == 0:
                    logger.debug(
                        (
                            f"need_img_based_video_noise, condition_latents={condition_latents.shape},"
                            f"batch_size={batch_size}, noise={noise.shape}, video_length={video_length}"
                        )
                    )
                condition_latents = condition_latents.mean(dim=2, keepdim=True)
                condition_latents = repeat(
                    condition_latents, "b c t h w->b c (t x) h w", x=video_length
                )
                noise = (
                    img_weight**0.5 * condition_latents
                    + (1 - img_weight) ** 0.5 * noise
                )
                if self.print_idx == 0:
                    logger.debug(f"noise={noise.shape}")

        if image is not None:
            if image.ndim == 5:
                image = rearrange(image, "b c t h w->(b t) c h w")
            image = image.to(device=device, dtype=dtype)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                init_latents = [
                    # self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    self.vae.encode(image[i : i + 1]).latent_dist.mean
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                # init_latents = self.vae.encode(image).latent_dist.sample(generator)
                init_latents = self.vae.encode(image).latent_dist.mean
            init_latents = self.vae.config.scaling_factor * init_latents
            # scale the initial noise by the standard deviation required by the scheduler
            if (
                batch_size > init_latents.shape[0]
                and batch_size % init_latents.shape[0] == 0
            ):
                # expand init_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate(
                    "len(prompt) != len(image)",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                additional_image_per_prompt = batch_size // init_latents.shape[0]
                init_latents = torch.cat(
                    [init_latents] * additional_image_per_prompt, dim=0
                )
            elif (
                batch_size > init_latents.shape[0]
                and batch_size % init_latents.shape[0] != 0
            ):
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                init_latents = torch.cat([init_latents], dim=0)
            if init_latents.shape[2] != shape[3] and init_latents.shape[3] != shape[4]:
                init_latents = torch.nn.functional.interpolate(
                    init_latents,
                    size=(shape[3], shape[4]),
                    mode="bilinear",
                )
            init_latents = rearrange(
                init_latents, "(b t) c h w-> b c t h w", t=video_length
            )
            if self.print_idx == 0:
                logger.debug(f"init_latensts={init_latents.shape}")
        if latents is None:
            if image is None:
                latents = noise * self.scheduler.init_noise_sigma
            else:
                if self.print_idx == 0:
                    logger.debug(f"prepare latents, image is not None")
                latents = self.scheduler.add_noise(init_latents, noise, timestep)
        else:
            if isinstance(latents, np.ndarray):
                latents = torch.from_numpy(latents)
            latents = latents.to(device=device, dtype=dtype)
            if add_latents_noise:
                latents = self.scheduler.add_noise(latents, noise, timestep)
            else:
                latents = latents * self.scheduler.init_noise_sigma
        if latents.shape != shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {shape}"
            )
        latents = latents.to(device, dtype=dtype)
        return latents

    def prepare_image(
        self,
        image,  # b c t h w
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        width=None,
        height=None,
    ):
        return prepare_image(
            image=image,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=dtype,
            width=width,
            height=height,
            image_processor=self.image_processor,
        )

    def prepare_control_image(
        self,
        image,  # b c t h w
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = prepare_image(
            image=image,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=dtype,
            width=width,
            height=height,
            image_processor=self.control_image_processor,
        )
        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)
        return image

    def check_inputs(
        self,
        prompt,
        image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1,
        control_guidance_start=0,
        control_guidance_end=1,
    ):
        # TODO: to implement
        if image is not None:
            return super().check_inputs(
                prompt,
                image,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                controlnet_conditioning_scale,
                control_guidance_start,
                control_guidance_end,
            )

    def hist_match_with_vis_cond(
        self, video: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """
        video: b c t1 h w
        target: b c t2(=1) h w
        """
        video = hist_match_video_bcthw(video, target, value=255.0)
        return video

    def get_facein_image_emb(
        self, refer_face_image, device, dtype, batch_size, do_classifier_free_guidance
    ):
        # refer_face_image and its face_emb
        if self.print_idx == 0:
            logger.debug(
                f"face_emb_extractor={type(self.face_emb_extractor)}, facein_image_proj={type(self.facein_image_proj)}, refer_face_image={type(refer_face_image)},  "
            )
        if (
            self.face_emb_extractor is not None
            and self.facein_image_proj is not None
            and refer_face_image is not None
        ):
            if self.print_idx == 0:
                logger.debug(f"refer_face_image={refer_face_image.shape}")
            if isinstance(refer_face_image, np.ndarray):
                refer_face_image = torch.from_numpy(refer_face_image)
            refer_face_image_facein = refer_face_image
            n_refer_face_image = refer_face_image_facein.shape[2]
            refer_face_image_facein = rearrange(
                refer_face_image, "b c t h w-> (b t) h w c"
            )
            # refer_face_image_emb： bt d或者 bt h w d
            (
                refer_face_image_emb,
                refer_align_face_image,
            ) = self.face_emb_extractor.extract_images(
                refer_face_image_facein, return_type="torch"
            )
            refer_face_image_emb = refer_face_image_emb.to(device=device, dtype=dtype)
            if self.print_idx == 0:
                logger.debug(f"refer_face_image_emb={refer_face_image_emb.shape}")
            if refer_face_image_emb.shape == 2:
                refer_face_image_emb = rearrange(refer_face_image_emb, "bt d-> bt 1 d")
            elif refer_face_image_emb.shape == 4:
                refer_face_image_emb = rearrange(
                    refer_face_image_emb, "bt h w d-> bt (h w) d"
                )
            refer_face_image_emb_bk = refer_face_image_emb
            refer_face_image_emb = self.facein_image_proj(refer_face_image_emb)
            # Todo:当前不支持 IPAdapterPlus的vision_clip的输出
            refer_face_image_emb = rearrange(
                refer_face_image_emb,
                "(b t) n q-> b (t n) q",
                t=n_refer_face_image,
            )
            refer_face_image_emb = align_repeat_tensor_single_dim(
                refer_face_image_emb, target_length=batch_size, dim=0
            )
            if do_classifier_free_guidance:
                # TODO：固定特征，有优化空间
                # TODO: fix the feature, there is optimization space
                uncond_refer_face_image_emb = self.facein_image_proj(
                    torch.zeros_like(refer_face_image_emb_bk).to(
                        device=device, dtype=dtype
                    )
                )
                # Todo:当前可能不支持 IPAdapterPlus的vision_clip的输出
                # TODO: do not support IPAdapterPlus's vision_clip's output
                uncond_refer_face_image_emb = rearrange(
                    uncond_refer_face_image_emb,
                    "(b t) n q-> b (t n) q",
                    t=n_refer_face_image,
                )
                uncond_refer_face_image_emb = align_repeat_tensor_single_dim(
                    uncond_refer_face_image_emb, target_length=batch_size, dim=0
                )
                if self.print_idx == 0:
                    logger.debug(
                        f"uncond_refer_face_image_emb, {uncond_refer_face_image_emb.shape}"
                    )
                    logger.debug(f"refer_face_image_emb, {refer_face_image_emb.shape}")
                refer_face_image_emb = torch.concat(
                    [
                        uncond_refer_face_image_emb,
                        refer_face_image_emb,
                    ],
                )
        else:
            refer_face_image_emb = None
        if self.print_idx == 0:
            logger.debug(f"refer_face_image_emb={type(refer_face_image_emb)}")

        return refer_face_image_emb

    def get_ip_adapter_face_emb(
        self, refer_face_image, device, dtype, batch_size, do_classifier_free_guidance
    ):
        # refer_face_image and its ip_adapter_face_emb
        if self.print_idx == 0:
            logger.debug(
                f"face_emb_extractor={type(self.face_emb_extractor)}, ip_adapter__image_proj={type(self.facein_image_proj)}, refer_face_image={type(refer_face_image)},  "
            )
        if (
            self.ip_adapter_face_emb_extractor is not None
            and self.ip_adapter_face_image_proj is not None
            and refer_face_image is not None
        ):
            if self.print_idx == 0:
                logger.debug(f"refer_face_image={refer_face_image.shape}")
            if isinstance(refer_face_image, np.ndarray):
                refer_face_image = torch.from_numpy(refer_face_image)
            refer_ip_adapter_face_image = refer_face_image
            n_refer_face_image = refer_ip_adapter_face_image.shape[2]
            refer_ip_adapter_face_image = rearrange(
                refer_ip_adapter_face_image, "b c t h w-> (b t) h w c"
            )
            # refer_face_image_emb： bt d or bt h w d
            (
                refer_face_image_emb,
                refer_align_face_image,
            ) = self.ip_adapter_face_emb_extractor.extract_images(
                refer_ip_adapter_face_image, return_type="torch"
            )
            refer_face_image_emb = refer_face_image_emb.to(device=device, dtype=dtype)
            if self.print_idx == 0:
                logger.debug(f"refer_face_image_emb={refer_face_image_emb.shape}")
            if refer_face_image_emb.shape == 2:
                refer_face_image_emb = rearrange(refer_face_image_emb, "bt d-> bt 1 d")
            elif refer_face_image_emb.shape == 4:
                refer_face_image_emb = rearrange(
                    refer_face_image_emb, "bt h w d-> bt (h w) d"
                )
            refer_face_image_emb_bk = refer_face_image_emb
            refer_face_image_emb = self.ip_adapter_face_image_proj(refer_face_image_emb)

            refer_face_image_emb = rearrange(
                refer_face_image_emb,
                "(b t) n q-> b (t n) q",
                t=n_refer_face_image,
            )
            refer_face_image_emb = align_repeat_tensor_single_dim(
                refer_face_image_emb, target_length=batch_size, dim=0
            )
            if do_classifier_free_guidance:
                # TODO：固定特征，有优化空间
                # TODO: fix the feature, there is optimization space
                uncond_refer_face_image_emb = self.ip_adapter_face_image_proj(
                    torch.zeros_like(refer_face_image_emb_bk).to(
                        device=device, dtype=dtype
                    )
                )
                # TODO: 当前可能不支持 IPAdapterPlus的vision_clip的输出
                # TODO: do not support IPAdapterPlus's vision_clip's output
                uncond_refer_face_image_emb = rearrange(
                    uncond_refer_face_image_emb,
                    "(b t) n q-> b (t n) q",
                    t=n_refer_face_image,
                )
                uncond_refer_face_image_emb = align_repeat_tensor_single_dim(
                    uncond_refer_face_image_emb, target_length=batch_size, dim=0
                )
                if self.print_idx == 0:
                    logger.debug(
                        f"uncond_refer_face_image_emb, {uncond_refer_face_image_emb.shape}"
                    )
                    logger.debug(f"refer_face_image_emb, {refer_face_image_emb.shape}")
                refer_face_image_emb = torch.concat(
                    [
                        uncond_refer_face_image_emb,
                        refer_face_image_emb,
                    ],
                )
        else:
            refer_face_image_emb = None
        if self.print_idx == 0:
            logger.debug(f"ip_adapter_face_emb={type(refer_face_image_emb)}")

        return refer_face_image_emb

    def get_ip_adapter_image_emb(
        self,
        ip_adapter_image,
        device,
        dtype,
        batch_size,
        do_classifier_free_guidance,
        height,
        width,
    ):
        # refer_image vision_clip and its ipadapter_emb
        if self.print_idx == 0:
            logger.debug(
                f"vision_clip_extractor={type(self.vision_clip_extractor)},"
                f"ip_adapter_image_proj={type(self.ip_adapter_image_proj)},"
                f"ip_adapter_image={type(ip_adapter_image)},"
            )
        if self.vision_clip_extractor is not None and ip_adapter_image is not None:
            if self.print_idx == 0:
                logger.debug(f"ip_adapter_image={ip_adapter_image.shape}")
            if isinstance(ip_adapter_image, np.ndarray):
                ip_adapter_image = torch.from_numpy(ip_adapter_image)
            # ip_adapter_image = ip_adapter_image.to(device=device, dtype=dtype)
            n_ip_adapter_image = ip_adapter_image.shape[2]
            ip_adapter_image = rearrange(ip_adapter_image, "b c t h w-> (b t) h w c")
            ip_adapter_image_emb = self.vision_clip_extractor.extract_images(
                ip_adapter_image,
                target_height=height,
                target_width=width,
                return_type="torch",
            )
            if ip_adapter_image_emb.ndim == 2:
                ip_adapter_image_emb = rearrange(ip_adapter_image_emb, "b q-> b 1 q")

            ip_adapter_image_emb_bk = ip_adapter_image_emb
            # 存在只需要image_prompt、但不需要 proj的场景，如使用image_prompt替代text_prompt
            # There are scenarios where only image_prompt is needed, but proj is not needed, such as using image_prompt instead of text_prompt
            if self.ip_adapter_image_proj is not None:
                logger.debug(f"ip_adapter_image_proj is None, ")
                ip_adapter_image_emb = self.ip_adapter_image_proj(ip_adapter_image_emb)
            # TODO: 当前不支持 IPAdapterPlus的vision_clip的输出
            # TODO: do not support IPAdapterPlus's vision_clip's output
            ip_adapter_image_emb = rearrange(
                ip_adapter_image_emb,
                "(b t) n q-> b (t n) q",
                t=n_ip_adapter_image,
            )
            ip_adapter_image_emb = align_repeat_tensor_single_dim(
                ip_adapter_image_emb, target_length=batch_size, dim=0
            )
            if do_classifier_free_guidance:
                # TODO：固定特征，有优化空间
                # TODO: fix the feature, there is optimization space
                if self.ip_adapter_image_proj is not None:
                    uncond_ip_adapter_image_emb = self.ip_adapter_image_proj(
                        torch.zeros_like(ip_adapter_image_emb_bk).to(
                            device=device, dtype=dtype
                        )
                    )
                    if self.print_idx == 0:
                        logger.debug(
                            f"uncond_ip_adapter_image_emb use ip_adapter_image_proj(zero_like)"
                        )
                else:
                    uncond_ip_adapter_image_emb = torch.zeros_like(ip_adapter_image_emb)
                    if self.print_idx == 0:
                        logger.debug(f"uncond_ip_adapter_image_emb  use zero_like")
                # TODO:当前可能不支持 IPAdapterPlus的vision_clip的输出
                # TODO: do not support IPAdapterPlus's vision_clip's output
                uncond_ip_adapter_image_emb = rearrange(
                    uncond_ip_adapter_image_emb,
                    "(b t) n q-> b (t n) q",
                    t=n_ip_adapter_image,
                )
                uncond_ip_adapter_image_emb = align_repeat_tensor_single_dim(
                    uncond_ip_adapter_image_emb, target_length=batch_size, dim=0
                )
                if self.print_idx == 0:
                    logger.debug(
                        f"uncond_ip_adapter_image_emb, {uncond_ip_adapter_image_emb.shape}"
                    )
                    logger.debug(f"ip_adapter_image_emb, {ip_adapter_image_emb.shape}")
                # uncond_ip_adapter_image_emb = torch.zeros_like(ip_adapter_image_emb)
                ip_adapter_image_emb = torch.concat(
                    [
                        uncond_ip_adapter_image_emb,
                        ip_adapter_image_emb,
                    ],
                )

        else:
            ip_adapter_image_emb = None
        if self.print_idx == 0:
            logger.debug(f"ip_adapter_image_emb={type(ip_adapter_image_emb)}")
        return ip_adapter_image_emb

    def get_referencenet_image_vae_emb(
        self,
        refer_image,
        batch_size,
        num_videos_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance,
        width: int = None,
        height: int = None,
    ):
        # prepare_referencenet_emb
        if self.print_idx == 0:
            logger.debug(
                f"referencenet={type(self.referencenet)}, refer_image={type(refer_image)}"
            )
        if self.referencenet is not None and refer_image is not None:
            n_refer_image = refer_image.shape[2]
            refer_image_vae = self.prepare_image(
                refer_image,
                batch_size=batch_size * num_videos_per_prompt,
                num_images_per_prompt=num_videos_per_prompt,
                device=device,
                dtype=dtype,
                width=width,
                height=height,
            )
            # ref_hidden_states = self.vae.encode(refer_image_vae).latent_dist.sample()
            refer_image_vae_emb = self.vae.encode(refer_image_vae).latent_dist.mean
            refer_image_vae_emb = self.vae.config.scaling_factor * refer_image_vae_emb

            logger.debug(f"refer_image_vae_emb={refer_image_vae_emb.shape}")

            if do_classifier_free_guidance:
                # 1. zeros_like image
                # uncond_refer_image_vae_emb = self.vae.encode(
                #     torch.zeros_like(refer_image_vae)
                # ).latent_dist.mean
                # uncond_refer_image_vae_emb = (
                #     self.vae.config.scaling_factor * uncond_refer_image_vae_emb
                # )

                # 2. zeros_like image vae emb
                # uncond_refer_image_vae_emb = torch.zeros_like(refer_image_vae_emb)

                # uncond_refer_image_vae_emb = rearrange(
                #     uncond_refer_image_vae_emb,
                #     "(b t) c h w-> b c t h w",
                #     t=n_refer_image,
                # )

                # refer_image_vae_emb = rearrange(
                #     refer_image_vae_emb, "(b t) c h w-> b c t h w", t=n_refer_image
                # )
                # refer_image_vae_emb = torch.concat(
                #     [uncond_refer_image_vae_emb, refer_image_vae_emb], dim=0
                # )
                # refer_image_vae_emb = rearrange(
                #     refer_image_vae_emb, "b c t h w-> (b t) c h w"
                # )
                # logger.debug(f"refer_image_vae_emb={refer_image_vae_emb.shape}")

                # 3. uncond_refer_image_vae_emb = refer_image_vae_emb
                uncond_refer_image_vae_emb = refer_image_vae_emb

                uncond_refer_image_vae_emb = rearrange(
                    uncond_refer_image_vae_emb,
                    "(b t) c h w-> b c t h w",
                    t=n_refer_image,
                )

                refer_image_vae_emb = rearrange(
                    refer_image_vae_emb, "(b t) c h w-> b c t h w", t=n_refer_image
                )
                refer_image_vae_emb = torch.concat(
                    [uncond_refer_image_vae_emb, refer_image_vae_emb], dim=0
                )
                refer_image_vae_emb = rearrange(
                    refer_image_vae_emb, "b c t h w-> (b t) c h w"
                )
                logger.debug(f"refer_image_vae_emb={refer_image_vae_emb.shape}")
        else:
            refer_image_vae_emb = None
        return refer_image_vae_emb

    def get_referencenet_emb(
        self,
        refer_image_vae_emb,
        refer_image,
        batch_size,
        num_videos_per_prompt,
        device,
        dtype,
        ip_adapter_image_emb,
        do_classifier_free_guidance,
        prompt_embeds,
        ref_timestep_int: int = 0,
    ):
        # prepare_referencenet_emb
        if self.print_idx == 0:
            logger.debug(
                f"referencenet={type(self.referencenet)}, refer_image={type(refer_image)}"
            )
        if (
            self.referencenet is not None
            and refer_image_vae_emb is not None
            and refer_image is not None
        ):
            n_refer_image = refer_image.shape[2]
            # ref_timestep = (
            #     torch.ones((refer_image_vae_emb.shape[0],), device=device)
            #     * ref_timestep_int
            # )
            ref_timestep = torch.zeros_like(ref_timestep_int)
            # referencenet 优先使用 ip_adapter 中图像提取到的 clip_vision_emb
            if ip_adapter_image_emb is not None:
                refer_prompt_embeds = ip_adapter_image_emb
            else:
                refer_prompt_embeds = prompt_embeds
            if self.print_idx == 0:
                logger.debug(
                    f"use referencenet: n_refer_image={n_refer_image}, refer_image_vae_emb={refer_image_vae_emb.shape}, ref_timestep={ref_timestep.shape}"
                )
                if prompt_embeds is not None:
                    logger.debug(f"prompt_embeds={prompt_embeds.shape},")

            # refer_image_vae_emb = self.scheduler.scale_model_input(
            #     refer_image_vae_emb, ref_timestep
            # )
            # self.scheduler._step_index = None
            # self.scheduler.is_scale_input_called = False
            referencenet_params = {
                "sample": refer_image_vae_emb,
                "encoder_hidden_states": refer_prompt_embeds,
                "timestep": ref_timestep,
                "num_frames": n_refer_image,
                "return_ndim": 5,
            }
            (
                down_block_refer_embs,
                mid_block_refer_emb,
                refer_self_attn_emb,
            ) = self.referencenet(**referencenet_params)

            # many ways to prepare negative referencenet emb
            # mode 1
            # zero shape like ref_image
            # if do_classifier_free_guidance:
            #     # mode 2:
            #     # if down_block_refer_embs is not None:
            #     #     down_block_refer_embs = [
            #     #         torch.cat([x] * 2) for x in down_block_refer_embs
            #     #     ]
            #     # if mid_block_refer_emb is not None:
            #     #     mid_block_refer_emb = torch.cat([mid_block_refer_emb] * 2)
            #     # if refer_self_attn_emb is not None:
            #     #     refer_self_attn_emb = [
            #     #         torch.cat([x] * 2) for x in refer_self_attn_emb
            #     #     ]

            #     #  mode 3
            #     if down_block_refer_embs is not None:
            #         down_block_refer_embs = [
            #             torch.cat([torch.zeros_like(x), x])
            #             for x in down_block_refer_embs
            #         ]
            #     if mid_block_refer_emb is not None:
            #         mid_block_refer_emb = torch.cat(
            #             [torch.zeros_like(mid_block_refer_emb), mid_block_refer_emb] * 2
            #         )
            #     if refer_self_attn_emb is not None:
            #         refer_self_attn_emb = [
            #             torch.cat([torch.zeros_like(x), x]) for x in refer_self_attn_emb
            #         ]
        else:
            down_block_refer_embs = None
            mid_block_refer_emb = None
            refer_self_attn_emb = None
        if self.print_idx == 0:
            logger.debug(f"down_block_refer_embs={type(down_block_refer_embs)}")
            logger.debug(f"mid_block_refer_emb={type(mid_block_refer_emb)}")
            logger.debug(f"refer_self_attn_emb={type(refer_self_attn_emb)}")
        return down_block_refer_embs, mid_block_refer_emb, refer_self_attn_emb

    def prepare_condition_latents_and_index(
        self,
        condition_images,
        condition_latents,
        video_length,
        batch_size,
        dtype,
        device,
        latent_index,
        vision_condition_latent_index,
    ):
        # prepare condition_latents
        if condition_images is not None and condition_latents is None:
            # condition_latents = self.vae.encode(condition_images).latent_dist.sample()
            condition_latents = self.vae.encode(condition_images).latent_dist.mean
            condition_latents = self.vae.config.scaling_factor * condition_latents
            condition_latents = rearrange(
                condition_latents, "(b t) c h w-> b c t h w", b=batch_size
            )
            if self.print_idx == 0:
                logger.debug(
                    f"condition_latents from condition_images, shape is condition_latents={condition_latents.shape}",
                )
        if condition_latents is not None:
            total_frames = condition_latents.shape[2] + video_length
            if isinstance(condition_latents, np.ndarray):
                condition_latents = torch.from_numpy(condition_latents)
            condition_latents = condition_latents.to(dtype=dtype, device=device)
            # if condition is None, mean condition_latents head, generated video is tail
            if vision_condition_latent_index is not None:
                # vision_condition_latent_index should be list, whose length is condition_latents.shape[2]
                # -1 -> will be converted to condition_latents.shape[2]+video_length
                vision_condition_latent_index_lst = [
                    i_v if i_v != -1 else total_frames - 1
                    for i_v in vision_condition_latent_index
                ]
                vision_condition_latent_index = torch.LongTensor(
                    vision_condition_latent_index_lst,
                ).to(device=device)
                if self.print_idx == 0:
                    logger.debug(
                        f"vision_condition_latent_index {type(vision_condition_latent_index)}, {vision_condition_latent_index}"
                    )
            else:
                # [0, condition_latents.shape[2]]
                vision_condition_latent_index = torch.arange(
                    condition_latents.shape[2], dtype=torch.long, device=device
                )
                vision_condition_latent_index_lst = (
                    vision_condition_latent_index.tolist()
                )
            if latent_index is None:
                # [condition_latents.shape[2], condition_latents.shape[2]+video_length]
                latent_index_lst = sorted(
                    list(
                        set(range(total_frames))
                        - set(vision_condition_latent_index_lst)
                    )
                )
                latent_index = torch.LongTensor(
                    latent_index_lst,
                ).to(device=device)

        if vision_condition_latent_index is not None:
            vision_condition_latent_index = vision_condition_latent_index.to(
                device=device
            )
            if self.print_idx == 0:
                logger.debug(
                    f"pipeline vision_condition_latent_index ={vision_condition_latent_index.shape}, {vision_condition_latent_index}"
                )
        if latent_index is not None:
            latent_index = latent_index.to(device=device)
            if self.print_idx == 0:
                logger.debug(
                    f"pipeline latent_index ={latent_index.shape}, {latent_index}"
                )
        logger.debug(f"condition_latents={type(condition_latents)}")
        logger.debug(f"latent_index={type(latent_index)}")
        logger.debug(
            f"vision_condition_latent_index={type(vision_condition_latent_index)}"
        )
        return condition_latents, latent_index, vision_condition_latent_index

    def prepare_controlnet_and_guidance_parameter(
        self, control_guidance_start, control_guidance_end
    ):
        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(
            control_guidance_end, list
        ):
            control_guidance_start = len(control_guidance_end) * [
                control_guidance_start
            ]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            mult = (
                len(controlnet.nets)
                if isinstance(controlnet, MultiControlNetModel)
                else 1
            )
            control_guidance_start, control_guidance_end = mult * [
                control_guidance_start
            ], mult * [control_guidance_end]
        return controlnet, control_guidance_start, control_guidance_end

    def prepare_controlnet_guess_mode(self, controlnet, guess_mode):
        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions
        return guess_mode

    def prepare_controlnet_image_and_latents(
        self,
        controlnet,
        width,
        height,
        batch_size,
        num_videos_per_prompt,
        device,
        dtype,
        controlnet_latents=None,
        controlnet_condition_latents=None,
        control_image=None,
        controlnet_condition_images=None,
        guess_mode=False,
        do_classifier_free_guidance=False,
    ):
        if isinstance(controlnet, ControlNetModel):
            if controlnet_latents is not None:
                if isinstance(controlnet_latents, np.ndarray):
                    controlnet_latents = torch.from_numpy(controlnet_latents)
                if controlnet_condition_latents is not None:
                    if isinstance(controlnet_condition_latents, np.ndarray):
                        controlnet_condition_latents = torch.from_numpy(
                            controlnet_condition_latents
                        )
                    # TODO：使用index进行concat
                    controlnet_latents = torch.concat(
                        [controlnet_condition_latents, controlnet_latents], dim=2
                    )
                if not guess_mode and do_classifier_free_guidance:
                    controlnet_latents = torch.concat([controlnet_latents] * 2, dim=0)
                controlnet_latents = rearrange(
                    controlnet_latents, "b c t h w->(b t) c h w"
                )
                controlnet_latents = controlnet_latents.to(device=device, dtype=dtype)
                if self.print_idx == 0:
                    logger.debug(
                        f"call, controlnet_latents.shape, f{controlnet_latents.shape}"
                    )
            else:
                # TODO: concat with index
                if isinstance(control_image, np.ndarray):
                    control_image = torch.from_numpy(control_image)
                if controlnet_condition_images is not None:
                    if isinstance(controlnet_condition_images, np.ndarray):
                        controlnet_condition_images = torch.from_numpy(
                            controlnet_condition_images
                        )
                    control_image = torch.concatenate(
                        [controlnet_condition_images, control_image], dim=2
                    )
                control_image = self.prepare_control_image(
                    image=control_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_videos_per_prompt,
                    num_images_per_prompt=num_videos_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = control_image.shape[-2:]
                if self.print_idx == 0:
                    logger.debug(f"call, control_image.shape , {control_image.shape}")

        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []
            # TODO: directly support contronet_latent instead of frames
            if (
                controlnet_latents is not None
                and controlnet_condition_latents is not None
            ):
                raise NotImplementedError
            for i, control_image_ in enumerate(control_image):
                if controlnet_condition_images is not None and isinstance(
                    controlnet_condition_images, list
                ):
                    if isinstance(controlnet_condition_images[i], np.ndarray):
                        control_image_ = np.concatenate(
                            [controlnet_condition_images[i], control_image_], axis=2
                        )
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_videos_per_prompt,
                    num_images_per_prompt=num_videos_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
            height, width = control_image[0].shape[-2:]
        else:
            assert False
        if control_image is not None:
            if not isinstance(control_image, list):
                if self.print_idx == 0:
                    logger.debug(f"control_image shape is {control_image.shape}")
            else:
                if self.print_idx == 0:
                    logger.debug(f"control_image shape is {control_image[0].shape}")

        return control_image, controlnet_latents

    def get_controlnet_emb(
        self,
        run_controlnet,
        guess_mode,
        do_classifier_free_guidance,
        latents,
        prompt_embeds,
        latent_model_input,
        controlnet_keep,
        controlnet_conditioning_scale,
        control_image,
        controlnet_latents,
        i,
        t,
    ):
        if run_controlnet and self.pose_guider is None:
            # controlnet(s) inference
            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(
                    control_model_input, t
                )
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
            if isinstance(controlnet_keep[i], list):
                cond_scale = [
                    c * s
                    for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])
                ]
            else:
                cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
            control_model_input_reshape = rearrange(
                control_model_input, "b c t h w -> (b t) c h w"
            )
            logger.debug(
                f"control_model_input_reshape={control_model_input_reshape.shape}, controlnet_prompt_embeds={controlnet_prompt_embeds.shape}"
            )
            encoder_hidden_states_repeat = align_repeat_tensor_single_dim(
                controlnet_prompt_embeds,
                target_length=control_model_input_reshape.shape[0],
                dim=0,
            )

            if self.print_idx == 0:
                logger.debug(
                    f"control_model_input_reshape={control_model_input_reshape.shape}, "
                    f"encoder_hidden_states_repeat={encoder_hidden_states_repeat.shape}, "
                )
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input_reshape,
                t,
                encoder_hidden_states_repeat,
                controlnet_cond=control_image,
                controlnet_cond_latents=controlnet_latents,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )
            if self.print_idx == 0:
                logger.debug(
                    f"controlnet, len(down_block_res_samples, {len(down_block_res_samples)}",
                )
                for i_tmp, tmp in enumerate(down_block_res_samples):
                    logger.debug(
                        f"controlnet down_block_res_samples i={i_tmp}, down_block_res_sample={tmp.shape}"
                    )
                logger.debug(
                    f"controlnet mid_block_res_sample, {mid_block_res_sample.shape}"
                )
            if guess_mode and do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [
                    torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
                ]
                mid_block_res_sample = torch.cat(
                    [
                        torch.zeros_like(mid_block_res_sample),
                        mid_block_res_sample,
                    ]
                )
        else:
            down_block_res_samples = None
            mid_block_res_sample = None

        return down_block_res_samples, mid_block_res_sample

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video_length: Optional[int],
        prompt: Union[str, List[str]] = None,
        # b c t h w
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        control_image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        # b c t(1) ho wo
        condition_images: Optional[torch.FloatTensor] = None,
        condition_latents: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        add_latents_noise: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_scale_end: float = None,
        guidance_scale_method: str = "linear",
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # b c t(1) hi wi
        controlnet_condition_images: Optional[torch.FloatTensor] = None,
        # b c t(1) ho wo
        controlnet_condition_latents: Optional[torch.FloatTensor] = None,
        controlnet_latents: Union[torch.FloatTensor, np.ndarray] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        need_middle_latents: bool = False,
        w_ind_noise: float = 0.5,
        initial_common_latent: Optional[torch.FloatTensor] = None,
        latent_index: torch.LongTensor = None,
        vision_condition_latent_index: torch.LongTensor = None,
        # noise parameters
        noise_type: str = "random",
        need_img_based_video_noise: bool = False,
        skip_temporal_layer: bool = False,
        img_weight: float = 1e-3,
        need_hist_match: bool = False,
        motion_speed: float = 8.0,
        refer_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        ip_adapter_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        refer_face_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        ip_adapter_scale: float = 1.0,
        facein_scale: float = 1.0,
        ip_adapter_face_scale: float = 1.0,
        ip_adapter_face_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        prompt_only_use_image_prompt: bool = False,
        # serial_denoise parameter start
        record_mid_video_noises: bool = False,
        last_mid_video_noises: List[torch.Tensor] = None,
        record_mid_video_latents: bool = False,
        last_mid_video_latents: List[torch.TensorType] = None,
        video_overlap: int = 1,
        # serial_denoise parameter end
        # parallel_denoise parameter start
        # refer to https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/src/pipelines/pipeline_pose2vid_long.py#L354
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        # parallel_denoise parameter end
        decoder_t_segment: int = 200,
    ):
        r"""
        旨在兼容text2video、text2image、img2img、video2video、是否有controlnet等的通用pipeline。目前仅不支持img2img、video2video。
        支持多片段同时denoise，交叉部分加权平均

        当 skip_temporal_layer 为 False 时, unet 起 video 生成作用；skip_temporal_layer为True时，unet起原image作用。
        当controlnet的所有入参为None，等价于走的是text2video pipeline；
        当 condition_latents、controlnet_condition_images、controlnet_condition_latents为None时，表示不走首帧条件生成的时序condition pipeline
        现在没有考虑对 `num_videos_per_prompt` 的兼容性，不是1可能报错；

        if skip_temporal_layer is False, unet motion layer works, else unet only run text2image layers.
        if parameters about controlnet are None, means text2video pipeline;
        if ondition_latents、controlnet_condition_images、controlnet_condition_latents are None, means only run text2video without vision condition images.
        By now, code works well with `num_videos_per_prpmpt=1`, !=1 may be wrong.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            condition_latents：
                与latents相对应，是Latents的时序condition，一般为首帧，b c t(1) ho wo
                be corresponding to latents, vision condtion latents, usually first frame, should be b c t(1) ho wo.
            controlnet_latents:
                与image二选一，image会被转化成controlnet_latents
                Choose either image or controlnet_latents. If image is chosen, it will be converted to controlnet_latents.
            controlnet_condition_images:
                Optional[torch.FloatTensor]# b c t(1) ho wo，与image相对应，会和image在t通道concat一起，然后转化成 controlnet_latents
                b c t(1) ho wo, corresponding to image, will be concatenated along the t channel with image and then converted to controlnet_latents.
            controlnet_condition_latents: Optional[torch.FloatTensor]:#
                b c t(1) ho wo，会和 controlnet_latents 在t 通道concat一起，转化成 controlnet_latents
                b c t(1) ho wo will be concatenated along the t channel with controlnet_latents and converted to controlnet_latents.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the controlnet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the controlnet stops applying.
            skip_temporal_layer (`bool`: default to False) 为False时，unet起video生成作用,会运行时序生成的block；skip_temporal_layer为True时，unet起原image作用，跳过时序生成的block。
            need_img_based_video_noise: bool = False, 当只有首帧latents时，是否需要扩展为video noise;
            num_videos_per_prompt: now only support 1.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        run_controlnet = control_image is not None or controlnet_latents is not None

        if run_controlnet:
            (
                controlnet,
                control_guidance_start,
                control_guidance_end,
            ) = self.prepare_controlnet_and_guidance_parameter(
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            control_image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.unet.dtype
        # print("pipeline unet dtype", dtype)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if run_controlnet:
            if isinstance(controlnet, MultiControlNetModel) and isinstance(
                controlnet_conditioning_scale, float
            ):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                    controlnet.nets
                )
            guess_mode = self.prepare_controlnet_guess_mode(
                controlnet=controlnet,
                guess_mode=guess_mode,
            )

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        if self.text_encoder is not None:
            prompt_embeds = encode_weighted_prompt(
                self,
                prompt,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                #             lora_scale=text_encoder_lora_scale,
            )
            logger.debug(f"use text_encoder prepare prompt_emb={prompt_embeds.shape}")
        else:
            prompt_embeds = None
        if image is not None:
            image = self.prepare_image(
                image,
                width=width,
                height=height,
                batch_size=batch_size * num_videos_per_prompt,
                num_images_per_prompt=num_videos_per_prompt,
                device=device,
                dtype=dtype,
            )
            if self.print_idx == 0:
                logger.debug(f"image={image.shape}")
        if condition_images is not None:
            condition_images = self.prepare_image(
                condition_images,
                width=width,
                height=height,
                batch_size=batch_size * num_videos_per_prompt,
                num_images_per_prompt=num_videos_per_prompt,
                device=device,
                dtype=dtype,
            )
            if self.print_idx == 0:
                logger.debug(f"condition_images={condition_images.shape}")
        # 4. Prepare image
        if run_controlnet:
            (
                control_image,
                controlnet_latents,
            ) = self.prepare_controlnet_image_and_latents(
                controlnet=controlnet,
                width=width,
                height=height,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                device=device,
                dtype=dtype,
                controlnet_condition_latents=controlnet_condition_latents,
                control_image=control_image,
                controlnet_condition_images=controlnet_condition_images,
                guess_mode=guess_mode,
                do_classifier_free_guidance=do_classifier_free_guidance,
                controlnet_latents=controlnet_latents,
            )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if strength and (image is not None and latents is not None):
            if self.print_idx == 0:
                logger.debug(
                    f"prepare timesteps, with get_timesteps strength={strength}, num_inference_steps={num_inference_steps}"
                )
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps, strength, device
            )
        else:
            if self.print_idx == 0:
                logger.debug(f"prepare timesteps, without get_timesteps")
            timesteps = self.scheduler.timesteps
        latent_timestep = timesteps[:1].repeat(
            batch_size * num_videos_per_prompt
        )  # 6. Prepare latent variables

        (
            condition_latents,
            latent_index,
            vision_condition_latent_index,
        ) = self.prepare_condition_latents_and_index(
            condition_images=condition_images,
            condition_latents=condition_latents,
            video_length=video_length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            latent_index=latent_index,
            vision_condition_latent_index=vision_condition_latent_index,
        )
        if vision_condition_latent_index is None:
            n_vision_cond = 0
        else:
            n_vision_cond = vision_condition_latent_index.shape[0]

        num_channels_latents = self.unet.config.in_channels
        if self.print_idx == 0:
            logger.debug(f"pipeline controlnet, start prepare latents")

        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            video_length=video_length,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
            image=image,
            timestep=latent_timestep,
            w_ind_noise=w_ind_noise,
            initial_common_latent=initial_common_latent,
            noise_type=noise_type,
            add_latents_noise=add_latents_noise,
            need_img_based_video_noise=need_img_based_video_noise,
            condition_latents=condition_latents,
            img_weight=img_weight,
        )
        if self.print_idx == 0:
            logger.debug(f"pipeline controlnet, finish prepare latents={latents.shape}")

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if noise_type == "video_fusion" and "noise_type" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        ):
            extra_step_kwargs["w_ind_noise"] = w_ind_noise
            extra_step_kwargs["noise_type"] = noise_type
            # extra_step_kwargs["noise_offset"] = noise_offset

        # 7.1 Create tensor stating which controlnets to keep
        if run_controlnet:
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(
                    keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
                )
        else:
            controlnet_keep = None
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if skip_temporal_layer:
            self.unet.set_skip_temporal_layers(True)

        n_timesteps = len(timesteps)
        guidance_scale_lst = generate_parameters_with_timesteps(
            start=guidance_scale,
            stop=guidance_scale_end,
            num=n_timesteps,
            method=guidance_scale_method,
        )
        if self.print_idx == 0:
            logger.debug(
                f"guidance_scale_lst, {guidance_scale_method}, {guidance_scale}, {guidance_scale_end}, {guidance_scale_lst}"
            )

        ip_adapter_image_emb = self.get_ip_adapter_image_emb(
            ip_adapter_image=ip_adapter_image,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            height=height,
            width=width,
        )

        # 当前仅当没有ip_adapter时，按照参数 prompt_only_use_image_prompt 要求是否完全替换 image_prompt_emb
        # only if ip_adapter is None and prompt_only_use_image_prompt is True, use image_prompt_emb replace text_prompt
        if (
            ip_adapter_image_emb is not None
            and prompt_only_use_image_prompt
            and not self.unet.ip_adapter_cross_attn
        ):
            prompt_embeds = ip_adapter_image_emb
            logger.debug(f"use ip_adapter_image_emb replace prompt_embeds")
        refer_face_image_emb = self.get_facein_image_emb(
            refer_face_image=refer_face_image,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        ip_adapter_face_emb = self.get_ip_adapter_face_emb(
            refer_face_image=ip_adapter_face_image,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        refer_image_vae_emb = self.get_referencenet_image_vae_emb(
            refer_image=refer_image,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            batch_size=batch_size,
            width=width,
            height=height,
        )

        if self.pose_guider is not None and control_image is not None:
            if self.print_idx == 0:
                logger.debug(f"pose_guider, controlnet_image={control_image.shape}")
            control_image = rearrange(
                control_image, " (b t) c h w->b c t h w", t=video_length
            )
            pose_guider_emb = self.pose_guider(control_image)
            pose_guider_emb = rearrange(pose_guider_emb, "b c t h w-> (b t) c h w")
        else:
            pose_guider_emb = None
        logger.debug(f"prompt_embeds={prompt_embeds.shape}")

        if control_image is not None:
            if isinstance(control_image, list):
                logger.debug(f"control_imageis list, num={len(control_image)}")
                control_image = [
                    rearrange(
                        control_image_tmp,
                        " (b t) c h w->b c t h w",
                        b=(int(do_classifier_free_guidance) * 1 + 1) * batch_size,
                    )
                    for control_image_tmp in control_image
                ]
            else:
                logger.debug(f"control_image={control_image.shape}, before")
                control_image = rearrange(
                    control_image,
                    " (b t) c h w->b c t h w",
                    b=(int(do_classifier_free_guidance) * 1 + 1) * batch_size,
                )
                logger.debug(f"control_image={control_image.shape}, after")

        if controlnet_latents is not None:
            if isinstance(controlnet_latents, list):
                logger.debug(
                    f"controlnet_latents is list, num={len(controlnet_latents)}"
                )
                controlnet_latents = [
                    rearrange(
                        controlnet_latents_tmp,
                        " (b t) c h w->b c t h w",
                        b=(int(do_classifier_free_guidance) * 1 + 1) * batch_size,
                    )
                    for controlnet_latents_tmp in controlnet_latents
                ]
            else:
                logger.debug(f"controlnet_latents={controlnet_latents.shape}, before")
                controlnet_latents = rearrange(
                    controlnet_latents,
                    " (b t) c h w->b c t h w",
                    b=(int(do_classifier_free_guidance) * 1 + 1) * batch_size,
                )
                logger.debug(f"controlnet_latents={controlnet_latents.shape}, after")

        videos_mid = []
        mid_video_noises = [] if record_mid_video_noises else None
        mid_video_latents = [] if record_mid_video_latents else None

        global_context = prepare_global_context(
            context_schedule=context_schedule,
            num_inference_steps=num_inference_steps,
            time_size=latents.shape[2],
            context_frames=context_frames,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_batch_size=context_batch_size,
        )
        logger.debug(
            f"context_schedule={context_schedule}, time_size={latents.shape[2]}, context_frames={context_frames}, context_stride={context_stride}, context_overlap={context_overlap}, context_batch_size={context_batch_size}"
        )
        logger.debug(f"global_context={global_context}")
        # iterative denoise
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 使用 last_mid_video_latents 来影响初始化latent，该部分效果较差，暂留代码
                # use last_mide_video_latents to affect initial latent. works bad, Temporarily reserved
                if i == 0:
                    if record_mid_video_latents:
                        mid_video_latents.append(latents[:, :, -video_overlap:])
                    if record_mid_video_noises:
                        mid_video_noises.append(None)
                    if (
                        last_mid_video_latents is not None
                        and len(last_mid_video_latents) > 0
                    ):
                        if self.print_idx == 1:
                            logger.debug(
                                f"{i}, last_mid_video_latents={last_mid_video_latents[i].shape}"
                            )
                        latents = fuse_part_tensor(
                            last_mid_video_latents[0],
                            latents,
                            video_overlap,
                            weight=0.1,
                            skip_step=0,
                        )
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                if i == 0:
                    (
                        down_block_refer_embs,
                        mid_block_refer_emb,
                        refer_self_attn_emb,
                    ) = self.get_referencenet_emb(
                        refer_image_vae_emb=refer_image_vae_emb,
                        refer_image=refer_image,
                        device=device,
                        dtype=dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        num_videos_per_prompt=num_videos_per_prompt,
                        prompt_embeds=prompt_embeds,
                        ip_adapter_image_emb=ip_adapter_image_emb,
                        batch_size=batch_size,
                        ref_timestep_int=t,
                    )
                for context in global_context:
                    # expand the latents if we are doing classifier free guidance
                    latents_c = torch.cat([latents[:, :, c] for c in context])
                    latent_index_c = (
                        torch.cat([latent_index[c] for c in context])
                        if latent_index is not None
                        else None
                    )
                    latent_model_input = latents_c.to(device).repeat(
                        2 if do_classifier_free_guidance else 1, 1, 1, 1, 1
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    sub_latent_index_c = (
                        torch.LongTensor(
                            torch.arange(latent_index_c.shape[-1]) + n_vision_cond
                        ).to(device=latents_c.device)
                        if latent_index is not None
                        else None
                    )
                    if condition_latents is not None:
                        latent_model_condition = (
                            torch.cat([condition_latents] * 2)
                            if do_classifier_free_guidance
                            else latents
                        )

                        if self.print_idx == 0:
                            logger.debug(
                                f"vision_condition_latent_index, {vision_condition_latent_index.shape}, vision_condition_latent_index"
                            )
                            logger.debug(
                                f"latent_model_condition, {latent_model_condition.shape}"
                            )
                            logger.debug(f"latent_index, {latent_index_c.shape}")
                            logger.debug(
                                f"latent_model_input, {latent_model_input.shape}"
                            )
                            logger.debug(f"sub_latent_index_c, {sub_latent_index_c}")
                        latent_model_input = batch_concat_two_tensor_with_index(
                            data1=latent_model_condition,
                            data1_index=vision_condition_latent_index,
                            data2=latent_model_input,
                            data2_index=sub_latent_index_c,
                            dim=2,
                        )
                    if control_image is not None:
                        if vision_condition_latent_index is not None:
                            # 获取 vision_condition 对应的 control_imgae/control_latent 部分
                            # generate control_image/control_latent corresponding to vision_condition
                            controlnet_condtion_latent_index = (
                                vision_condition_latent_index.clone().cpu().tolist()
                            )
                            if self.print_idx == 0:
                                logger.debug(
                                    f"context={context}, controlnet_condtion_latent_index={controlnet_condtion_latent_index}"
                                )
                            controlnet_context = [
                                controlnet_condtion_latent_index
                                + [c_i + n_vision_cond for c_i in c]
                                for c in context
                            ]
                        else:
                            controlnet_context = context
                        if self.print_idx == 0:
                            logger.debug(
                                f"controlnet_context={controlnet_context}, latent_model_input={latent_model_input.shape}"
                            )
                        if isinstance(control_image, list):
                            control_image_c = [
                                torch.cat(
                                    [
                                        control_image_tmp[:, :, c]
                                        for c in controlnet_context
                                    ]
                                )
                                for control_image_tmp in control_image
                            ]
                            control_image_c = [
                                rearrange(control_image_tmp, " b c t h w-> (b t) c h w")
                                for control_image_tmp in control_image_c
                            ]
                        else:
                            control_image_c = torch.cat(
                                [control_image[:, :, c] for c in controlnet_context]
                            )
                            control_image_c = rearrange(
                                control_image_c, " b c t h w-> (b t) c h w"
                            )
                    else:
                        control_image_c = None
                    if controlnet_latents is not None:
                        if vision_condition_latent_index is not None:
                            # 获取 vision_condition 对应的 control_imgae/control_latent 部分
                            # generate control_image/control_latent corresponding to vision_condition
                            controlnet_condtion_latent_index = (
                                vision_condition_latent_index.clone().cpu().tolist()
                            )
                            if self.print_idx == 0:
                                logger.debug(
                                    f"context={context}, controlnet_condtion_latent_index={controlnet_condtion_latent_index}"
                                )
                            controlnet_context = [
                                controlnet_condtion_latent_index
                                + [c_i + n_vision_cond for c_i in c]
                                for c in context
                            ]
                        else:
                            controlnet_context = context
                        if self.print_idx == 0:
                            logger.debug(
                                f"controlnet_context={controlnet_context}, controlnet_latents={controlnet_latents.shape}, latent_model_input={latent_model_input.shape},"
                            )
                        controlnet_latents_c = torch.cat(
                            [controlnet_latents[:, :, c] for c in controlnet_context]
                        )
                        controlnet_latents_c = rearrange(
                            controlnet_latents_c, " b c t h w-> (b t) c h w"
                        )
                    else:
                        controlnet_latents_c = None
                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.get_controlnet_emb(
                        run_controlnet=run_controlnet,
                        guess_mode=guess_mode,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        latents=latents_c,
                        prompt_embeds=prompt_embeds,
                        latent_model_input=latent_model_input,
                        control_image=control_image_c,
                        controlnet_latents=controlnet_latents_c,
                        controlnet_keep=controlnet_keep,
                        t=t,
                        i=i,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                    )
                    if self.print_idx == 0:
                        logger.debug(
                            f"{i}, latent_model_input={latent_model_input.shape}, sub_latent_index_c={sub_latent_index_c}"
                            f"{vision_condition_latent_index}"
                        )
                    # time.sleep(10)
                    noise_pred_c = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                        sample_index=sub_latent_index_c,
                        vision_conditon_frames_sample_index=vision_condition_latent_index,
                        sample_frame_rate=motion_speed,
                        down_block_refer_embs=down_block_refer_embs,
                        mid_block_refer_emb=mid_block_refer_emb,
                        refer_self_attn_emb=refer_self_attn_emb,
                        vision_clip_emb=ip_adapter_image_emb,
                        face_emb=refer_face_image_emb,
                        ip_adapter_scale=ip_adapter_scale,
                        facein_scale=facein_scale,
                        ip_adapter_face_emb=ip_adapter_face_emb,
                        ip_adapter_face_scale=ip_adapter_face_scale,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        pose_guider_emb=pose_guider_emb,
                    )[0]
                    if condition_latents is not None:
                        noise_pred_c = batch_index_select(
                            noise_pred_c, dim=2, index=sub_latent_index_c
                        ).contiguous()
                    if self.print_idx == 0:
                        logger.debug(
                            f"{i}, latent_model_input={latent_model_input.shape}, noise_pred_c={noise_pred_c.shape}, {len(context)}, {len(context[0])}"
                        )
                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + noise_pred_c
                        counter[:, :, c] = counter[:, :, c] + 1
                noise_pred = noise_pred / counter

                if (
                    last_mid_video_noises is not None
                    and len(last_mid_video_noises) > 0
                    and i <= num_inference_steps // 2  # 是个超参数 super paramter
                ):
                    if self.print_idx == 1:
                        logger.debug(
                            f"{i}, last_mid_video_noises={last_mid_video_noises[i].shape}"
                        )
                    noise_pred = fuse_part_tensor(
                        last_mid_video_noises[i + 1],
                        noise_pred,
                        video_overlap,
                        weight=0.01,
                        skip_step=1,
                    )
                if record_mid_video_noises:
                    mid_video_noises.append(noise_pred[:, :, -video_overlap:])

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale_lst[i] * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.print_idx == 0:
                    logger.debug(
                        f"before step, noise_pred={noise_pred.shape}, {noise_pred.device}, latents={latents.shape}, {latents.device}, t={t}"
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                ).prev_sample

                if (
                    last_mid_video_latents is not None
                    and len(last_mid_video_latents) > 0
                    and i <= 1  # 超参数, super parameter
                ):
                    if self.print_idx == 1:
                        logger.debug(
                            f"{i}, last_mid_video_latents={last_mid_video_latents[i].shape}"
                        )
                    latents = fuse_part_tensor(
                        last_mid_video_latents[i + 1],
                        latents,
                        video_overlap,
                        weight=0.1,
                        skip_step=0,
                    )
                if record_mid_video_latents:
                    mid_video_latents.append(latents[:, :, -video_overlap:])

                if need_middle_latents is True:
                    videos_mid.append(self.decode_latents(latents))
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                self.print_idx += 1

        if condition_latents is not None:
            latents = batch_concat_two_tensor_with_index(
                data1=condition_latents,
                data1_index=vision_condition_latent_index,
                data2=latents,
                data2_index=latent_index,
                dim=2,
            )
        b, c, t, h, w = latents.shape
        num_segments = (t + decoder_t_segment - 1) // decoder_t_segment

        video_segments = []
        # to avoid t chanel too large causing gpu memory error
        # split video latents in slices along t channel, decode each slice, and then concatenate them
        for i in range(num_segments):
            logger.debug(f"Decoding {i} th segment")
            start_t = i * decoder_t_segment
            end_t = min((i + 1) * decoder_t_segment, t)
            latents_segment = latents[:, :, start_t:end_t, :, :]
            video_segment = self.decode_latents(latents_segment)
            video_segments.append(video_segment)
        video_segments_np = np.concatenate(video_segments, axis=2)
        video = torch.from_numpy(video_segments_np)

        if skip_temporal_layer:
            self.unet.set_skip_temporal_layers(False)
        if need_hist_match:
            video[:, :, latent_index, :, :] = self.hist_match_with_vis_cond(
                batch_index_select(video, index=latent_index, dim=2),
                batch_index_select(video, index=vision_condition_latent_index, dim=2),
            )
        # Convert to tensor
        if output_type == "tensor":
            videos_mid = [torch.from_numpy(x) for x in videos_mid]
            video = torch.from_numpy(video)
        else:
            latents = latents.cpu().numpy()

        if not return_dict:
            return (
                video,
                latents,
                videos_mid,
                mid_video_latents,
                mid_video_noises,
            )

        return VideoPipelineOutput(
            videos=video,
            latents=latents,
            videos_mid=videos_mid,
            mid_video_latents=mid_video_latents,
            mid_video_noises=mid_video_noises,
        )
