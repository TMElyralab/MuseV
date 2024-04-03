import copy
from typing import Any, Callable, Dict, Iterable, Union
import PIL
import cv2
import torch
import argparse
import datetime
import logging
import inspect
import math
import os
import shutil
from typing import Dict, List, Optional, Tuple
from pprint import pformat, pprint
from collections import OrderedDict
from dataclasses import dataclass
import gc
import time

import numpy as np
from omegaconf import OmegaConf
from omegaconf import SCMode
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
import pandas as pd
import h5py
from diffusers.models.autoencoder_kl import AutoencoderKL

from diffusers.models.modeling_utils import load_state_dict
from diffusers.utils import (
    logging,
    BaseOutput,
    logging,
)
from diffusers.utils.dummy_pt_objects import ConsistencyDecoderVAE
from diffusers.utils.import_utils import is_xformers_available

from mmcm.utils.seed_util import set_all_seed
from mmcm.vision.data.video_dataset import DecordVideoDataset
from mmcm.vision.process.correct_color import hist_match_video_bcthw
from mmcm.vision.process.image_process import (
    batch_dynamic_crop_resize_images,
    batch_dynamic_crop_resize_images_v2,
)
from mmcm.vision.utils.data_type_util import is_video
from mmcm.vision.feature_extractor.controlnet import load_controlnet_model

from ..schedulers import (
    EulerDiscreteScheduler,
    LCMScheduler,
    DDIMScheduler,
    DDPMScheduler,
)
from ..models.unet_3d_condition import UNet3DConditionModel
from .pipeline_controlnet import (
    MusevControlNetPipeline,
    VideoPipelineOutput as PipelineVideoPipelineOutput,
)
from ..utils.util import save_videos_grid_with_opencv
from ..utils.model_util import (
    update_pipeline_basemodel,
    update_pipeline_lora_model,
    update_pipeline_lora_models,
    update_pipeline_model_parameters,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    latents: Union[torch.Tensor, np.ndarray]
    videos_mid: Union[torch.Tensor, np.ndarray]
    controlnet_cond: Union[torch.Tensor, np.ndarray]
    generated_videos: Union[torch.Tensor, np.ndarray]


def update_controlnet_processor_params(
    src: Union[Dict, List[Dict]], dst: Union[Dict, List[Dict]]
):
    """merge dst into src"""
    if isinstance(src, list) and not isinstance(dst, List):
        dst = [dst] * len(src)
    if isinstance(src, list) and isinstance(dst, list):
        return [
            update_controlnet_processor_params(src[i], dst[i]) for i in range(len(src))
        ]
    if src is None:
        dct = {}
    else:
        dct = copy.deepcopy(src)
    if dst is None:
        dst = {}
    dct.update(dst)
    return dct


class DiffusersPipelinePredictor(object):
    """wraper of diffusers pipeline, support generation function interface. support
    1. text2video: inputs include text, image(optional), refer_image(optional)
    2. video2video:
        1. use controlnet to control spatial
        2. or use video fuse noise to denoise
    """

    def __init__(
        self,
        sd_model_path: str,
        unet: nn.Module,
        controlnet_name: Union[str, List[str]] = None,
        controlnet: nn.Module = None,
        lora_dict: Dict[str, Dict] = None,
        requires_safety_checker: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        # controlnet parameters start
        need_controlnet_processor: bool = True,
        need_controlnet: bool = True,
        image_resolution: int = 512,
        detect_resolution: int = 512,
        include_body: bool = True,
        hand_and_face: bool = None,
        include_face: bool = False,
        include_hand: bool = True,
        negative_embedding: List = None,
        # controlnet parameters end
        enable_xformers_memory_efficient_attention: bool = True,
        lcm_lora_dct: Dict = None,
        referencenet: nn.Module = None,
        ip_adapter_image_proj: nn.Module = None,
        vision_clip_extractor: nn.Module = None,
        face_emb_extractor: nn.Module = None,
        facein_image_proj: nn.Module = None,
        ip_adapter_face_emb_extractor: nn.Module = None,
        ip_adapter_face_image_proj: nn.Module = None,
        vae_model: Optional[Tuple[nn.Module, str]] = None,
        pose_guider: Optional[nn.Module] = None,
        enable_zero_snr: bool = False,
    ) -> None:
        self.sd_model_path = sd_model_path
        self.unet = unet
        self.controlnet_name = controlnet_name
        self.controlnet = controlnet
        self.requires_safety_checker = requires_safety_checker
        self.device = device
        self.dtype = dtype
        self.need_controlnet_processor = need_controlnet_processor
        self.need_controlnet = need_controlnet
        self.need_controlnet_processor = need_controlnet_processor
        self.image_resolution = image_resolution
        self.detect_resolution = detect_resolution
        self.include_body = include_body
        self.hand_and_face = hand_and_face
        self.include_face = include_face
        self.include_hand = include_hand
        self.negative_embedding = negative_embedding
        self.device = device
        self.dtype = dtype
        self.lcm_lora_dct = lcm_lora_dct
        if controlnet is None and controlnet_name is not None:
            controlnet, controlnet_processor, processor_params = load_controlnet_model(
                controlnet_name,
                device=device,
                dtype=dtype,
                need_controlnet_processor=need_controlnet_processor,
                need_controlnet=need_controlnet,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
                include_body=include_body,
                include_face=include_face,
                hand_and_face=hand_and_face,
                include_hand=include_hand,
            )
            self.controlnet_processor = controlnet_processor
            self.controlnet_processor_params = processor_params
            logger.debug(f"init controlnet controlnet_name={controlnet_name}")

        if controlnet is not None:
            controlnet = controlnet.to(device=device, dtype=dtype)
            controlnet.eval()
        if pose_guider is not None:
            pose_guider = pose_guider.to(device=device, dtype=dtype)
            pose_guider.eval()
        unet.to(device=device, dtype=dtype)
        unet.eval()
        if referencenet is not None:
            referencenet.to(device=device, dtype=dtype)
            referencenet.eval()
        if ip_adapter_image_proj is not None:
            ip_adapter_image_proj.to(device=device, dtype=dtype)
            ip_adapter_image_proj.eval()
        if vision_clip_extractor is not None:
            vision_clip_extractor.to(device=device, dtype=dtype)
            vision_clip_extractor.eval()
        if face_emb_extractor is not None:
            face_emb_extractor.to(device=device, dtype=dtype)
            face_emb_extractor.eval()
        if facein_image_proj is not None:
            facein_image_proj.to(device=device, dtype=dtype)
            facein_image_proj.eval()

        if isinstance(vae_model, str):
            # TODO: poor implementation, to improve
            if "consistency" in vae_model:
                vae = ConsistencyDecoderVAE.from_pretrained(vae_model)
            else:
                vae = AutoencoderKL.from_pretrained(vae_model)
        elif isinstance(vae_model, nn.Module):
            vae = vae_model
        else:
            vae = None
        if vae is not None:
            vae.to(device=device, dtype=dtype)
            vae.eval()
        if ip_adapter_face_emb_extractor is not None:
            ip_adapter_face_emb_extractor.to(device=device, dtype=dtype)
            ip_adapter_face_emb_extractor.eval()
        if ip_adapter_face_image_proj is not None:
            ip_adapter_face_image_proj.to(device=device, dtype=dtype)
            ip_adapter_face_image_proj.eval()
        params = {
            "pretrained_model_name_or_path": sd_model_path,
            "controlnet": controlnet,
            "unet": unet,
            "requires_safety_checker": requires_safety_checker,
            "torch_dtype": dtype,
            "torch_device": device,
            "referencenet": referencenet,
            "ip_adapter_image_proj": ip_adapter_image_proj,
            "vision_clip_extractor": vision_clip_extractor,
            "facein_image_proj": facein_image_proj,
            "face_emb_extractor": face_emb_extractor,
            "ip_adapter_face_emb_extractor": ip_adapter_face_emb_extractor,
            "ip_adapter_face_image_proj": ip_adapter_face_image_proj,
            "pose_guider": pose_guider,
        }
        if vae is not None:
            params["vae"] = vae
        pipeline = MusevControlNetPipeline.from_pretrained(**params)
        pipeline = pipeline.to(torch_device=device, torch_dtype=dtype)
        logger.debug(
            f"init pipeline from sd_model_path={sd_model_path}, device={device}, dtype={dtype}"
        )
        if (
            negative_embedding is not None
            and pipeline.text_encoder is not None
            and pipeline.tokenizer is not None
        ):
            for neg_emb_path, neg_token in negative_embedding:
                pipeline.load_textual_inversion(neg_emb_path, token=neg_token)

        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()
        if not enable_zero_snr:
            pipeline.scheduler = EulerDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )
            # pipeline.scheduler = DDIMScheduler.from_config(
            #     pipeline.scheduler.config,
            # 该部分会影响生成视频的亮度，不适用于首帧给定的视频生成
            # this part will change brightness of video, not suitable for image2video mode
            # rescale_betas_zero_snr affect the brightness of the generated video, not suitable for vision condition images mode
            #     # rescale_betas_zero_snr=True,
            # )
            # pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        else:
            # moore scheduler, just for codetest
            pipeline.scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="linear",
                clip_sample=False,
                steps_offset=1,
                ### Zero-SNR params
                prediction_type="v_prediction",
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
            )

        pipeline.enable_vae_slicing()
        self.enable_xformers_memory_efficient_attention = (
            enable_xformers_memory_efficient_attention
        )
        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                pipeline.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
        self.pipeline = pipeline
        self.unload_dict = []  # keep lora state
        if lora_dict is not None:
            self.load_lora(lora_dict=lora_dict)
            logger.debug("load lora {}".format(" ".join(list(lora_dict.keys()))))

        if lcm_lora_dct is not None:
            self.pipeline.scheduler = LCMScheduler.from_config(
                self.pipeline.scheduler.config
            )
            self.load_lora(lora_dict=lcm_lora_dct)
            logger.debug("load lcm lora {}".format(" ".join(list(lcm_lora_dct.keys()))))

        # logger.debug("Unet3Model Parameters")
        # logger.debug(pformat(self.__dict__))

    def load_lora(
        self,
        lora_dict: Dict[str, Dict],
    ):
        self.pipeline, unload_dict = update_pipeline_lora_models(
            self.pipeline, lora_dict, device=self.device
        )
        self.unload_dict += unload_dict

    def unload_lora(self):
        for layer_data in self.unload_dict:
            layer = layer_data["layer"]
            added_weight = layer_data["added_weight"]
            layer.weight.data -= added_weight
        self.unload_dict = []
        gc.collect()
        torch.cuda.empty_cache()

    def update_unet(self, unet: nn.Module):
        self.pipeline.unet = unet.to(device=self.device, dtype=self.dtype)

    def update_sd_model(self, model_path: str, text_model_path: str):
        self.pipeline = update_pipeline_basemodel(
            self.pipeline,
            model_path,
            text_sd_model_path=text_model_path,
            device=self.device,
        )

    def update_sd_model_and_unet(
        self, lora_sd_path: str, lora_path: str, sd_model_path: str = None
    ):
        self.pipeline = update_pipeline_model_parameters(
            self.pipeline,
            model_path=lora_sd_path,
            lora_path=lora_path,
            text_model_path=sd_model_path,
            device=self.device,
        )

    def update_controlnet(self, controlnet_name=Union[str, List[str]]):
        self.pipeline.controlnet = load_controlnet_model(controlnet_name).to(
            device=self.device, dtype=self.dtype
        )

    def run_pipe_text2video(
        self,
        video_length: int,
        prompt: Union[str, List[str]] = None,
        # b c t h w
        height: Optional[int] = None,
        width: Optional[int] = None,
        video_num_inference_steps: int = 50,
        video_guidance_scale: float = 7.5,
        video_guidance_scale_end: float = 3.5,
        video_guidance_scale_method: str = "linear",
        strength: float = 0.8,
        video_negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        same_seed: Optional[Union[int, List[int]]] = None,
        # b c t(1) ho wo
        condition_latents: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        need_middle_latents: bool = False,
        w_ind_noise: float = 0.5,
        initial_common_latent: Optional[torch.FloatTensor] = None,
        latent_index: torch.LongTensor = None,
        vision_condition_latent_index: torch.LongTensor = None,
        n_vision_condition: int = 1,
        noise_type: str = "random",
        max_batch_num: int = 30,
        need_img_based_video_noise: bool = False,
        condition_images: torch.Tensor = None,
        fix_condition_images: bool = False,
        redraw_condition_image: bool = False,
        img_weight: float = 1e-3,
        motion_speed: float = 8.0,
        need_hist_match: bool = False,
        refer_image: Optional[
            Tuple[np.ndarray, torch.Tensor, List[str], List[np.ndarray]]
        ] = None,
        ip_adapter_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        fixed_refer_image: bool = True,
        fixed_ip_adapter_image: bool = True,
        redraw_condition_image_with_ipdapter: bool = True,
        redraw_condition_image_with_referencenet: bool = True,
        refer_face_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        fixed_refer_face_image: bool = True,
        redraw_condition_image_with_facein: bool = True,
        ip_adapter_scale: float = 1.0,
        redraw_condition_image_with_ip_adapter_face: bool = True,
        facein_scale: float = 1.0,
        ip_adapter_face_scale: float = 1.0,
        prompt_only_use_image_prompt: bool = False,
        # serial_denoise parameter start
        record_mid_video_noises: bool = False,
        record_mid_video_latents: bool = False,
        video_overlap: int = 1,
        # serial_denoise parameter end
        # parallel_denoise parameter start
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        # parallel_denoise parameter end
    ):
        """
        generate long video with end2end mode
        1. prepare vision condition image by assingning, redraw, or generation with text2image module with skip_temporal_layer=True;
        2. use image or latest of vision condition image to generate first shot;
        3. use last n (1) image or last latent of last shot as new vision condition latent to generate next shot
        4. repeat n_batch times between 2 and 3

        类似img2img pipeline
        refer_image和ip_adapter_image的来源：
        1. 输入给定；
        2. 当未输入时，纯text2video生成首帧，并赋值更新refer_image和ip_adapter_image;
        3. 当有输入，但是因为redraw更新了首帧时，也需要赋值更新refer_image和ip_adapter_image;

        refer_image和ip_adapter_image的作用：
        1. 当无首帧图像时，用于生成首帧；
        2. 用于生成视频。


        similar to diffusers img2img pipeline.
        three ways to prepare refer_image  and ip_adapter_image
        1. from input parameter
        2. when input paramter is None, use text2video to generate vis cond image, and use as refer_image and ip_adapter_image too.
        3. given from input paramter, but still redraw, update with redrawn vis cond image.
        """
        # crop resize images
        if condition_images is not None:
            logger.debug(
                f"center crop resize condition_images={condition_images.shape}, to height={height}, width={width}"
            )
            condition_images = batch_dynamic_crop_resize_images_v2(
                condition_images,
                target_height=height,
                target_width=width,
            )
        if refer_image is not None:
            logger.debug(
                f"center crop resize refer_image to height={height}, width={width}"
            )
            refer_image = batch_dynamic_crop_resize_images_v2(
                refer_image,
                target_height=height,
                target_width=width,
            )
        if ip_adapter_image is not None:
            logger.debug(
                f"center crop resize ip_adapter_image to height={height}, width={width}"
            )
            ip_adapter_image = batch_dynamic_crop_resize_images_v2(
                ip_adapter_image,
                target_height=height,
                target_width=width,
            )
        if refer_face_image is not None:
            logger.debug(
                f"center crop resize refer_face_image to height={height}, width={width}"
            )
            refer_face_image = batch_dynamic_crop_resize_images_v2(
                refer_face_image,
                target_height=height,
                target_width=width,
            )
        run_video_length = video_length
        # generate vision condition frame start
        # if condition_images is None, generate with refer_image, ip_adapter_image
        # if condition_images not None and need redraw, according to redraw_condition_image_with_ipdapter, redraw_condition_image_with_referencenet, refer_image, ip_adapter_image
        if n_vision_condition > 0:
            if condition_images is None and condition_latents is None:
                logger.debug("run_pipe_text2video, generate first_image")
                (
                    condition_images,
                    condition_latents,
                    _,
                    _,
                    _,
                ) = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    video_length=1,
                    height=height,
                    width=width,
                    return_dict=False,
                    skip_temporal_layer=True,
                    output_type="np",
                    generator=generator,
                    w_ind_noise=w_ind_noise,
                    need_img_based_video_noise=need_img_based_video_noise,
                    refer_image=refer_image
                    if redraw_condition_image_with_referencenet
                    else None,
                    ip_adapter_image=ip_adapter_image
                    if redraw_condition_image_with_ipdapter
                    else None,
                    refer_face_image=refer_face_image
                    if redraw_condition_image_with_facein
                    else None,
                    ip_adapter_scale=ip_adapter_scale,
                    facein_scale=facein_scale,
                    ip_adapter_face_scale=ip_adapter_face_scale,
                    ip_adapter_face_image=refer_face_image
                    if redraw_condition_image_with_ip_adapter_face
                    else None,
                    prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                )
                run_video_length = video_length - 1
            elif (
                condition_images is not None
                and redraw_condition_image
                and condition_latents is None
            ):
                logger.debug("run_pipe_text2video, redraw first_image")

                (
                    condition_images,
                    condition_latents,
                    _,
                    _,
                    _,
                ) = self.pipeline(
                    prompt=prompt,
                    image=condition_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    video_length=condition_images.shape[2],
                    height=height,
                    width=width,
                    return_dict=False,
                    skip_temporal_layer=True,
                    output_type="np",
                    generator=generator,
                    w_ind_noise=w_ind_noise,
                    need_img_based_video_noise=need_img_based_video_noise,
                    refer_image=refer_image
                    if redraw_condition_image_with_referencenet
                    else None,
                    ip_adapter_image=ip_adapter_image
                    if redraw_condition_image_with_ipdapter
                    else None,
                    refer_face_image=refer_face_image
                    if redraw_condition_image_with_facein
                    else None,
                    ip_adapter_scale=ip_adapter_scale,
                    facein_scale=facein_scale,
                    ip_adapter_face_scale=ip_adapter_face_scale,
                    ip_adapter_face_image=refer_face_image
                    if redraw_condition_image_with_ip_adapter_face
                    else None,
                    prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                )
        else:
            condition_images = None
            condition_latents = None
        # generate vision condition frame end

        # refer_image and ip_adapter_image, update mode from 2 and 3 as mentioned above start
        if (
            refer_image is not None
            and redraw_condition_image
            and condition_images is not None
        ):
            refer_image = condition_images * 255.0
            logger.debug(f"update refer_image because of redraw_condition_image")
        elif (
            refer_image is None
            and self.pipeline.referencenet is not None
            and condition_images is not None
        ):
            refer_image = condition_images * 255.0
            logger.debug(f"update refer_image because of generate first_image")

        # ipadapter_image
        if (
            ip_adapter_image is not None
            and redraw_condition_image
            and condition_images is not None
        ):
            ip_adapter_image = condition_images * 255.0
            logger.debug(f"update ip_adapter_image because of redraw_condition_image")
        elif (
            ip_adapter_image is None
            and self.pipeline.ip_adapter_image_proj is not None
            and condition_images is not None
        ):
            ip_adapter_image = condition_images * 255.0
            logger.debug(f"update ip_adapter_image because of generate first_image")
        # refer_image and ip_adapter_image, update mode from 2 and 3 as mentioned above end

        # refer_face_image, update mode from 2 and 3 as mentioned above start
        if (
            refer_face_image is not None
            and redraw_condition_image
            and condition_images is not None
        ):
            refer_face_image = condition_images * 255.0
            logger.debug(f"update refer_face_image because of redraw_condition_image")
        elif (
            refer_face_image is None
            and self.pipeline.facein_image_proj is not None
            and condition_images is not None
        ):
            refer_face_image = condition_images * 255.0
            logger.debug(f"update face_image because of generate first_image")
            # refer_face_image, update mode from 2 and 3 as mentioned above end

        last_mid_video_noises = None
        last_mid_video_latents = None
        initial_common_latent = None

        out_videos = []
        for i_batch in range(max_batch_num):
            logger.debug(f"sd_pipeline_predictor, run_pipe_text2video: {i_batch}")
            if max_batch_num is not None and i_batch == max_batch_num:
                break

            if i_batch == 0:
                result_overlap = 0
            else:
                if n_vision_condition > 0:
                    # ignore condition_images if condition_latents is not None in pipeline
                    if not fix_condition_images:
                        logger.debug(f"{i_batch}, update condition_latents")
                        condition_latents = out_latents_batch[
                            :, :, -n_vision_condition:, :, :
                        ]
                    else:
                        logger.debug(f"{i_batch}, do not update condition_latents")
                result_overlap = n_vision_condition

                if not fixed_refer_image and n_vision_condition > 0:
                    logger.debug("ref_image use last frame of last generated out video")
                    refer_image = out_batch[:, :, -n_vision_condition:, :, :] * 255.0
                else:
                    logger.debug("use given fixed ref_image")

                if not fixed_ip_adapter_image and n_vision_condition > 0:
                    logger.debug(
                        "ip_adapter_image use last frame of last generated out video"
                    )
                    ip_adapter_image = (
                        out_batch[:, :, -n_vision_condition:, :, :] * 255.0
                    )
                else:
                    logger.debug("use given fixed ip_adapter_image")

                if not fixed_refer_face_image and n_vision_condition > 0:
                    logger.debug(
                        "refer_face_image use last frame of last generated out video"
                    )
                    refer_face_image = (
                        out_batch[:, :, -n_vision_condition:, :, :] * 255.0
                    )
                else:
                    logger.debug("use given fixed ip_adapter_image")

                run_video_length = video_length
            if same_seed is not None:
                _, generator = set_all_seed(same_seed)

            out = self.pipeline(
                video_length=run_video_length,  # int
                prompt=prompt,
                num_inference_steps=video_num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                condition_images=condition_images,
                condition_latents=condition_latents,  # b co t(1) ho wo
                skip_temporal_layer=False,
                output_type="np",
                noise_type=noise_type,
                negative_prompt=video_negative_prompt,
                guidance_scale=video_guidance_scale,
                guidance_scale_end=video_guidance_scale_end,
                guidance_scale_method=video_guidance_scale_method,
                w_ind_noise=w_ind_noise,
                need_img_based_video_noise=need_img_based_video_noise,
                img_weight=img_weight,
                motion_speed=motion_speed,
                vision_condition_latent_index=vision_condition_latent_index,
                refer_image=refer_image,
                ip_adapter_image=ip_adapter_image,
                refer_face_image=refer_face_image,
                ip_adapter_scale=ip_adapter_scale,
                facein_scale=facein_scale,
                ip_adapter_face_scale=ip_adapter_face_scale,
                ip_adapter_face_image=refer_face_image,
                prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                initial_common_latent=initial_common_latent,
                # serial_denoise parameter start
                record_mid_video_noises=record_mid_video_noises,
                last_mid_video_noises=last_mid_video_noises,
                record_mid_video_latents=record_mid_video_latents,
                last_mid_video_latents=last_mid_video_latents,
                video_overlap=video_overlap,
                # serial_denoise parameter end
                # parallel_denoise parameter start
                context_schedule=context_schedule,
                context_frames=context_frames,
                context_stride=context_stride,
                context_overlap=context_overlap,
                context_batch_size=context_batch_size,
                interpolation_factor=interpolation_factor,
                # parallel_denoise parameter end
            )
            logger.debug(
                f"run_pipe_text2video, out.videos.shape, i_batch={i_batch}, videos={out.videos.shape}, result_overlap={result_overlap}"
            )
            out_batch = out.videos[:, :, result_overlap:, :, :]
            out_latents_batch = out.latents[:, :, result_overlap:, :, :]
            out_videos.append(out_batch)

        out_videos = np.concatenate(out_videos, axis=2)
        if need_hist_match:
            out_videos[:, :, 1:, :, :] = hist_match_video_bcthw(
                out_videos[:, :, 1:, :, :], out_videos[:, :, :1, :, :], value=255.0
            )
        return out_videos

    def run_pipe_with_latent_input(
        self,
    ):
        pass

    def run_pipe_middle2video_with_middle(self, middle: Tuple[str, Iterable]):
        pass

    def run_pipe_video2video(
        self,
        video: Tuple[str, Iterable],
        time_size: int = None,
        sample_rate: int = None,
        overlap: int = None,
        step: int = None,
        prompt: Union[str, List[str]] = None,
        # b c t h w
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        video_num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        video_guidance_scale: float = 7.5,
        video_guidance_scale_end: float = 3.5,
        video_guidance_scale_method: str = "linear",
        video_negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        controlnet_latents: Union[torch.FloatTensor, np.ndarray] = None,
        # b c t(1) hi wi
        controlnet_condition_images: Optional[torch.FloatTensor] = None,
        # b c t(1) ho wo
        controlnet_condition_latents: Optional[torch.FloatTensor] = None,
        # b c t(1) ho wo
        condition_latents: Optional[torch.FloatTensor] = None,
        condition_images: Optional[torch.FloatTensor] = None,
        fix_condition_images: bool = False,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
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
        img_weight: float = 0.001,
        initial_common_latent: Optional[torch.FloatTensor] = None,
        latent_index: torch.LongTensor = None,
        vision_condition_latent_index: torch.LongTensor = None,
        noise_type: str = "random",
        controlnet_processor_params: Dict = None,
        need_return_videos: bool = False,
        need_return_condition: bool = False,
        max_batch_num: int = 30,
        strength: float = 0.8,
        video_strength: float = 0.8,
        need_video2video: bool = False,
        need_img_based_video_noise: bool = False,
        need_hist_match: bool = False,
        end_to_end: bool = True,
        refer_image: Optional[
            Tuple[np.ndarray, torch.Tensor, List[str], List[np.ndarray]]
        ] = None,
        ip_adapter_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        fixed_refer_image: bool = True,
        fixed_ip_adapter_image: bool = True,
        redraw_condition_image: bool = False,
        redraw_condition_image_with_ipdapter: bool = True,
        redraw_condition_image_with_referencenet: bool = True,
        refer_face_image: Optional[Tuple[torch.Tensor, np.array]] = None,
        fixed_refer_face_image: bool = True,
        redraw_condition_image_with_facein: bool = True,
        ip_adapter_scale: float = 1.0,
        facein_scale: float = 1.0,
        ip_adapter_face_scale: float = 1.0,
        redraw_condition_image_with_ip_adapter_face: bool = False,
        n_vision_condition: int = 1,
        prompt_only_use_image_prompt: bool = False,
        motion_speed: float = 8.0,
        # serial_denoise parameter start
        record_mid_video_noises: bool = False,
        record_mid_video_latents: bool = False,
        video_overlap: int = 1,
        # serial_denoise parameter end
        # parallel_denoise parameter start
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        # parallel_denoise parameter end
        # 支持 video_path 时多种输入
        # TODO:// video_has_condition =False，当且仅支持 video_is_middle=True, 待后续重构
        # TODO:// when video_has_condition =False, video_is_middle should be True.
        video_is_middle: bool = False,
        video_has_condition: bool = True,
    ):
        """
        类似controlnet text2img pipeline。 输入视频，用视频得到controlnet condition。
        目前仅支持time_size == step，overlap=0
        输出视频长度=输入视频长度

        similar to controlnet text2image pipeline, generate video with controlnet condition from given video.
        By now, sliding window only support time_size == step, overlap = 0.
        """
        if isinstance(video, str):
            video_reader = DecordVideoDataset(
                video,
                time_size=time_size,
                step=step,
                overlap=overlap,
                sample_rate=sample_rate,
                device="cpu",
                data_type="rgb",
                channels_order="c t h w",
                drop_last=True,
            )
        else:
            video_reader = video
        videos = [] if need_return_videos else None
        out_videos = []
        out_condition = (
            []
            if need_return_condition and self.pipeline.controlnet is not None
            else None
        )
        # crop resize images
        if condition_images is not None:
            logger.debug(
                f"center crop resize condition_images={condition_images.shape}, to height={height}, width={width}"
            )
            condition_images = batch_dynamic_crop_resize_images_v2(
                condition_images,
                target_height=height,
                target_width=width,
            )
        if refer_image is not None:
            logger.debug(
                f"center crop resize refer_image to height={height}, width={width}"
            )
            refer_image = batch_dynamic_crop_resize_images_v2(
                refer_image,
                target_height=height,
                target_width=width,
            )
        if ip_adapter_image is not None:
            logger.debug(
                f"center crop resize ip_adapter_image to height={height}, width={width}"
            )
            ip_adapter_image = batch_dynamic_crop_resize_images_v2(
                ip_adapter_image,
                target_height=height,
                target_width=width,
            )
        if refer_face_image is not None:
            logger.debug(
                f"center crop resize refer_face_image to height={height}, width={width}"
            )
            refer_face_image = batch_dynamic_crop_resize_images_v2(
                refer_face_image,
                target_height=height,
                target_width=width,
            )
        first_image = None
        last_mid_video_noises = None
        last_mid_video_latents = None
        initial_common_latent = None
        # initial_common_latent = torch.randn((1, 4, 1, 112, 64)).to(
        #     device=self.device, dtype=self.dtype
        # )

        for i_batch, item in enumerate(video_reader):
            logger.debug(f"\n sd_pipeline_predictor, run_pipe_video2video: {i_batch}")
            if max_batch_num is not None and i_batch == max_batch_num:
                break
            # read and prepare video batch
            batch = item.data
            batch = batch_dynamic_crop_resize_images(
                batch,
                target_height=height,
                target_width=width,
            )

            batch = batch[np.newaxis, ...]
            batch_size, channel, video_length, video_height, video_width = batch.shape
            # extract controlnet middle
            if self.pipeline.controlnet is not None:
                batch = rearrange(batch, "b c t h w-> (b t) h w c")
                controlnet_processor_params = update_controlnet_processor_params(
                    src=self.controlnet_processor_params,
                    dst=controlnet_processor_params,
                )
                if not video_is_middle:
                    batch_condition = self.controlnet_processor(
                        data=batch,
                        data_channel_order="b h w c",
                        target_height=height,
                        target_width=width,
                        return_type="np",
                        return_data_channel_order="b c h w",
                        input_rgb_order="rgb",
                        processor_params=controlnet_processor_params,
                    )
                else:
                    # TODO: 临时用于可视化输入的 controlnet middle 序列，后续待拆到 middl2video中，也可以增加参数支持
                    # TODO: only use video_path is controlnet middle output, to improved
                    batch_condition = rearrange(
                        copy.deepcopy(batch), " b h w c-> b c h w"
                    )

                # 当前仅当 输入是 middle、condition_image的pose在middle首帧之前，需要重新生成condition_images的pose并绑定到middle_batch上
                # when video_path is middle seq and condition_image is not aligned with middle seq,
                # regenerate codntion_images pose, and then concat into middle_batch,
                if (
                    i_batch == 0
                    and not video_has_condition
                    and video_is_middle
                    and condition_images is not None
                ):
                    condition_images_reshape = rearrange(
                        condition_images, "b c t h w-> (b t) h w c"
                    )
                    condition_images_condition = self.controlnet_processor(
                        data=condition_images_reshape,
                        data_channel_order="b h w c",
                        target_height=height,
                        target_width=width,
                        return_type="np",
                        return_data_channel_order="b c h w",
                        input_rgb_order="rgb",
                        processor_params=controlnet_processor_params,
                    )
                    condition_images_condition = rearrange(
                        condition_images_condition,
                        "(b t) c h w-> b c t h w",
                        b=batch_size,
                    )
                else:
                    condition_images_condition = None
                if not isinstance(batch_condition, list):
                    batch_condition = rearrange(
                        batch_condition, "(b t) c h w-> b c t h w", b=batch_size
                    )
                    if condition_images_condition is not None:
                        batch_condition = np.concatenate(
                            [
                                condition_images_condition,
                                batch_condition,
                            ],
                            axis=2,
                        )
                        # 此时 batch_condition 比 batch 多了一帧，为了最终视频能 concat 存储，替换下
                        # 当前仅适用于  condition_images_condition 不为None
                        # when condition_images_condition is not None,  batch_condition has more frames than batch
                        batch = rearrange(batch_condition, "b c t h w ->(b t) h w c")
                else:
                    batch_condition = [
                        rearrange(x, "(b t) c h w-> b c t h w", b=batch_size)
                        for x in batch_condition
                    ]
                    if condition_images_condition is not None:
                        batch_condition = [
                            np.concatenate(
                                [condition_images_condition, batch_condition_tmp],
                                axis=2,
                            )
                            for batch_condition_tmp in batch_condition
                        ]
                batch = rearrange(batch, "(b t) h w c -> b c t h w", b=batch_size)
            else:
                batch_condition = None
            # condition [0,255]
            # latent: [0,1]
            # 按需求生成多个片段，
            # generate multi video_shot
            # 第一个片段 会特殊处理，需要生成首帧
            # first shot is special because of first frame.
            # 后续片段根据拿前一个片段结果，首尾相连的方式生成。
            # use last frame of last shot as the first frame of the current shot
            # TODO: 当前独立拆开实现，待后续融合到一起实现
            # TODO: to optimize  implementation way
            if n_vision_condition == 0:
                actual_video_length = video_length
                control_image = batch_condition
                first_image_controlnet_condition = None
                first_image_latents = None
                if need_video2video:
                    video = batch
                else:
                    video = None
                result_overlap = 0
            else:
                if i_batch == 0:
                    if self.pipeline.controlnet is not None:
                        if not isinstance(batch_condition, list):
                            first_image_controlnet_condition = batch_condition[
                                :, :, :1, :, :
                            ]
                        else:
                            first_image_controlnet_condition = [
                                x[:, :, :1, :, :] for x in batch_condition
                            ]
                    else:
                        first_image_controlnet_condition = None
                    if need_video2video:
                        if condition_images is None:
                            video = batch[:, :, :1, :, :]
                        else:
                            video = condition_images
                    else:
                        video = None
                    if condition_images is not None and not redraw_condition_image:
                        first_image = condition_images
                        first_image_latents = None
                    else:
                        (
                            first_image,
                            first_image_latents,
                            _,
                            _,
                            _,
                        ) = self.pipeline(
                            prompt=prompt,
                            image=video,
                            control_image=first_image_controlnet_condition,
                            num_inference_steps=num_inference_steps,
                            video_length=1,
                            height=height,
                            width=width,
                            return_dict=False,
                            skip_temporal_layer=True,
                            output_type="np",
                            generator=generator,
                            negative_prompt=negative_prompt,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            control_guidance_start=control_guidance_start,
                            control_guidance_end=control_guidance_end,
                            w_ind_noise=w_ind_noise,
                            strength=strength,
                            refer_image=refer_image
                            if redraw_condition_image_with_referencenet
                            else None,
                            ip_adapter_image=ip_adapter_image
                            if redraw_condition_image_with_ipdapter
                            else None,
                            refer_face_image=refer_face_image
                            if redraw_condition_image_with_facein
                            else None,
                            ip_adapter_scale=ip_adapter_scale,
                            facein_scale=facein_scale,
                            ip_adapter_face_scale=ip_adapter_face_scale,
                            ip_adapter_face_image=refer_face_image
                            if redraw_condition_image_with_ip_adapter_face
                            else None,
                            prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                        )
                        if refer_image is not None:
                            refer_image = first_image * 255.0
                        if ip_adapter_image is not None:
                            ip_adapter_image = first_image * 255.0
                        # 首帧用于后续推断可以直接用first_image_latent不需要 first_image了
                        first_image = None
                    if self.pipeline.controlnet is not None:
                        if not isinstance(batch_condition, list):
                            control_image = batch_condition[:, :, 1:, :, :]
                            logger.debug(f"control_image={control_image.shape}")
                        else:
                            control_image = [x[:, :, 1:, :, :] for x in batch_condition]
                    else:
                        control_image = None

                    actual_video_length = time_size - int(video_has_condition)
                    if need_video2video:
                        video = batch[:, :, 1:, :, :]
                    else:
                        video = None

                    result_overlap = 0
                else:
                    actual_video_length = time_size
                    if self.pipeline.controlnet is not None:
                        if not fix_condition_images:
                            logger.debug(
                                f"{i_batch}, update first_image_controlnet_condition"
                            )

                            if not isinstance(last_batch_condition, list):
                                first_image_controlnet_condition = last_batch_condition[
                                    :, :, -1:, :, :
                                ]
                            else:
                                first_image_controlnet_condition = [
                                    x[:, :, -1:, :, :] for x in last_batch_condition
                                ]
                        else:
                            logger.debug(
                                f"{i_batch}, do not update first_image_controlnet_condition"
                            )
                        control_image = batch_condition
                    else:
                        control_image = None
                        first_image_controlnet_condition = None
                    if not fix_condition_images:
                        logger.debug(f"{i_batch}, update condition_images")
                        first_image_latents = out_latents_batch[:, :, -1:, :, :]
                    else:
                        logger.debug(f"{i_batch}, do not update condition_images")

                    if need_video2video:
                        video = batch
                    else:
                        video = None
                    result_overlap = 1

                    # 更新 ref_image和 ipadapter_image
                    if not fixed_refer_image:
                        logger.debug(
                            "ref_image use last frame of last generated out video"
                        )
                        refer_image = (
                            out_batch[:, :, -n_vision_condition:, :, :] * 255.0
                        )
                    else:
                        logger.debug("use given fixed ref_image")

                    if not fixed_ip_adapter_image:
                        logger.debug(
                            "ip_adapter_image use last frame of last generated out video"
                        )
                        ip_adapter_image = (
                            out_batch[:, :, -n_vision_condition:, :, :] * 255.0
                        )
                    else:
                        logger.debug("use given fixed ip_adapter_image")

                    # face image
                    if not fixed_ip_adapter_image:
                        logger.debug(
                            "refer_face_image use last frame of last generated out video"
                        )
                        refer_face_image = (
                            out_batch[:, :, -n_vision_condition:, :, :] * 255.0
                        )
                    else:
                        logger.debug("use given fixed ip_adapter_image")

            out = self.pipeline(
                video_length=actual_video_length,  # int
                prompt=prompt,
                num_inference_steps=video_num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                image=video,
                control_image=control_image,  # b ci(3) t hi wi
                controlnet_condition_images=first_image_controlnet_condition,  # b ci(3) t(1) hi wi
                # controlnet_condition_images=np.zeros_like(
                #     first_image_controlnet_condition
                # ),  # b ci(3) t(1) hi wi
                condition_images=first_image,
                condition_latents=first_image_latents,  # b co t(1) ho wo
                skip_temporal_layer=False,
                output_type="np",
                noise_type=noise_type,
                negative_prompt=video_negative_prompt,
                need_img_based_video_noise=need_img_based_video_noise,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                w_ind_noise=w_ind_noise,
                img_weight=img_weight,
                motion_speed=video_reader.sample_rate,
                guidance_scale=video_guidance_scale,
                guidance_scale_end=video_guidance_scale_end,
                guidance_scale_method=video_guidance_scale_method,
                strength=video_strength,
                refer_image=refer_image,
                ip_adapter_image=ip_adapter_image,
                refer_face_image=refer_face_image,
                ip_adapter_scale=ip_adapter_scale,
                facein_scale=facein_scale,
                ip_adapter_face_scale=ip_adapter_face_scale,
                ip_adapter_face_image=refer_face_image,
                prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                initial_common_latent=initial_common_latent,
                # serial_denoise parameter start
                record_mid_video_noises=record_mid_video_noises,
                last_mid_video_noises=last_mid_video_noises,
                record_mid_video_latents=record_mid_video_latents,
                last_mid_video_latents=last_mid_video_latents,
                video_overlap=video_overlap,
                # serial_denoise parameter end
                # parallel_denoise parameter start
                context_schedule=context_schedule,
                context_frames=context_frames,
                context_stride=context_stride,
                context_overlap=context_overlap,
                context_batch_size=context_batch_size,
                interpolation_factor=interpolation_factor,
                # parallel_denoise parameter end
            )
            last_batch = batch
            last_batch_condition = batch_condition
            last_mid_video_latents = out.mid_video_latents
            last_mid_video_noises = out.mid_video_noises
            out_batch = out.videos[:, :, result_overlap:, :, :]
            out_latents_batch = out.latents[:, :, result_overlap:, :, :]
            out_videos.append(out_batch)
            if need_return_videos:
                videos.append(batch)
            if out_condition is not None:
                out_condition.append(batch_condition)

        out_videos = np.concatenate(out_videos, axis=2)
        if need_return_videos:
            videos = np.concatenate(videos, axis=2)
        if out_condition is not None:
            if not isinstance(out_condition[0], list):
                out_condition = np.concatenate(out_condition, axis=2)
            else:
                out_condition = [
                    [out_condition[j][i] for j in range(len(out_condition))]
                    for i in range(len(out_condition[0]))
                ]
                out_condition = [np.concatenate(x, axis=2) for x in out_condition]
        if need_hist_match:
            videos[:, :, 1:, :, :] = hist_match_video_bcthw(
                videos[:, :, 1:, :, :], videos[:, :, :1, :, :], value=255.0
            )
        return out_videos, out_condition, videos
