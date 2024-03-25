from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import warnings
import os


import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
import PIL
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.init as init
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import is_compiled_module


class ControlnetPredictor(object):
    def __init__(self, controlnet_model_path: str, *args, **kwargs):
        """Controlnet 推断函数，用于提取 controlnet backbone的emb，避免训练时重复抽取
            Controlnet inference predictor, used to extract the emb of the controlnet backbone to avoid repeated extraction during training
        Args:
            controlnet_model_path (str): controlnet 模型路径. controlnet model path.
        """
        super(ControlnetPredictor, self).__init__(*args, **kwargs)
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path,
        )

    def prepare_image(
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
        if height is None:
            height = image.shape[-2]
        if width is None:
            width = image.shape[-1]
        width, height = (
            x - x % self.control_image_processor.vae_scale_factor
            for x in (width, height)
        )
        image = rearrange(image, "b c t h w-> (b t) c h w")
        image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0

        image = (
            torch.nn.functional.interpolate(
                image,
                size=(height, width),
                mode="bilinear",
            ),
        )

        do_normalize = self.control_image_processor.config.do_normalize
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.control_image_processor.normalize(image)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: str,
        dtype: torch.dtype,
        timesteps: List[float],
        i: int,
        scheduler: KarrasDiffusionSchedulers,
        prompt_embeds: torch.Tensor,
        do_classifier_free_guidance: bool = False,
        # 2b co t ho wo
        latent_model_input: torch.Tensor = None,
        # b co t ho wo
        latents: torch.Tensor = None,
        # b c t h w
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        # b c t(1) hi wi
        controlnet_condition_frames: Optional[torch.FloatTensor] = None,
        # b c t ho wo
        controlnet_latents: Union[torch.FloatTensor, np.ndarray] = None,
        # b c t(1) ho wo
        controlnet_condition_latents: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        return_dict: bool = True,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        latent_index: torch.LongTensor = None,
        vision_condition_latent_index: torch.LongTensor = None,
        **kwargs,
    ):
        assert (
            image is None and controlnet_latents is None
        ), "should set one of image and controlnet_latents"

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

        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            if (
                controlnet_latents is not None
                and controlnet_condition_latents is not None
            ):
                if isinstance(controlnet_latents, np.ndarray):
                    controlnet_latents = torch.from_numpy(controlnet_latents)
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
            else:
                # TODO：使用index进行concat
                # TODO: concat with index
                if controlnet_condition_frames is not None:
                    if isinstance(controlnet_condition_frames, np.ndarray):
                        image = np.concatenate(
                            [controlnet_condition_frames, image], axis=2
                        )
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_videos_per_prompt,
                    num_images_per_prompt=num_videos_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []
            # TODO: 支持直接使用controlnet_latent而不是frames
            # TODO: support using controlnet_latent directly instead of frames
            if controlnet_latents is not None:
                raise NotImplementedError
            else:
                for i, image_ in enumerate(image):
                    if controlnet_condition_frames is not None and isinstance(
                        controlnet_condition_frames, list
                    ):
                        if isinstance(controlnet_condition_frames[i], np.ndarray):
                            image_ = np.concatenate(
                                [controlnet_condition_frames[i], image_], axis=2
                            )
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_videos_per_prompt,
                        num_images_per_prompt=num_videos_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
        else:
            assert False

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        t = timesteps[i]

        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latents
            control_model_input = scheduler.scale_model_input(control_model_input, t)
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        else:
            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds
        if isinstance(controlnet_keep[i], list):
            cond_scale = [
                c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])
            ]
        else:
            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
        control_model_input_reshape = rearrange(
            control_model_input, "b c t h w -> (b t) c h w"
        )
        encoder_hidden_states_repeat = repeat(
            controlnet_prompt_embeds,
            "b n q->(b t) n q",
            t=control_model_input.shape[2],
        )

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input_reshape,
            t,
            encoder_hidden_states_repeat,
            controlnet_cond=image,
            controlnet_cond_latents=controlnet_latents,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )

        return down_block_res_samples, mid_block_res_sample


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
        print(
            f"loaded PoseGuider's pretrained weights from {pretrained_model_path} ..."
        )

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = PoseGuider(
            conditioning_embedding_channels=conditioning_embedding_channels,
            conditioning_channels=conditioning_channels,
            block_out_channels=block_out_channels,
        )

        m, u = model.load_state_dict(state_dict, strict=False)
        # print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        params = [p.numel() for n, p in model.named_parameters()]
        print(f"### PoseGuider's Parameters: {sum(params) / 1e6} M")

        return model
