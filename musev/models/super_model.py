from __future__ import annotations

import logging

from typing import Any, Dict, Tuple, Union, Optional
from einops import rearrange, repeat
from torch import nn
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin, load_state_dict

from ..data.data_util import align_repeat_tensor_single_dim

from .unet_3d_condition import UNet3DConditionModel
from .referencenet import ReferenceNet2D
from ip_adapter.ip_adapter import ImageProjModel

logger = logging.getLogger(__name__)


class SuperUNet3DConditionModel(nn.Module):
    """封装了各种子模型的超模型，与 diffusers 的 pipeline 很像，只不过这里是模型定义。
    主要作用
    1. 将支持controlnet、referencenet等功能的计算封装起来，简洁些；
    2. 便于 accelerator 的分布式训练；

    wrap the sub-models, such as unet, referencenet, controlnet, vae, text_encoder, tokenizer, text_emb_extractor, clip_vision_extractor, ip_adapter_image_proj
    1. support controlnet, referencenet, etc.
    2. support accelerator distributed training
    """

    _supports_gradient_checkpointing = True
    print_idx = 0

    # @register_to_config
    def __init__(
        self,
        unet: nn.Module,
        referencenet: nn.Module = None,
        controlnet: nn.Module = None,
        vae: nn.Module = None,
        text_encoder: nn.Module = None,
        tokenizer: nn.Module = None,
        text_emb_extractor: nn.Module = None,
        clip_vision_extractor: nn.Module = None,
        ip_adapter_image_proj: nn.Module = None,
    ) -> None:
        """_summary_

        Args:
            unet (nn.Module): _description_
            referencenet (nn.Module, optional): _description_. Defaults to None.
            controlnet (nn.Module, optional): _description_. Defaults to None.
            vae (nn.Module, optional): _description_. Defaults to None.
            text_encoder (nn.Module, optional): _description_. Defaults to None.
            tokenizer (nn.Module, optional): _description_. Defaults to None.
            text_emb_extractor (nn.Module, optional): wrap text_encoder and tokenizer for str2emb. Defaults to None.
            clip_vision_extractor (nn.Module, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.unet = unet
        self.referencenet = referencenet
        self.controlnet = controlnet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_emb_extractor = text_emb_extractor
        self.clip_vision_extractor = clip_vision_extractor
        self.ip_adapter_image_proj = ip_adapter_image_proj

    def forward(
        self,
        unet_params: Dict,
        encoder_hidden_states: torch.Tensor,
        referencenet_params: Dict = None,
        controlnet_params: Dict = None,
        controlnet_scale: float = 1.0,
        vision_clip_emb: Union[torch.Tensor, None] = None,
        prompt_only_use_image_prompt: bool = False,
    ):
        """_summary_

        Args:
            unet_params (Dict): _description_
            encoder_hidden_states (torch.Tensor): b t n d
            referencenet_params (Dict, optional): _description_. Defaults to None.
            controlnet_params (Dict, optional): _description_. Defaults to None.
            controlnet_scale (float, optional): _description_. Defaults to 1.0.
            vision_clip_emb (Union[torch.Tensor, None], optional): b t d. Defaults to None.
            prompt_only_use_image_prompt (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        batch_size = unet_params["sample"].shape[0]
        time_size = unet_params["sample"].shape[2]

        # ip_adapter_cross_attn, prepare image prompt
        if vision_clip_emb is not None:
            # b t n d -> b t n d
            if self.print_idx == 0:
                logger.debug(
                    f"vision_clip_emb, before ip_adapter_image_proj, shape={vision_clip_emb.shape} mean={torch.mean(vision_clip_emb)}"
                )
            if vision_clip_emb.ndim == 3:
                vision_clip_emb = rearrange(vision_clip_emb, "b t d-> b t 1 d")
            if self.ip_adapter_image_proj is not None:
                vision_clip_emb = rearrange(vision_clip_emb, "b t n d ->(b t) n d")
                vision_clip_emb = self.ip_adapter_image_proj(vision_clip_emb)
                if self.print_idx == 0:
                    logger.debug(
                        f"vision_clip_emb, after ip_adapter_image_proj shape={vision_clip_emb.shape} mean={torch.mean(vision_clip_emb)}"
                    )
                if vision_clip_emb.ndim == 2:
                    vision_clip_emb = rearrange(vision_clip_emb, "b d-> b 1 d")
                vision_clip_emb = rearrange(
                    vision_clip_emb, "(b t) n d -> b t n d", b=batch_size
                )
            vision_clip_emb = align_repeat_tensor_single_dim(
                vision_clip_emb, target_length=time_size, dim=1
            )
            if self.print_idx == 0:
                logger.debug(
                    f"vision_clip_emb, after reshape shape={vision_clip_emb.shape} mean={torch.mean(vision_clip_emb)}"
                )

        if vision_clip_emb is None and encoder_hidden_states is not None:
            vision_clip_emb = encoder_hidden_states
        if vision_clip_emb is not None and encoder_hidden_states is None:
            encoder_hidden_states = vision_clip_emb
        # 当 prompt_only_use_image_prompt 为True时，
        # 1. referencenet 都使用 vision_clip_emb
        # 2. unet 如果没有dual_cross_attn，使用vision_clip_emb，有时不更新
        # 3. controlnet 当前使用 text_prompt

        # when prompt_only_use_image_prompt True,
        # 1. referencenet use vision_clip_emb
        # 2. unet use vision_clip_emb if no dual_cross_attn, sometimes not update
        # 3. controlnet use text_prompt

        # extract referencenet emb
        if self.referencenet is not None and referencenet_params is not None:
            referencenet_encoder_hidden_states = align_repeat_tensor_single_dim(
                vision_clip_emb,
                target_length=referencenet_params["num_frames"],
                dim=1,
            )
            referencenet_params["encoder_hidden_states"] = rearrange(
                referencenet_encoder_hidden_states, "b t n d->(b t) n d"
            )
            referencenet_out = self.referencenet(**referencenet_params)
            (
                down_block_refer_embs,
                mid_block_refer_emb,
                refer_self_attn_emb,
            ) = referencenet_out
            if down_block_refer_embs is not None:
                if self.print_idx == 0:
                    logger.debug(
                        f"len(down_block_refer_embs)={len(down_block_refer_embs)}"
                    )
                for i, down_emb in enumerate(down_block_refer_embs):
                    if self.print_idx == 0:
                        logger.debug(
                            f"down_emb, {i}, {down_emb.shape}, mean={down_emb.mean()}"
                        )
            else:
                if self.print_idx == 0:
                    logger.debug(f"down_block_refer_embs is None")
            if mid_block_refer_emb is not None:
                if self.print_idx == 0:
                    logger.debug(
                        f"mid_block_refer_emb, {mid_block_refer_emb.shape}, mean={mid_block_refer_emb.mean()}"
                    )
            else:
                if self.print_idx == 0:
                    logger.debug(f"mid_block_refer_emb is None")
            if refer_self_attn_emb is not None:
                if self.print_idx == 0:
                    logger.debug(f"refer_self_attn_emb, num={len(refer_self_attn_emb)}")
                for i, self_attn_emb in enumerate(refer_self_attn_emb):
                    if self.print_idx == 0:
                        logger.debug(
                            f"referencenet, self_attn_emb, {i}th, shape={self_attn_emb.shape}, mean={self_attn_emb.mean()}"
                        )
            else:
                if self.print_idx == 0:
                    logger.debug(f"refer_self_attn_emb is None")
        else:
            down_block_refer_embs, mid_block_refer_emb, refer_self_attn_emb = (
                None,
                None,
                None,
            )

        # extract controlnet emb
        if self.controlnet is not None and controlnet_params is not None:
            controlnet_encoder_hidden_states = align_repeat_tensor_single_dim(
                encoder_hidden_states,
                target_length=unet_params["sample"].shape[2],
                dim=1,
            )
            controlnet_params["encoder_hidden_states"] = rearrange(
                controlnet_encoder_hidden_states, " b t n d -> (b t) n d"
            )
            (
                down_block_additional_residuals,
                mid_block_additional_residual,
            ) = self.controlnet(**controlnet_params)
            if controlnet_scale != 1.0:
                down_block_additional_residuals = [
                    x * controlnet_scale for x in down_block_additional_residuals
                ]
                mid_block_additional_residual = (
                    mid_block_additional_residual * controlnet_scale
                )
            for i, down_block_additional_residual in enumerate(
                down_block_additional_residuals
            ):
                if self.print_idx == 0:
                    logger.debug(
                        f"{i}, down_block_additional_residual mean={torch.mean(down_block_additional_residual)}"
                    )

            if self.print_idx == 0:
                logger.debug(
                    f"mid_block_additional_residual mean={torch.mean(mid_block_additional_residual)}"
                )
        else:
            down_block_additional_residuals = None
            mid_block_additional_residual = None

        if prompt_only_use_image_prompt and vision_clip_emb is not None:
            encoder_hidden_states = vision_clip_emb

        # run unet
        out = self.unet(
            **unet_params,
            down_block_refer_embs=down_block_refer_embs,
            mid_block_refer_emb=mid_block_refer_emb,
            refer_self_attn_emb=refer_self_attn_emb,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            encoder_hidden_states=encoder_hidden_states,
            vision_clip_emb=vision_clip_emb,
        )
        self.print_idx += 1
        return out

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (UNet3DConditionModel, ReferenceNet2D)):
            module.gradient_checkpointing = value
