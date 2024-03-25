# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/transformer_temporal.py
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional
import logging

import torch
from torch import nn
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformer_temporal import (
    TransformerTemporalModelOutput,
    TransformerTemporalModel as DiffusersTransformerTemporalModel,
)
from diffusers.models.attention_processor import AttnProcessor

from mmcm.utils.gpu_util import get_gpu_status
from ..data.data_util import (
    batch_concat_two_tensor_with_index,
    batch_index_fill,
    batch_index_select,
    concat_two_tensor,
    align_repeat_tensor_single_dim,
)
from ..utils.attention_util import generate_sparse_causcal_attn_mask
from .attention import BasicTransformerBlock
from .attention_processor import (
    BaseIPAttnProcessor,
)
from . import Model_Register

# https://github.com/facebookresearch/xformers/issues/845
# 输入bs*n_frames*w*h太高，xformers报错。因此将transformer_temporal的allow_xformers均关掉
# if bs*n_frames*w*h to large, xformers will raise error. So we close the allow_xformers in transformer_temporal
logger = logging.getLogger(__name__)


@Model_Register.register
class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each TransformerBlock should contain two self-attention layers
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        femb_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        allow_xformers: bool = False,
        only_cross_attention: bool = False,
        keep_content_condition: bool = False,
        need_spatial_position_emb: bool = False,
        need_temporal_weight: bool = True,
        self_attn_mask: str = None,
        # TODO: 运行参数，有待改到forward里面去
        # TODO: running parameters, need to be moved to forward
        image_scale: float = 1.0,
        processor: AttnProcessor | None = None,
        remove_femb_non_linear: bool = False,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 2. Define temporal positional embedding
        self.frame_emb_proj = torch.nn.Linear(femb_channels, inner_dim)
        self.remove_femb_non_linear = remove_femb_non_linear
        if not remove_femb_non_linear:
            self.nonlinearity = nn.SiLU()

        # spatial_position_emb 使用femb_的参数配置
        self.need_spatial_position_emb = need_spatial_position_emb
        if need_spatial_position_emb:
            self.spatial_position_emb_proj = torch.nn.Linear(femb_channels, inner_dim)
        # 3. Define transformers blocks
        # TODO： 该实现方式不好，待优化
        # TODO: bad implementation, need to be optimized
        self.need_ipadapter = False
        self.cross_attn_temporal_cond = False
        self.allow_xformers = allow_xformers
        if processor is not None and isinstance(processor, BaseIPAttnProcessor):
            self.cross_attn_temporal_cond = True
            self.allow_xformers = False
            if "NonParam" not in processor.__class__.__name__:
                self.need_ipadapter = True

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    allow_xformers=allow_xformers,
                    only_cross_attention=only_cross_attention,
                    cross_attn_temporal_cond=self.need_ipadapter,
                    image_scale=image_scale,
                    processor=processor,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.need_temporal_weight = need_temporal_weight
        if need_temporal_weight:
            self.temporal_weight = nn.Parameter(
                torch.tensor(
                    [
                        1e-5,
                    ]
                )
            )  # initialize parameter with 0
        self.skip_temporal_layers = False  # Whether to skip temporal layer
        self.keep_content_condition = keep_content_condition
        self.self_attn_mask = self_attn_mask
        self.only_cross_attention = only_cross_attention
        self.double_self_attention = double_self_attention
        self.cross_attention_dim = cross_attention_dim
        self.image_scale = image_scale
        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        hidden_states,
        femb,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        num_frames=1,
        cross_attention_kwargs=None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        spatial_position_emb: torch.Tensor = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] or `tuple`:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
        if self.skip_temporal_layers is True:
            if not return_dict:
                return (hidden_states,)

            return TransformerTemporalModelOutput(sample=hidden_states)

        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = rearrange(
            hidden_states, "(b t) c h w -> b c t h w", b=batch_size
        )
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        hidden_states = rearrange(hidden_states, "b c t h w -> (b h w) t c")

        hidden_states = self.proj_in(hidden_states)

        # 2 Positional embedding
        # adapted from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/resnet.py#L574
        if not self.remove_femb_non_linear:
            femb = self.nonlinearity(femb)
        femb = self.frame_emb_proj(femb)
        femb = align_repeat_tensor_single_dim(femb, hidden_states.shape[0], dim=0)
        hidden_states = hidden_states + femb

        # 3. Blocks
        if (
            (self.only_cross_attention or not self.double_self_attention)
            and self.cross_attention_dim is not None
            and encoder_hidden_states is not None
        ):
            encoder_hidden_states = align_repeat_tensor_single_dim(
                encoder_hidden_states,
                hidden_states.shape[0],
                dim=0,
                n_src_base_length=batch_size,
            )

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 4. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(
            hidden_states, "(b h w) t c -> b c t h w", b=batch_size, h=height, w=width
        ).contiguous()

        # 保留condition对应的frames，便于保持前序内容帧，提升一致性
        # keep the frames corresponding to the condition to maintain the previous content frames and improve consistency
        if (
            vision_conditon_frames_sample_index is not None
            and self.keep_content_condition
        ):
            mask = torch.ones_like(hidden_states, device=hidden_states.device)
            mask = batch_index_fill(
                mask, dim=2, index=vision_conditon_frames_sample_index, value=0
            )
            if self.need_temporal_weight:
                output = (
                    residual + torch.abs(self.temporal_weight) * mask * hidden_states
                )
            else:
                output = residual + mask * hidden_states
        else:
            if self.need_temporal_weight:
                output = residual + torch.abs(self.temporal_weight) * hidden_states
            else:
                output = residual + mask * hidden_states

        # output = torch.abs(self.temporal_weight) * hidden_states + residual
        output = rearrange(output, "b c t h w -> (b t) c h w")
        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
