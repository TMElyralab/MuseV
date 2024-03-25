# Copyright 2023 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
# Copyright 2023 The ModelScope Team.
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

# Adapted from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/unet_3d_condition.py

# 1. 增加了from_pretrained，将模型从2D blocks改为3D blocks
# 1. add from_pretrained, change model from 2D blocks to 3D blocks

from copy import deepcopy
from dataclasses import dataclass
import inspect
from pprint import pprint, pformat
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import os
import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange, repeat
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput

# from diffusers.utils import logging
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin, load_state_dict
from diffusers import __version__
from diffusers.utils import (
    CONFIG_NAME,
    DIFFUSERS_CACHE,
    FLAX_WEIGHTS_NAME,
    HF_HUB_OFFLINE,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
)
from diffusers.utils.import_utils import _safetensors_available
from diffusers.models.unet_3d_condition import (
    UNet3DConditionOutput,
    UNet3DConditionModel as DiffusersUNet3DConditionModel,
)
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    AttnProcessor,
    AttnProcessor2_0,
    XFormersAttnProcessor,
)

from ..models import Model_Register

from .resnet import TemporalConvLayer
from .temporal_transformer import (
    TransformerTemporalModel,
)
from .embeddings import get_2d_sincos_pos_embed, resize_spatial_position_emb
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from ..data.data_util import (
    adaptive_instance_normalization,
    align_repeat_tensor_single_dim,
    batch_adain_conditioned_tensor,
    batch_concat_two_tensor_with_index,
    concat_two_tensor,
    concat_two_tensor_with_index,
)
from .attention_processor import BaseIPAttnProcessor
from .attention_processor import ReferEmbFuseAttention
from .transformer_2d import Transformer2DModel
from .attention import BasicTransformerBlock


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# if is_torch_version(">=", "1.9.0"):
#     _LOW_CPU_MEM_USAGE_DEFAULT = True
# else:
#     _LOW_CPU_MEM_USAGE_DEFAULT = False
_LOW_CPU_MEM_USAGE_DEFAULT = False

if is_accelerate_available():
    import accelerate
    from accelerate.utils import set_module_tensor_to_device
    from accelerate.utils.versions import is_torch_version


import safetensors


def hack_t2i_sd_layer_attn_with_ip(
    unet: nn.Module,
    self_attn_class: BaseIPAttnProcessor = None,
    cross_attn_class: BaseIPAttnProcessor = None,
):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        if "temp_attentions" in name or "transformer_in" in name:
            continue
        if name.endswith("attn1.processor") and self_attn_class is not None:
            attn_procs[name] = self_attn_class()
            if unet.print_idx == 0:
                logger.debug(
                    f"hack attn_processor of {name} to {attn_procs[name].__class__.__name__}"
                )
        elif name.endswith("attn2.processor") and cross_attn_class is not None:
            attn_procs[name] = cross_attn_class()
            if unet.print_idx == 0:
                logger.debug(
                    f"hack attn_processor of {name} to {attn_procs[name].__class__.__name__}"
                )
    unet.set_attn_processor(attn_procs, strict=False)


def convert_2D_to_3D(
    module_names,
    valid_modules=(
        "CrossAttnDownBlock2D",
        "CrossAttnUpBlock2D",
        "DownBlock2D",
        "UNetMidBlock2DCrossAttn",
        "UpBlock2D",
    ),
):
    if not isinstance(module_names, list):
        return module_names.replace("2D", "3D")

    return_modules = []
    for module_name in module_names:
        if module_name in valid_modules:
            return_modules.append(module_name.replace("2D", "3D"))
        else:
            return_modules.append(module_name)
    return return_modules


def insert_spatial_self_attn_idx(unet):
    pass


@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    The output of [`UNet3DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    UNet3DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
    """

    _supports_gradient_checkpointing = True
    print_idx = 0

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        temporal_conv_block: str = "TemporalConvLayer",
        temporal_transformer: str = "TransformerTemporalModel",
        need_spatial_position_emb: bool = False,
        need_transformer_in: bool = True,
        need_t2i_ip_adapter: bool = False,  # self_attn,  t2i.attn1
        need_adain_temporal_cond: bool = False,
        t2i_ip_adapter_attn_processor: str = "NonParamT2ISelfReferenceXFormersAttnProcessor",
        keep_vision_condtion: bool = False,
        use_anivv1_cfg: bool = False,
        resnet_2d_skip_time_act: bool = False,
        need_zero_vis_cond_temb: bool = True,
        norm_spatial_length: bool = False,
        spatial_max_length: int = 2048,
        need_refer_emb: bool = False,
        ip_adapter_cross_attn: bool = False,  # cross_attn, t2i.attn2
        t2i_crossattn_ip_adapter_attn_processor: str = "T2IReferencenetIPAdapterXFormersAttnProcessor",
        need_t2i_facein: bool = False,
        need_t2i_ip_adapter_face: bool = False,
        need_vis_cond_mask: bool = False,
    ):
        """_summary_

        Args:
            sample_size (Optional[int], optional): _description_. Defaults to None.
            in_channels (int, optional): _description_. Defaults to 4.
            out_channels (int, optional): _description_. Defaults to 4.
            down_block_types (Tuple[str], optional): _description_. Defaults to ( "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D", ).
            up_block_types (Tuple[str], optional): _description_. Defaults to ( "UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", ).
            block_out_channels (Tuple[int], optional): _description_. Defaults to (320, 640, 1280, 1280).
            layers_per_block (int, optional): _description_. Defaults to 2.
            downsample_padding (int, optional): _description_. Defaults to 1.
            mid_block_scale_factor (float, optional): _description_. Defaults to 1.
            act_fn (str, optional): _description_. Defaults to "silu".
            norm_num_groups (Optional[int], optional): _description_. Defaults to 32.
            norm_eps (float, optional): _description_. Defaults to 1e-5.
            cross_attention_dim (int, optional): _description_. Defaults to 1024.
            attention_head_dim (Union[int, Tuple[int]], optional): _description_. Defaults to 8.
            temporal_conv_block (str, optional): 3D卷积字符串，需要注册在 Model_Register. Defaults to "TemporalConvLayer".
            temporal_transformer (str, optional): 时序 Transformer block字符串，需要定义在 Model_Register. Defaults to "TransformerTemporalModel".
            need_spatial_position_emb (bool, optional): 是否需要 spatial hw 的emb，需要配合 thw attn使用. Defaults to False.
            need_transformer_in (bool, optional): 是否需要 第一个 temporal_transformer_block. Defaults to True.
            need_t2i_ip_adapter (bool, optional): T2I 模块是否需要面向视觉条件帧的 attn. Defaults to False.
            need_adain_temporal_cond (bool, optional): 是否需要面向首帧 使用Adain. Defaults to False.
            t2i_ip_adapter_attn_processor (str, optional):
                t2i attn_processor的优化版，需配合need_t2i_ip_adapter使用，
                有 NonParam 表示无参ReferenceOnly-attn，没有表示有参 IpAdapter.
                Defaults to "NonParamT2ISelfReferenceXFormersAttnProcessor".
            keep_vision_condtion (bool, optional): 是否对视觉条件帧不加 timestep emb. Defaults to False.
            use_anivv1_cfg (bool, optional): 一些基本配置 是否延续AnivV设计. Defaults to False.
            resnet_2d_skip_time_act (bool, optional): 配合use_anivv1_cfg，修改 transformer 2d block. Defaults to False.
            need_zero_vis_cond_temb (bool, optional): 目前无效参数. Defaults to True.
            norm_spatial_length (bool, optional): 是否需要 norm_spatial_length，只有当 need_spatial_position_emb= True时,才有效. Defaults to False.
            spatial_max_length (int, optional):  归一化长度. Defaults to 2048.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        super(UNet3DConditionModel, self).__init__()
        self.keep_vision_condtion = keep_vision_condtion
        self.use_anivv1_cfg = use_anivv1_cfg
        self.sample_size = sample_size
        self.resnet_2d_skip_time_act = resnet_2d_skip_time_act
        self.need_zero_vis_cond_temb = need_zero_vis_cond_temb
        self.norm_spatial_length = norm_spatial_length
        self.spatial_max_length = spatial_max_length
        self.need_refer_emb = need_refer_emb
        self.ip_adapter_cross_attn = ip_adapter_cross_attn
        self.need_t2i_facein = need_t2i_facein
        self.need_t2i_ip_adapter_face = need_t2i_ip_adapter_face

        logger.debug(f"need_t2i_ip_adapter_face={need_t2i_ip_adapter_face}")
        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )
        if use_anivv1_cfg:
            self.time_nonlinearity = nn.SiLU()

        # frame
        frame_embed_dim = block_out_channels[0] * 4
        self.frame_proj = Timesteps(block_out_channels[0], True, 0)
        frame_input_dim = block_out_channels[0]
        if temporal_transformer is not None:
            self.frame_embedding = TimestepEmbedding(
                frame_input_dim,
                frame_embed_dim,
                act_fn=act_fn,
            )
        else:
            self.frame_embedding = None
        if use_anivv1_cfg:
            self.femb_nonlinearity = nn.SiLU()

        # spatial_position_emb
        self.need_spatial_position_emb = need_spatial_position_emb
        if need_spatial_position_emb:
            self.spatial_position_input_dim = block_out_channels[0] * 2
            self.spatial_position_embed_dim = block_out_channels[0] * 4

            self.spatial_position_embedding = TimestepEmbedding(
                self.spatial_position_input_dim,
                self.spatial_position_embed_dim,
                act_fn=act_fn,
            )

        # 从模型注册表中获取 模型类
        temporal_conv_block = (
            Model_Register[temporal_conv_block]
            if isinstance(temporal_conv_block, str)
            and temporal_conv_block.lower() != "none"
            else None
        )
        self.need_transformer_in = need_transformer_in

        temporal_transformer = (
            Model_Register[temporal_transformer]
            if isinstance(temporal_transformer, str)
            and temporal_transformer.lower() != "none"
            else None
        )
        self.need_vis_cond_mask = need_vis_cond_mask

        if need_transformer_in and temporal_transformer is not None:
            self.transformer_in = temporal_transformer(
                num_attention_heads=attention_head_dim,
                attention_head_dim=block_out_channels[0] // attention_head_dim,
                in_channels=block_out_channels[0],
                num_layers=1,
                femb_channels=frame_embed_dim,
                need_spatial_position_emb=need_spatial_position_emb,
                cross_attention_dim=cross_attention_dim,
            )

        # class embedding
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        self.need_t2i_ip_adapter = need_t2i_ip_adapter
        # 确定T2I Attn 是否加入 ReferenceOnly机制或Ipadaper机制
        # TODO:有待更好的实现机制,
        need_t2i_ip_adapter_param = (
            t2i_ip_adapter_attn_processor is not None
            and "NonParam" not in t2i_ip_adapter_attn_processor
            and need_t2i_ip_adapter
        )
        self.need_adain_temporal_cond = need_adain_temporal_cond
        self.t2i_ip_adapter_attn_processor = t2i_ip_adapter_attn_processor

        if need_refer_emb:
            self.first_refer_emb_attns = ReferEmbFuseAttention(
                query_dim=block_out_channels[0],
                heads=attention_head_dim[0],
                dim_head=block_out_channels[0] // attention_head_dim[0],
                dropout=0,
                bias=False,
                cross_attention_dim=None,
                upcast_attention=False,
            )
            self.mid_block_refer_emb_attns = ReferEmbFuseAttention(
                query_dim=block_out_channels[-1],
                heads=attention_head_dim[-1],
                dim_head=block_out_channels[-1] // attention_head_dim[-1],
                dropout=0,
                bias=False,
                cross_attention_dim=None,
                upcast_attention=False,
            )
        else:
            self.first_refer_emb_attns = None
            self.mid_block_refer_emb_attns = None
        # down
        output_channel = block_out_channels[0]
        self.layers_per_block = layers_per_block
        self.block_out_channels = block_out_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                femb_channels=frame_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
                temporal_conv_block=temporal_conv_block,
                temporal_transformer=temporal_transformer,
                need_spatial_position_emb=need_spatial_position_emb,
                need_t2i_ip_adapter=need_t2i_ip_adapter_param,
                ip_adapter_cross_attn=ip_adapter_cross_attn,
                need_t2i_facein=need_t2i_facein,
                need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                need_adain_temporal_cond=need_adain_temporal_cond,
                resnet_2d_skip_time_act=resnet_2d_skip_time_act,
                need_refer_emb=need_refer_emb,
            )
            self.down_blocks.append(down_block)
        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            femb_channels=frame_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
            temporal_conv_block=temporal_conv_block,
            temporal_transformer=temporal_transformer,
            need_spatial_position_emb=need_spatial_position_emb,
            need_t2i_ip_adapter=need_t2i_ip_adapter_param,
            ip_adapter_cross_attn=ip_adapter_cross_attn,
            need_t2i_facein=need_t2i_facein,
            need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
            need_adain_temporal_cond=need_adain_temporal_cond,
            resnet_2d_skip_time_act=resnet_2d_skip_time_act,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                femb_channels=frame_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=False,
                temporal_conv_block=temporal_conv_block,
                temporal_transformer=temporal_transformer,
                need_spatial_position_emb=need_spatial_position_emb,
                need_t2i_ip_adapter=need_t2i_ip_adapter_param,
                ip_adapter_cross_attn=ip_adapter_cross_attn,
                need_t2i_facein=need_t2i_facein,
                need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                need_adain_temporal_cond=need_adain_temporal_cond,
                resnet_2d_skip_time_act=resnet_2d_skip_time_act,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding,
        )
        self.insert_spatial_self_attn_idx()

        # 根据需要hack attn_processor，实现ip_adapter等功能
        if need_t2i_ip_adapter or ip_adapter_cross_attn:
            hack_t2i_sd_layer_attn_with_ip(
                self,
                self_attn_class=Model_Register[t2i_ip_adapter_attn_processor]
                if t2i_ip_adapter_attn_processor is not None and need_t2i_ip_adapter
                else None,
                cross_attn_class=Model_Register[t2i_crossattn_ip_adapter_attn_processor]
                if t2i_crossattn_ip_adapter_attn_processor is not None
                and (
                    ip_adapter_cross_attn or need_t2i_facein or need_t2i_ip_adapter_face
                )
                else None,
            )
            # logger.debug(pformat(self.attn_processors))

            # 非参数AttnProcessor，就不需要to_k_ip、to_v_ip参数了
            if (
                t2i_ip_adapter_attn_processor is None
                or "NonParam" in t2i_ip_adapter_attn_processor
            ):
                need_t2i_ip_adapter = False

        if self.print_idx == 0:
            logger.debug("Unet3Model Parameters")
            # logger.debug(pformat(self.__dict__))

        # 会在 set_skip_temporal_layers 设置 skip_refer_downblock_emb
        # 当为 True 时，会跳过 referencenet_block_emb的影响，主要用于首帧生成
        self.skip_refer_downblock_emb = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = (
            num_sliceable_layers * [slice_size]
            if not isinstance(slice_size, list)
            else slice_size
        )

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(
            module: torch.nn.Module, slice_size: List[int]
        ):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self,
        processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]],
        strict: bool = True,
    ):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count and strict:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    logger.debug(
                        f"module {name} set attn processor {processor.__class__.__name__}"
                    )
                    module.set_processor(processor)
                else:
                    if f"{name}.processor" in processor:
                        logger.debug(
                            "module {} set attn processor {}".format(
                                name, processor[f"{name}.processor"].__class__.__name__
                            )
                        )
                        module.set_processor(processor.pop(f"{name}.processor"))
                    else:
                        logger.debug(
                            f"module {name} has no new target attn_processor, still use {module.processor.__class__.__name__} "
                        )
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)
        ):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        sample_index: torch.LongTensor = None,
        vision_condition_frames_sample: torch.Tensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        sample_frame_rate: int = 10,
        skip_temporal_layers: bool = None,
        frame_index: torch.LongTensor = None,
        down_block_refer_embs: Optional[Tuple[torch.Tensor]] = None,
        mid_block_refer_emb: Optional[torch.Tensor] = None,
        refer_self_attn_emb: Optional[List[torch.Tensor]] = None,
        refer_self_attn_emb_mode: Literal["read", "write"] = "read",
        vision_clip_emb: torch.Tensor = None,
        ip_adapter_scale: float = 1.0,
        face_emb: torch.Tensor = None,
        facein_scale: float = 1.0,
        ip_adapter_face_emb: torch.Tensor = None,
        ip_adapter_face_scale: float = 1.0,
        do_classifier_free_guidance: bool = False,
        pose_guider_emb: torch.Tensor = None,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        """_summary_

        Args:
            sample (torch.FloatTensor): _description_
            timestep (Union[torch.Tensor, float, int]): _description_
            encoder_hidden_states (torch.Tensor): _description_
            class_labels (Optional[torch.Tensor], optional): _description_. Defaults to None.
            timestep_cond (Optional[torch.Tensor], optional): _description_. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
            down_block_additional_residuals (Optional[Tuple[torch.Tensor]], optional): _description_. Defaults to None.
            mid_block_additional_residual (Optional[torch.Tensor], optional): _description_. Defaults to None.
            return_dict (bool, optional): _description_. Defaults to True.
            sample_index (torch.LongTensor, optional): _description_. Defaults to None.
            vision_condition_frames_sample (torch.Tensor, optional): _description_. Defaults to None.
            vision_conditon_frames_sample_index (torch.LongTensor, optional): _description_. Defaults to None.
            sample_frame_rate (int, optional): _description_. Defaults to 10.
            skip_temporal_layers (bool, optional): _description_. Defaults to None.
            frame_index (torch.LongTensor, optional): _description_. Defaults to None.
            up_block_additional_residual (Optional[torch.Tensor], optional): 用于up_block的 参考latent. Defaults to None.
            down_block_refer_embs (Optional[torch.Tensor], optional): 用于 download 的 参考latent. Defaults to None.
            how_fuse_referencenet_emb (Literal, optional): 如何融合 参考 latent. Defaults to ["add", "attn"]="add".
                add: 要求 additional_latent 和 latent hw 同尺寸. hw of addtional_latent should be same as of latent
                attn:   concat bt*h1w1*c and bt*h2w2*c into bt*(h1w1+h2w2)*c, and then as key,value into attn
        Raises:
            ValueError: _description_

        Returns:
            Union[UNet3DConditionOutput, Tuple]: _description_
        """

        if skip_temporal_layers is not None:
            self.set_skip_temporal_layers(skip_temporal_layers)
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            #             logger.debug("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size = sample.shape[0]

        # when vision_condition_frames_sample is not None and  vision_conditon_frames_sample_index is not None
        # if not None, b c t h w -> b c (t + n_content ) h w

        if vision_condition_frames_sample is not None:
            sample = batch_concat_two_tensor_with_index(
                sample,
                sample_index,
                vision_condition_frames_sample,
                vision_conditon_frames_sample_index,
                dim=2,
            )

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, channel, num_frames, height, width = sample.shape

        # 准备 timestep emb
        timesteps = timesteps.expand(sample.shape[0])
        temb = self.time_proj(timesteps)
        temb = temb.to(dtype=self.dtype)
        emb = self.time_embedding(temb, timestep_cond)
        if self.use_anivv1_cfg:
            emb = self.time_nonlinearity(emb)
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)

        # 一致性保持，使条件时序帧的 首帧 timesteps emb 为 0，即不影响视觉条件帧
        # keep consistent with the first frame of vision condition frames
        if (
            self.keep_vision_condtion
            and num_frames > 1
            and sample_index is not None
            and vision_conditon_frames_sample_index is not None
        ):
            emb = rearrange(emb, "(b t) d -> b t d", t=num_frames)
            emb[:, vision_conditon_frames_sample_index, :] = 0
            emb = rearrange(emb, "b t d->(b t) d")

        # temporal positional embedding
        femb = None
        if self.temporal_transformer is not None:
            if frame_index is None:
                frame_index = torch.arange(
                    num_frames, dtype=torch.long, device=sample.device
                )
                if self.use_anivv1_cfg:
                    frame_index = (frame_index * sample_frame_rate).to(dtype=torch.long)
                femb = self.frame_proj(frame_index)
                if self.print_idx == 0:
                    logger.debug(
                        f"unet prepare frame_index, {femb.shape}, {batch_size}"
                    )
                femb = repeat(femb, "t d-> b t d", b=batch_size)
            else:
                # b t -> b t d
                assert frame_index.ndim == 2, ValueError(
                    "ndim of given frame_index should be 2, but {frame_index.ndim}"
                )
                femb = torch.stack(
                    [self.frame_proj(frame_index[i]) for i in range(batch_size)], dim=0
                )
        if self.temporal_transformer is not None:
            femb = femb.to(dtype=self.dtype)
            femb = self.frame_embedding(
                femb,
            )
            if self.use_anivv1_cfg:
                femb = self.femb_nonlinearity(femb)
        if encoder_hidden_states.ndim == 3:
            encoder_hidden_states = align_repeat_tensor_single_dim(
                encoder_hidden_states, target_length=emb.shape[0], dim=0
            )
        elif encoder_hidden_states.ndim == 4:
            encoder_hidden_states = rearrange(
                encoder_hidden_states, "b t n q-> (b t) n q"
            )
        else:
            raise ValueError(
                f"only support ndim in [3, 4], but given {encoder_hidden_states.ndim}"
            )
        if vision_clip_emb is not None:
            if vision_clip_emb.ndim == 4:
                vision_clip_emb = rearrange(vision_clip_emb, "b t n q-> (b t) n q")
        # 准备 hw 层面的 spatial positional embedding
        # prepare spatial_position_emb
        if self.need_spatial_position_emb:
            # height * width, self.spatial_position_input_dim
            spatial_position_emb = get_2d_sincos_pos_embed(
                embed_dim=self.spatial_position_input_dim,
                grid_size_w=width,
                grid_size_h=height,
                cls_token=False,
                norm_length=self.norm_spatial_length,
                max_length=self.spatial_max_length,
            )
            spatial_position_emb = torch.from_numpy(spatial_position_emb).to(
                device=sample.device, dtype=self.dtype
            )
            # height * width, self.spatial_position_embed_dim
            spatial_position_emb = self.spatial_position_embedding(spatial_position_emb)
        else:
            spatial_position_emb = None

        # prepare cross_attention_kwargs，ReferenceOnly/IpAdapter的attn_processor需要这些参数 进行 latenst和viscond_latents拆分运算
        if (
            self.need_t2i_ip_adapter
            or self.ip_adapter_cross_attn
            or self.need_t2i_facein
            or self.need_t2i_ip_adapter_face
        ):
            if cross_attention_kwargs is None:
                cross_attention_kwargs = {}
            cross_attention_kwargs["num_frames"] = num_frames
            cross_attention_kwargs[
                "do_classifier_free_guidance"
            ] = do_classifier_free_guidance
            cross_attention_kwargs["sample_index"] = sample_index
            cross_attention_kwargs[
                "vision_conditon_frames_sample_index"
            ] = vision_conditon_frames_sample_index
            if self.ip_adapter_cross_attn:
                cross_attention_kwargs["vision_clip_emb"] = vision_clip_emb
                cross_attention_kwargs["ip_adapter_scale"] = ip_adapter_scale
            if self.need_t2i_facein:
                if self.print_idx == 0:
                    logger.debug(
                        f"face_emb={type(face_emb)}, facein_scale={facein_scale}"
                    )
                cross_attention_kwargs["face_emb"] = face_emb
                cross_attention_kwargs["facein_scale"] = facein_scale
            if self.need_t2i_ip_adapter_face:
                if self.print_idx == 0:
                    logger.debug(
                        f"ip_adapter_face_emb={type(ip_adapter_face_emb)}, ip_adapter_face_scale={ip_adapter_face_scale}"
                    )
                cross_attention_kwargs["ip_adapter_face_emb"] = ip_adapter_face_emb
                cross_attention_kwargs["ip_adapter_face_scale"] = ip_adapter_face_scale
        # 2. pre-process
        sample = rearrange(sample, "b c t h w -> (b t) c h w")
        sample = self.conv_in(sample)

        if pose_guider_emb is not None:
            if self.print_idx == 0:
                logger.debug(
                    f"sample={sample.shape}, pose_guider_emb={pose_guider_emb.shape}"
                )
            sample = sample + pose_guider_emb

        if self.print_idx == 0:
            logger.debug(f"after conv in sample={sample.mean()}")
        if spatial_position_emb is not None:
            if self.print_idx == 0:
                logger.debug(
                    f"unet3d, transformer_in, spatial_position_emb={spatial_position_emb.shape}"
                )
        if self.print_idx == 0:
            logger.debug(
                f"unet vision_conditon_frames_sample_index, {type(vision_conditon_frames_sample_index)}",
            )
        if vision_conditon_frames_sample_index is not None:
            if self.print_idx == 0:
                logger.debug(
                    f"vision_conditon_frames_sample_index shape {vision_conditon_frames_sample_index.shape}",
                )
        if self.print_idx == 0:
            logger.debug(f"unet sample_index {type(sample_index)}")
        if sample_index is not None:
            if self.print_idx == 0:
                logger.debug(f"sample_index shape {sample_index.shape}")
        if self.need_transformer_in:
            if self.print_idx == 0:
                logger.debug(f"unet3d, transformer_in, sample={sample.shape}")
            sample = self.transformer_in(
                sample,
                femb=femb,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_hidden_states=encoder_hidden_states,
                sample_index=sample_index,
                vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                spatial_position_emb=spatial_position_emb,
            ).sample
        if (
            self.need_refer_emb
            and down_block_refer_embs is not None
            and not self.skip_refer_downblock_emb
        ):
            if self.print_idx == 0:
                logger.debug(
                    f"self.first_refer_emb_attns, {self.first_refer_emb_attns.__class__.__name__} {down_block_refer_embs[0].shape}"
                )
            sample = self.first_refer_emb_attns(
                sample, down_block_refer_embs[0], num_frames=num_frames
            )
            if self.print_idx == 0:
                logger.debug(
                    f"first_refer_emb_attns, sample is_leaf={sample.is_leaf}, requires_grad={sample.requires_grad}, down_block_refer_embs, {down_block_refer_embs[0].is_leaf}, {down_block_refer_embs[0].requires_grad},"
                )
        else:
            if self.print_idx == 0:
                logger.debug(f"first_refer_emb_attns, no this step")
        # 将 refer_self_attn_emb 转化成字典，增加一个当前index，表示block 的对应关系
        # convert refer_self_attn_emb to dict, add a current index to represent the corresponding relationship of the block

        # 3. down
        down_block_res_samples = (sample,)
        for i_down_block, downsample_block in enumerate(self.down_blocks):
            # 使用 attn 的方式 来融合 refer_emb，这里是准备 downblock 对应的 refer_emb
            # fuse refer_emb with attn, here is to prepare the refer_emb corresponding to downblock
            if (
                not self.need_refer_emb
                or down_block_refer_embs is None
                or self.skip_refer_downblock_emb
            ):
                this_down_block_refer_embs = None
                if self.print_idx == 0:
                    logger.debug(
                        f"{i_down_block}, prepare this_down_block_refer_embs, is None"
                    )
            else:
                is_final_block = i_down_block == len(self.block_out_channels) - 1
                num_block = self.layers_per_block + int(not is_final_block * 1)
                this_downblock_start_idx = 1 + num_block * i_down_block
                this_down_block_refer_embs = down_block_refer_embs[
                    this_downblock_start_idx : this_downblock_start_idx + num_block
                ]
                if self.print_idx == 0:
                    logger.debug(
                        f"prepare this_down_block_refer_embs, {len(this_down_block_refer_embs)}, {this_down_block_refer_embs[0].shape}"
                    )
            if self.print_idx == 0:
                logger.debug(f"downsample_block {i_down_block}, sample={sample.mean()}")
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    femb=femb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    sample_index=sample_index,
                    vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    spatial_position_emb=spatial_position_emb,
                    refer_embs=this_down_block_refer_embs,
                    refer_self_attn_emb=refer_self_attn_emb,
                    refer_self_attn_emb_mode=refer_self_attn_emb_mode,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    femb=femb,
                    num_frames=num_frames,
                    sample_index=sample_index,
                    vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    spatial_position_emb=spatial_position_emb,
                    refer_embs=this_down_block_refer_embs,
                    refer_self_attn_emb=refer_self_attn_emb,
                    refer_self_attn_emb_mode=refer_self_attn_emb_mode,
                )

            # resize spatial_position_emb
            if self.need_spatial_position_emb:
                has_downblock = i_down_block < len(self.down_blocks) - 1
                if has_downblock:
                    spatial_position_emb = resize_spatial_position_emb(
                        spatial_position_emb,
                        scale=0.5,
                        height=sample.shape[2] * 2,
                        width=sample.shape[3] * 2,
                    )
            down_block_res_samples += res_samples
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                hidden_states=sample,
                temb=emb,
                femb=femb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                sample_index=sample_index,
                vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                spatial_position_emb=spatial_position_emb,
                refer_self_attn_emb=refer_self_attn_emb,
                refer_self_attn_emb_mode=refer_self_attn_emb_mode,
            )
        # 使用 attn 的方式 来融合 mid_block_refer_emb
        # fuse mid_block_refer_emb with attn
        if (
            self.mid_block_refer_emb_attns is not None
            and mid_block_refer_emb is not None
            and not self.skip_refer_downblock_emb
        ):
            if self.print_idx == 0:
                logger.debug(
                    f"self.mid_block_refer_emb_attns={self.mid_block_refer_emb_attns}, mid_block_refer_emb={mid_block_refer_emb.shape}"
                )
            sample = self.mid_block_refer_emb_attns(
                sample, mid_block_refer_emb, num_frames=num_frames
            )
            if self.print_idx == 0:
                logger.debug(
                    f"mid_block_refer_emb_attns, sample is_leaf={sample.is_leaf}, requires_grad={sample.requires_grad}, mid_block_refer_emb, {mid_block_refer_emb[0].is_leaf}, {mid_block_refer_emb[0].requires_grad},"
                )
        else:
            if self.print_idx == 0:
                logger.debug(f"mid_block_refer_emb_attns, no this step")
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i_up_block, upsample_block in enumerate(self.up_blocks):
            is_final_block = i_up_block == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    femb=femb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    sample_index=sample_index,
                    vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    spatial_position_emb=spatial_position_emb,
                    refer_self_attn_emb=refer_self_attn_emb,
                    refer_self_attn_emb_mode=refer_self_attn_emb_mode,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    femb=femb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                    sample_index=sample_index,
                    vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    spatial_position_emb=spatial_position_emb,
                    refer_self_attn_emb=refer_self_attn_emb,
                    refer_self_attn_emb_mode=refer_self_attn_emb_mode,
                )
            # resize spatial_position_emb
            if self.need_spatial_position_emb:
                has_upblock = i_up_block < len(self.up_blocks) - 1
                if has_upblock:
                    spatial_position_emb = resize_spatial_position_emb(
                        spatial_position_emb,
                        scale=2,
                        height=int(sample.shape[2] / 2),
                        width=int(sample.shape[3] / 2),
                    )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)

        sample = self.conv_out(sample)
        sample = rearrange(sample, "(b t) c h w -> b c t h w", t=num_frames)

        # if self.need_adain_temporal_cond and num_frames > 1:
        #     sample = batch_adain_conditioned_tensor(
        #         sample,
        #         num_frames=num_frames,
        #         need_style_fidelity=False,
        #         src_index=sample_index,
        #         dst_index=vision_conditon_frames_sample_index,
        #     )
        self.print_idx += 1

        if skip_temporal_layers is not None:
            self.set_skip_temporal_layers(not skip_temporal_layers)
        if not return_dict:
            return (sample,)
        else:
            return UNet3DConditionOutput(sample=sample)

    # from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/modeling_utils.py#L328
    @classmethod
    def from_pretrained_2d(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
    ):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_flax`.
            use_safetensors (`bool`, *optional* ):
                If set to `True`, the pipeline will forcibly load the models from `safetensors` weights. If set to
                `None` (the default). The pipeline will load using `safetensors` if safetensors weights are available
                *and* if `safetensors` is installed. If the to `False` the pipeline will *not* use `safetensors`.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        strict = kwargs.pop("strict", True)

        allow_pickle = False
        if use_safetensors is None:
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set"
                " `device_map=None`. You can install accelerate with `pip install accelerate`."
            )

        # Check if we can handle device_map and dispatching the weights
        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            user_agent=user_agent,
            **kwargs,
        )

        config["_class_name"] = cls.__name__
        config["down_block_types"] = convert_2D_to_3D(config["down_block_types"])
        if "mid_block_type" in config:
            config["mid_block_type"] = convert_2D_to_3D(config["mid_block_type"])
        else:
            config["mid_block_type"] = "UNetMidBlock3DCrossAttn"
        config["up_block_types"] = convert_2D_to_3D(config["up_block_types"])

        # load model
        model_file = None
        if from_flax:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=FLAX_WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            model = cls.from_config(config, **unused_kwargs)

            # Convert the weights
            from diffusers.models.modeling_pytorch_flax_utils import (
                load_flax_checkpoint_in_pytorch_model,
            )

            model = load_flax_checkpoint_in_pytorch_model(model, model_file)
        else:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )
            except IOError as e:
                if not allow_pickle:
                    raise e
                pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )

            if low_cpu_mem_usage:
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **unused_kwargs)

                # if device_map is None, load the state dict and move the params from meta device to the cpu
                if device_map is None:
                    param_device = "cpu"
                    state_dict = load_state_dict(model_file, variant=variant)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(
                        state_dict.keys()
                    )
                    if len(missing_keys) > 0:
                        if strict:
                            raise ValueError(
                                f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"
                                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                                " those weights or else make sure your checkpoint file is correct."
                            )
                        else:
                            logger.warning(
                                f"model{cls}  has no target pretrained paramter from {pretrained_model_name_or_path},  {', '.join(missing_keys)}"
                            )

                    empty_state_dict = model.state_dict()
                    for param_name, param in state_dict.items():
                        accepts_dtype = "dtype" in set(
                            inspect.signature(
                                set_module_tensor_to_device
                            ).parameters.keys()
                        )

                        if empty_state_dict[param_name].shape != param.shape:
                            raise ValueError(
                                f"Cannot load {pretrained_model_name_or_path} because {param_name} expected shape {empty_state_dict[param_name]}, but got {param.shape}. If you want to instead overwrite randomly initialized weights, please make sure to pass both `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example."
                            )

                        if accepts_dtype:
                            set_module_tensor_to_device(
                                model,
                                param_name,
                                param_device,
                                value=param,
                                dtype=torch_dtype,
                            )
                        else:
                            set_module_tensor_to_device(
                                model, param_name, param_device, value=param
                            )
                else:  # else let accelerate handle loading and dispatching.
                    # Load weights and dispatch according to the device_map
                    # by default the device_map is None and the weights are loaded on the CPU
                    accelerate.load_checkpoint_and_dispatch(
                        model, model_file, device_map, dtype=torch_dtype
                    )

                loading_info = {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            else:
                model = cls.from_config(config, **unused_kwargs)

                state_dict = load_state_dict(model_file, variant=variant)

                (
                    model,
                    missing_keys,
                    unexpected_keys,
                    mismatched_keys,
                    error_msgs,
                ) = cls._load_pretrained_model(
                    model,
                    state_dict,
                    model_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )

                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model

    def set_skip_temporal_layers(
        self,
        valid: bool,
    ) -> None:  # turn 3Dunet to 2Dunet
        # Recursively walk through all the children.
        # Any children which exposes the skip_temporal_layers parameter gets the message

        # 推断时使用参数控制refer_image和ip_adapter_image来控制，不需要这里了
        # if hasattr(self, "skip_refer_downblock_emb"):
        #     self.skip_refer_downblock_emb = valid

        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "skip_temporal_layers"):
                module.skip_temporal_layers = valid
            # if hasattr(module, "skip_refer_downblock_emb"):
            #     module.skip_refer_downblock_emb = valid

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def insert_spatial_self_attn_idx(self):
        attns, basic_transformers = self.spatial_self_attns
        self.self_attn_num = len(attns)
        for i, (name, layer) in enumerate(attns):
            logger.debug(
                f"{self.__class__.__name__}, {i}, {name}, {layer.__class__.__name__}"
            )
            layer.spatial_self_attn_idx = i
        for i, (name, layer) in enumerate(basic_transformers):
            logger.debug(
                f"{self.__class__.__name__}, {i}, {name}, {layer.__class__.__name__}"
            )
            layer.spatial_self_attn_idx = i

    @property
    def spatial_self_attns(
        self,
    ) -> List[Tuple[str, Attention]]:
        attns, spatial_transformers = self.get_attns(
            include="attentions", exclude="temp_attentions", attn_name="attn1"
        )
        attns = sorted(attns)
        spatial_transformers = sorted(spatial_transformers)
        return attns, spatial_transformers

    @property
    def spatial_cross_attns(
        self,
    ) -> List[Tuple[str, Attention]]:
        attns, spatial_transformers = self.get_attns(
            include="attentions", exclude="temp_attentions", attn_name="attn2"
        )
        attns = sorted(attns)
        spatial_transformers = sorted(spatial_transformers)
        return attns, spatial_transformers

    def get_attns(
        self,
        attn_name: str,
        include: str = None,
        exclude: str = None,
    ) -> List[Tuple[str, Attention]]:
        r"""
        Returns:
            `dict` of attention attns: A dictionary containing all attention attns used in the model with
            indexed by its weight name.
        """
        # set recursively
        attns = []
        spatial_transformers = []

        def fn_recursive_add_attns(
            name: str,
            module: torch.nn.Module,
            attns: List[Tuple[str, Attention]],
            spatial_transformers: List[Tuple[str, BasicTransformerBlock]],
        ):
            is_target = False
            if isinstance(module, BasicTransformerBlock) and hasattr(module, attn_name):
                is_target = True
                if include is not None:
                    is_target = include in name
                if exclude is not None:
                    is_target = exclude not in name
            if is_target:
                attns.append([f"{name}.{attn_name}", getattr(module, attn_name)])
                spatial_transformers.append([f"{name}", module])
            for sub_name, child in module.named_children():
                fn_recursive_add_attns(
                    f"{name}.{sub_name}", child, attns, spatial_transformers
                )

            return attns

        for name, module in self.named_children():
            fn_recursive_add_attns(name, module, attns, spatial_transformers)

        return attns, spatial_transformers
