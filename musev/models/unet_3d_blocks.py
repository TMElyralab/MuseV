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

# Adapted from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/unet_3d_blocks.py

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import logging

import torch
from torch import nn

from diffusers.utils import is_torch_version
from diffusers.models.transformer_2d import (
    Transformer2DModel as DiffusersTransformer2DModel,
)
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from ..data.data_util import batch_adain_conditioned_tensor

from .resnet import TemporalConvLayer
from .temporal_transformer import TransformerTemporalModel
from .transformer_2d import Transformer2DModel
from .attention_processor import ReferEmbFuseAttention


logger = logging.getLogger(__name__)

# 注：
#   (1) 原代码的`use_linear_projection`默认值均为True，与2D-SD模型不符，load时报错。因此均改为False
#   (2) 原代码调用`Transformer2DModel`的输入参数顺序为n_channels // attn_num_head_channels, attn_num_head_channels,
#       与2D-SD模型不符。因此把顺序交换
#   (3) 增加了temporal attention用的frame embedding输入

# note:
# 1. The default value of `use_linear_projection` in the original code is True, which is inconsistent with the 2D-SD model and causes an error when loading. Therefore, it is changed to False.
# 2. The original code calls `Transformer2DModel` with the input parameter order of n_channels // attn_num_head_channels, attn_num_head_channels, which is inconsistent with the 2D-SD model. Therefore, the order is reversed.
# 3. Added the frame embedding input used by the temporal attention


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    femb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    temporal_transformer: Union[nn.Module, None] = TransformerTemporalModel,
    temporal_conv_block: Union[nn.Module, None] = TemporalConvLayer,
    need_spatial_position_emb: bool = False,
    need_t2i_ip_adapter: bool = False,
    ip_adapter_cross_attn: bool = False,
    need_t2i_facein: bool = False,
    need_t2i_ip_adapter_face: bool = False,
    need_adain_temporal_cond: bool = False,
    resnet_2d_skip_time_act: bool = False,
    need_refer_emb: bool = False,
):
    if (isinstance(down_block_type, str) and down_block_type == "DownBlock3D") or (
        isinstance(down_block_type, nn.Module)
        and down_block_type.__name__ == "DownBlock3D"
    ):
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            femb_channels=femb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_conv_block=temporal_conv_block,
            need_adain_temporal_cond=need_adain_temporal_cond,
            resnet_2d_skip_time_act=resnet_2d_skip_time_act,
            need_refer_emb=need_refer_emb,
            attn_num_head_channels=attn_num_head_channels,
        )
    elif (
        isinstance(down_block_type, str) and down_block_type == "CrossAttnDownBlock3D"
    ) or (
        isinstance(down_block_type, nn.Module)
        and down_block_type.__name__ == "CrossAttnDownBlock3D"
    ):
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock3D"
            )
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            femb_channels=femb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_conv_block=temporal_conv_block,
            temporal_transformer=temporal_transformer,
            need_spatial_position_emb=need_spatial_position_emb,
            need_t2i_ip_adapter=need_t2i_ip_adapter,
            ip_adapter_cross_attn=ip_adapter_cross_attn,
            need_t2i_facein=need_t2i_facein,
            need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
            need_adain_temporal_cond=need_adain_temporal_cond,
            resnet_2d_skip_time_act=resnet_2d_skip_time_act,
            need_refer_emb=need_refer_emb,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    femb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    temporal_conv_block: Union[nn.Module, None] = TemporalConvLayer,
    temporal_transformer: Union[nn.Module, None] = TransformerTemporalModel,
    need_spatial_position_emb: bool = False,
    need_t2i_ip_adapter: bool = False,
    ip_adapter_cross_attn: bool = False,
    need_t2i_facein: bool = False,
    need_t2i_ip_adapter_face: bool = False,
    need_adain_temporal_cond: bool = False,
    resnet_2d_skip_time_act: bool = False,
):
    if (isinstance(up_block_type, str) and up_block_type == "UpBlock3D") or (
        isinstance(up_block_type, nn.Module) and up_block_type.__name__ == "UpBlock3D"
    ):
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            femb_channels=femb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_conv_block=temporal_conv_block,
            need_adain_temporal_cond=need_adain_temporal_cond,
            resnet_2d_skip_time_act=resnet_2d_skip_time_act,
        )
    elif (isinstance(up_block_type, str) and up_block_type == "CrossAttnUpBlock3D") or (
        isinstance(up_block_type, nn.Module)
        and up_block_type.__name__ == "CrossAttnUpBlock3D"
    ):
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock3D"
            )
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            femb_channels=femb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_conv_block=temporal_conv_block,
            temporal_transformer=temporal_transformer,
            need_spatial_position_emb=need_spatial_position_emb,
            need_t2i_ip_adapter=need_t2i_ip_adapter,
            ip_adapter_cross_attn=ip_adapter_cross_attn,
            need_t2i_facein=need_t2i_facein,
            need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
            need_adain_temporal_cond=need_adain_temporal_cond,
            resnet_2d_skip_time_act=resnet_2d_skip_time_act,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    print_idx = 0

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        femb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        temporal_conv_block: Union[nn.Module, None] = TemporalConvLayer,
        temporal_transformer: Union[nn.Module, None] = TransformerTemporalModel,
        need_spatial_position_emb: bool = False,
        need_t2i_ip_adapter: bool = False,
        ip_adapter_cross_attn: bool = False,
        need_t2i_facein: bool = False,
        need_t2i_ip_adapter_face: bool = False,
        need_adain_temporal_cond: bool = False,
        resnet_2d_skip_time_act: bool = False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                skip_time_act=resnet_2d_skip_time_act,
            )
        ]
        if temporal_conv_block is not None:
            temp_convs = [
                temporal_conv_block(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                    femb_channels=femb_channels,
                )
            ]
        else:
            temp_convs = [None]
        attentions = []
        temp_attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    cross_attn_temporal_cond=need_t2i_ip_adapter,
                    ip_adapter_cross_attn=ip_adapter_cross_attn,
                    need_t2i_facein=need_t2i_facein,
                    need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                )
            )
            if temporal_transformer is not None:
                temp_attention = temporal_transformer(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    femb_channels=femb_channels,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    need_spatial_position_emb=need_spatial_position_emb,
                )
            else:
                temp_attention = None
            temp_attentions.append(temp_attention)
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=resnet_2d_skip_time_act,
                )
            )
            if temporal_conv_block is not None:
                temp_convs.append(
                    temporal_conv_block(
                        in_channels,
                        in_channels,
                        dropout=0.1,
                        femb_channels=femb_channels,
                    )
                )
            else:
                temp_convs.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        self.need_adain_temporal_cond = need_adain_temporal_cond

    def forward(
        self,
        hidden_states,
        temb=None,
        femb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        spatial_position_emb: torch.Tensor = None,
        refer_self_attn_emb: List[torch.Tensor] = None,
        refer_self_attn_emb_mode: Literal["read", "write"] = "read",
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        if self.temp_convs[0] is not None:
            hidden_states = self.temp_convs[0](
                hidden_states,
                femb=femb,
                num_frames=num_frames,
                sample_index=sample_index,
                vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
            )
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                self_attn_block_embs=refer_self_attn_emb,
                self_attn_block_embs_mode=refer_self_attn_emb_mode,
            ).sample
            if temp_attn is not None:
                hidden_states = temp_attn(
                    hidden_states,
                    femb=femb,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_hidden_states=encoder_hidden_states,
                    sample_index=sample_index,
                    vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    spatial_position_emb=spatial_position_emb,
                ).sample
            hidden_states = resnet(hidden_states, temb)
            if temp_conv is not None:
                hidden_states = temp_conv(
                    hidden_states,
                    femb=femb,
                    num_frames=num_frames,
                    sample_index=sample_index,
                    vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                )
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
        self.print_idx += 1
        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    print_idx = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        femb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_conv_block: Union[nn.Module, None] = TemporalConvLayer,
        temporal_transformer: Union[nn.Module, None] = TransformerTemporalModel,
        need_spatial_position_emb: bool = False,
        need_t2i_ip_adapter: bool = False,
        ip_adapter_cross_attn: bool = False,
        need_t2i_facein: bool = False,
        need_t2i_ip_adapter_face: bool = False,
        need_adain_temporal_cond: bool = False,
        resnet_2d_skip_time_act: bool = False,
        need_refer_emb: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
        temp_convs = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        self.need_refer_emb = need_refer_emb
        if need_refer_emb:
            refer_emb_attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=resnet_2d_skip_time_act,
                )
            )
            if temporal_conv_block is not None:
                temp_convs.append(
                    temporal_conv_block(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                        femb_channels=femb_channels,
                    )
                )
            else:
                temp_convs.append(None)
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    cross_attn_temporal_cond=need_t2i_ip_adapter,
                    ip_adapter_cross_attn=ip_adapter_cross_attn,
                    need_t2i_facein=need_t2i_facein,
                    need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                )
            )
            if temporal_transformer is not None:
                temp_attention = temporal_transformer(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    femb_channels=femb_channels,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    need_spatial_position_emb=need_spatial_position_emb,
                )
            else:
                temp_attention = None
            temp_attentions.append(temp_attention)

            if need_refer_emb:
                refer_emb_attns.append(
                    ReferEmbFuseAttention(
                        query_dim=out_channels,
                        heads=attn_num_head_channels,
                        dim_head=out_channels // attn_num_head_channels,
                        dropout=0,
                        bias=False,
                        cross_attention_dim=None,
                        upcast_attention=False,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
            if need_refer_emb:
                refer_emb_attns.append(
                    ReferEmbFuseAttention(
                        query_dim=out_channels,
                        heads=attn_num_head_channels,
                        dim_head=out_channels // attn_num_head_channels,
                        dropout=0,
                        bias=False,
                        cross_attention_dim=None,
                        upcast_attention=False,
                    )
                )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False
        self.need_adain_temporal_cond = need_adain_temporal_cond
        if need_refer_emb:
            self.refer_emb_attns = nn.ModuleList(refer_emb_attns)
        logger.debug(f"cross attn downblock 3d need_refer_emb, {self.need_refer_emb}")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        femb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        spatial_position_emb: torch.Tensor = None,
        refer_embs: Optional[List[torch.Tensor]] = None,
        refer_self_attn_emb: List[torch.Tensor] = None,
        refer_self_attn_emb_mode: Literal["read", "write"] = "read",
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()
        for i_downblock, (resnet, temp_conv, attn, temp_attn) in enumerate(
            zip(self.resnets, self.temp_convs, self.attentions, self.temp_attentions)
        ):
            # print("crossattndownblock3d, attn,", type(attn), cross_attention_kwargs)
            if self.training and self.gradient_checkpointing:
                if self.print_idx == 0:
                    logger.debug(
                        f"self.training and self.gradient_checkpointing={self.training and self.gradient_checkpointing}"
                    )

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                if self.print_idx == 0:
                    logger.debug(f"unet3d after resnet {hidden_states.mean()}")
                if temp_conv is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        num_frames,
                        sample_index,
                        vision_conditon_frames_sample_index,
                        femb,
                        **ckpt_kwargs,
                    )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # added_cond_kwargs
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    refer_self_attn_emb,
                    refer_self_attn_emb_mode,
                    **ckpt_kwargs,
                )[0]
                if temp_attn is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False),
                        hidden_states,
                        femb,
                        # None,  # encoder_hidden_states,
                        encoder_hidden_states,
                        None,  # timestep
                        None,  # class_labels
                        num_frames,
                        cross_attention_kwargs,
                        sample_index,
                        vision_conditon_frames_sample_index,
                        spatial_position_emb,
                        **ckpt_kwargs,
                    )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                if self.print_idx == 0:
                    logger.debug(f"unet3d after resnet {hidden_states.mean()}")
                if temp_conv is not None:
                    hidden_states = temp_conv(
                        hidden_states,
                        femb=femb,
                        num_frames=num_frames,
                        sample_index=sample_index,
                        vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    self_attn_block_embs=refer_self_attn_emb,
                    self_attn_block_embs_mode=refer_self_attn_emb_mode,
                ).sample
                if temp_attn is not None:
                    hidden_states = temp_attn(
                        hidden_states,
                        femb=femb,
                        num_frames=num_frames,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        sample_index=sample_index,
                        vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                        spatial_position_emb=spatial_position_emb,
                    ).sample
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
            # 使用 attn 的方式 来融合 down_block_refer_emb
            if self.print_idx == 0:
                logger.debug(
                    f"downblock, {i_downblock}, self.need_refer_emb={self.need_refer_emb}"
                )
            if self.need_refer_emb and refer_embs is not None:
                if self.print_idx == 0:
                    logger.debug(
                        f"{i_downblock}, self.refer_emb_attns {refer_embs[i_downblock].shape}"
                    )
                hidden_states = self.refer_emb_attns[i_downblock](
                    hidden_states, refer_embs[i_downblock], num_frames=num_frames
                )
            else:
                if self.print_idx == 0:
                    logger.debug(f"crossattndownblock refer_emb_attns, no this step")
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
            # 使用 attn 的方式 来融合 down_block_refer_emb
            # TODO: adain和 refer_emb的顺序
            # TODO：adain 首帧特征还是refer_emb的
            if self.need_refer_emb and refer_embs is not None:
                i_downblock += 1
                hidden_states = self.refer_emb_attns[i_downblock](
                    hidden_states, refer_embs[i_downblock], num_frames=num_frames
                )
            output_states += (hidden_states,)
        self.print_idx += 1
        return hidden_states, output_states


class DownBlock3D(nn.Module):
    print_idx = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        femb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        temporal_conv_block: Union[nn.Module, None] = TemporalConvLayer,
        need_adain_temporal_cond: bool = False,
        resnet_2d_skip_time_act: bool = False,
        need_refer_emb: bool = False,
        attn_num_head_channels: int = 1,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        self.need_refer_emb = need_refer_emb
        if need_refer_emb:
            refer_emb_attns = []
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=resnet_2d_skip_time_act,
                )
            )
            if temporal_conv_block is not None:
                temp_convs.append(
                    temporal_conv_block(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                        femb_channels=femb_channels,
                    )
                )
            else:
                temp_convs.append(None)
            if need_refer_emb:
                refer_emb_attns.append(
                    ReferEmbFuseAttention(
                        query_dim=out_channels,
                        heads=attn_num_head_channels,
                        dim_head=out_channels // attn_num_head_channels,
                        dropout=0,
                        bias=False,
                        cross_attention_dim=None,
                        upcast_attention=False,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
            if need_refer_emb:
                refer_emb_attns.append(
                    ReferEmbFuseAttention(
                        query_dim=out_channels,
                        heads=attn_num_head_channels,
                        dim_head=out_channels // attn_num_head_channels,
                        dropout=0,
                        bias=False,
                        cross_attention_dim=None,
                        upcast_attention=False,
                    )
                )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False
        self.need_adain_temporal_cond = need_adain_temporal_cond
        if need_refer_emb:
            self.refer_emb_attns = nn.ModuleList(refer_emb_attns)

    def forward(
        self,
        hidden_states,
        temb=None,
        num_frames=1,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        spatial_position_emb: torch.Tensor = None,
        femb=None,
        refer_embs: Optional[Tuple[torch.Tensor]] = None,
        refer_self_attn_emb: List[torch.Tensor] = None,
        refer_self_attn_emb_mode: Literal["read", "write"] = "read",
    ):
        output_states = ()

        for i_downblock, (resnet, temp_conv) in enumerate(
            zip(self.resnets, self.temp_convs)
        ):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                if temp_conv is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        num_frames,
                        sample_index,
                        vision_conditon_frames_sample_index,
                        femb,
                        **ckpt_kwargs,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)
                if temp_conv is not None:
                    hidden_states = temp_conv(
                        hidden_states,
                        femb=femb,
                        num_frames=num_frames,
                        sample_index=sample_index,
                        vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    )
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
            if self.need_refer_emb and refer_embs is not None:
                hidden_states = self.refer_emb_attns[i_downblock](
                    hidden_states, refer_embs[i_downblock], num_frames=num_frames
                )
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
            if self.need_refer_emb and refer_embs is not None:
                i_downblock += 1
                hidden_states = self.refer_emb_attns[i_downblock](
                    hidden_states, refer_embs[i_downblock], num_frames=num_frames
                )
            output_states += (hidden_states,)
        self.print_idx += 1
        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    print_idx = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        femb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_conv_block: Union[nn.Module, None] = TemporalConvLayer,
        temporal_transformer: Union[nn.Module, None] = TransformerTemporalModel,
        need_spatial_position_emb: bool = False,
        need_t2i_ip_adapter: bool = False,
        ip_adapter_cross_attn: bool = False,
        need_t2i_facein: bool = False,
        need_t2i_ip_adapter_face: bool = False,
        need_adain_temporal_cond: bool = False,
        resnet_2d_skip_time_act: bool = False,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=resnet_2d_skip_time_act,
                )
            )
            if temporal_conv_block is not None:
                temp_convs.append(
                    temporal_conv_block(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                        femb_channels=femb_channels,
                    )
                )
            else:
                temp_convs.append(None)
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    cross_attn_temporal_cond=need_t2i_ip_adapter,
                    ip_adapter_cross_attn=ip_adapter_cross_attn,
                    need_t2i_facein=need_t2i_facein,
                    need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                )
            )
            if temporal_transformer is not None:
                temp_attention = temporal_transformer(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    femb_channels=femb_channels,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    need_spatial_position_emb=need_spatial_position_emb,
                )
            else:
                temp_attention = None
            temp_attentions.append(temp_attention)
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.need_adain_temporal_cond = need_adain_temporal_cond

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        femb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        spatial_position_emb: torch.Tensor = None,
        refer_self_attn_emb: List[torch.Tensor] = None,
        refer_self_attn_emb_mode: Literal["read", "write"] = "read",
    ):
        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                if temp_conv is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        num_frames,
                        sample_index,
                        vision_conditon_frames_sample_index,
                        femb,
                        **ckpt_kwargs,
                    )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # added_cond_kwargs
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    refer_self_attn_emb,
                    refer_self_attn_emb_mode,
                    **ckpt_kwargs,
                )[0]
                if temp_attn is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False),
                        hidden_states,
                        femb,
                        # None,  # encoder_hidden_states,
                        encoder_hidden_states,
                        None,  # timestep
                        None,  # class_labels
                        num_frames,
                        cross_attention_kwargs,
                        sample_index,
                        vision_conditon_frames_sample_index,
                        spatial_position_emb,
                        **ckpt_kwargs,
                    )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                if temp_conv is not None:
                    hidden_states = temp_conv(
                        hidden_states,
                        num_frames=num_frames,
                        femb=femb,
                        sample_index=sample_index,
                        vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    self_attn_block_embs=refer_self_attn_emb,
                    self_attn_block_embs_mode=refer_self_attn_emb_mode,
                ).sample
                if temp_attn is not None:
                    hidden_states = temp_attn(
                        hidden_states,
                        femb=femb,
                        num_frames=num_frames,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_hidden_states=encoder_hidden_states,
                        sample_index=sample_index,
                        vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                        spatial_position_emb=spatial_position_emb,
                    ).sample
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
        self.print_idx += 1
        return hidden_states


class UpBlock3D(nn.Module):
    print_idx = 0

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        femb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        temporal_conv_block: Union[nn.Module, None] = TemporalConvLayer,
        need_adain_temporal_cond: bool = False,
        resnet_2d_skip_time_act: bool = False,
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=resnet_2d_skip_time_act,
                )
            )
            if temporal_conv_block is not None:
                temp_convs.append(
                    temporal_conv_block(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                        femb_channels=femb_channels,
                    )
                )
            else:
                temp_convs.append(None)
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.need_adain_temporal_cond = need_adain_temporal_cond

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        num_frames=1,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        spatial_position_emb: torch.Tensor = None,
        femb=None,
        refer_self_attn_emb: List[torch.Tensor] = None,
        refer_self_attn_emb_mode: Literal["read", "write"] = "read",
    ):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                if temp_conv is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        num_frames,
                        sample_index,
                        vision_conditon_frames_sample_index,
                        femb,
                        **ckpt_kwargs,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)
                if temp_conv is not None:
                    hidden_states = temp_conv(
                        hidden_states,
                        num_frames=num_frames,
                        femb=femb,
                        sample_index=sample_index,
                        vision_conditon_frames_sample_index=vision_conditon_frames_sample_index,
                    )
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            if (
                self.need_adain_temporal_cond
                and num_frames > 1
                and sample_index is not None
            ):
                if self.print_idx == 0:
                    logger.debug(f"adain to vision_condition")
                hidden_states = batch_adain_conditioned_tensor(
                    hidden_states,
                    num_frames=num_frames,
                    need_style_fidelity=False,
                    src_index=sample_index,
                    dst_index=vision_conditon_frames_sample_index,
                )
        self.print_idx += 1
        return hidden_states
