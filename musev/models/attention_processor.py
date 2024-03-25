# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""该模型是自定义的attn_processor，实现特殊功能的 Attn功能。
    相对而言，开源代码经常会重新定义Attention 类，
    
    This module implements special AttnProcessor function with custom attn_processor class.
    While other open source code always modify Attention class.
"""
# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
from __future__ import annotations

import time
from typing import Any, Callable, Optional
import logging

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers
from diffusers.models.lora import LoRACompatibleLinear

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import (
    Attention as DiffusersAttention,
    AttnProcessor,
    AttnProcessor2_0,
)
from ..data.data_util import (
    batch_concat_two_tensor_with_index,
    batch_index_select,
    align_repeat_tensor_single_dim,
    batch_adain_conditioned_tensor,
)

from . import Model_Register

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class IPAttention(DiffusersAttention):
    r"""
    Modified Attention class which has special layer, like ip_apadapter_to_k, ip_apadapter_to_v,
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: str | None = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: int | None = None,
        norm_num_groups: int | None = None,
        spatial_norm_dim: int | None = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 0.00001,
        rescale_output_factor: float = 1,
        residual_connection: bool = False,
        _from_deprecated_attn_block=False,
        processor: AttnProcessor | None = None,
        cross_attn_temporal_cond: bool = False,
        image_scale: float = 1.0,
        ip_adapter_dim: int = None,
        need_t2i_facein: bool = False,
        facein_dim: int = None,
        need_t2i_ip_adapter_face: bool = False,
        ip_adapter_face_dim: int = None,
    ):
        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            added_kv_proj_dim,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
        )
        self.cross_attn_temporal_cond = cross_attn_temporal_cond
        self.image_scale = image_scale
        # 面向首帧的 ip_adapter
        # ip_apdater
        if cross_attn_temporal_cond:
            self.to_k_ip = LoRACompatibleLinear(ip_adapter_dim, query_dim, bias=False)
            self.to_v_ip = LoRACompatibleLinear(ip_adapter_dim, query_dim, bias=False)
        # facein
        self.need_t2i_facein = need_t2i_facein
        self.facein_dim = facein_dim
        if need_t2i_facein:
            raise NotImplementedError("facein")

        # ip_adapter_face
        self.need_t2i_ip_adapter_face = need_t2i_ip_adapter_face
        self.ip_adapter_face_dim = ip_adapter_face_dim
        if need_t2i_ip_adapter_face:
            self.ip_adapter_face_to_k_ip = LoRACompatibleLinear(
                ip_adapter_face_dim, query_dim, bias=False
            )
            self.ip_adapter_face_to_v_ip = LoRACompatibleLinear(
                ip_adapter_face_dim, query_dim, bias=False
            )

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op: Callable[..., Any] | None = None,
    ):
        if (
            "XFormers" in self.processor.__class__.__name__
            or "IP" in self.processor.__class__.__name__
        ):
            pass
        else:
            return super().set_use_memory_efficient_attention_xformers(
                use_memory_efficient_attention_xformers, attention_op
            )


@Model_Register.register
class BaseIPAttnProcessor(nn.Module):
    print_idx = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@Model_Register.register
class T2IReferencenetIPAdapterXFormersAttnProcessor(BaseIPAttnProcessor):
    r"""
    面向 ref_image的 self_attn的 IPAdapter
    """
    print_idx = 0

    def __init__(
        self,
        attention_op: Optional[Callable] = None,
    ):
        super().__init__()

        self.attention_op = attention_op

    def __call__(
        self,
        attn: IPAttention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: int = None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        refer_emb: torch.Tensor = None,
        vision_clip_emb: torch.Tensor = None,
        ip_adapter_scale: float = 1.0,
        face_emb: torch.Tensor = None,
        facein_scale: float = 1.0,
        ip_adapter_face_emb: torch.Tensor = None,
        ip_adapter_face_scale: float = 1.0,
        do_classifier_free_guidance: bool = False,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        encoder_hidden_states = align_repeat_tensor_single_dim(
            encoder_hidden_states, target_length=hidden_states.shape[0], dim=0
        )
        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        # for facein
        if self.print_idx == 0:
            logger.debug(
                f"T2IReferencenetIPAdapterXFormersAttnProcessor,type(face_emb)={type(face_emb)}, facein_scale={facein_scale}"
            )
        if facein_scale > 0 and face_emb is not None:
            raise NotImplementedError("facein")

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )

        # ip-adapter start
        if self.print_idx == 0:
            logger.debug(
                f"T2IReferencenetIPAdapterXFormersAttnProcessor,type(vision_clip_emb)={type(vision_clip_emb)}"
            )
        if ip_adapter_scale > 0 and vision_clip_emb is not None:
            if self.print_idx == 0:
                logger.debug(
                    f"T2I cross_attn, ipadapter, vision_clip_emb={vision_clip_emb.shape}, hidden_states={hidden_states.shape}, batch_size={batch_size}"
                )
            ip_key = attn.to_k_ip(vision_clip_emb)
            ip_value = attn.to_v_ip(vision_clip_emb)
            ip_key = align_repeat_tensor_single_dim(
                ip_key, target_length=batch_size, dim=0
            )
            ip_value = align_repeat_tensor_single_dim(
                ip_value, target_length=batch_size, dim=0
            )
            ip_key = attn.head_to_batch_dim(ip_key).contiguous()
            ip_value = attn.head_to_batch_dim(ip_value).contiguous()
            if self.print_idx == 0:
                logger.debug(
                    f"query={query.shape}, ip_key={ip_key.shape}, ip_value={ip_value.shape}"
                )
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            hidden_states_from_ip = xformers.ops.memory_efficient_attention(
                query,
                ip_key,
                ip_value,
                attn_bias=attention_mask,
                op=self.attention_op,
                scale=attn.scale,
            )
            hidden_states = hidden_states + ip_adapter_scale * hidden_states_from_ip
        # ip-adapter end

        # ip-adapter face start
        if self.print_idx == 0:
            logger.debug(
                f"T2IReferencenetIPAdapterXFormersAttnProcessor,type(ip_adapter_face_emb)={type(ip_adapter_face_emb)}"
            )
        if ip_adapter_face_scale > 0 and ip_adapter_face_emb is not None:
            if self.print_idx == 0:
                logger.debug(
                    f"T2I cross_attn, ipadapter face, ip_adapter_face_emb={vision_clip_emb.shape}, hidden_states={hidden_states.shape}, batch_size={batch_size}"
                )
            ip_key = attn.ip_adapter_face_to_k_ip(ip_adapter_face_emb)
            ip_value = attn.ip_adapter_face_to_v_ip(ip_adapter_face_emb)
            ip_key = align_repeat_tensor_single_dim(
                ip_key, target_length=batch_size, dim=0
            )
            ip_value = align_repeat_tensor_single_dim(
                ip_value, target_length=batch_size, dim=0
            )
            ip_key = attn.head_to_batch_dim(ip_key).contiguous()
            ip_value = attn.head_to_batch_dim(ip_value).contiguous()
            if self.print_idx == 0:
                logger.debug(
                    f"query={query.shape}, ip_key={ip_key.shape}, ip_value={ip_value.shape}"
                )
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            hidden_states_from_ip = xformers.ops.memory_efficient_attention(
                query,
                ip_key,
                ip_value,
                attn_bias=attention_mask,
                op=self.attention_op,
                scale=attn.scale,
            )
            hidden_states = (
                hidden_states + ip_adapter_face_scale * hidden_states_from_ip
            )
        # ip-adapter face end

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        self.print_idx += 1
        return hidden_states


@Model_Register.register
class NonParamT2ISelfReferenceXFormersAttnProcessor(BaseIPAttnProcessor):
    r"""
    面向首帧的 referenceonly attn,适用于 T2I的 self_attn
    referenceonly with vis_cond as key, value, in t2i self_attn.
    """
    print_idx = 0

    def __init__(
        self,
        attention_op: Optional[Callable] = None,
    ):
        super().__init__()

        self.attention_op = attention_op

    def __call__(
        self,
        attn: IPAttention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: int = None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        refer_emb: torch.Tensor = None,
        face_emb: torch.Tensor = None,
        vision_clip_emb: torch.Tensor = None,
        ip_adapter_scale: float = 1.0,
        facein_scale: float = 1.0,
        ip_adapter_face_emb: torch.Tensor = None,
        ip_adapter_face_scale: float = 1.0,
        do_classifier_free_guidance: bool = False,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        # vision_cond in same unet attn start
        if (
            vision_conditon_frames_sample_index is not None and num_frames > 1
        ) or refer_emb is not None:
            batchsize_timesize = hidden_states.shape[0]
            if self.print_idx == 0:
                logger.debug(
                    f"NonParamT2ISelfReferenceXFormersAttnProcessor 0, hidden_states={hidden_states.shape}, vision_conditon_frames_sample_index={vision_conditon_frames_sample_index}"
                )
            encoder_hidden_states = rearrange(
                hidden_states, "(b t) hw c -> b t hw c", t=num_frames
            )
            # if False:
            if vision_conditon_frames_sample_index is not None and num_frames > 1:
                ip_hidden_states = batch_index_select(
                    encoder_hidden_states,
                    dim=1,
                    index=vision_conditon_frames_sample_index,
                ).contiguous()
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor 1, vis_cond referenceonly, encoder_hidden_states={encoder_hidden_states.shape}, ip_hidden_states={ip_hidden_states.shape}"
                    )
                #
                ip_hidden_states = rearrange(
                    ip_hidden_states, "b t hw c -> b 1 (t hw) c"
                )
                ip_hidden_states = align_repeat_tensor_single_dim(
                    ip_hidden_states,
                    dim=1,
                    target_length=num_frames,
                )
                # b t hw c -> b t hw + hw c
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor 2, vis_cond referenceonly, encoder_hidden_states={encoder_hidden_states.shape}, ip_hidden_states={ip_hidden_states.shape}"
                    )
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states, ip_hidden_states], dim=2
                )
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor 3, hidden_states={hidden_states.shape}, ip_hidden_states={ip_hidden_states.shape}"
                    )
            # if False:
            if refer_emb is not None:  # and num_frames > 1:
                refer_emb = rearrange(refer_emb, "b c t h w->b 1 (t h w) c")
                refer_emb = align_repeat_tensor_single_dim(
                    refer_emb, target_length=num_frames, dim=1
                )
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor4, referencenet, encoder_hidden_states={encoder_hidden_states.shape}, refer_emb={refer_emb.shape}"
                    )
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states, refer_emb], dim=2
                )
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor5, referencenet, encoder_hidden_states={encoder_hidden_states.shape}, refer_emb={refer_emb.shape}"
                    )
            encoder_hidden_states = rearrange(
                encoder_hidden_states, "b t hw c -> (b t) hw c"
            )
        #  vision_cond in same unet attn end

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        encoder_hidden_states = align_repeat_tensor_single_dim(
            encoder_hidden_states, target_length=hidden_states.shape[0], dim=0
        )
        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        self.print_idx += 1

        return hidden_states


@Model_Register.register
class NonParamReferenceIPXFormersAttnProcessor(
    NonParamT2ISelfReferenceXFormersAttnProcessor
):
    def __init__(self, attention_op: Callable[..., Any] | None = None):
        super().__init__(attention_op)


@maybe_allow_in_graph
class ReferEmbFuseAttention(IPAttention):
    """使用 attention 融合 refernet 中的 emb 到 unet 对应的 latens 中
    # TODO: 目前只支持 bt hw c 的融合，后续考虑增加对 视频 bhw t c、b thw c的融合
    residual_connection: bool = True, 默认， 从不产生影响开始学习

    use attention to fuse referencenet emb into unet latents
    # TODO: by now, only support bt hw c, later consider to support bhw t c, b thw c
    residual_connection: bool = True, default, start from no effect

    Args:
        IPAttention (_type_): _description_
    """

    print_idx = 0

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: str | None = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: int | None = None,
        norm_num_groups: int | None = None,
        spatial_norm_dim: int | None = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 0.00001,
        rescale_output_factor: float = 1,
        residual_connection: bool = True,
        _from_deprecated_attn_block=False,
        processor: AttnProcessor | None = None,
        cross_attn_temporal_cond: bool = False,
        image_scale: float = 1,
    ):
        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            added_kv_proj_dim,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
            cross_attn_temporal_cond,
            image_scale,
        )
        self.processor = None
        # 配合residual,使一开始不影响之前结果
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: int = None,
    ) -> torch.Tensor:
        """fuse referencenet emb b c t2 h2 w2  into unet latents b c t1 h1 w1 with attn
        refer to musev/models/attention_processor.py::NonParamT2ISelfReferenceXFormersAttnProcessor

        Args:
            hidden_states (torch.FloatTensor): unet latents, (b t1) c h1 w1
            encoder_hidden_states (Optional[torch.FloatTensor], optional): referencenet emb b c2 t2 h2 w2. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            temb (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            scale (float, optional): _description_. Defaults to 1.0.
            num_frames (int, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        residual = hidden_states
        # start
        hidden_states = rearrange(
            hidden_states, "(b t) c h w -> b c t h w", t=num_frames
        )
        batch_size, channel, t1, height, width = hidden_states.shape
        if self.print_idx == 0:
            logger.debug(
                f"hidden_states={hidden_states.shape},encoder_hidden_states={encoder_hidden_states.shape}"
            )
        # concat  with hidden_states b c t1 h1 w1 in  hw channel into bt  (t2 + 1)hw c
        encoder_hidden_states = rearrange(
            encoder_hidden_states, " b c t2 h w-> b (t2 h w) c"
        )
        encoder_hidden_states = repeat(
            encoder_hidden_states, " b t2hw c -> (b t) t2hw c", t=t1
        )
        hidden_states = rearrange(hidden_states, " b c t h w-> (b t) (h w) c")
        # bt (t2+1)hw d
        encoder_hidden_states = torch.concat(
            [encoder_hidden_states, hidden_states], dim=1
        )
        # encoder_hidden_states = align_repeat_tensor_single_dim(
        #     encoder_hidden_states, target_length=hidden_states.shape[0], dim=0
        # )
        # end

        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        _, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = self.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = self.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = self.to_k(encoder_hidden_states, scale=scale)
        value = self.to_v(encoder_hidden_states, scale=scale)

        query = self.head_to_batch_dim(query).contiguous()
        key = self.head_to_batch_dim(key).contiguous()
        value = self.head_to_batch_dim(value).contiguous()

        # query: b t hw d
        # key/value: bt (t1+1)hw d
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            scale=self.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(
            hidden_states,
            "bt (h w) c-> bt c h w",
            h=height,
            w=width,
        )
        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor
        self.print_idx += 1
        return hidden_states
