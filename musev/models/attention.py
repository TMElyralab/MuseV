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

# Adapted from https://github.com/huggingface/diffusers/blob/64bf5d33b7ef1b1deac256bed7bd99b55020c4e0/src/diffusers/models/attention.py
from __future__ import annotations
from copy import deepcopy

from typing import Any, Dict, List, Literal, Optional, Callable, Tuple
import logging
from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention as DiffusersAttention
from diffusers.models.attention import (
    BasicTransformerBlock as DiffusersBasicTransformerBlock,
    AdaLayerNormZero,
    AdaLayerNorm,
    FeedForward,
)
from diffusers.models.attention_processor import AttnProcessor

from .attention_processor import IPAttention, BaseIPAttnProcessor


logger = logging.getLogger(__name__)


def not_use_xformers_anyway(
    use_memory_efficient_attention_xformers: bool,
    attention_op: Optional[Callable] = None,
):
    return None


@maybe_allow_in_graph
class BasicTransformerBlock(DiffusersBasicTransformerBlock):
    print_idx = 0

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        attention_type: str = "default",
        allow_xformers: bool = True,
        cross_attn_temporal_cond: bool = False,
        image_scale: float = 1.0,
        processor: AttnProcessor | None = None,
        ip_adapter_cross_attn: bool = False,
        need_t2i_facein: bool = False,
        need_t2i_ip_adapter_face: bool = False,
    ):
        if not only_cross_attention and double_self_attention:
            cross_attention_dim = None
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout,
            cross_attention_dim,
            activation_fn,
            num_embeds_ada_norm,
            attention_bias,
            only_cross_attention,
            double_self_attention,
            upcast_attention,
            norm_elementwise_affine,
            norm_type,
            final_dropout,
            attention_type,
        )

        self.attn1 = IPAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            cross_attn_temporal_cond=cross_attn_temporal_cond,
            image_scale=image_scale,
            ip_adapter_dim=cross_attention_dim
            if only_cross_attention
            else attention_head_dim,
            facein_dim=cross_attention_dim
            if only_cross_attention
            else attention_head_dim,
            processor=processor,
        )
        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )

            self.attn2 = IPAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim
                if not double_self_attention
                else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                cross_attn_temporal_cond=ip_adapter_cross_attn,
                need_t2i_facein=need_t2i_facein,
                need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                image_scale=image_scale,
                ip_adapter_dim=cross_attention_dim
                if not double_self_attention
                else attention_head_dim,
                facein_dim=cross_attention_dim
                if not double_self_attention
                else attention_head_dim,
                ip_adapter_face_dim=cross_attention_dim
                if not double_self_attention
                else attention_head_dim,
                processor=processor,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
        if self.attn1 is not None:
            if not allow_xformers:
                self.attn1.set_use_memory_efficient_attention_xformers = (
                    not_use_xformers_anyway
                )
        if self.attn2 is not None:
            if not allow_xformers:
                self.attn2.set_use_memory_efficient_attention_xformers = (
                    not_use_xformers_anyway
                )
        self.double_self_attention = double_self_attention
        self.only_cross_attention = only_cross_attention
        self.cross_attn_temporal_cond = cross_attn_temporal_cond
        self.image_scale = image_scale

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        self_attn_block_embs: Optional[Tuple[List[torch.Tensor], List[None]]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        # 特殊AttnProcessor需要的入参 在 cross_attention_kwargs 准备
        # special AttnProcessor needs input parameters in cross_attention_kwargs
        original_cross_attention_kwargs = {
            k: v
            for k, v in cross_attention_kwargs.items()
            if k
            not in [
                "num_frames",
                "sample_index",
                "vision_conditon_frames_sample_index",
                "vision_cond",
                "vision_clip_emb",
                "ip_adapter_scale",
                "face_emb",
                "facein_scale",
                "ip_adapter_face_emb",
                "ip_adapter_face_scale",
                "do_classifier_free_guidance",
            ]
        }

        if "do_classifier_free_guidance" in cross_attention_kwargs:
            do_classifier_free_guidance = cross_attention_kwargs[
                "do_classifier_free_guidance"
            ]
        else:
            do_classifier_free_guidance = False

        # 2. Prepare GLIGEN inputs
        original_cross_attention_kwargs = (
            original_cross_attention_kwargs.copy()
            if original_cross_attention_kwargs is not None
            else {}
        )
        gligen_kwargs = original_cross_attention_kwargs.pop("gligen", None)

        # 返回self_attn的结果，适用于referencenet的输出给其他Unet来使用
        # return the result of self_attn, which is suitable for the output of referencenet to be used by other Unet
        if (
            self_attn_block_embs is not None
            and self_attn_block_embs_mode.lower() == "write"
        ):
            # self_attn_block_emb = self.attn1.head_to_batch_dim(attn_output, out_dim=4)
            self_attn_block_emb = norm_hidden_states
            if not hasattr(self, "spatial_self_attn_idx"):
                raise ValueError(
                    "must call unet.insert_spatial_self_attn_idx to generate spatial attn index"
                )
            basick_transformer_idx = self.spatial_self_attn_idx
            if self.print_idx == 0:
                logger.debug(
                    f"self_attn_block_embs, self_attn_block_embs_mode={self_attn_block_embs_mode}, "
                    f"basick_transformer_idx={basick_transformer_idx}, length={len(self_attn_block_embs)}, shape={self_attn_block_emb.shape}, "
                    # f"attn1 processor, {type(self.attn1.processor)}"
                )
            self_attn_block_embs[basick_transformer_idx] = self_attn_block_emb

        # read and put referencenet emb into cross_attention_kwargs, which would be fused into attn_processor
        if (
            self_attn_block_embs is not None
            and self_attn_block_embs_mode.lower() == "read"
        ):
            basick_transformer_idx = self.spatial_self_attn_idx
            if not hasattr(self, "spatial_self_attn_idx"):
                raise ValueError(
                    "must call unet.insert_spatial_self_attn_idx to generate spatial attn index"
                )
            if self.print_idx == 0:
                logger.debug(
                    f"refer_self_attn_emb: , self_attn_block_embs_mode={self_attn_block_embs_mode}, "
                    f"length={len(self_attn_block_embs)}, idx={basick_transformer_idx}, "
                    # f"attn1 processor, {type(self.attn1.processor)}, "
                )
            ref_emb = self_attn_block_embs[basick_transformer_idx]
            cross_attention_kwargs["refer_emb"] = ref_emb
            if self.print_idx == 0:
                logger.debug(
                    f"unet attention read, {self.spatial_self_attn_idx}",
                )
                # ------------------------------warning-----------------------
                # 这两行由于使用了ref_emb会导致和checkpoint_train相关的训练错误，具体未知，留在这里作为警示
                # bellow annoated code will cause training error, keep it here as a warning
                # logger.debug(f"ref_emb shape,{ref_emb.shape}, {ref_emb.mean()}")
                # logger.debug(
                # f"norm_hidden_states shape, {norm_hidden_states.shape}, {norm_hidden_states.mean()}",
                # )
        if self.attn1 is None:
            self.print_idx += 1
            return norm_hidden_states
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **(
                cross_attention_kwargs
                if isinstance(self.attn1.processor, BaseIPAttnProcessor)
                else original_cross_attention_kwargs
            ),
        )

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 推断的时候，对于uncondition_部分独立生成，排除掉 refer_emb，
        # 首帧等的影响，避免生成参考了refer_emb、首帧等，又在uncond上去除了
        # in inference stage, eliminate influence of refer_emb, vis_cond on unconditionpart
        # to avoid use that, and then eliminate in pipeline
        # refer to moore-animate anyone

        # do_classifier_free_guidance = False
        if self.print_idx == 0:
            logger.debug(f"do_classifier_free_guidance={do_classifier_free_guidance},")
        if do_classifier_free_guidance:
            hidden_states_c = attn_output.clone()
            _uc_mask = (
                torch.Tensor(
                    [1] * (norm_hidden_states.shape[0] // 2)
                    + [0] * (norm_hidden_states.shape[0] // 2)
                )
                .to(norm_hidden_states.device)
                .bool()
            )
            hidden_states_c[_uc_mask] = self.attn1(
                norm_hidden_states[_uc_mask],
                encoder_hidden_states=norm_hidden_states[_uc_mask],
                attention_mask=attention_mask,
            )
            attn_output = hidden_states_c.clone()

        if "refer_emb" in cross_attention_kwargs:
            del cross_attention_kwargs["refer_emb"]

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        # 2.5 ends

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )

            # 特殊AttnProcessor需要的入参 在 cross_attention_kwargs 准备
            # special AttnProcessor needs input parameters in cross_attention_kwargs
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states
                if not self.double_self_attention
                else None,
                attention_mask=encoder_attention_mask,
                **(
                    original_cross_attention_kwargs
                    if not isinstance(self.attn2.processor, BaseIPAttnProcessor)
                    else cross_attention_kwargs
                ),
            )
            if self.print_idx == 0:
                logger.debug(
                    f"encoder_hidden_states, type={type(encoder_hidden_states)}"
                )
                if encoder_hidden_states is not None:
                    logger.debug(
                        f"encoder_hidden_states, ={encoder_hidden_states.shape}"
                    )

            # encoder_hidden_states_tmp = (
            #     encoder_hidden_states
            #     if not self.double_self_attention
            #     else norm_hidden_states
            # )
            # if do_classifier_free_guidance:
            #     hidden_states_c = attn_output.clone()
            #     _uc_mask = (
            #         torch.Tensor(
            #             [1] * (norm_hidden_states.shape[0] // 2)
            #             + [0] * (norm_hidden_states.shape[0] // 2)
            #         )
            #         .to(norm_hidden_states.device)
            #         .bool()
            #     )
            #     hidden_states_c[_uc_mask] = self.attn2(
            #         norm_hidden_states[_uc_mask],
            #         encoder_hidden_states=encoder_hidden_states_tmp[_uc_mask],
            #         attention_mask=attention_mask,
            #     )
            #     attn_output = hidden_states_c.clone()
            hidden_states = attn_output + hidden_states
        # 4. Feed-forward
        if self.norm3 is not None and self.ff is not None:
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                    raise ValueError(
                        f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                    )

                num_chunks = (
                    norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                )
                ff_output = torch.cat(
                    [
                        self.ff(hid_slice, scale=lora_scale)
                        for hid_slice in norm_hidden_states.chunk(
                            num_chunks, dim=self._chunk_dim
                        )
                    ],
                    dim=self._chunk_dim,
                )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states
        self.print_idx += 1
        return hidden_states
