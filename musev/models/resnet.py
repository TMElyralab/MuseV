# Copyright 2023 The HuggingFace Team. All rights reserved.
# `TemporalConvLayer` Copyright 2023 Alibaba DAMO-VILAB, The ModelScope Team and The HuggingFace Team. All rights reserved.
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

# Adapted from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/resnet.py
from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.models.resnet import TemporalConvLayer as DiffusersTemporalConvLayer
from ..data.data_util import batch_index_fill, batch_index_select
from . import Model_Register


@Model_Register.register
class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(
        self,
        in_dim,
        out_dim=None,
        dropout=0.0,
        keep_content_condition: bool = False,
        femb_channels: Optional[int] = None,
        need_temporal_weight: bool = True,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.keep_content_condition = keep_content_condition
        self.femb_channels = femb_channels
        self.need_temporal_weight = need_temporal_weight
        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        # zero out the last layer params,so the conv block is identity
        #         nn.init.zeros_(self.conv4[-1].weight)
        #         nn.init.zeros_(self.conv4[-1].bias)
        self.temporal_weight = nn.Parameter(
            torch.tensor(
                [
                    1e-5,
                ]
            )
        )  # initialize parameter with 0
        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)
        self.skip_temporal_layers = False  # Whether to skip temporal layer

    def forward(
        self,
        hidden_states,
        num_frames=1,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        femb: torch.Tensor = None,
    ):
        if self.skip_temporal_layers is True:
            return hidden_states
        hidden_states_dtype = hidden_states.dtype
        hidden_states = rearrange(
            hidden_states, "(b t) c h w -> b c t h w", t=num_frames
        )
        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)
        # 保留condition对应的frames，便于保持前序内容帧，提升一致性
        if self.keep_content_condition:
            mask = torch.ones_like(hidden_states, device=hidden_states.device)
            mask = batch_index_fill(
                mask, dim=2, index=vision_conditon_frames_sample_index, value=0
            )
            if self.need_temporal_weight:
                hidden_states = (
                    identity + torch.abs(self.temporal_weight) * mask * hidden_states
                )
            else:
                hidden_states = identity + mask * hidden_states
        else:
            if self.need_temporal_weight:
                hidden_states = (
                    identity + torch.abs(self.temporal_weight) * hidden_states
                )
            else:
                hidden_states = identity + hidden_states
        hidden_states = rearrange(hidden_states, " b c t h w -> (b t) c h w")
        hidden_states = hidden_states.to(dtype=hidden_states_dtype)
        return hidden_states
