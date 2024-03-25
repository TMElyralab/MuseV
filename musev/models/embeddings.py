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

from einops import rearrange
import torch
from torch.nn import functional as F
import numpy as np

from diffusers.models.embeddings import get_2d_sincos_pos_embed_from_grid


# ref diffusers.models.embeddings.get_2d_sincos_pos_embed
def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size_w,
    grid_size_h,
    cls_token=False,
    extra_tokens=0,
    norm_length: bool = False,
    max_length: float = 2048,
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if norm_length and grid_size_h <= max_length and grid_size_w <= max_length:
        grid_h = np.linspace(0, max_length, grid_size_h)
        grid_w = np.linspace(0, max_length, grid_size_w)
    else:
        grid_h = np.arange(grid_size_h, dtype=np.float32)
        grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_w)  # here h goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def resize_spatial_position_emb(
    emb: torch.Tensor,
    height: int,
    width: int,
    scale: float = None,
    target_height: int = None,
    target_width: int = None,
) -> torch.Tensor:
    """_summary_

    Args:
        emb (torch.Tensor): b ( h w) d
        height (int): _description_
        width (int): _description_
        scale (float, optional): _description_. Defaults to None.
        target_height (int, optional): _description_. Defaults to None.
        target_width (int, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: b (target_height target_width) d
    """
    if scale is not None:
        target_height = int(height * scale)
        target_width = int(width * scale)
    emb = rearrange(emb, "(h w) (b d) ->b d h w", h=height, b=1)
    emb = F.interpolate(
        emb,
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    )
    emb = rearrange(emb, "b d h w-> (h w) (b d)")
    return emb
