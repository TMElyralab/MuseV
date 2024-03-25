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
from pprint import pprint
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
from diffusers.models.modeling_utils import load_state_dict
from diffusers.utils import (
    logging,
)
from diffusers.utils.import_utils import is_xformers_available

from .referencenet import ReferenceNet2D
from .unet_loader import update_unet_with_sd


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_referencenet(
    sd_referencenet_model: Tuple[str, nn.Module],
    sd_model: nn.Module = None,
    need_self_attn_block_embs: bool = False,
    need_block_embs: bool = False,
    dtype: torch.dtype = torch.float16,
    cross_attention_dim: int = 768,
    subfolder: str = "unet",
):
    """
    Loads the ReferenceNet model.

    Args:
        sd_referencenet_model (Tuple[str, nn.Module] or str): The pretrained ReferenceNet model or the path to the model.
        sd_model (nn.Module, optional): The sd_model to update the ReferenceNet with. Defaults to None.
        need_self_attn_block_embs (bool, optional): Whether to compute self-attention block embeddings. Defaults to False.
        need_block_embs (bool, optional): Whether to compute block embeddings. Defaults to False.
        dtype (torch.dtype, optional): The data type of the tensors. Defaults to torch.float16.
        cross_attention_dim (int, optional): The dimension of the cross-attention. Defaults to 768.
        subfolder (str, optional): The subfolder of the model. Defaults to "unet".

    Returns:
        nn.Module: The loaded ReferenceNet model.
    """

    if isinstance(sd_referencenet_model, str):
        referencenet = ReferenceNet2D.from_pretrained(
            sd_referencenet_model,
            subfolder=subfolder,
            need_self_attn_block_embs=need_self_attn_block_embs,
            need_block_embs=need_block_embs,
            torch_dtype=dtype,
            cross_attention_dim=cross_attention_dim,
        )
    elif isinstance(sd_referencenet_model, nn.Module):
        referencenet = sd_referencenet_model
    if sd_model is not None:
        referencenet = update_unet_with_sd(referencenet, sd_model)
    return referencenet


def load_referencenet_by_name(
    model_name: str,
    sd_referencenet_model: Tuple[str, nn.Module],
    sd_model: nn.Module = None,
    cross_attention_dim: int = 768,
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """通过模型名字 初始化 referencenet，载入预训练参数，
        如希望后续通过简单名字就可以使用预训练模型，需要在这里完成定义
        init referencenet with model_name.
        if you want to use pretrained model with simple name, you need to define it here.
    Args:
        model_name (str): _description_
        sd_unet_model (Tuple[str, nn.Module]): _description_
        sd_model (Tuple[str, nn.Module]): _description_
        cross_attention_dim (int, optional): _description_. Defaults to 768.
        dtype (torch.dtype, optional): _description_. Defaults to torch.float16.

    Raises:
        ValueError: _description_

    Returns:
        nn.Module: _description_
    """
    if model_name in [
        "musev_referencenet",
    ]:
        unet = load_referencenet(
            sd_referencenet_model=sd_referencenet_model,
            sd_model=sd_model,
            cross_attention_dim=cross_attention_dim,
            dtype=dtype,
            need_self_attn_block_embs=False,
            need_block_embs=True,
            subfolder="referencenet",
        )
    else:
        raise ValueError(
            f"unsupport model_name={model_name}, only support ReferenceNet_V0_block13, ReferenceNet_V1_block13, ReferenceNet_V2_block13, ReferenceNet_V0_sefattn16"
        )
    return unet
