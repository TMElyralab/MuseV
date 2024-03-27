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

from ..models.unet_3d_condition import UNet3DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def update_unet_with_sd(
    unet: nn.Module, sd_model: Tuple[str, nn.Module], subfolder: str = "unet"
):
    """更新T2V模型中的T2I参数. update t2i parameters in t2v model

    Args:
        unet (nn.Module): _description_
        sd_model (Tuple[str, nn.Module]): _description_

    Returns:
        _type_: _description_
    """
    # dtype = unet.dtype
    # TODO: in this way, sd_model_path must be absolute path, to be more dynamic
    if isinstance(sd_model, str):
        if os.path.isdir(sd_model):
            unet_state_dict = load_state_dict(
                os.path.join(sd_model, subfolder, "diffusion_pytorch_model.bin"),
            )
        elif os.path.isfile(sd_model):
            if sd_model.endswith("pth"):
                unet_state_dict = torch.load(sd_model, map_location="cpu")
                print(f"referencenet successful load ={sd_model} with torch.load")
            else:
                try:
                    unet_state_dict = load_state_dict(sd_model)
                    print(
                        f"referencenet successful load with {sd_model} with load_state_dict"
                    )
                except Exception as e:
                    print(e)

    elif isinstance(sd_model, nn.Module):
        unet_state_dict = sd_model.state_dict()
    else:
        raise ValueError(f"given {type(sd_model)}, but only support nn.Module or str")
    missing, unexpected = unet.load_state_dict(unet_state_dict, strict=False)
    assert len(unexpected) == 0, f"unet load_state_dict error, unexpected={unexpected}"
    # unet.to(dtype=dtype)
    return unet


def load_unet(
    sd_unet_model: Tuple[str, nn.Module],
    sd_model: Tuple[str, nn.Module] = None,
    cross_attention_dim: int = 768,
    temporal_transformer: str = "TransformerTemporalModel",
    temporal_conv_block: str = "TemporalConvLayer",
    need_spatial_position_emb: bool = False,
    need_transformer_in: bool = True,
    need_t2i_ip_adapter: bool = False,
    need_adain_temporal_cond: bool = False,
    t2i_ip_adapter_attn_processor: str = "IPXFormersAttnProcessor",
    keep_vision_condtion: bool = False,
    use_anivv1_cfg: bool = False,
    resnet_2d_skip_time_act: bool = False,
    dtype: torch.dtype = torch.float16,
    need_zero_vis_cond_temb: bool = True,
    norm_spatial_length: bool = True,
    spatial_max_length: int = 2048,
    need_refer_emb: bool = False,
    ip_adapter_cross_attn=False,
    t2i_crossattn_ip_adapter_attn_processor="T2IReferencenetIPAdapterXFormersAttnProcessor",
    need_t2i_facein: bool = False,
    need_t2i_ip_adapter_face: bool = False,
    strict: bool = True,
):
    """通过模型名字 初始化Unet，载入预训练参数. init unet with model_name.
    该部分都是通过 models.unet_3d_condition.py:UNet3DConditionModel 定义、训练的模型
    model is defined and trained in models.unet_3d_condition.py:UNet3DConditionModel

    Args:
        sd_unet_model (Tuple[str, nn.Module]): _description_
        sd_model (Tuple[str, nn.Module]): _description_
        cross_attention_dim (int, optional): _description_. Defaults to 768.
        temporal_transformer (str, optional): _description_. Defaults to "TransformerTemporalModel".
        temporal_conv_block (str, optional): _description_. Defaults to "TemporalConvLayer".
        need_spatial_position_emb (bool, optional): _description_. Defaults to False.
        need_transformer_in (bool, optional): _description_. Defaults to True.
        need_t2i_ip_adapter (bool, optional): _description_. Defaults to False.
        need_adain_temporal_cond (bool, optional): _description_. Defaults to False.
        t2i_ip_adapter_attn_processor (str, optional): _description_. Defaults to "IPXFormersAttnProcessor".
        keep_vision_condtion (bool, optional): _description_. Defaults to False.
        use_anivv1_cfg (bool, optional): _description_. Defaults to False.
        resnet_2d_skip_time_act (bool, optional): _description_. Defaults to False.
        dtype (torch.dtype, optional): _description_. Defaults to torch.float16.
        need_zero_vis_cond_temb (bool, optional): _description_. Defaults to True.
        norm_spatial_length (bool, optional): _description_. Defaults to True.
        spatial_max_length (int, optional): _description_. Defaults to 2048.

    Returns:
        _type_: _description_
    """
    if isinstance(sd_unet_model, str):
        unet = UNet3DConditionModel.from_pretrained_2d(
            sd_unet_model,
            subfolder="unet",
            temporal_transformer=temporal_transformer,
            temporal_conv_block=temporal_conv_block,
            cross_attention_dim=cross_attention_dim,
            need_spatial_position_emb=need_spatial_position_emb,
            need_transformer_in=need_transformer_in,
            need_t2i_ip_adapter=need_t2i_ip_adapter,
            need_adain_temporal_cond=need_adain_temporal_cond,
            t2i_ip_adapter_attn_processor=t2i_ip_adapter_attn_processor,
            keep_vision_condtion=keep_vision_condtion,
            use_anivv1_cfg=use_anivv1_cfg,
            resnet_2d_skip_time_act=resnet_2d_skip_time_act,
            torch_dtype=dtype,
            need_zero_vis_cond_temb=need_zero_vis_cond_temb,
            norm_spatial_length=norm_spatial_length,
            spatial_max_length=spatial_max_length,
            need_refer_emb=need_refer_emb,
            ip_adapter_cross_attn=ip_adapter_cross_attn,
            t2i_crossattn_ip_adapter_attn_processor=t2i_crossattn_ip_adapter_attn_processor,
            need_t2i_facein=need_t2i_facein,
            strict=strict,
            need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
        )
    elif isinstance(sd_unet_model, nn.Module):
        unet = sd_unet_model
    if sd_model is not None:
        unet = update_unet_with_sd(unet, sd_model)
    return unet


def load_unet_custom_unet(
    sd_unet_model: Tuple[str, nn.Module],
    sd_model: Tuple[str, nn.Module],
    unet_class: nn.Module,
):
    """
    通过模型名字 初始化Unet，载入预训练参数. init unet with model_name.
    该部分都是通过 不通过models.unet_3d_condition.py:UNet3DConditionModel 定义、训练的模型
    model is not defined in models.unet_3d_condition.py:UNet3DConditionModel
    Args:
        sd_unet_model (Tuple[str, nn.Module]): _description_
        sd_model (Tuple[str, nn.Module]): _description_
        unet_class (nn.Module): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(sd_unet_model, str):
        unet = unet_class.from_pretrained(
            sd_unet_model,
            subfolder="unet",
        )
    elif isinstance(sd_unet_model, nn.Module):
        unet = sd_unet_model

    # TODO: in this way, sd_model_path must be absolute path, to be more dynamic
    if isinstance(sd_model, str):
        unet_state_dict = load_state_dict(
            os.path.join(sd_model, "unet/diffusion_pytorch_model.bin"),
        )
    elif isinstance(sd_model, nn.Module):
        unet_state_dict = sd_model.state_dict()
    missing, unexpected = unet.load_state_dict(unet_state_dict, strict=False)
    assert (
        len(unexpected) == 0
    ), "unet load_state_dict error"  # Load scheduler, tokenizer and models.
    return unet


def load_unet_by_name(
    model_name: str,
    sd_unet_model: Tuple[str, nn.Module],
    sd_model: Tuple[str, nn.Module] = None,
    cross_attention_dim: int = 768,
    dtype: torch.dtype = torch.float16,
    need_t2i_facein: bool = False,
    need_t2i_ip_adapter_face: bool = False,
    strict: bool = True,
) -> nn.Module:
    """通过模型名字 初始化Unet，载入预训练参数. init unet with model_name.
        如希望后续通过简单名字就可以使用预训练模型，需要在这里完成定义
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
    if model_name in ["musev"]:
        unet = load_unet(
            sd_unet_model=sd_unet_model,
            sd_model=sd_model,
            need_spatial_position_emb=False,
            cross_attention_dim=cross_attention_dim,
            need_t2i_ip_adapter=True,
            need_adain_temporal_cond=True,
            t2i_ip_adapter_attn_processor="NonParamReferenceIPXFormersAttnProcessor",
            dtype=dtype,
        )
    elif model_name in [
        "musev_referencenet",
        "musev_referencenet_pose",
    ]:
        unet = load_unet(
            sd_unet_model=sd_unet_model,
            sd_model=sd_model,
            cross_attention_dim=cross_attention_dim,
            temporal_conv_block="TemporalConvLayer",
            need_transformer_in=False,
            temporal_transformer="TransformerTemporalModel",
            use_anivv1_cfg=True,
            resnet_2d_skip_time_act=True,
            need_t2i_ip_adapter=True,
            need_adain_temporal_cond=True,
            keep_vision_condtion=True,
            t2i_ip_adapter_attn_processor="NonParamReferenceIPXFormersAttnProcessor",
            dtype=dtype,
            need_refer_emb=True,
            need_zero_vis_cond_temb=True,
            ip_adapter_cross_attn=True,
            t2i_crossattn_ip_adapter_attn_processor="T2IReferencenetIPAdapterXFormersAttnProcessor",
            need_t2i_facein=need_t2i_facein,
            strict=strict,
            need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
        )
    else:
        raise ValueError(
            f"unsupport model_name={model_name}, only support musev, musev_referencenet, musev_referencenet_pose"
        )
    return unet
