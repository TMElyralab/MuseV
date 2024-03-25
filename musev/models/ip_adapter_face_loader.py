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

from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.ip_adapter_faceid import ProjPlusModel, MLPProjModel

from mmcm.vision.feature_extractor.clip_vision_extractor import (
    ImageClipVisionFeatureExtractor,
    ImageClipVisionFeatureExtractorV2,
)
from mmcm.vision.feature_extractor.insight_face_extractor import (
    InsightFaceExtractorNormEmb,
)


from .unet_loader import update_unet_with_sd
from .unet_3d_condition import UNet3DConditionModel
from .ip_adapter_loader import ip_adapter_keys_list

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# refer https://github.com/tencent-ailab/IP-Adapter/issues/168#issuecomment-1846771651
unet_keys_list = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
]


UNET2IPAadapter_Keys_MAPIING = {
    k: v for k, v in zip(unet_keys_list, ip_adapter_keys_list)
}


def load_ip_adapter_face_extractor_and_proj_by_name(
    model_name: str,
    ip_ckpt: Tuple[str, nn.Module],
    ip_image_encoder: Tuple[str, nn.Module] = None,
    cross_attention_dim: int = 768,
    clip_embeddings_dim: int = 1024,
    clip_extra_context_tokens: int = 4,
    ip_scale: float = 0.0,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    unet: nn.Module = None,
) -> nn.Module:
    if model_name == "IPAdapterFaceID":
        if ip_image_encoder is not None:
            ip_adapter_face_emb_extractor = InsightFaceExtractorNormEmb(
                pretrained_model_name_or_path=ip_image_encoder,
                dtype=dtype,
                device=device,
            )
        else:
            ip_adapter_face_emb_extractor = None
        ip_adapter_image_proj = MLPProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=clip_embeddings_dim,
            num_tokens=clip_extra_context_tokens,
        ).to(device, dtype=dtype)
    else:
        raise ValueError(
            f"unsupport model_name={model_name}, only support IPAdapter, IPAdapterPlus, IPAdapterFaceID"
        )
    ip_adapter_state_dict = torch.load(
        ip_ckpt,
        map_location="cpu",
    )
    ip_adapter_image_proj.load_state_dict(ip_adapter_state_dict["image_proj"])
    if unet is not None and "ip_adapter" in ip_adapter_state_dict:
        update_unet_ip_adapter_cross_attn_param(
            unet,
            ip_adapter_state_dict["ip_adapter"],
        )
        logger.info(
            f"update unet.spatial_cross_attn_ip_adapter parameter with {ip_ckpt}"
        )
    return (
        ip_adapter_face_emb_extractor,
        ip_adapter_image_proj,
    )


def update_unet_ip_adapter_cross_attn_param(
    unet: UNet3DConditionModel, ip_adapter_state_dict: Dict
) -> None:
    """use independent ip_adapter attn 中的 to_k, to_v in unet
    ip_adapter： like ['1.to_k_ip.weight', '1.to_v_ip.weight', '3.to_k_ip.weight']


    Args:
        unet (UNet3DConditionModel): _description_
        ip_adapter_state_dict (Dict): _description_
    """
    unet_spatial_cross_atnns = unet.spatial_cross_attns[0]
    unet_spatial_cross_atnns_dct = {k: v for k, v in unet_spatial_cross_atnns}
    for i, (unet_key_more, ip_adapter_key) in enumerate(
        UNET2IPAadapter_Keys_MAPIING.items()
    ):
        ip_adapter_value = ip_adapter_state_dict[ip_adapter_key]
        unet_key_more_spit = unet_key_more.split(".")
        unet_key = ".".join(unet_key_more_spit[:-3])
        suffix = ".".join(unet_key_more_spit[-3:])
        logger.debug(
            f"{i}: unet_key_more = {unet_key_more}, {unet_key}=unet_key, suffix={suffix}",
        )
        if ".ip_adapter_face_to_k" in suffix:
            with torch.no_grad():
                unet_spatial_cross_atnns_dct[
                    unet_key
                ].ip_adapter_face_to_k_ip.weight.copy_(ip_adapter_value.data)
        else:
            with torch.no_grad():
                unet_spatial_cross_atnns_dct[
                    unet_key
                ].ip_adapter_face_to_v_ip.weight.copy_(ip_adapter_value.data)
