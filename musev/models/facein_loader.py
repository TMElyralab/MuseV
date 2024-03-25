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

from mmcm.vision.feature_extractor.clip_vision_extractor import (
    ImageClipVisionFeatureExtractor,
    ImageClipVisionFeatureExtractorV2,
)
from mmcm.vision.feature_extractor.insight_face_extractor import InsightFaceExtractor

from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel

from .unet_loader import update_unet_with_sd
from .unet_3d_condition import UNet3DConditionModel
from .ip_adapter_loader import ip_adapter_keys_list

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# refer https://github.com/tencent-ailab/IP-Adapter/issues/168#issuecomment-1846771651
unet_keys_list = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
]


UNET2IPAadapter_Keys_MAPIING = {
    k: v for k, v in zip(unet_keys_list, ip_adapter_keys_list)
}


def load_facein_extractor_and_proj_by_name(
    model_name: str,
    ip_ckpt: Tuple[str, nn.Module],
    ip_image_encoder: Tuple[str, nn.Module] = None,
    cross_attention_dim: int = 768,
    clip_embeddings_dim: int = 512,
    clip_extra_context_tokens: int = 1,
    ip_scale: float = 0.0,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    unet: nn.Module = None,
) -> nn.Module:
    pass


def update_unet_facein_cross_attn_param(
    unet: UNet3DConditionModel, ip_adapter_state_dict: Dict
) -> None:
    """use independent ip_adapter attn 中的 to_k, to_v in unet
    ip_adapter： like ['1.to_k_ip.weight', '1.to_v_ip.weight', '3.to_k_ip.weight']的字典


    Args:
        unet (UNet3DConditionModel): _description_
        ip_adapter_state_dict (Dict): _description_
    """
    pass
