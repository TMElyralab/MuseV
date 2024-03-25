import gc
import os
from typing import Any, Callable, List, Literal, Union, Dict, Tuple
import logging

from safetensors.torch import load_file
from safetensors import safe_open
import torch
from torch import nn
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from .convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint,
)
from .convert_lora_safetensor_to_diffusers import convert_motion_lora_ckpt_to_diffusers

logger = logging.getLogger(__name__)


def update_pipeline_model_parameters(
    pipeline: DiffusionPipeline,
    model_path: str = None,
    lora_dict: Dict[str, Dict] = None,
    text_model_path: str = None,
    device="cuda",
    need_unload: bool = False,
):
    if model_path is not None:
        pipeline = update_pipeline_basemodel(
            pipeline, model_path, text_sd_model_path=text_model_path, device=device
        )
    if lora_dict is not None:
        pipeline, unload_dict = update_pipeline_lora_models(
            pipeline,
            lora_dict,
            device=device,
            need_unload=need_unload,
        )
        if need_unload:
            return pipeline, unload_dict
    return pipeline


def update_pipeline_basemodel(
    pipeline: DiffusionPipeline,
    model_path: str,
    text_sd_model_path: str,
    device: str = "cuda",
):
    """使用model_path更新pipeline中的基础参数

    Args:
        pipeline (DiffusionPipeline): _description_
        model_path (str): _description_
        text_sd_model_path (str): _description_
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        _type_: _description_
    """
    # load base
    if model_path.endswith(".ckpt"):
        state_dict = torch.load(model_path, map_location=device)
        pipeline.unet.load_state_dict(state_dict)
        print("update sd_model", model_path)
    elif model_path.endswith(".safetensors"):
        base_state_dict = {}
        with safe_open(model_path, framework="pt", device=device) as f:
            for key in f.keys():
                base_state_dict[key] = f.get_tensor(key)

        is_lora = all("lora" in k for k in base_state_dict.keys())
        assert is_lora == False, "Base model cannot be LoRA: {}".format(model_path)

        # vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(
            base_state_dict, pipeline.vae.config
        )
        pipeline.vae.load_state_dict(converted_vae_checkpoint)
        # unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            base_state_dict, pipeline.unet.config
        )
        pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # text_model
        pipeline.text_encoder = convert_ldm_clip_checkpoint(
            base_state_dict, text_sd_model_path
        )
        print("update sd_model", model_path)
    pipeline.to(device)
    return pipeline


# ref https://git.woa.com/innovative_tech/GenerationGroup/VirtualIdol/VidolImageDraw/blob/master/cfg.yaml
LORA_BLOCK_WEIGHT_MAP = {
    "FACE": [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "DEFACE": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "ALL": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "MIDD": [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "OUTALL": [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}


# ref https://git.woa.com/innovative_tech/GenerationGroup/VirtualIdol/VidolImageDraw/blob/master/pipeline/draw_pipe.py
def update_pipeline_lora_model(
    pipeline: DiffusionPipeline,
    lora: Union[str, Dict],
    alpha: float = 0.75,
    device: str = "cuda",
    lora_prefix_unet: str = "lora_unet",
    lora_prefix_text_encoder: str = "lora_te",
    lora_unet_layers=[
        "lora_unet_down_blocks_0_attentions_0",
        "lora_unet_down_blocks_0_attentions_1",
        "lora_unet_down_blocks_1_attentions_0",
        "lora_unet_down_blocks_1_attentions_1",
        "lora_unet_down_blocks_2_attentions_0",
        "lora_unet_down_blocks_2_attentions_1",
        "lora_unet_mid_block_attentions_0",
        "lora_unet_up_blocks_1_attentions_0",
        "lora_unet_up_blocks_1_attentions_1",
        "lora_unet_up_blocks_1_attentions_2",
        "lora_unet_up_blocks_2_attentions_0",
        "lora_unet_up_blocks_2_attentions_1",
        "lora_unet_up_blocks_2_attentions_2",
        "lora_unet_up_blocks_3_attentions_0",
        "lora_unet_up_blocks_3_attentions_1",
        "lora_unet_up_blocks_3_attentions_2",
    ],
    lora_block_weight_str: Literal["FACE", "ALL"] = "ALL",
    need_unload: bool = False,
):
    """使用 lora 更新pipeline中的unet相关参数

    Args:
        pipeline (DiffusionPipeline): _description_
        lora (Union[str, Dict]): _description_
        alpha (float, optional): _description_. Defaults to 0.75.
        device (str, optional): _description_. Defaults to "cuda".
        lora_prefix_unet (str, optional): _description_. Defaults to "lora_unet".
        lora_prefix_text_encoder (str, optional): _description_. Defaults to "lora_te".
        lora_unet_layers (list, optional): _description_. Defaults to [ "lora_unet_down_blocks_0_attentions_0", "lora_unet_down_blocks_0_attentions_1", "lora_unet_down_blocks_1_attentions_0", "lora_unet_down_blocks_1_attentions_1", "lora_unet_down_blocks_2_attentions_0", "lora_unet_down_blocks_2_attentions_1", "lora_unet_mid_block_attentions_0", "lora_unet_up_blocks_1_attentions_0", "lora_unet_up_blocks_1_attentions_1", "lora_unet_up_blocks_1_attentions_2", "lora_unet_up_blocks_2_attentions_0", "lora_unet_up_blocks_2_attentions_1", "lora_unet_up_blocks_2_attentions_2", "lora_unet_up_blocks_3_attentions_0", "lora_unet_up_blocks_3_attentions_1", "lora_unet_up_blocks_3_attentions_2", ].
        lora_block_weight_str (Literal[&quot;FACE&quot;, &quot;ALL&quot;], optional): _description_. Defaults to "ALL".
        need_unload (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # ref https://git.woa.com/innovative_tech/GenerationGroup/VirtualIdol/VidolImageDraw/blob/master/pipeline/tool.py#L20
    if lora_block_weight_str is not None:
        lora_block_weight = LORA_BLOCK_WEIGHT_MAP[lora_block_weight_str.upper()]
    if lora_block_weight:
        assert len(lora_block_weight) == 17
    # load lora weight
    if isinstance(lora, str):
        state_dict = load_file(lora, device=device)
    else:
        for k in lora:
            lora[k] = lora[k].to(device)
        state_dict = lora  # state_dict = {}

    visited = set()
    unload_dict = []
    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(lora_prefix_text_encoder + "_")[-1].split("_")
            )
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(lora_prefix_unet + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
            alpha_key = key.replace("lora_down.weight", "alpha")
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))
            alpha_key = key.replace("lora_up.weight", "alpha")

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            if alpha_key in state_dict:
                weight_scale = state_dict[alpha_key].item() / weight_up.shape[1]
            else:
                weight_scale = 1.0
            # adding_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            if len(weight_up.shape) == len(weight_down.shape):
                adding_weight = (
                    alpha
                    * weight_scale
                    * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                )
            else:
                adding_weight = (
                    alpha
                    * weight_scale
                    * torch.einsum("a b, b c h w -> a c h w", weight_up, weight_down)
                )
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            if alpha_key in state_dict:
                weight_scale = state_dict[alpha_key].item() / weight_up.shape[1]
            else:
                weight_scale = 1.0
            adding_weight = alpha * weight_scale * torch.mm(weight_up, weight_down)
        adding_weight = adding_weight.to(torch.float16)
        if lora_block_weight:
            if "text" in key:
                adding_weight *= lora_block_weight[0]
            else:
                for idx, layer in enumerate(lora_unet_layers):
                    if layer in key:
                        adding_weight *= lora_block_weight[idx + 1]
                        break

        curr_layer_unload_data = {"layer": curr_layer, "added_weight": adding_weight}
        curr_layer.weight.data += adding_weight

        unload_dict.append(curr_layer_unload_data)
        # update visited list
        for item in pair_keys:
            visited.add(item)
    if need_unload:
        return pipeline, unload_dict
    else:
        return pipeline


# ref https://git.woa.com/innovative_tech/GenerationGroup/VirtualIdol/VidolImageDraw/blob/master/pipeline/draw_pipe.py
def update_pipeline_lora_model_old(
    pipeline: DiffusionPipeline,
    lora: Union[str, Dict],
    alpha: float = 0.75,
    device: str = "cuda",
    lora_prefix_unet: str = "lora_unet",
    lora_prefix_text_encoder: str = "lora_te",
    lora_unet_layers=[
        "lora_unet_down_blocks_0_attentions_0",
        "lora_unet_down_blocks_0_attentions_1",
        "lora_unet_down_blocks_1_attentions_0",
        "lora_unet_down_blocks_1_attentions_1",
        "lora_unet_down_blocks_2_attentions_0",
        "lora_unet_down_blocks_2_attentions_1",
        "lora_unet_mid_block_attentions_0",
        "lora_unet_up_blocks_1_attentions_0",
        "lora_unet_up_blocks_1_attentions_1",
        "lora_unet_up_blocks_1_attentions_2",
        "lora_unet_up_blocks_2_attentions_0",
        "lora_unet_up_blocks_2_attentions_1",
        "lora_unet_up_blocks_2_attentions_2",
        "lora_unet_up_blocks_3_attentions_0",
        "lora_unet_up_blocks_3_attentions_1",
        "lora_unet_up_blocks_3_attentions_2",
    ],
    lora_block_weight_str: Literal["FACE", "ALL"] = "ALL",
    need_unload: bool = False,
):
    """使用 lora 更新pipeline中的unet相关参数

    Args:
        pipeline (DiffusionPipeline): _description_
        lora (Union[str, Dict]): _description_
        alpha (float, optional): _description_. Defaults to 0.75.
        device (str, optional): _description_. Defaults to "cuda".
        lora_prefix_unet (str, optional): _description_. Defaults to "lora_unet".
        lora_prefix_text_encoder (str, optional): _description_. Defaults to "lora_te".
        lora_unet_layers (list, optional): _description_. Defaults to [ "lora_unet_down_blocks_0_attentions_0", "lora_unet_down_blocks_0_attentions_1", "lora_unet_down_blocks_1_attentions_0", "lora_unet_down_blocks_1_attentions_1", "lora_unet_down_blocks_2_attentions_0", "lora_unet_down_blocks_2_attentions_1", "lora_unet_mid_block_attentions_0", "lora_unet_up_blocks_1_attentions_0", "lora_unet_up_blocks_1_attentions_1", "lora_unet_up_blocks_1_attentions_2", "lora_unet_up_blocks_2_attentions_0", "lora_unet_up_blocks_2_attentions_1", "lora_unet_up_blocks_2_attentions_2", "lora_unet_up_blocks_3_attentions_0", "lora_unet_up_blocks_3_attentions_1", "lora_unet_up_blocks_3_attentions_2", ].
        lora_block_weight_str (Literal[&quot;FACE&quot;, &quot;ALL&quot;], optional): _description_. Defaults to "ALL".
        need_unload (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # ref https://git.woa.com/innovative_tech/GenerationGroup/VirtualIdol/VidolImageDraw/blob/master/pipeline/tool.py#L20
    if lora_block_weight_str is not None:
        lora_block_weight = LORA_BLOCK_WEIGHT_MAP[lora_block_weight_str.upper()]
    if lora_block_weight:
        assert len(lora_block_weight) == 17
    # load lora weight
    if isinstance(lora, str):
        state_dict = load_file(lora, device=device)
    else:
        for k in lora:
            lora[k] = lora[k].to(device)
        state_dict = lora  # state_dict = {}

    visited = set()
    unload_dict = []
    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(lora_prefix_text_encoder + "_")[-1].split("_")
            )
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(lora_prefix_unet + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            adding_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(
                2
            ).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            adding_weight = alpha * torch.mm(weight_up, weight_down)

        if lora_block_weight:
            if "text" in key:
                adding_weight *= lora_block_weight[0]
            else:
                for idx, layer in enumerate(lora_unet_layers):
                    if layer in key:
                        adding_weight *= lora_block_weight[idx + 1]
                        break

        curr_layer_unload_data = {"layer": curr_layer, "added_weight": adding_weight}
        curr_layer.weight.data += adding_weight

        unload_dict.append(curr_layer_unload_data)
        # update visited list
        for item in pair_keys:
            visited.add(item)
    if need_unload:
        return pipeline, unload_dict
    else:
        return pipeline


def update_pipeline_lora_models(
    pipeline: DiffusionPipeline,
    lora_dict: Dict[str, Dict],
    device: str = "cuda",
    need_unload: bool = True,
    lora_prefix_unet: str = "lora_unet",
    lora_prefix_text_encoder: str = "lora_te",
    lora_unet_layers=[
        "lora_unet_down_blocks_0_attentions_0",
        "lora_unet_down_blocks_0_attentions_1",
        "lora_unet_down_blocks_1_attentions_0",
        "lora_unet_down_blocks_1_attentions_1",
        "lora_unet_down_blocks_2_attentions_0",
        "lora_unet_down_blocks_2_attentions_1",
        "lora_unet_mid_block_attentions_0",
        "lora_unet_up_blocks_1_attentions_0",
        "lora_unet_up_blocks_1_attentions_1",
        "lora_unet_up_blocks_1_attentions_2",
        "lora_unet_up_blocks_2_attentions_0",
        "lora_unet_up_blocks_2_attentions_1",
        "lora_unet_up_blocks_2_attentions_2",
        "lora_unet_up_blocks_3_attentions_0",
        "lora_unet_up_blocks_3_attentions_1",
        "lora_unet_up_blocks_3_attentions_2",
    ],
):
    """使用 lora 更新pipeline中的unet相关参数

    Args:
        pipeline (DiffusionPipeline): _description_
        lora_dict (Dict[str, Dict]): _description_
        device (str, optional): _description_. Defaults to "cuda".
        lora_prefix_unet (str, optional): _description_. Defaults to "lora_unet".
        lora_prefix_text_encoder (str, optional): _description_. Defaults to "lora_te".
        lora_unet_layers (list, optional): _description_. Defaults to [ "lora_unet_down_blocks_0_attentions_0", "lora_unet_down_blocks_0_attentions_1", "lora_unet_down_blocks_1_attentions_0", "lora_unet_down_blocks_1_attentions_1", "lora_unet_down_blocks_2_attentions_0", "lora_unet_down_blocks_2_attentions_1", "lora_unet_mid_block_attentions_0", "lora_unet_up_blocks_1_attentions_0", "lora_unet_up_blocks_1_attentions_1", "lora_unet_up_blocks_1_attentions_2", "lora_unet_up_blocks_2_attentions_0", "lora_unet_up_blocks_2_attentions_1", "lora_unet_up_blocks_2_attentions_2", "lora_unet_up_blocks_3_attentions_0", "lora_unet_up_blocks_3_attentions_1", "lora_unet_up_blocks_3_attentions_2", ].

    Returns:
        _type_: _description_
    """
    unload_dicts = []
    for lora, value in lora_dict.items():
        lora_name = os.path.basename(lora).replace(".safetensors", "")
        strength_offset = value.get("strength_offset", 0.0)
        alpha = value.get("strength", 1.0)
        alpha += strength_offset
        lora_weight_str = value.get("lora_block_weight", "ALL")
        lora = load_file(lora)
        pipeline, unload_dict = update_pipeline_lora_model(
            pipeline,
            lora=lora,
            device=device,
            alpha=alpha,
            lora_prefix_unet=lora_prefix_unet,
            lora_prefix_text_encoder=lora_prefix_text_encoder,
            lora_unet_layers=lora_unet_layers,
            lora_block_weight_str=lora_weight_str,
            need_unload=True,
        )
        print(
            "Update LoRA {} with alpha {} and weight {}".format(
                lora_name, alpha, lora_weight_str
            )
        )
    unload_dicts += unload_dict
    return pipeline, unload_dicts


def unload_lora(unload_dict: List[Dict[str, nn.Module]]):
    for layer_data in unload_dict:
        layer = layer_data["layer"]
        added_weight = layer_data["added_weight"]
        layer.weight.data -= added_weight

    gc.collect()
    torch.cuda.empty_cache()


def load_motion_lora_weights(
    animation_pipeline,
    motion_module_lora_configs=[],
):
    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = (
            motion_module_lora_config["path"],
            motion_module_lora_config["alpha"],
        )
        print(f"load motion LoRA from {path}")

        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = (
            motion_lora_state_dict["state_dict"]
            if "state_dict" in motion_lora_state_dict
            else motion_lora_state_dict
        )

        animation_pipeline = convert_motion_lora_ckpt_to_diffusers(
            animation_pipeline, motion_lora_state_dict, alpha
        )

    return animation_pipeline
