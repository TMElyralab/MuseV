# Modify from https://github.com/huggingface/diffusers/blob/v0.16.0/examples/text_to_image/train_text_to_image.py
# and https://github.com/showlab/Tune-A-Video/blob/f0c5d9b7c186a40e02c52129a7e10b9d906f6063/train_tuneavideo.py
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import load_state_dict
from musev import logger as v_logger
import multiprocessing as mp

import argparse
import datetime
import logging
import inspect
import math
import os
import shutil
from typing import Dict, List, Optional, Tuple
from pprint import pformat, pprint
from collections import OrderedDict

import numpy as np
from omegaconf import OmegaConf
from omegaconf import SCMode
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.profiler import profile, record_function, ProfilerActivity
import accelerate

from musev.data.sampler_util import NumVisCondBatchSampler
from musev.models import Model_Register
from musev.models.ip_adapter_loader import (
    load_ip_adapter_image_proj_by_name,
    load_ip_adapter_vision_clip_encoder_by_name,
    load_vision_clip_encoder_by_name,
)
from musev.models.referencenet import ReferenceNet2D
from musev.models.text_model import TextEmbExtractor
from musev.pipelines.pipeline_controlnet_predictor import (
    DiffusersPipelinePredictor,
)
from musev.models.unet_loader import (
    load_unet,
    load_unet_by_name,
    update_unet_with_sd,
)
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.autoencoder_kl import AutoencoderKL
from accelerate.logging import get_logger

from diffusers.models.controlnet import ControlNetModel

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.vae import DiagonalGaussianDistribution
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat
import time
import pandas as pd
import h5py


from mmcm.vision.data.video_dataset import DecordVideoDataset
from mmcm.vision.feature_extractor.controlnet import (
    ControlnetProcessor,
    PoseKPs2ImgConverter,
    get_controlnet_params,
    load_controlnet_model,
)

from musev.schedulers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from musev.models.unet_3d_condition import UNet3DConditionModel
from musev.models.super_model import SuperUNet3DConditionModel
from musev.data.PreextractH5pyDataset import (
    PreextractH5pyDataset,
    add_text2_prompt,
    mv_batch_head_latent2viscond,
    mv_batch_tail_latent2viscond,
)
from musev.data.data_util import (
    align_repeat_tensor_single_dim,
    batch_concat_two_tensor_with_index,
    batch_index_select,
    sample_tensor_by_idx,
    split_index,
    split_tensor,
)
from musev.pipelines.pipeline_controlnet import MusevControlNetPipeline
from musev.utils.noise_util import random_noise, video_fusion_noise
from musev.utils.util import (
    save_videos_grid_with_opencv,
    fn_recursive_search,
)
from musev import logger as v_logger
from mmcm.vision.process.image_process import batch_dynamic_crop_resize_images
from mmcm.vision.utils.data_type_util import is_image, is_video, read_image_as_5d
from musev.utils.vae_util import decode_unet_latents_with_vae
from musev.loss.video_loss import (
    cal_video_inter_frames_loss,
    cal_viscond_video_latents_loss,
)
from accelerate import DistributedDataParallelKwargs

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# 定义一个构造器函数，用于解析 include 标记
def include_constructor(path):
    # 加载包含文件
    path = os.path.join(os.path.dirname(__file__), "configs/train", path)
    return OmegaConf.load(path)


# 注册 include 标记
OmegaConf.register_new_resolver("include", include_constructor)


def log_validation(
    unet: nn.Module,
    device: str,
    weight_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    output_dir: Dict,
    validation_data: Dict,
    model: Dict,
    dataset: Dict,
    train: Dict,
    diffusion: Dict,
    referencenet: nn.Module = None,
    ip_adapter_image_proj: nn.Module = None,
):
    # Get the validation pipeline
    logger.info(
        f"validation,device={device}, weight_dtype={weight_dtype}",
        main_process_only=True,
    )
    # unet = load_unet(sd_unet_model=unet, sd_model=model.pretrained_model_path)
    need_transformer_in = model.get("need_transformer_in", True)
    need_t2i_ip_adapter = model.get("need_t2i_ip_adapter", False)
    need_adain_temporal_cond = model.get("need_adain_temporal_cond", False)
    keep_vision_condtion = model.get("keep_vision_condtion", False)
    t2i_ip_adapter_attn_processor = model.get(
        "t2i_ip_adapter_attn_processor", "IPXFormersAttnProcessor"
    )
    use_anivv1_cfg = model.get("use_anivv1_cfg", False)
    resnet_2d_skip_time_act = model.get("resnet_2d_skip_time_act", False)
    norm_spatial_length = model.get("norm_spatial_length", False)
    spatial_max_length = model.get("spatial_max_length", 2048)
    sample_frame_rate = validation_data.get("sample_frame_rate")
    video2emb_sr = dataset.get("video2emb_sr", 2)
    use_referencenet = model.get("use_referencenet", False)
    need_block_embs = False
    need_self_attn_block_embs = False
    if use_referencenet:
        need_self_attn_block_embs = model.referencenet_params.get(
            "need_self_attn_block_embs", False
        )
        need_block_embs = model.referencenet_params.get("need_block_embs", False)
    need_ip_adapter_cross_attn = model.get("need_ip_adapter_cross_attn", False)

    t2i_crossattn_ip_adapter_attn_processor = model.get(
        "t2i_crossattn_ip_adapter_attn_processor", None
    )

    actual_sample_frame_rate = sample_frame_rate * video2emb_sr
    prompt_only_use_image_prompt = not need_ip_adapter_cross_attn and model.get(
        "prompt_only_use_image_prompt", False
    )
    # TODO: 按理说直接取 accelerate.unwrap的unet/supernet就行，不需要重新生成定义，待确定
    unet_copy = UNet3DConditionModel.from_pretrained_2d(
        model.pretrained_model_path,
        subfolder="unet",
        temporal_transformer=model.temporal_transformer,
        temporal_conv_block=model.temporal_conv_block,
        cross_attention_dim=model.cross_attention_dim,
        need_spatial_position_emb=model.need_spatial_position_emb,
        need_transformer_in=need_transformer_in,
        need_t2i_ip_adapter=need_t2i_ip_adapter,
        need_adain_temporal_cond=need_adain_temporal_cond,
        t2i_ip_adapter_attn_processor=t2i_ip_adapter_attn_processor,
        keep_vision_condtion=keep_vision_condtion,
        use_anivv1_cfg=use_anivv1_cfg,
        resnet_2d_skip_time_act=resnet_2d_skip_time_act,
        norm_spatial_length=norm_spatial_length,
        spatial_max_length=spatial_max_length,
        need_refer_emb=need_block_embs
        and not model.get("only_need_block_embs_adain", False),
        ip_adapter_cross_attn=need_ip_adapter_cross_attn,
        t2i_crossattn_ip_adapter_attn_processor=t2i_crossattn_ip_adapter_attn_processor,
        need_vis_cond_mask=model.get("need_vis_cond_mask", False),
    )
    unet_copy.load_state_dict(unet.state_dict())
    unet_copy.to(dtype=weight_dtype, device=device)
    unet_copy.eval()
    unet = unet_copy

    # referencenet
    if use_referencenet and referencenet is not None:
        referencenet_class = Model_Register[model.referencenet_params.referencenet]
        referencenet_copy = referencenet_class.from_pretrained(
            model.pretrained_model_path,
            subfolder="unet",
            cross_attention_dim=model.cross_attention_dim,
            need_self_attn_block_embs=need_self_attn_block_embs,
            need_block_embs=need_block_embs,
        )
        referencenet_copy.load_state_dict(referencenet.state_dict())
        referencenet_copy.to(dtype=weight_dtype, device=device)
        referencenet_copy.eval()
        referencenet = referencenet_copy

    # vision_clip_extractor
    if model.get("vision_clip_extractor_class_name", None) is not None:
        vision_clip_extractor = load_vision_clip_encoder_by_name(
            ip_image_encoder=model.get("vision_clip_model_path"),
            vision_clip_extractor_class_name=model.get(
                "vision_clip_extractor_class_name"
            ),
        )
        vision_clip_extractor.to(dtype=weight_dtype, device=device)
        vision_clip_extractor.eval()
    else:
        vision_clip_extractor = None

    # ip_adapter_image_proj
    # 训练和推断所使用的image_proj可能不一样
    ip_adapter_cross_attn = validation_data.get(
        "ip_adapter_cross_attn", model.get("ip_adapter_cross_attn", False)
    )
    if ip_adapter_cross_attn:
        ip_adapter_image_proj_copy = load_ip_adapter_image_proj_by_name(
            model_name=model.ip_adapter_params.ip_adapter_model_name,
            ip_image_encoder=model.ip_adapter_params.ip_image_encoder,
            ip_ckpt=model.ip_adapter_params.pretrained_model_path,
            cross_attention_dim=model.cross_attention_dim,
            clip_embeddings_dim=model.ip_adapter_params.clip_embeddings_dim,
            clip_extra_context_tokens=model.ip_adapter_params.clip_extra_context_tokens,
            ip_scale=model.ip_adapter_params.ip_scale,
            device=device,
        )
        ip_adapter_image_proj_copy.load_state_dict(ip_adapter_image_proj.state_dict())
        ip_adapter_image_proj_copy.to(dtype=weight_dtype, device=device)
        ip_adapter_image_proj_copy.eval()
        ip_adapter_image_proj = ip_adapter_image_proj_copy
    else:
        ip_adapter_image_proj = None

    logger.debug(
        f"log_validation: referencenet = {type(referencenet)}, ip_adapter_image_proj={type(ip_adapter_image_proj)}, vision_clip_extractor={type(vision_clip_extractor)} "
    )
    sd_predictor = DiffusersPipelinePredictor(
        sd_model_path=validation_data.get("sd_model", model.pretrained_model_path),
        lora_dict=validation_data.get("lora_dict", None),
        unet=unet,
        controlnet_name=model.controlnet_name,
        device=device,
        dtype=weight_dtype,
        negative_embedding=validation_data.negative_embedding,
        referencenet=referencenet,
        ip_adapter_image_proj=ip_adapter_image_proj,
        vision_clip_extractor=vision_clip_extractor,
        vae_model=model.get("vae_model_path", None),
    )

    noise_type = validation_data.noise_type
    width = validation_data.width
    height = validation_data.height
    num_inference_steps = validation_data.num_inference_steps
    n_vision_condition = validation_data.get("n_vision_condition", 1)
    video_num_inference_steps = validation_data.get("video_num_inference_steps", 10)
    num_inference_steps = validation_data.get("num_inference_steps", 30)
    guidance_scale = validation_data.get("guidance_scale", 7.5)
    video_guidance_scale = validation_data.get("video_guidance_scale", 3.5)
    redraw_condition_image = validation_data.get("redraw_condition_image", False)
    for idx, test_data in tqdm(enumerate(validation_data.datas)):
        samples = []
        texts = []
        data_name = test_data.name
        video_path = test_data.get("video_path", None)
        condition_images = test_data.get("condition_images", None)
        prompt = test_data.prompt
        generator = torch.Generator(device=device).manual_seed(train.seed)
        test_data_height = test_data.get("height", height)
        test_data_width = test_data.get("width", width)
        test_data_redraw_condition_image = test_data.get(
            "redraw_condition_image", redraw_condition_image
        )
        save_path = os.path.join(
            output_dir.samples,
            # f"step={global_step}_name={data_name}_w={test_data_width}_h={test_data_height}_prompt={prompt[:20]}.{validation_data.output_file_format}",
            f"step={global_step}_name={data_name}_w={test_data_width}_h={test_data_height}_n_batch={validation_data.n_video_batch}_prompt={prompt[:20]}_.{validation_data.output_file_format}",
        )
        logger.info(
            f"idx={idx}, test_data_name={data_name}, video_path={video_path}, condition_images={condition_images}, save_path={save_path}",
            main_process_only=True,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path) and not validation_data.overwrite:
            logger.info(f"existed: save_path={save_path}", main_process_only=True)
            continue
        prompt = validation_data.prefix_prompt + prompt + validation_data.suffix_prompt

        # text2video
        if condition_images is not None:
            condition_images = read_image_as_5d(condition_images)

        test_data_refer_image_path = test_data.get("refer_image", None)
        test_data_ipadapter_image_path = test_data.get("ipadapter_image", None)
        test_data_video_is_middle = test_data.get("video_is_middle", False)

        # 准备 test_data_refer_image
        if referencenet is not None:
            if test_data_refer_image_path is None:
                test_data_refer_image = condition_images
                logger.debug(f"test_data_refer_image use test_data_condition_image")
            else:
                test_data_refer_image = read_image_as_5d(test_data_refer_image_path)
                logger.debug(f"test_data_refer_image use {test_data_refer_image_path}")
        else:
            test_data_refer_image = None
            logger.debug(f"test_data_refer_image is None")

        # 准备 test_data_ipadapter_image
        if ip_adapter_image_proj is not None or vision_clip_extractor is not None:
            if test_data_ipadapter_image_path is None:
                test_data_ipadapter_image = condition_images
                logger.debug(f"test_data_ipadapter_image use test_data_condition_image")
            else:
                test_data_ipadapter_image = read_image_as_5d(
                    test_data_ipadapter_image_path
                )
                logger.debug(
                    f"test_data_ipadapter_image use f{test_data_ipadapter_image_path}"
                )
        else:
            test_data_ipadapter_image = None
            logger.debug(f"test_data_ipadapter_image is None")

        logger.debug(
            f"test_data_refer_image,  type(refer_image)={type(test_data_refer_image)}"
        )
        logger.debug(
            f"test_data_ipadapter_image,  type(test_data_ipadapter_image)={type(test_data_ipadapter_image)}"
        )
        # text2video testcase
        t2v_videos = sd_predictor.run_pipe_text2video(
            video_length=validation_data.video_length - 1,
            prompt=prompt,
            width=test_data_width,
            height=test_data_height,
            generator=generator,
            noise_type=noise_type,
            video_negative_prompt=validation_data.negative_prompt,
            guidance_scale=guidance_scale,
            video_guidance_scale=video_guidance_scale,
            num_inference_steps=num_inference_steps,
            video_num_inference_steps=video_num_inference_steps,
            negative_prompt=validation_data.negative_prompt,
            max_batch_num=validation_data.n_video_batch,
            need_img_based_video_noise=validation_data.need_img_based_video_noise,
            condition_images=condition_images,
            redraw_condition_image=test_data_redraw_condition_image,
            motion_speed=actual_sample_frame_rate,
            n_vision_condition=n_vision_condition,
            refer_image=test_data_refer_image,
            ip_adapter_image=test_data_ipadapter_image,
            prompt_only_use_image_prompt=prompt_only_use_image_prompt,
        )
        samples.append(t2v_videos)
        texts.append("text2video")
        # controlnet middle2video
        if video_path is not None and is_video(video_path):
            controlnet_processor_params = {
                "detect_resolution": min(test_data_height, test_data_width),
                "image_resolution": min(test_data_height, test_data_width),
                "return_pose_only": False,
            }
            out_videos, out_condition, videos = sd_predictor.run_pipe_video2video(
                video=video_path,
                time_size=validation_data.video_length,
                step=validation_data.video_length,
                sample_rate=validation_data.sample_frame_rate,
                prompt=prompt,
                width=test_data_width,
                height=test_data_height,
                need_return_videos=True,
                need_return_condition=True,
                generator=generator,
                noise_type=noise_type,
                negative_prompt=validation_data.negative_prompt,
                max_batch_num=validation_data.n_video_batch,
                need_img_based_video_noise=validation_data.need_img_based_video_noise,
                controlnet_processor_params=controlnet_processor_params,
                guidance_scale=guidance_scale,
                video_guidance_scale=video_guidance_scale,
                num_inference_steps=num_inference_steps,
                video_num_inference_steps=video_num_inference_steps,
                refer_image=test_data_refer_image,
                ip_adapter_image=test_data_ipadapter_image,
                need_video2video=validation_data.get("need_video2video", False),
                prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                n_vision_condition=n_vision_condition,
                condition_images=condition_images,
                video_is_middle=test_data_video_is_middle,
            )
            samples = [videos / 255.0, out_condition / 255.0, out_videos] + samples
            texts = [
                prompt,
                model.controlnet_name,
                "videomiddle2video",
            ] + texts
        samples = np.concatenate(samples, axis=0)
        prompt = "null" if prompt == "" else prompt
        if test_data_width / test_data_height > 1.0:
            n_cols = 1
        else:
            n_cols = len(samples)
        save_videos_grid_with_opencv(
            samples,
            save_path,
            texts=texts,
            n_cols=n_cols,
            fps=validation_data.fps,
            split_size_or_sections=validation_data.split_size_or_sections,
            write_info=validation_data.get("write_info", False),
        )
        logger.info(f"save to {save_path}", main_process_only=True)
    del sd_predictor
    torch.cuda.empty_cache()
    return samples


def main(
    output_dir: Dict,
    model: Dict,
    dataset: Dict,
    validation_data: Dict,
    train: Dict,
    diffusion: Dict,
):
    if output_dir.add_time_to_exp_dir:
        dirname, basename = os.path.dirname(output_dir.exp_dir), os.path.basename(
            output_dir.exp_dir
        )
        basename = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S") + "_" + basename
        output_dir.exp_dir = os.path.join(dirname, basename)
    *_, config = inspect.getargvalues(inspect.currentframe())

    pretrained_model_path = model.pretrained_model_path
    logging_dir = output_dir.log_dir
    validation_steps = validation_data.validation_steps
    trainable_modules = model.trainable_modules
    train_batch_size = train.train_batch_size
    n_sample_condition_frames = validation_data.n_sample_condition_frames
    dataloader_num_workers = train.dataloader_num_workers
    max_train_steps = train.max_train_steps
    learning_rate = train.learning_rate
    scale_lr = train.scale_lr
    lr_scheduler = train.lr_scheduler
    lr_warmup_steps = train.lr_warmup_steps
    adam_beta1 = train.adam_beta1
    adam_beta2 = train.adam_beta2
    adam_weight_decay = train.adam_weight_decay
    adam_epsilon = train.adam_epsilon
    max_grad_norm = train.max_grad_norm
    gradient_accumulation_steps = train.gradient_accumulation_steps
    gradient_checkpointing = train.gradient_checkpointing
    report_to = output_dir.report_to
    checkpointing_steps = train.checkpointing_steps
    save_checkpoint = train.get("save_checkpoint", True)
    resume_from_checkpoint = train.resume_from_checkpoint
    checkpoints_total_limit = train.checkpoints_total_limit
    mixed_precision = train.mixed_precision
    use_8bit_adam = train.use_8bit_adam
    enable_xformers_memory_efficient_attention = (
        train.enable_xformers_memory_efficient_attention
    )
    noise_offset = train.noise_offset
    noise_type = train.noise_type
    seed = train.seed
    testing_speed = train.testing_speed
    get_midway_ckpt = train.get_midway_ckpt
    only_validation = output_dir.only_validation

    need_transformer_in = model.get("need_transformer_in", True)
    need_t2i_ip_adapter = model.get("need_t2i_ip_adapter", False)
    need_adain_temporal_cond = model.get("need_adain_temporal_cond", False)
    keep_vision_condtion = model.get("keep_vision_condtion", False)
    t2i_ip_adapter_attn_processor = model.get(
        "t2i_ip_adapter_attn_processor", "IPXFormersAttnProcessor"
    )
    use_anivv1_cfg = model.get("use_anivv1_cfg", False)
    resnet_2d_skip_time_act = model.get("resnet_2d_skip_time_act", False)

    sample_frame_rate = dataset.get("sample_frame_rate")
    video2emb_sr = dataset.get("video2emb_sr", 2)
    actual_sample_frame_rate = sample_frame_rate * video2emb_sr
    # 10 是 AnivV1使用的默认参数，向前兼容值
    norm_spatial_length = model.get("norm_spatial_length", False)
    spatial_max_length = model.get("spatial_max_length", 2048)
    need_video_viscond_loss = model.get("need_video_viscond_loss", False)
    video_viscond_loss_weight = model.get("video_viscond_loss_weight", 0.1)
    pixel_video_viscond_loss_weight = model.get(
        "pixel_video_viscond_loss_weight", video_viscond_loss_weight
    )
    need_viscond_loss = model.get("need_viscond_loss", False)
    viscond_loss_weight = model.get("viscond_loss_weight", 0.1)
    video_inter_frames_timestep_th = model.get("video_inter_frames_timestep_th", 0.3)
    need_video_inter_frames_loss = model.get("need_video_inter_frames_loss", False)
    video_inter_frames_loss_weight = model.get("video_inter_frames_loss_weight", 0.1)
    pixel_video_inter_frames_loss_weight = model.get(
        "pixel_video_inter_frames_loss_weight", video_inter_frames_loss_weight
    )
    # use_pixel_loss True 时容易爆显存，需要设置 vae.enable_slice()
    use_pixel_loss = model.get("use_pixel_loss", False)
    video_inter_frames_loss_type = model.get("video_inter_frames_loss_type", "MSE")
    static_video_prompt = dataset.get("static_video_prompt", None)
    dynamic_video_prompt = dataset.get("dynamic_video_prompt", None)
    use_dynamic_sr = dataset.get("use_dynamic_sr", False)
    max_sample_frame_rate = dataset.get(
        "max_sample_frame_rate", dataset.get("sample_frame_rate")
    )
    n_vision_condition = dataset.get(
        "n_vision_condition", dataset.get("n_sample_condition_frames", 1)
    )
    max_n_vision_condition = dataset.get("max_n_vision_condition", None)
    first_element_prob = dataset.get("first_element_prob", 0.5)
    sample_n_viscond_method = dataset.get("sample_n_viscond_method", "union")
    change_n_viscond_in_dataset = dataset.get("change_n_viscond_in_dataset", False)

    tail_frame_prob = dataset.get("tail_frame_prob", 0)
    add_viscond_timestep_noise = dataset.get("add_viscond_timestep_noise", False)
    viscond_timestep_noise_min = dataset.get("viscond_timestep_noise_min", 0)
    viscond_timestep_noise_max = dataset.get("viscond_timestep_noise_max", 100)
    # MakePixelDance: https://arxiv.org/abs/2311.10982

    use_referencenet = model.get("use_referencenet", False)

    pretrained_referencenet_path = None
    need_self_attn_block_embs = False
    need_block_embs = False
    if use_referencenet:
        pretrained_referencenet_path = model.referencenet_params.get(
            "pretrained_model_path", None
        )
        need_self_attn_block_embs = model.referencenet_params.get(
            "need_self_attn_block_embs", False
        )
        need_block_embs = model.referencenet_params.get("need_block_embs", False)
    need_ip_adapter_cross_attn = model.get("need_ip_adapter_cross_attn", False)
    ip_adapter_cross_attn = model.get("ip_adapter_cross_attn", False)
    t2i_crossattn_ip_adapter_attn_processor = model.get(
        "t2i_crossattn_ip_adapter_attn_processor", None
    )
    # 当不需要ip_adapter_cross_attn时 且依然有ip_adapter的相关参数，说明主要目的是使用 image_prompt
    # TODO：目前命名存在概念混淆，待优化
    prompt_only_use_image_prompt = not need_ip_adapter_cross_attn and model.get(
        "prompt_only_use_image_prompt", False
    )

    accelerator_project_config = ProjectConfiguration(
        total_limit=checkpoints_total_limit,
        project_dir=logging_dir,
    )

    # 当模型存在未参与训练的层时，要么加这个，要么模型定义中去掉用不到的层。
    # 使用这个会多消耗计算资源
    # kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    kwargs_handlers = None
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Handle the output folder creation
    if accelerator.is_main_process:
        if not os.path.exists(output_dir.exp_dir):
            # 注意工作流里不够权限删除此文件夹里的文件，需要提前手动删除，不然会报无权限的错误！！！
            os.makedirs(output_dir.exp_dir, exist_ok=True)
            os.makedirs(output_dir.samples, exist_ok=True)
            os.makedirs(output_dir.checkpoints, exist_ok=True)
            os.makedirs(output_dir.log_dir, exist_ok=True)
            OmegaConf.save(config, os.path.join(output_dir.exp_dir, "config.yaml"))
            shutil.copytree(
                "musev",
                os.path.join(output_dir.exp_dir, "musev"),
                dirs_exist_ok=True,
            )

            shutil.copy(os.path.basename(__file__), f"{output_dir.exp_dir}/")

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler"
    )
    # prepare
    # text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        # model_max_length=512
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
        # max_position_embeddings=512,
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_emb_extractor = TextEmbExtractor(
        tokenizer=tokenizer, text_encoder=text_encoder
    )

    # prepare vae
    vae_model_path = model.get("vae_model_path", None)
    if vae_model_path is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        logger.info(f"vae_model_path={pretrained_model_path}")
    else:
        vae = AutoencoderKL.from_pretrained(vae_model_path)
        logger.info(f"vae_model_path={vae_model_path}")

    vae.requires_grad_(False)
    vae.encoder = None
    vae_scaling_factor = vae.config.scaling_factor
    vae_block_out_channels = vae.config.block_out_channels
    vae.enable_slicing()
    vae.to(accelerator.device, dtype=weight_dtype)
    if not model.get("need_vae_module", True) and not model.get(
        "save_final_pipeline", True
    ):
        del vae
    torch.cuda.empty_cache()

    # referencenet
    logger.info(f"use_referencenet={use_referencenet}", main_process_only=True)
    train_referencenet = False
    if use_referencenet:
        referencenet_class = Model_Register[model.referencenet_params.referencenet]
        referencenet = referencenet_class.from_pretrained(
            pretrained_model_path,
            subfolder="unet",
            cross_attention_dim=model.cross_attention_dim,
            need_self_attn_block_embs=need_self_attn_block_embs,
            need_block_embs=need_block_embs,
        )
        referencenet_pretrained_model_path = model.referencenet_params.get(
            "pretrained_model_path", None
        )
        logger.info(
            f"referencenet_pretrained_model_path={referencenet_pretrained_model_path}",
            main_process_only=True,
        )
        if referencenet_pretrained_model_path is not None:
            logger.info(
                f"update referencenet parameters with {referencenet_pretrained_model_path}"
            )
            referencenet = update_unet_with_sd(
                referencenet, referencenet_pretrained_model_path
            )

        train_referencenet = model.referencenet_params.is_train
        referencenet.requires_grad_(train_referencenet)
        if train_referencenet:
            referencenet.train()
        else:
            referencenet.eval()
        logger.info(
            f"use_referencenet, {type(referencenet)}, need_block_embs={need_block_embs}, need_self_attn_block_embs={need_self_attn_block_embs}, train_referencenet={train_referencenet}",
            main_process_only=True,
        )
    else:
        referencenet = None
        train_referencenet = False

    # 固定 controlnet，用于训练 T2V，如需训练controlnet，可以把这里的是否训练配置表化
    logger.info(
        f"use_controlnet_pipeline: {model.use_controlnet_pipeline}",
        main_process_only=True,
    )
    if model.use_controlnet_pipeline:
        controlnet, controlnet_processor, processor_params = load_controlnet_model(
            model.controlnet_name,
            device=accelerator.device,
            dtype=weight_dtype,
            need_controlnet_processor=False,
            need_controlnet=True,
            include_body=validation_data.include_body,
            include_face=validation_data.include_face,
            hand_and_face=validation_data.hand_and_face,
            include_hand=validation_data.include_hand,
        )
        controlnet.eval()
        controlnet.requires_grad_(False)
        # controlnet = accelerator.prepare(controlnet)
        logger.info(
            f"accelerator.prepare controlnet {type(controlnet)}", main_process_only=True
        )
    else:
        controlnet = None

    # 根据pretrained的来源需要选择 from_pretrained_2d 还是 from_pretrained，但代码上来看 from_pretrained_2d是支持 3d的
    unet = UNet3DConditionModel.from_pretrained_2d(
        model.pretrained_model_path,
        subfolder="unet",
        temporal_transformer=model.temporal_transformer,
        temporal_conv_block=model.temporal_conv_block,
        cross_attention_dim=model.cross_attention_dim,
        need_spatial_position_emb=model.need_spatial_position_emb,
        need_transformer_in=need_transformer_in,
        need_t2i_ip_adapter=need_t2i_ip_adapter,
        need_adain_temporal_cond=need_adain_temporal_cond,
        t2i_ip_adapter_attn_processor=t2i_ip_adapter_attn_processor,
        keep_vision_condtion=keep_vision_condtion,
        use_anivv1_cfg=use_anivv1_cfg,
        resnet_2d_skip_time_act=resnet_2d_skip_time_act,
        norm_spatial_length=norm_spatial_length,
        spatial_max_length=spatial_max_length,
        need_refer_emb=need_block_embs
        and not model.get("only_need_block_embs_adain", False),
        ip_adapter_cross_attn=need_ip_adapter_cross_attn,
        t2i_crossattn_ip_adapter_attn_processor=t2i_crossattn_ip_adapter_attn_processor,
        need_vis_cond_mask=model.get("need_vis_cond_mask", False),
    )
    pretrained_unet_path = model.get("pretrained_unet_path", None)
    if pretrained_unet_path is not None:
        logger.info(f"update unet parameters with {pretrained_unet_path}")
        unet = update_unet_with_sd(unet, pretrained_unet_path)
    unet.requires_grad_(False)
    # 根据配置表选择训练哪些参数，当有自定义层时，一定一定要注意
    train_unet = model.get("train_unet", True)
    if train_unet:
        for name, module in unet.named_modules():
            if name.endswith(tuple(trainable_modules)):
                logger.info(
                    f"unet trainable_modules, name={name}, module={type(module)}",
                    main_process_only=True,
                )
                for params in module.parameters():
                    params.requires_grad = True
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            if use_referencenet:
                referencenet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    logger.debug(
        "after enable_xformers_memory_efficient_attention", main_process_only=True
    )
    logger.debug(pformat(unet.attn_processors), main_process_only=True)

    logger.info(
        f"ip_adapter_cross_attn: {ip_adapter_cross_attn}", main_process_only=True
    )

    #   ip_adapter_image_proj
    train_ip_adapter_image_proj = False
    if ip_adapter_cross_attn:
        ip_adapter_image_proj = load_ip_adapter_image_proj_by_name(
            model_name=model.ip_adapter_params.ip_adapter_model_name,
            ip_image_encoder=model.ip_adapter_params.ip_image_encoder,
            ip_ckpt=model.ip_adapter_params.pretrained_model_path,
            cross_attention_dim=model.cross_attention_dim,
            clip_embeddings_dim=model.ip_adapter_params.clip_embeddings_dim,
            clip_extra_context_tokens=model.ip_adapter_params.clip_extra_context_tokens,
            ip_scale=model.ip_adapter_params.ip_scale,
            unet=unet,
            device=accelerator.device,
        )
        train_ip_adapter_image_proj = (
            model.ip_adapter_params.train_ip_adapter_image_proj
        )
        ip_adapter_image_proj.requires_grad_(train_ip_adapter_image_proj)
        logger.info(
            f"ip_adapter_image_proj requires ={model.ip_adapter_params.train_ip_adapter_image_proj}"
        )
        # ip_adapter_image_proj.to(dtype=weight_dtype)
        # TODO: 根据是否train ip_adapter_attn 修改unet对应 attn 的 requires_grad
    else:
        ip_adapter_image_proj = None
        train_ip_adapter_image_proj = False

    supernet = SuperUNet3DConditionModel(
        unet=unet,
        referencenet=referencenet,
        controlnet=controlnet,
        ip_adapter_image_proj=ip_adapter_image_proj,
    )
    # supernet.requires_grad_(False)
    # # 根据配置表选择训练哪些参数，当有自定义层时，一定一定要注意
    # for name, module in supernet.named_modules():
    #     if name.endswith(tuple(trainable_modules)):
    #         logger.info(f"trainable_modules, name={name}", main_process_only=True)
    #         for params in module.parameters():
    #             params.requires_grad = True
    #     if "referencenet" in name and use_referencenet and train_referencenet:
    #         logger.info(f"trainable_modules, name={name}", main_process_only=True)
    #         for params in module.parameters():
    #             params.requires_grad = True

    if gradient_checkpointing:
        # supernet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()
        if use_referencenet:
            logger.debug(f"referencenet.enable_gradient_checkpointing()")
            referencenet.enable_gradient_checkpointing()

    # 配置 acclelerate 的模型存储与load
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice forma

        def save_model_hook(models, weights, output_dir):
            logger.debug(
                f"save_model_hook, len(models)={len(models)}, len(weights)={len(weights)}",
                main_process_only=True,
            )
            for i, hook_model in enumerate(models):
                hook_model_class = hook_model.__class__.__name__.lower()
                logger.debug(
                    f"save_model_hook i={i}, model = {type(hook_model)}, hook_model_class={hook_model_class}"
                )
                if (
                    "unet" in hook_model_class
                    and "super" not in hook_model_class
                    and train_unet
                ):
                    # load diffusers style into model
                    if hook_model is not None:
                        hook_model.save_pretrained(
                            os.path.join(output_dir, "unet"),
                            safe_serialization=False,
                        )
                # 绝大部分情况下 referencenet和controlnet都不需要重新更新参数，所以不需要单独再做处理
                elif "referencenet" in hook_model_class and train_referencenet:
                    if hook_model is not None:
                        hook_model.save_pretrained(
                            os.path.join(output_dir, "referencenet"),
                            safe_serialization=False,
                        )
                elif "super" in hook_model_class:
                    if hook_model.unet is not None and train_unet:
                        hook_model.unet.save_pretrained(
                            os.path.join(output_dir, "unet"), safe_serialization=False
                        )
                    if hook_model.referencenet is not None and train_referencenet:
                        hook_model.referencenet.save_pretrained(
                            os.path.join(output_dir, "referencenet"),
                            safe_serialization=False,
                        )
                    if (
                        hook_model.ip_adapter_image_proj is not None
                        and train_ip_adapter_image_proj
                    ):
                        torch.save(
                            {
                                "image_proj": hook_model.ip_adapter_image_proj.state_dict()
                            },
                            os.path.join(output_dir, "ip_adapter_image_proj.bin"),
                        )
                else:
                    logger.warn(
                        f"now only support unet, referencenet, controlnet, but given {hook_model_class}"
                    )
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            logger.debug(
                f"load_model_hook, len(models)={len(models)}, input_dir={input_dir}",
                main_process_only=True,
            )

            for i in range(len(models)):
                # pop models so that they are not loaded again
                hook_model = models.pop()

                hook_model_class = hook_model.__class__.__name__.lower()
                logger.info(
                    f"load_model_hook={i}, hook_model={type(hook_model)}, input_dir={input_dir}, hook_model_class={hook_model_class}",
                    main_process_only=True,
                )
                if "unet" in hook_model_class and "super" not in hook_model_class:
                    # load diffusers style into model
                    load_model = UNet3DConditionModel.from_pretrained(
                        input_dir,
                        subfolder="unet",
                        temporal_conv_block=model.temporal_conv_block,
                        temporal_transformer=model.temporal_transformer,
                        cross_attention_dim=model.cross_attention_dim,
                        need_spatial_position_emb=model.need_spatial_position_emb,
                        need_transformer_in=need_transformer_in,
                        need_t2i_ip_adapter=need_t2i_ip_adapter,
                        need_adain_temporal_cond=need_adain_temporal_cond,
                        t2i_ip_adapter_attn_processor=t2i_ip_adapter_attn_processor,
                        keep_vision_condtion=keep_vision_condtion,
                        use_anivv1_cfg=use_anivv1_cfg,
                        resnet_2d_skip_time_act=resnet_2d_skip_time_act,
                        norm_spatial_length=norm_spatial_length,
                        spatial_max_length=spatial_max_length,
                        need_refer_emb=need_block_embs
                        and not model.get("only_need_block_embs_adain", False),
                        ip_adapter_cross_attn=need_ip_adapter_cross_attn,
                        t2i_crossattn_ip_adapter_attn_processor=t2i_crossattn_ip_adapter_attn_processor,
                        need_vis_cond_mask=model.get("need_vis_cond_mask", False),
                    )
                    hook_model.register_to_config(**load_model.config)
                    hook_model.load_state_dict(load_model.state_dict())
                    del load_model
                    logger.info(
                        f"load_model_hook={i}, UNet3DConditionModel",
                        main_process_only=True,
                    )
                # 绝大部分情况下 referencenet和controlnet都不需要重新更新参数，所以不需要单独再做处理
                elif "referencenet" in hook_model_class:
                    load_model = referencenet_class.from_pretrained(
                        input_dir,
                        subfolder="referencenet",
                        cross_attention_dim=model.cross_attention_dim,
                        need_self_attn_block_embs=need_self_attn_block_embs,
                        need_block_embs=need_block_embs,
                    )
                    hook_model.register_to_config(**load_model.config)
                    hook_model.load_state_dict(load_model.state_dict())
                    del load_model
                    logger.info(
                        f"load_model_hook={i}, referencenet",
                        main_process_only=True,
                    )
                elif "super" in hook_model_class:
                    if os.path.exists(os.path.join(input_dir, "unet")):
                        logger.info(
                            f"load_model_hook={i}, supernet.unet, input_dir={input_dir}",
                            main_process_only=True,
                        )
                        load_model = UNet3DConditionModel.from_pretrained(
                            input_dir,
                            subfolder="unet",
                            temporal_conv_block=model.temporal_conv_block,
                            temporal_transformer=model.temporal_transformer,
                            cross_attention_dim=model.cross_attention_dim,
                            need_spatial_position_emb=model.need_spatial_position_emb,
                            need_transformer_in=need_transformer_in,
                            need_t2i_ip_adapter=need_t2i_ip_adapter,
                            need_adain_temporal_cond=need_adain_temporal_cond,
                            t2i_ip_adapter_attn_processor=t2i_ip_adapter_attn_processor,
                            keep_vision_condtion=keep_vision_condtion,
                            use_anivv1_cfg=use_anivv1_cfg,
                            resnet_2d_skip_time_act=resnet_2d_skip_time_act,
                            norm_spatial_length=norm_spatial_length,
                            spatial_max_length=spatial_max_length,
                            need_refer_emb=need_block_embs
                            and not model.get("only_need_block_embs_adain", False),
                            ip_adapter_cross_attn=need_ip_adapter_cross_attn,
                            t2i_crossattn_ip_adapter_attn_processor=t2i_crossattn_ip_adapter_attn_processor,
                            need_vis_cond_mask=model.get("need_vis_cond_mask", False),
                        )
                        hook_model.unet.register_to_config(**load_model.config)
                        hook_model.unet.load_state_dict(load_model.state_dict())
                        del load_model

                    if os.path.exists(os.path.join(input_dir, "referencenet")):
                        logger.info(
                            f"load_model_hook={i}, supernet.referencenet, input_dir={input_dir}",
                            main_process_only=True,
                        )
                        load_model = referencenet_class.from_pretrained(
                            input_dir,
                            subfolder="referencenet",
                            cross_attention_dim=model.cross_attention_dim,
                            need_self_attn_block_embs=need_self_attn_block_embs,
                            need_block_embs=need_block_embs,
                        )
                        hook_model.referencenet.register_to_config(**load_model.config)
                        hook_model.referencenet.load_state_dict(load_model.state_dict())
                        del load_model

                    if (
                        ip_adapter_cross_attn
                        and os.path.exists(
                            os.path.join(input_dir, "ip_adapter_image_proj.bin")
                        )
                        and train_ip_adapter_image_proj
                    ):
                        logger.info(
                            f"load_model_hook={i}, supernet.ip_adapter_image_proj, input_dir={input_dir}",
                            main_process_only=True,
                        )
                        (
                            _,
                            load_model,
                        ) = load_ip_adapter_vision_clip_encoder_by_name(
                            model_name=model.ip_adapter_params.ip_adapter_model_name,
                            ip_ckpt=os.path.join(
                                input_dir, "ip_adapter_image_proj.bin"
                            ),
                            cross_attention_dim=model.cross_attention_dim,
                            clip_embeddings_dim=model.ip_adapter_params.clip_embeddings_dim,
                            clip_extra_context_tokens=model.ip_adapter_params.clip_extra_context_tokens,
                            ip_scale=model.ip_adapter_params.ip_scale,
                            device=accelerator.device,
                        )
                        hook_model.ip_adapter_image_proj.load_state_dict(
                            load_model.state_dict()
                        )
                        del load_model

                else:
                    logger.warn(
                        f"now only support unet, referencenet, controlnet, but given {hook_model_class}"
                    )

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        supernet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Preprocessing unconditional text
    uncond_embeddings = (
        text_emb_extractor(dataset.uncond_prompt)[0][0]
        .clone()
        .detach()
        .cpu()
        .to(dtype=weight_dtype)
    )
    # Get the training dataset

    # 当有pose等关键点训练的时候，存的是kps，需要一个转换器将其转换成controlent需要的输入
    need_dwposekps2poseimg_converter = False
    if model.use_controlnet_pipeline and dataset.contronet_emb_keys is not None:
        for dct in dataset.contronet_emb_keys:
            if "pose" in dct["name"]:
                need_dwposekps2poseimg_converter = True
    if not need_dwposekps2poseimg_converter:
        dwposekps2poseimg_converter = None
    else:
        logger.info(f"dataset has pose, use PoseKPs2ImgConverter")
        dwposekps2poseimg_converter = PoseKPs2ImgConverter(
            target_width=dataset.dataset.width,
            target_height=dataset.dataset.height,
            num_candidates=dataset.get("controlnet_kps_num_candidates", 20),
            image_processor=VaeImageProcessor(
                vae_scale_factor=2 ** (len(vae_block_out_channels) - 1),
                do_convert_rgb=True,
                do_normalize=False,
            ),
            include_body=model.get("include_body", True),
            include_face=model.get("include_face", False),
            hand_and_face=model.get("hand_and_face", None),
            include_hand=model.get("include_hand", True),
        )

    # change_n_viscond_in_dataset, 变化vision_cond数量的dataset需要batch_sampler配合，在分布式训练中不稳定，所以改为在collate中实现
    train_dataset = PreextractH5pyDataset(
        csv_path=dataset.dataset.csv_path,
        h5py_key=dataset.dataset.h5py_key,
        sep=dataset.dataset.sep,
        text_idx=dataset.text_idx,
        feat_key=dataset.feat_key,
        prompt_emb_key=dataset.prompt_emb_key,
        n_sample_frames=dataset.n_sample_frames,
        sample_start_idx=dataset.sample_start_idx,
        sample_frame_rate=dataset.sample_frame_rate,
        n_vision_condition=n_vision_condition if change_n_viscond_in_dataset else 0,
        max_n_vision_condition=max_n_vision_condition
        if change_n_viscond_in_dataset
        else 0,
        condition_sample_method=dataset.condition_sample_method,
        shuffle=dataset.shuffle,
        contronet_emb_keys=dataset.contronet_emb_keys,
        prob_uncond=dataset.prob_uncond,
        uncond_embeddings=uncond_embeddings,
        prob_static_video=dataset.prob_static_video,
        static_video_prompt=static_video_prompt,
        dynamic_video_prompt=dynamic_video_prompt,
        use_dynamic_sr=use_dynamic_sr,
        max_sample_frame_rate=max_sample_frame_rate,
        video2emb_sr=video2emb_sr,
        tail_frame_prob=tail_frame_prob if change_n_viscond_in_dataset else 0,
        first_element_prob=first_element_prob,
        sample_n_viscond_method=sample_n_viscond_method,
        vision_clip_emb_key=dataset.vision_clip_emb_key
        if hasattr(dataset, "vision_clip_emb_key")
        else None,
        n_refer_image=dataset.get("n_refer_image", 0),
        dwposekps2poseimg_converter=dwposekps2poseimg_converter,
    )
    logger.info("finish prepare train_dataset", main_process_only=True)

    # 自定义collate_fn 如何将一个batch 的 datasetru和拼装
    def collate_fn(examples: List[Dict]):
        dct = {}
        tensor_keys = ["latents", "vision_condition_latents", "vision_clip_emb"]
        if dataset.contronet_emb_keys is not None:
            controlnet_keys = [x.name for x in dataset.contronet_emb_keys]
            tensor_keys.extend(controlnet_keys)
        for k in examples[0].keys():
            if k in ["ids", "n_vision_condition"]:
                continue
            elif examples[0][k] is None:
                dct[k] = None
            elif isinstance(examples[0][k], (str, bool)):
                dct[k] = [example[k] for example in examples]
            elif isinstance(examples[0][k], torch.Tensor):
                dct[k] = torch.stack([example[k] for example in examples]).to(
                    memory_format=torch.contiguous_format, dtype=examples[0][k].dtype
                )
            else:
                for example in examples:
                    print(k, example[k])
                    print(example[k].type, example[k].shape)
                dct[k] = torch.stack([example[k] for example in examples])
        latents = dct["latents"]
        vision_condition_latents = dct["vision_condition_latents"]
        latent_index = dct["latent_index"]
        vision_condition_latent_index = dct["vision_condition_latent_index"]
        if (
            max_n_vision_condition is not None
            and max_n_vision_condition >= n_vision_condition
            and not change_n_viscond_in_dataset
        ):
            (
                latents,
                vision_condition_latents,
                latent_index,
                vision_condition_latent_index,
            ) = mv_batch_head_latent2viscond(
                latents,
                vision_condition_latents,
                latent_index,
                vision_condition_latent_index,
                n_vision_condition=n_vision_condition,
                max_n_vision_condition=max_n_vision_condition,
                sample_n_viscond_method=sample_n_viscond_method,
                first_element_prob=first_element_prob,
            )
        if tail_frame_prob > 0 and not change_n_viscond_in_dataset:
            (
                latents,
                vision_condition_latents,
                latent_index,
                vision_condition_latent_index,
            ) = mv_batch_tail_latent2viscond(
                latents,
                vision_condition_latents,
                latent_index,
                vision_condition_latent_index,
            )
        dct["latents"] = latents.contiguous()
        dct["latent_index"] = latent_index.contiguous()
        if vision_condition_latents is not None:
            dct["vision_condition_latents"] = vision_condition_latents.contiguous()
        if vision_condition_latent_index is not None:
            dct[
                "vision_condition_latent_index"
            ] = vision_condition_latent_index.contiguous()
            n_vision_cond = dct["vision_condition_latent_index"].shape[-1]
            logger.debug(f"collate_fn: change n_vision_cond={n_vision_cond}")
        return dct

    # 当使用动态视觉条件帧时，需要自定义batch_sampler，从而按照视觉条件帧数量 组batch
    if (n_vision_condition > 0 or tail_frame_prob > 0) and change_n_viscond_in_dataset:
        use_num_vis_cond_batch_sampler = True
    else:
        use_num_vis_cond_batch_sampler = False
    drop_last = False
    if use_num_vis_cond_batch_sampler:
        batch_sampler = NumVisCondBatchSampler(
            train_dataset, batch_size=train_batch_size, drop_last=drop_last
        )
        shuffle = False
        batch_size = 1
        drop_last = None
    else:
        shuffle = True
        batch_sampler = None
        batch_size = train_batch_size
        drop_last = drop_last

    logger.info(
        f"dataset batchsampler use {batch_sampler.__class__.__name__}",
        main_process_only=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        batch_sampler=batch_sampler,
        drop_last=drop_last,
    )
    logger.info(f"train_dataloader num of batch", main_process_only=True)
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        supernet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        noise_scheduler,
    ) = accelerator.prepare(
        supernet, optimizer, train_dataloader, lr_scheduler, noise_scheduler
    )
    logger.info(
        "accelerator.prepare supernet, optimizer, train_dataloader, lr_scheduler",
        main_process_only=True,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video_accelerator_tensorboard")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    # Train!
    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )
    logger.info("***** Running training *****", main_process_only=True)

    logger.info(f"  Num examples = {len(train_dataset)}", main_process_only=True)
    logger.info(f"  Num Epochs = {num_train_epochs}", main_process_only=True)
    logger.info(
        f"  Distributed environment = {accelerator.use_distributed}, distributed_type={accelerator.distributed_type}",
        main_process_only=True,
    )
    logger.info(
        f"  Instantaneous batch size per device = {train_batch_size}",
        main_process_only=True,
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}",
        main_process_only=True,
    )
    logger.info(
        f"  Gradient Accumulation steps = {gradient_accumulation_steps}",
        main_process_only=True,
    )
    logger.info(
        f"  Total optimization steps = {max_train_steps}", main_process_only=True
    )
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if not os.path.exists(output_dir.checkpoints):
                logger.info(
                    f"Checkpoint {output_dir.checkpoints} does not exist. Start training from beginning.",
                    main_process_only=True,
                )
                path = None
                resume_from_checkpoint = None
            else:
                dirs = os.listdir(output_dir.checkpoints)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                if len(dirs) == 0:
                    logger.info(
                        f"Checkpoint latest does not exist. Start training from beginning.",
                        main_process_only=True,
                    )
                    path = None
                    resume_from_checkpoint = None
                else:
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1]

        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir.checkpoints, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * gradient_accumulation_steps
            )
            accelerator.print(
                f"Resuming from first_epoch={first_epoch}, resume_step={resume_step}"
            )
            if model.temporal_transformer is not None:
                accelerator.print(
                    accelerator.unwrap_model(supernet)
                    .unet.up_blocks[2]
                    .temp_attentions[0]
                    .transformer_blocks[0]
                    .attn1.to_v.weight
                )
            # if resume, close profiler
            output_dir.profiler.use = False

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    if get_midway_ckpt is True:
        num_train_epochs = first_epoch

    # 是否一开始就跑 validation
    if output_dir.track_init and validation_data.do_validation:
        if accelerator.is_main_process:
            logger.info(
                "Running validation... after model initialization and before training",
                main_process_only=True,
            )
            log_validation(
                unet=accelerator.unwrap_model(supernet).unet,
                device=accelerator.device,
                weight_dtype=weight_dtype,
                epoch=first_epoch,
                global_step=global_step,
                output_dir=output_dir,
                validation_data=validation_data,
                model=model,
                train=train,
                dataset=dataset,
                diffusion=diffusion,
                referencenet=accelerator.unwrap_model(supernet).referencenet
                if use_referencenet
                else None,
                ip_adapter_image_proj=accelerator.unwrap_model(
                    supernet
                ).ip_adapter_image_proj
                if ip_adapter_cross_attn
                else None,
            )

    # 是否使用 torch.profiler跟进训练进度，由于数据太多，tensorboard往往很卡，
    # TODO： 移到独立的脚本来跟进模型计算会更好
    if output_dir.profiler.use:
        if accelerator.is_main_process:
            prof = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=output_dir.profiler.wait,
                    warmup=output_dir.profiler.warmup,
                    active=output_dir.profiler.active,
                    repeat=output_dir.profiler.repeat,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    output_dir.log_dir
                ),
                record_shapes=output_dir.profiler.record_shapes,
                with_stack=output_dir.profiler.with_stack,
                profile_memory=output_dir.profiler.profile_memory,
                with_flops=output_dir.profiler.with_flops,
                with_modules=output_dir.profiler.with_modules,
            )
            prof_step_counter = 0
            prof_total_step_num = (
                output_dir.profiler.wait
                + output_dir.profiler.warmup
                + output_dir.profiler.active
            ) * output_dir.profiler.repeat
            prof.start()

    if testing_speed:
        num_train_epochs = first_epoch + 1
    elapsed_time = []

    if only_validation:
        return
    logger.info(
        "====> start train \n",
        main_process_only=True,
    )

    for epoch in range(first_epoch, num_train_epochs):
        supernet.train()
        train_loss = 0.0
        step_start = time.time()
        global_step_start = time.time()
        start = time.time()
        global_step_time = 0
        # deprecate
        # dataset重新生成一遍 n_vision_cond等每个epoch需要不同的配置
        if use_num_vis_cond_batch_sampler:
            train_dataloader.dataset.prepare_init_datas()
        logger.debug(f"epoch={epoch} finish prepare_init_datas ")
        t0 = time.time()
        for step, batch in enumerate(train_dataloader):
            epoch_step = epoch * len(train_dataloader) + step
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                t1 = time.time()
                print(
                    f"========================== resume_from_checkpoint, step: {step}, iterate to resume_step:{resume_step} TIME COST: {t1-t0}"
                )
                continue
            logger.debug("\n")
            with accelerator.accumulate(supernet):
                # logger.debug(
                #     f"torch model address, {torch.cuda.current_device()}, {next(text_encoder.parameters()).storage().data_ptr()}",
                # )
                dataloader_time = time.time() - start
                start = time.time()
                # Load latent space
                # batch_size * channel * time_size * height * width
                latents = batch["latents"].to(dtype=weight_dtype)
                latents = latents * vae_scaling_factor
                vision_condition_latents = batch["vision_condition_latents"]
                frame_index = batch["frame_index"]
                vision_clip_emb = batch["vision_clip_emb"]
                if vision_clip_emb is not None and ip_adapter_cross_attn:
                    vision_clip_emb = vision_clip_emb.to(dtype=weight_dtype)
                refer_image_latents = batch["refer_image_latents"]
                if refer_image_latents is not None:
                    refer_image_latents = refer_image_latents.to(dtype=weight_dtype)
                logger.debug(f"frame_index, {step}, {frame_index}")
                if vision_condition_latents is not None:
                    vision_condition_latents = vision_condition_latents.to(
                        dtype=weight_dtype
                    )
                    vision_condition_latents = (
                        vision_condition_latents * vae_scaling_factor
                    )

                # 现阶段 一个batch的index都是相同的，使用一维可以加速相关数据处理
                latent_index = batch["latent_index"][0]
                vision_condition_latent_index = batch["vision_condition_latent_index"]
                if vision_condition_latent_index is not None:
                    vision_condition_latent_index = vision_condition_latent_index[0]
                logger.debug(f"train, latent_index, {latent_index}")
                logger.debug(
                    f"train, vision_condition_latent_index, {vision_condition_latent_index}"
                )
                if model.use_controlnet_pipeline:
                    # TODO: support multi controlnets
                    logger.debug(
                        f"batch[dataset.contronet_emb_keys[0].name={batch[dataset.contronet_emb_keys[0].name].shape}"
                    )
                    controlnet_latents_dct = {
                        k.name: rearrange(batch[k.name], "b c t h w->(b t) c h w")
                        for k in dataset.contronet_emb_keys
                    }
                    # 目前仅支持Controlnet-pose，不支持MultiControlnet
                    # 使用在 赋值 controlnet_cond，不使用controlnet_cond_latents
                    controlnet_latents = controlnet_latents_dct[
                        dataset.contronet_emb_keys[0].name
                    ].to(dtype=weight_dtype)
                    logger.debug(f"controlnet_latents={controlnet_latents.shape}")
                # Sample noise that we'll add to the latents
                if noise_type == "video_fusion":
                    noise = video_fusion_noise(
                        latents, w_ind_noise=torch.tensor(0.5, device=latents.device)
                    )
                else:
                    noise = random_noise(latents, noise_offset=noise_offset)
                noise = noise.to(dtype=weight_dtype)
                logger.debug(f"noise, shape={noise.shape}, mean={noise.mean()}")
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # add noise to viscond start
                if vision_condition_latents is not None and add_viscond_timestep_noise:
                    viscond_timestep = torch.randint(
                        viscond_timestep_noise_min,
                        viscond_timestep_noise_max,
                        (bsz,),
                        device=latents.device,
                    )
                    viscond_noise = random_noise(
                        vision_condition_latents, noise_offset=noise_offset
                    ).to(dtype=weight_dtype)
                    # Sample a random timestep for each viscond
                    viscond_timestep = viscond_timestep.long()
                    vision_condition_latents = noise_scheduler.add_noise(
                        vision_condition_latents, viscond_noise, viscond_timestep
                    )
                # add noise to viscond end

                # Get the text embedding for conditioning
                encoder_hidden_states = batch["prompt_clip_emb"]
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)
                    v_logger.debug(
                        f"encoder_hidden_states, type={type(encoder_hidden_states)}, shape={encoder_hidden_states.shape}"
                    )
                    # update_prompt_clip_emb, 是否需要更新 text_emb
                    update_prompt_clip_emb_lst = batch["update_prompt_clip_emb"]
                    v_logger.debug(
                        f"update_prompt_clip_emb_lst, {update_prompt_clip_emb_lst}"
                    )
                    if any(update_prompt_clip_emb_lst):
                        prompts = batch["prompt"]
                        logger.debug(f"new_prompts, {prompts}")
                        encoder_hidden_states = text_emb_extractor(prompts)[0].to(
                            dtype=weight_dtype
                        )
                        logger.debug(
                            f"encoder_hidden_states, {encoder_hidden_states.shape}, {dataset.n_sample_frames}"
                        )
                        encoder_hidden_states = repeat(
                            encoder_hidden_states,
                            "b n q ->b t n q",
                            t=dataset.n_sample_frames,
                        )
                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                if vision_condition_latents is not None:
                    noisy_latents = batch_concat_two_tensor_with_index(
                        data1=noisy_latents,
                        data1_index=latent_index,
                        data2=vision_condition_latents,
                        data2_index=vision_condition_latent_index,
                        dim=2,
                    )
                if model.use_controlnet_pipeline:
                    noisy_latents_reshape = rearrange(
                        noisy_latents, "b c t h w->(b t) c h w"
                    )
                    timesteps_reshape = repeat(
                        timesteps, "b -> (b t)", t=dataset.n_sample_frames
                    )
                    controlnet_inference_params = {
                        "sample": noisy_latents_reshape,
                        "timestep": timesteps_reshape,
                        "controlnet_cond": controlnet_latents,
                        "controlnet_cond_latents": None,
                        "return_dict": False,
                    }
                else:
                    controlnet_inference_params = None
                logger.debug(f"train before supernet latent_index ={latent_index} ")
                # text_prompt_emb和image_prompt_emb统一在supernet中处理
                if use_referencenet:
                    ref_timestep_int = 0
                    if refer_image_latents is not None:
                        n_refer_image = refer_image_latents.shape[2]
                        ref_sample = rearrange(
                            refer_image_latents, "b c t h w-> (b t) c h w"
                        )
                    else:
                        n_refer_image = vision_condition_latent_index.shape[-1]
                        ref_sample = rearrange(
                            vision_condition_latents.clone(), "b c t h w-> (b t) c h w"
                        )

                    ref_timestep = (
                        torch.ones((ref_sample.shape[0],), device=accelerator.device)
                        * ref_timestep_int
                    ).long()
                    logger.debug(f"referencenet: ref_sample={ref_sample.shape}")
                    logger.debug(f"referencenet: ref_timestep={ref_timestep.shape}")
                    logger.debug(f"referencenet: n_refer_image={n_refer_image}")
                    referencenet_inference_params = {
                        "sample": ref_sample,
                        "timestep": ref_timestep,
                        "num_frames": n_refer_image,
                        "return_ndim": 5,
                    }
                else:
                    referencenet_inference_params = None

                unet_inference_params = {
                    "sample": noisy_latents,
                    "timestep": timesteps,
                    "sample_index": latent_index,
                    "vision_conditon_frames_sample_index": vision_condition_latent_index,
                    "frame_index": frame_index,
                }
                model_pred = supernet(
                    unet_params=unet_inference_params,
                    referencenet_params=referencenet_inference_params,
                    controlnet_params=controlnet_inference_params,
                    controlnet_scale=model.controlnet_scale,
                    vision_clip_emb=vision_clip_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                ).sample
                logger.debug(
                    f"model_pred, shape={model_pred.shape}, mean={model_pred.mean()}"
                )
                logger.debug(f"target, shape={target.shape}, mean={target.mean()}")
                if n_sample_condition_frames > 0:
                    viscond_pred = batch_index_select(
                        model_pred, dim=2, index=vision_condition_latent_index
                    )
                    model_pred = batch_index_select(
                        model_pred, dim=2, index=latent_index
                    )
                    if need_video_viscond_loss:
                        noisy_latents = batch_index_select(
                            noisy_latents, dim=2, index=latent_index
                        )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.log({"loss/denoise": loss.detach().item()}, step=epoch_step)
                # 一致性 loss start
                last_timesteps_weight = (
                    (timesteps / noise_scheduler.config.num_train_timesteps)
                    <= video_inter_frames_timestep_th
                ) * 1.0
                if need_video_viscond_loss:
                    # TODO: 支持不同的 noise_type  用于支持一个 batch 不同的 timesteps , 暂不支持
                    pred_video = torch.stack(
                        [
                            noise_scheduler.step(
                                model_pred[i, ...],
                                timesteps.cpu()[i],
                                noisy_latents[i, ...],
                            ).pred_original_sample
                            for i in range(len(model_pred))
                        ],
                        dim=0,
                    ).to(dtype=weight_dtype)

                    video_viscond_loss = cal_viscond_video_latents_loss(
                        pred_video,
                        vision_condition_latents,
                        timesteps_weight=last_timesteps_weight,
                    )
                    accelerator.log(
                        {"loss/video_viscond_loss": video_viscond_loss.detach().item()},
                        step=epoch_step,
                    )
                    loss += video_viscond_loss_weight * video_viscond_loss
                    if use_pixel_loss:
                        pred_video_pixel = decode_unet_latents_with_vae(vae, pred_video)
                        vis_cond_video_pixel = decode_unet_latents_with_vae(
                            vae, vision_condition_latents
                        )
                        pixel_video_viscond_loss = cal_viscond_video_latents_loss(
                            pred_video_pixel,
                            vis_cond_video_pixel,
                            timesteps_weight=last_timesteps_weight,
                        )
                        accelerator.log(
                            {
                                "loss/pixel_video_viscond_loss": pixel_video_viscond_loss.detach().item()
                            },
                            step=epoch_step,
                        )

                        loss += (
                            pixel_video_viscond_loss_weight * pixel_video_viscond_loss
                        )
                    else:
                        pred_video_pixel = None
                        vis_cond_video_pixel = None
                else:
                    pred_video = None
                    pred_video_pixel = None
                    vis_cond_video_pixel = None

                if need_video_inter_frames_loss:
                    if pred_video is None:
                        pred_video = torch.stack(
                            [
                                noise_scheduler.step(
                                    model_pred[i, ...],
                                    timesteps.cpu()[i],
                                    noisy_latents[i, ...],
                                ).pred_original_sample
                                for i in range(len(model_pred))
                            ],
                            dim=0,
                        ).to(dtype=weight_dtype)
                    video = batch_concat_two_tensor_with_index(
                        pred_video,
                        latent_index,
                        vision_condition_latents,
                        vision_condition_latent_index,
                        dim=2,
                    )
                    video_inter_frames_loss = cal_video_inter_frames_loss(
                        video,
                        loss_type=video_inter_frames_loss_type,
                        timesteps_weight=last_timesteps_weight,
                    )
                    accelerator.log(
                        {
                            "loss/video_inter_frames_loss": video_inter_frames_loss.detach().item()
                        },
                        step=epoch_step,
                    )
                    loss += video_inter_frames_loss_weight * video_inter_frames_loss
                    if use_pixel_loss:
                        if pred_video_pixel is None or vis_cond_video_pixel is None:
                            pred_video_pixel = decode_unet_latents_with_vae(
                                vae, pred_video
                            )
                            vis_cond_video_pixel = decode_unet_latents_with_vae(
                                vae, vision_condition_latents
                            )

                        video = batch_concat_two_tensor_with_index(
                            pred_video_pixel,
                            latent_index,
                            vis_cond_video_pixel,
                            vision_condition_latent_index,
                            dim=0,
                        )
                        pixel_video_inter_frames_loss = cal_video_inter_frames_loss(
                            video,
                            loss_type=video_inter_frames_loss_type,
                            timesteps_weight=last_timesteps_weight,
                        )
                        accelerator.log(
                            {
                                "loss/pixel_video_inter_frames_loss": pixel_video_inter_frames_loss.detach().item()
                            },
                            step=epoch_step,
                        )
                        loss += (
                            pixel_video_inter_frames_loss_weight
                            * pixel_video_inter_frames_loss
                        )
                if need_viscond_loss:
                    unet_inference_params = {
                        "sample": vision_condition_latents,
                        "timestep": timesteps * 0,
                        "sample_frame_rate": actual_sample_frame_rate,
                        "skip_temporal_layers": True,
                    }
                    viscond_model_pred = supernet(
                        unet_params=unet_inference_params
                    ).sample
                    viscond_loss = F.mse_loss(
                        viscond_pred,
                        viscond_model_pred,
                    )
                    accelerator.log(
                        {"loss/viscond_loss": viscond_loss.detach().item()},
                        step=epoch_step,
                    )
                    loss += viscond_loss_weight * viscond_loss

                # 一致性 loss end

                unet_elapsed_time = time.time() - start
                start = time.time()
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(supernet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                backward_elapsed_time = time.time() - start
                start = time.time()

                if testing_speed is True and accelerator.is_main_process:
                    elapsed_time.append(
                        [dataloader_time, unet_elapsed_time, backward_elapsed_time]
                    )

            # Checks if the accelerator has performed an optimization step behind the scenes
            sync_gradients_satrt = time.time()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step_time = time.time() - global_step_start
                global_step_start = time.time()
                global_step += 1
                train_loss = 0.0
                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        fn_recursive_search(
                            name="unet",
                            module=accelerator.unwrap_model(supernet).unet,
                            target="temporal_weight",
                            print_method=accelerator.print,
                            print_name="data",
                        )
                        # if (
                        #     model.temporal_transformer
                        #     is not None
                        #     # and model.temporal_transformer
                        # ):
                        #     unet_temporal_attention_to_v_param = (
                        #         accelerator.unwrap_model(supernet)
                        #         .unet.up_blocks[2]
                        #         .temp_attentions[0]
                        #         .transformer_blocks[0]
                        #         .attn1.to_v.weight  # .detach()
                        #     )
                        #     logger.debug(
                        #         (
                        #             f"epoch={epoch}, step={step}, global_step={global_step}, unet_temporal_attention_to_v_param, is_leaf={unet_temporal_attention_to_v_param.is_leaf}, "
                        #             f"requires_grad={unet_temporal_attention_to_v_param.requires_grad}, grad={type(unet_temporal_attention_to_v_param.grad)}"
                        #         ),
                        #         main_process_only=True,
                        #     )
                        #     unet_temporal_attention_to_v_weight = torch.mean(
                        #         unet_temporal_attention_to_v_param
                        #     )
                        #     unet_temporal_attention_to_v_weight_grad = (
                        #         torch.mean(unet_temporal_attention_to_v_param.grad)
                        #         if train_unet
                        #         else None
                        #     )
                        #     logger.debug(
                        #         (
                        #             f"epoch={epoch}, step={step}, global_step={global_step},"
                        #             f"parameter/unet_temporal_attention_to_v_weight_mean ={unet_temporal_attention_to_v_weight}, "
                        #             f"unet_temporal_attention_to_v_weight_grad_mean={unet_temporal_attention_to_v_weight_grad}"
                        #         ),
                        #         main_process_only=True,
                        #     )
                        #     accelerator.log(
                        #         {
                        #             "weight/unet_temporal_attention_to_v_weight": unet_temporal_attention_to_v_weight
                        #         },
                        #         step=global_step,
                        #     )

                        # if referencenet is not None and referencenet.requires_grad:
                        #     referencenet_param = (
                        #         accelerator.unwrap_model(
                        #             supernet
                        #         ).referencenet.conv_in.weight
                        #         if train_referencenet
                        #         else None
                        #     )  # .detach()
                        #     logger.debug(
                        #         (
                        #             f"epoch={epoch}, step={step}, global_step={global_step}, accelerator.unwrap_model(supernet).referencenet.conv_in.weight.grad, "
                        #             f"requires_grad={referencenet_param.requires_grad}, grad={type(referencenet_param.grad)}, is_leaf={referencenet_param.is_leaf}, "
                        #         ),
                        #         main_process_only=True,
                        #     )
                        #     logger.debug(
                        #         f"epoch={epoch}, step={step}, global_step={global_step},  referencenet={torch.mean(referencenet_param)}",
                        #         main_process_only=True,
                        #     )
                        #     if (
                        #         train_referencenet
                        #         and referencenet_param.grad is not None
                        #     ):
                        #         logger.debug(
                        #             f"epoch={epoch}, step={step}, global_step={global_step},  referencenet_grad={torch.mean(referencenet_param.grad)}",
                        #             main_process_only=True,
                        #         )
                        #     accelerator.log(
                        #         {
                        #             "weight/referencenet_conv_in_weight": torch.mean(
                        #                 referencenet_param
                        #             )
                        #         },
                        #         step=global_step,
                        #     )
                        save_path = os.path.join(
                            output_dir.checkpoints, f"checkpoint-{global_step}"
                        )
                        if save_checkpoint:
                            accelerator.save_state(save_path, safe_serialization=False)
                        logger.info(
                            f"Saved state to {save_path}",
                            main_process_only=True,
                        )

                if (
                    global_step % validation_steps == 0
                    and validation_data.do_validation
                ):
                    if accelerator.is_main_process:
                        logger.info(
                            f"Running validation... epoch={epoch}, global_step={global_step}",
                            main_process_only=True,
                        )
                        log_validation(
                            unet=accelerator.unwrap_model(supernet).unet,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            epoch=epoch,
                            global_step=global_step,
                            output_dir=output_dir,
                            validation_data=validation_data,
                            model=model,
                            train=train,
                            dataset=dataset,
                            diffusion=diffusion,
                            referencenet=accelerator.unwrap_model(supernet).referencenet
                            if use_referencenet
                            else None,
                            ip_adapter_image_proj=accelerator.unwrap_model(
                                supernet
                            ).ip_adapter_image_proj
                            if ip_adapter_cross_attn
                            else None,
                        )
                        logger.info(
                            f"Saved samples to {output_dir.samples}",
                            main_process_only=True,
                        )
                    start = time.time()
                if (
                    testing_speed is True
                    and accelerator.is_main_process
                    and global_step == 100
                ):
                    df_time = pd.DataFrame(
                        elapsed_time,
                        columns=["data_loader_time", "unet_time", "backward_time"],
                    )
                    df_time.to_csv("elapsed_time.csv", sep=",", index=False)
                    break

            # logger.debug(
            #     f"accelerator sync_gradients_satrt time={time.time()-sync_gradients_satrt}"
            # )

            # profiler
            if accelerator.is_main_process:
                if output_dir.profiler.use:
                    prof.step()
                    prof_step_counter += 1
                    if prof_step_counter == prof_total_step_num:
                        prof.stop()
                        torch.autograd.profiler.KinetoStepTracker.erase_step_count(
                            torch.profiler.profiler.PROFILER_STEP_NAME
                        )
                        logger.info("profiler stop", main_process_only=True)
                step_time = time.time() - step_start
                logs = OrderedDict(
                    {
                        "step_loss": loss.detach().item(),
                        "train_loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "data": dataloader_time,
                        "unet": unet_elapsed_time,
                        "backward": backward_elapsed_time,
                        "step_time": step_time,
                        "time/global_step_time": global_step_time,
                    }
                )
                # progress_bar.set_postfix(**logs)
                log_start = time.time()
                accelerator.log(
                    {
                        "loss/step_loss": logs["step_loss"],
                        "loss/train_loss": logs["train_loss"],
                        "parameter/lr": logs["lr"],
                        "time/unet_forward_time": unet_elapsed_time,
                        "time/unet_backward_time": backward_elapsed_time,
                        "time/data_time": dataloader_time,
                        "time/step_time": step_time,
                        "time/global_step_time": global_step_time,
                    },
                    step=global_step,
                )
                logger.debug(f"accelerator logtime={time.time()-log_start}")
                logger.info(
                    "epoch={}, step={}, global_step={}, {}".format(
                        epoch,
                        step,
                        global_step,
                        ", ".join(
                            [
                                "{}={:.6f}".format(k.replace("/", "_"), v)
                                for k, v in logs.items()
                            ]
                        ),
                    ),
                    main_process_only=True,
                )
                step_start = time.time()
                start = time.time()
            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and train.get("save_final_pipeline", True):
        unet = accelerator.unwrap_model(supernet).unet
        pipeline = MusevControlNetPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            controlnet=None,
        )
        if get_midway_ckpt is True:
            final_dir = os.path.join(output_dir.checkpoints, path, "models")
            os.makedirs(final_dir, exist_ok=True)
            pipeline.save_pretrained(final_dir, safe_serialization=False)
        else:
            pipeline.save_pretrained(output_dir.checkpoints, safe_serialization=False)
    if output_dir.profiler.use:
        if accelerator.is_main_process:
            prof.stop()
    accelerator.end_training()


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train/musev_referencenet_train_template.yaml",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
    )
    args = parser.parse_args()
    v_logger.setLevel(args.log_level)
    logger.setLevel(args.log_level)
    config = OmegaConf.load(args.config)
    dct = OmegaConf.to_container(
        config, structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )
    pprint(dct)
    main(**config)
