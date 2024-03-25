import os
import imageio
import numpy as np
from typing import Literal, Union, List, Dict, Tuple

import torch
import torchvision
import cv2
from PIL import Image

from tqdm import tqdm
from einops import rearrange
import webp
import subprocess

from .. import logger


def save_videos_to_images(videos: np.array, path: str, image_type="png") -> None:
    """save video batch to images into image_type

    Args:
        videos (np.array): [h w c]
        path (str): image directory path
    """
    os.makedirs(path, exist_ok=True)
    for i, video in enumerate(videos):
        imageio.imsave(os.path.join(path, f"{i:04d}.{image_type}"), video)


def save_videos_grid(
    videos: torch.Tensor,
    path: str,
    rescale=False,
    n_rows=4,  # 一行多少个视频
    fps=8,
    save_type="webp",
) -> None:
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        if x.dtype != torch.uint8:
            x = (x * 255).numpy().astype(np.uint8)

        if save_type == "webp":
            outputs.append(Image.fromarray(x))
        else:
            outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if "gif" in path or save_type == "gif":
        params = {
            "duration": int(1000 * 1.0 / fps),
            "loop": 0,
        }
    elif save_type == "mp4":
        params = {
            "quality": 9,
            "fps": fps,
            "pixelformat": "yuv420p",
        }
    else:
        params = {
            "quality": 9,
            "fps": fps,
        }

    if save_type == "webp":
        webp.save_images(outputs, path, fps=fps, lossless=True)
    else:
        imageio.mimsave(path, outputs, **params)


def make_grid_with_opencv(
    batch: Union[torch.Tensor, np.ndarray],
    nrows: int,
    texts: List[str] = None,
    rescale: bool = False,
    font_size: float = 0.05,
    font_thickness: int = 1,
    font_color: Tuple[int] = (255, 0, 0),
    tensor_order: str = "b c h w",
    write_info: bool = False,
) -> np.ndarray:
    """read tensor batch and make a grid with opencv

    Args:
        batch (Union[torch.Tensor, np.ndarray]): 4 dim tensor, like b c h w
        nrows (int): how many rows in the grid
        texts (List[str], optional): text to write in video . Defaults to None.
        rescale (bool, optional): whether rescale [0,1] from [-1, 1]. Defaults to False.
        font_size (float, optional): font size. Defaults to 0.05.
        font_thickness (int, optional): font_thickness . Defaults to 1.
        font_color (Tuple[int], optional): text color. Defaults to (255, 0, 0).
        tensor_order (str, optional): batch channel order. Defaults to "b c h w".
        write_info (bool, optional): whether write text into video. Defaults to True.

    Returns:
        np.ndarray: h w c
    """
    if isinstance(batch, torch.Tensor):
        batch = batch.cpu().numpy()
    # batch: (B, C, H, W)
    batch = rearrange(batch, f"{tensor_order} -> b h w c")
    b, h, w, c = batch.shape
    ncols = int(np.ceil(b / nrows))
    grid = np.zeros((h * nrows, w * ncols, c), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, x in enumerate(batch):
        i_row, i_col = i // ncols, i % ncols
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).astype(np.uint8)
        # 没有这行会报错
        # ref: https://stackoverflow.com/questions/72327137/opencv4-5-5-error-5bad-argument-in-function-puttext
        x = x.copy()
        if texts is not None and write_info:
            x = cv2.putText(
                x,
                texts[i],
                (5, 20),
                font,
                fontScale=font_size,
                color=font_color,
                thickness=font_thickness,
            )
        grid[i_row * h : (i_row + 1) * h, i_col * w : (i_col + 1) * w, :] = x
    return grid


def save_videos_grid_with_opencv(
    videos: Union[torch.Tensor, np.ndarray],
    path: str,
    n_cols: int,
    texts: List[str] = None,
    rescale: bool = False,
    fps: int = 8,
    font_size: int = 0.6,
    font_thickness: int = 1,
    font_color: Tuple[int] = (255, 0, 0),
    tensor_order: str = "b c t h w",
    batch_dim: int = 0,
    split_size_or_sections: int = None,  # split batch to avoid large video
    write_info: bool = False,
    save_filetype: Literal["gif", "mp4", "webp"] = "mp4",
    save_images: bool = False,
) -> None:
    """存储tensor视频为gif、mp4等

    Args:
        videos (Union[torch.Tensor, np.ndarray]): 五维视频tensor， 如 b c t h w，值范围[0-1]
        path (str): 视频存储路径，后缀会影响存储方式
        n_cols (int): 由于b可能特别大，所以会分成几列
        texts (List[str], optional): b长度，会写在每个b视频左上角. Defaults to None.
        rescale (bool, optional): 输入是[-1,1]时，应该为True. Defaults to False.
        fps (int, optional): 存储视频的fps. Defaults to 8.
        font_size (int, optional): text对应的字体大小. Defaults to 0.6.
        font_thickness (int, optional): 字体宽度. Defaults to 1.
        font_color (Tuple[int], optional): 字体颜色. Defaults to (255, 0, 0).
        tensor_order (str, optional): 输入tensor的顺序，如果不是 `b c t h w`，会被转换成 b c t h w，. Defaults to "b c t h w".
        batch_dim (int, optional): 有时候b特别大，这时候一个视频就太大了，就可以分成几个视频存储. Defaults to 0.
        split_size_or_sections (int, optional): 不为None时，与batch_dim配套，一个存储视频最多支持几个子视频。会按照n_cols截断向上取整数. Defaults to None.
        write_info (bool, False): 是否也些提示信息在视频上
    """
    if split_size_or_sections is not None:
        split_size_or_sections = int(np.ceil(split_size_or_sections / n_cols)) * n_cols
        if isinstance(videos, np.ndarray):
            videos = torch.from_numpy(videos)
        # 比np.array_split更适合
        videos_split = torch.split(videos, split_size_or_sections, dim=batch_dim)
        videos_split = [videos.cpu().numpy() for videos in videos_split]
    else:
        videos_split = [videos]
    n_videos_split = len(videos_split)
    dirname, basename = os.path.dirname(path), os.path.basename(path)
    filename, ext = os.path.splitext(basename)
    os.makedirs(dirname, exist_ok=True)

    for i_video, videos in enumerate(videos_split):
        videos = rearrange(videos, f"{tensor_order} -> t b c h w")
        outputs = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        batch_size = videos.shape[1]
        n_rows = int(np.ceil(batch_size / n_cols))
        for t, x in enumerate(videos):
            x = make_grid_with_opencv(
                x,
                n_rows,
                texts,
                rescale,
                font_size,
                font_thickness,
                font_color,
                write_info=write_info,
            )
            h, w, c = x.shape
            x = x.copy()
            if write_info:
                x = cv2.putText(
                    x,
                    str(t),
                    (5, h - 20),
                    font,
                    fontScale=2,
                    color=font_color,
                    thickness=font_thickness,
                )
            outputs.append(x)
        logger.debug(f"outputs[0].shape: {outputs[0].shape}")
        # TODO: 有待更新实现方式
        if i_video == 0 and n_videos_split == 1:
            pass
        else:
            path = os.path.join(dirname, "{}_{}{}".format(filename, i_video, ext))
        if save_filetype == "gif":
            params = {
                "duration": int(1000 * 1.0 / fps),
                "loop": 0,
            }
            imageio.mimsave(path, outputs, **params)
        elif save_filetype == "mp4":
            params = {
                "quality": 9,
                "fps": fps,
            }
            imageio.mimsave(path, outputs, **params)
        elif save_filetype == "webp":
            outputs = [Image.fromarray(x_tmp) for x_tmp in outputs]
            webp.save_images(outputs, path, fps=fps, lossless=True)
        else:
            raise ValueError(f"Unsupported file type: {save_filetype}")
        if save_images:
            images_path = os.path.join(dirname, filename)
            os.makedirs(images_path, exist_ok=True)
            save_videos_to_images(outputs, images_path)


def export_to_video(videos: torch.Tensor, output_video_path: str, fps=8):
    tmp_path = output_video_path.replace(".mp4", "_tmp.mp4")

    videos = rearrange(videos, "b c t h w -> b t h w c")
    videos = videos.squeeze()
    videos = (videos * 255).cpu().detach().numpy().astype(np.uint8)  # tensor -> numpy
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = videos[0].shape
    video_writer = cv2.VideoWriter(
        tmp_path, fourcc, fps=fps, frameSize=(w, h), isColor=True
    )
    for i in range(len(videos)):
        img = cv2.cvtColor(videos[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    video_writer.release()  # 要释放video writer，否则无法播放
    cmd = f"ffmpeg -y -i {tmp_path} -c:v libx264 -c:a aac -strict -2 {output_video_path} -loglevel quiet"
    subprocess.run(cmd, shell=True)
    os.remove(tmp_path)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipeline.text_encoder(
        uncond_input.input_ids.to(pipeline.device)
    )[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
    ddim_scheduler,
):
    timestep, next_timestep = (
        min(
            timestep
            - ddim_scheduler.config.num_train_timesteps
            // ddim_scheduler.num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = (
        ddim_scheduler.alphas_cumprod[timestep]
        if timestep >= 0
        else ddim_scheduler.final_alpha_cumprod
    )
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (
        sample - beta_prod_t**0.5 * model_output
    ) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = (
        alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    )
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(
        pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt
    )
    return ddim_latents


def fn_recursive_search(
    name: str,
    module: torch.nn.Module,
    target: str,
    print_method=print,
    print_name: str = "data",
):
    if hasattr(module, target):
        print_method(
            [
                name + "." + target + "." + print_name,
                getattr(getattr(module, target), print_name)[0].cpu().detach().numpy(),
            ]
        )

    parent_name = name
    for name, child in module.named_children():
        fn_recursive_search(
            parent_name + "." + name, child, target, print_method, print_name
        )


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg
