from einops import rearrange

from torch import nn
import torch


def decode_unet_latents_with_vae(vae: nn.Module, latents: torch.tensor):
    n_dim = latents.ndim
    batch_size = latents.shape[0]
    if n_dim == 5:
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
    latents = 1 / vae.config.scaling_factor * latents
    video = vae.decode(latents, return_dict=False)[0]
    video = (video / 2 + 0.5).clamp(0, 1)
    if n_dim == 5:
        latents = rearrange(latents, "(b f) h w c -> b c f h w", b=batch_size)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    return video
