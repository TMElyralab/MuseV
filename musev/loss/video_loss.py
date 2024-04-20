from einops import repeat
import torch
from torch.nn import functional as F

# from ..data.data_util import align_repeat_tensor_single_dim


def weighted_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    diff = (input - target) ** 2
    if weight is not None:
        weight_shape = weight.shape
        # 将weight的形状扩展为和target一样的形状
        weight2target_shape = (weight_shape[-1],) + (1,) * (target.ndim - 1)
        weight = weight.view(weight2target_shape)
        weight = weight.expand_as(diff)
        diff = weight * diff
    if reduction == "mean":
        out = torch.mean(diff)
    elif reduction == "sum":
        out = torch.sum(diff)
    else:
        raise ValueError(f"only support mean or sum, but given {reduction}")
    return out


def cal_video_inter_frames_mse_loss(
    video: torch.Tensor,
    timesteps_weight: torch.Tensor = None,
) -> torch.Tensor:
    video_diff = torch.diff(video, dim=2)
    loss = weighted_mse_loss(
        video_diff,
        torch.zeros_like(video_diff, device=video_diff.device, dtype=video_diff.dtype),
        weight=timesteps_weight,
    )
    return loss


def cal_video_inter_frames_loss(
    video: torch.Tensor,
    loss_type: str = "MSE",
    timesteps_weight: torch.Tensor = None,
) -> torch.Tensor:
    if loss_type == "MSE":
        loss = cal_video_inter_frames_mse_loss(video, timesteps_weight)
    else:
        raise NotImplementedError(f"given {loss_type}, but now only support MSE")
    return loss


def cal_viscond_video_loss(video: torch.Tensor, viscond: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(
        video,
        viscond,
        reduction="mean",
    )
    return loss


def align_latents_viscond_latents(
    latents: torch.Tensor,
    vision_condition_latents: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Args:
        latents (torch.Tensor):  b c t1 h w
        vision_condition_latents (torch.Tensor): b c t2 h w, h2 < h1

    Returns:
        torch.Tensor: b c (t1 t2) h w
        torch.Tensor: b c (t2 t1) h w
    """
    t1 = latents.shape[2]
    t2 = vision_condition_latents.shape[2]
    latents = repeat(latents, "b c t1 h w ->b c (t1 t2) h w", t2=t2)
    vision_condition_latents = repeat(
        vision_condition_latents, "b c t2 h w ->b c (t2 t1) h w", t1=t1
    )
    return latents, vision_condition_latents


def cal_viscond_video_latents_loss(
    latents: torch.Tensor,
    vision_condition_latents: torch.Tensor,
    timesteps_weight: torch.Tensor = None,
) -> torch.Tensor:
    """_summary_

    Args:
        latents (torch.Tensor):  b c t1 h w
        vision_condition_latents (torch.Tensor): b c t2 h w, h2 < h1

    Returns:
        torch.Tensor: _description_
    """
    latents, vision_condition_latents = align_latents_viscond_latents(
        latents, vision_condition_latents
    )
    video_viscond_loss = weighted_mse_loss(
        vision_condition_latents, latents, weight=timesteps_weight
    )
    return video_viscond_loss
