from typing import List, Optional, Tuple, Union
import torch


from diffusers.utils.torch_utils import randn_tensor


def random_noise(
    tensor: torch.Tensor = None,
    shape: Tuple[int] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    noise_offset: Optional[float] = None,  # typical value is 0.1
) -> torch.Tensor:
    if tensor is not None:
        shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
    if isinstance(device, str):
        device = torch.device(device)
    noise = randn_tensor(shape, dtype=dtype, device=device, generator=generator)
    if noise_offset is not None:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += noise_offset * torch.randn(
            (tensor.shape[0], tensor.shape[1], 1, 1, 1), device
        )
    return noise


def video_fusion_noise(
    tensor: torch.Tensor = None,
    shape: Tuple[int] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
    w_ind_noise: float = 0.5,
    generator: Optional[Union[List[torch.Generator], torch.Generator]] = None,
    initial_common_noise: torch.Tensor = None,
) -> torch.Tensor:
    if tensor is not None:
        shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
    if isinstance(device, str):
        device = torch.device(device)
    batch_size, c, t, h, w = shape
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )
    if not isinstance(generator, list):
        if initial_common_noise is not None:
            common_noise = initial_common_noise.to(device, dtype=dtype)
        else:
            common_noise = randn_tensor(
                (shape[0], shape[1], 1, shape[3], shape[4]),
                generator=generator,
                device=device,
                dtype=dtype,
            )  # common noise
        ind_noise = randn_tensor(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )  # individual noise
        s = torch.tensor(w_ind_noise, device=device, dtype=dtype)
        latents = torch.sqrt(1 - s) * common_noise + torch.sqrt(s) * ind_noise
    else:
        latents = []
        for i in range(batch_size):
            latent = video_fusion_noise(
                shape=(1, c, t, h, w),
                dtype=dtype,
                device=device,
                w_ind_noise=w_ind_noise,
                generator=generator[i],
                initial_common_noise=initial_common_noise,
            )
            latents.append(latent)
        latents = torch.cat(latents, dim=0).to(device)
    return latents
