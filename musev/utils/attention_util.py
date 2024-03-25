from typing import Tuple, Union, Literal

from einops import repeat
import torch
import numpy as np


def get_diags_indices(
    shape: Union[int, Tuple[int, int]], k_min: int = 0, k_max: int = 0
):
    if isinstance(shape, int):
        shape = (shape, shape)
    rows, cols = np.indices(shape)
    diag = cols - rows
    return np.where((diag >= k_min) & (diag <= k_max))


def generate_mask_from_indices(
    shape: Tuple[int, int],
    indices: Tuple[np.ndarray, np.ndarray],
    big_value: float = 0,
    small_value: float = -1e9,
):
    matrix = np.ones(shape) * small_value
    matrix[indices] = big_value
    return matrix


def generate_sparse_causcal_attn_mask(
    batch_size: int,
    n: int,
    n_near: int = 1,
    big_value: float = 0,
    small_value: float = -1e9,
    out_type: Literal["torch", "numpy"] = "numpy",
    expand: int = 1,
) -> np.ndarray:
    """generate b (n expand) (n expand) mask，
        where value of diag (0<=<=n_near) and first column  of shape mat (n n) is set as big_value, others as small value
        expand的概念：
            attn 是 b n d 时，mask 是 b n n, 当 attn 是 b (expand n) d 时， mask 是 b (n expand) (n expand)
    Args:
        batch_size (int): _description_
        n (int): _description_
        n_near (int, optional): _description_. Defaults to 1.
        big_value (float, optional): _description_. Defaults to 0.
        small_value (float, optional): _description_. Defaults to -1e9.
        out_type (Literal[&quot;torch&quot;, &quot;numpy&quot;], optional): _description_. Defaults to "numpy".
        expand (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    shape = (n, n)
    diag_indices = get_diags_indices(n, k_min=-n_near, k_max=0)
    first_column = (np.arange(n), np.zeros(n).astype(np.int))
    indices = (
        np.concatenate([diag_indices[0], first_column[0]]),
        np.concatenate([diag_indices[1], first_column[1]]),
    )
    mask = generate_mask_from_indices(
        shape=shape, indices=indices, big_value=big_value, small_value=small_value
    )
    mask = repeat(mask, "m n-> b m n", b=batch_size)
    if expand > 1:
        mask = repeat(
            mask,
            "b m n -> b (m d1) (n d2)",
            d1=expand,
            d2=expand,
        )
    if out_type == "torch":
        mask = torch.from_numpy(mask)
    return mask
