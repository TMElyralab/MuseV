from typing import List, Dict, Literal, Union, Tuple
import os
import string
import logging

import torch
import numpy as np
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


def generate_tasks_of_dir(
    path: str,
    output_dir: str,
    exts: Tuple[str],
    same_dir_name: bool = False,
    **kwargs,
) -> List[Dict]:
    """covert video directory into tasks

    Args:
        path (str): _description_
        output_dir (str): _description_
        exts (Tuple[str]): _description_
        same_dir_name (bool, optional): 存储路径是否保留和源视频相同的父文件名. Defaults to False.
            whether keep the same parent dir name as the source video
    Returns:
        List[Dict]: _description_
    """
    tasks = []
    for rootdir, dirs, files in os.walk(path):
        for basename in files:
            if basename.lower().endswith(exts):
                video_path = os.path.join(rootdir, basename)
                filename, ext = basename.split(".")
                rootdir_name = os.path.basename(rootdir)
                if same_dir_name:
                    save_path = os.path.join(
                        output_dir, rootdir_name, f"{filename}.h5py"
                    )
                    save_dir = os.path.join(output_dir, rootdir_name)
                else:
                    save_path = os.path.join(output_dir, f"{filename}.h5py")
                    save_dir = output_dir
                task = {
                    "video_path": video_path,
                    "output_path": save_path,
                    "output_dir": save_dir,
                    "filename": filename,
                    "ext": ext,
                }
                task.update(kwargs)
                tasks.append(task)
    return tasks


def sample_by_idx(
    T: int,
    n_sample: int,
    sample_rate: int,
    sample_start_idx: int = None,
    change_sample_rate: bool = False,
    seed: int = None,
    whether_random: bool = True,
    n_independent: int = 0,
) -> List[int]:
    """given a int to represent candidate list, sample n_sample with sample_rate from the candidate list

    Args:
        T (int): _description_
        n_sample (int): 目标采样数目. sample number
        sample_rate (int): 采样率, 每隔sample_rate个采样一个. sample interval, pick one per sample_rate number
        sample_start_idx (int, optional): 采样开始位置的选择. start position to sample . Defaults to 0.
        change_sample_rate (bool, optional): 是否可以通过降低sample_rate的方式来完成采样. whether allow changing sample_rate to finish sample process. Defaults to False.
        whether_random (bool, optional): 是否最后随机选择开始点. whether randomly choose sample start position. Defaults to False.

    Raises:
        ValueError: T / sample_rate should be larger than n_sample
    Returns:
        List[int]: 采样的索引位置. sampled index position
    """
    if T < n_sample:
        raise ValueError(f"T({T}) < n_sample({n_sample})")
    else:
        if T / sample_rate < n_sample:
            if not change_sample_rate:
                raise ValueError(
                    f"T({T}) / sample_rate({sample_rate}) < n_sample({n_sample})"
                )
            else:
                while T / sample_rate < n_sample:
                    sample_rate -= 1
                    logger.error(
                        f"sample_rate{sample_rate+1} is too large, decrease to {sample_rate}"
                    )
                    if sample_rate == 0:
                        raise ValueError("T / sample_rate < n_sample")

    if sample_start_idx is None:
        if whether_random:
            sample_start_idx_candidates = np.arange(T - n_sample * sample_rate)
            if seed is not None:
                np.random.seed(seed)
            sample_start_idx = np.random.choice(sample_start_idx_candidates, 1)[0]

        else:
            sample_start_idx = 0
    sample_end_idx = sample_start_idx + sample_rate * n_sample
    sample = list(range(sample_start_idx, sample_end_idx, sample_rate))
    if n_independent == 0:
        n_independent_sample = None
    else:
        left_candidate = np.array(
            list(range(0, sample_start_idx)) + list(range(sample_end_idx, T))
        )
        if len(left_candidate) >= n_independent:
            # 使用两端的剩余空间采样, use the left space to sample
            n_independent_sample = np.random.choice(left_candidate, n_independent)
        else:
            # 当两端没有剩余采样空间时，使用任意不是sample中的帧
            # if no enough space to sample, use any frame not in sample
            left_candidate = np.array(list(set(range(T) - set(sample))))
            n_independent_sample = np.random.choice(left_candidate, n_independent)

    return sample, sample_rate, n_independent_sample


def sample_tensor_by_idx(
    tensor: Union[torch.Tensor, np.ndarray],
    n_sample: int,
    sample_rate: int,
    sample_start_idx: int = 0,
    change_sample_rate: bool = False,
    seed: int = None,
    dim: int = 0,
    return_type: Literal["numpy", "torch"] = "torch",
    whether_random: bool = True,
    n_independent: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """sample sub_tensor

    Args:
        tensor (Union[torch.Tensor, np.ndarray]): _description_
        n_sample (int): _description_
        sample_rate (int): _description_
        sample_start_idx (int, optional): _description_. Defaults to 0.
        change_sample_rate (bool, optional): _description_. Defaults to False.
        seed (int, optional): _description_. Defaults to None.
        dim (int, optional): _description_. Defaults to 0.
        return_type (Literal[&quot;numpy&quot;, &quot;torch&quot;], optional): _description_. Defaults to "torch".
        whether_random (bool, optional): _description_. Defaults to True.
        n_independent (int, optional): 独立于n_sample的采样数量. Defaults to 0.
            n_independent sample number that is independent of n_sample

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]: sampled tensor
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    T = tensor.shape[dim]
    sample_idx, sample_rate, independent_sample_idx = sample_by_idx(
        T,
        n_sample,
        sample_rate,
        sample_start_idx,
        change_sample_rate,
        seed,
        whether_random=whether_random,
        n_independent=n_independent,
    )
    sample_idx = torch.LongTensor(sample_idx)
    sample = torch.index_select(tensor, dim, sample_idx)
    if independent_sample_idx is not None:
        independent_sample_idx = torch.LongTensor(independent_sample_idx)
        independent_sample = torch.index_select(tensor, dim, independent_sample_idx)
    else:
        independent_sample = None
        independent_sample_idx = None
    if return_type == "numpy":
        sample = sample.cpu().numpy()
    return sample, sample_idx, sample_rate, independent_sample, independent_sample_idx


def concat_two_tensor(
    data1: torch.Tensor,
    data2: torch.Tensor,
    dim: int,
    method: Literal[
        "first_in_first_out", "first_in_last_out", "intertwine", "index"
    ] = "first_in_first_out",
    data1_index: torch.long = None,
    data2_index: torch.long = None,
    return_index: bool = False,
):
    """concat two tensor along dim with given method

    Args:
        data1 (torch.Tensor): first in data
        data2 (torch.Tensor): last in  data
        dim (int): _description_
        method (Literal[ &quot;first_in_first_out&quot;, &quot;first_in_last_out&quot;, &quot;intertwine&quot; ], optional): _description_. Defaults to "first_in_first_out".

    Raises:
        NotImplementedError: unsupported method
        ValueError: unsupported method

    Returns:
        _type_: _description_
    """
    len_data1 = data1.shape[dim]
    len_data2 = data2.shape[dim]

    if method == "first_in_first_out":
        res = torch.concat([data1, data2], dim=dim)
        data1_index = range(len_data1)
        data2_index = [len_data1 + x for x in range(len_data2)]
    elif method == "first_in_last_out":
        res = torch.concat([data2, data1], dim=dim)
        data2_index = range(len_data2)
        data1_index = [len_data2 + x for x in range(len_data1)]
    elif method == "intertwine":
        raise NotImplementedError("intertwine")
    elif method == "index":
        res = concat_two_tensor_with_index(
            data1=data1,
            data1_index=data1_index,
            data2=data2,
            data2_index=data2_index,
            dim=dim,
        )
    else:
        raise ValueError(
            "only support first_in_first_out, first_in_last_out, intertwine, index"
        )
    if return_index:
        return res, data1_index, data2_index
    else:
        return res


def concat_two_tensor_with_index(
    data1: torch.Tensor,
    data1_index: torch.LongTensor,
    data2: torch.Tensor,
    data2_index: torch.LongTensor,
    dim: int,
) -> torch.Tensor:
    """_summary_

    Args:
        data1 (torch.Tensor): b1*c1*h1*w1*...
        data1_index (torch.LongTensor): N, if dim=1, N=c1
        data2 (torch.Tensor): b2*c2*h2*w2*...
        data2_index (torch.LongTensor): M, if dim=1, M=c2
        dim (int): int

    Returns:
        torch.Tensor: b*c*h*w*..., if dim=1, b=b1=b2, c=c1+c2, h=h1=h2, w=w1=w2,...
    """
    shape1 = list(data1.shape)
    shape2 = list(data2.shape)
    target_shape = list(shape1)
    target_shape[dim] = shape1[dim] + shape2[dim]
    target = torch.zeros(target_shape, device=data1.device, dtype=data1.dtype)
    target = batch_index_copy(target, dim=dim, index=data1_index, source=data1)
    target = batch_index_copy(target, dim=dim, index=data2_index, source=data2)
    return target


def repeat_index_to_target_size(
    index: torch.LongTensor, target_size: int
) -> torch.LongTensor:
    if len(index.shape) == 1:
        index = repeat(index, "n -> b n", b=target_size)
    if len(index.shape) == 2:
        remainder = target_size % index.shape[0]
        assert (
            remainder == 0
        ), f"target_size % index.shape[0] must be zero, but give {target_size % index.shape[0]}"
        index = repeat(index, "b n -> (b c) n", c=int(target_size / index.shape[0]))
    return index


def batch_concat_two_tensor_with_index(
    data1: torch.Tensor,
    data1_index: torch.LongTensor,
    data2: torch.Tensor,
    data2_index: torch.LongTensor,
    dim: int,
) -> torch.Tensor:
    return concat_two_tensor_with_index(data1, data1_index, data2, data2_index, dim)


def interwine_two_tensor(
    data1: torch.Tensor,
    data2: torch.Tensor,
    dim: int,
    return_index: bool = False,
) -> torch.Tensor:
    shape1 = list(data1.shape)
    shape2 = list(data2.shape)
    target_shape = list(shape1)
    target_shape[dim] = shape1[dim] + shape2[dim]
    target = torch.zeros(target_shape, device=data1.device, dtype=data1.dtype)
    data1_reshape = torch.swapaxes(data1, 0, dim)
    data2_reshape = torch.swapaxes(data2, 0, dim)
    target = torch.swapaxes(target, 0, dim)
    total_index = set(range(target_shape[dim]))
    data1_index = range(0, 2 * shape1[dim], 2)
    data2_index = sorted(list(set(total_index) - set(data1_index)))
    data1_index = torch.LongTensor(data1_index)
    data2_index = torch.LongTensor(data2_index)
    target[data1_index, ...] = data1_reshape
    target[data2_index, ...] = data2_reshape
    target = torch.swapaxes(target, 0, dim)
    if return_index:
        return target, data1_index, data2_index
    else:
        return target


def split_index(
    indexs: torch.Tensor,
    n_first: int = None,
    n_last: int = None,
    method: Literal[
        "first_in_first_out", "first_in_last_out", "intertwine", "index", "random"
    ] = "first_in_first_out",
):
    """_summary_

    Args:
        indexs (List): _description_
        n_first (int): _description_
        n_last (int): _description_
        method (Literal[ &quot;first_in_first_out&quot;, &quot;first_in_last_out&quot;, &quot;intertwine&quot;, &quot;index&quot; ], optional): _description_. Defaults to "first_in_first_out".

    Raises:
        NotImplementedError: _description_

    Returns:
        first_index: _description_
        last_index:
    """
    # assert (
    #     n_first is None and n_last is None
    # ), "must assign one value for n_first or n_last"
    n_total = len(indexs)
    if n_first is None:
        n_first = n_total - n_last
    if n_last is None:
        n_last = n_total - n_first
    assert len(indexs) == n_first + n_last
    if method == "first_in_first_out":
        first_index = indexs[:n_first]
        last_index = indexs[n_first:]
    elif method == "first_in_last_out":
        first_index = indexs[n_last:]
        last_index = indexs[:n_last]
    elif method == "intertwine":
        raise NotImplementedError
    elif method == "random":
        idx_ = torch.randperm(len(indexs))
        first_index = indexs[idx_[:n_first]]
        last_index = indexs[idx_[n_first:]]
    return first_index, last_index


def split_tensor(
    tensor: torch.Tensor,
    dim: int,
    n_first=None,
    n_last=None,
    method: Literal[
        "first_in_first_out", "first_in_last_out", "intertwine", "index", "random"
    ] = "first_in_first_out",
    need_return_index: bool = False,
):
    device = tensor.device
    total = tensor.shape[dim]
    if n_first is None:
        n_first = total - n_last
    if n_last is None:
        n_last = total - n_first
    indexs = torch.arange(
        total,
        dtype=torch.long,
        device=device,
    )
    (
        first_index,
        last_index,
    ) = split_index(
        indexs=indexs,
        n_first=n_first,
        method=method,
    )
    first_tensor = torch.index_select(tensor, dim=dim, index=first_index)
    last_tensor = torch.index_select(tensor, dim=dim, index=last_index)
    if need_return_index:
        return (
            first_tensor,
            last_tensor,
            first_index,
            last_index,
        )
    else:
        return (first_tensor, last_tensor)


# TODO: 待确定batch_index_select的优化
def batch_index_select(
    tensor: torch.Tensor, index: torch.LongTensor, dim: int
) -> torch.Tensor:
    """_summary_

    Args:
        tensor (torch.Tensor): D1*D2*D3*D4...
        index (torch.LongTensor): D1*N or N, N<= tensor.shape[dim]
        dim (int): dim to select

    Returns:
        torch.Tensor: D1*...*N*...
    """
    # TODO: now only support N same for every d1
    if len(index.shape) == 1:
        return torch.index_select(tensor, dim=dim, index=index)
    else:
        index = repeat_index_to_target_size(index, tensor.shape[0])
        out = []
        for i in torch.arange(tensor.shape[0]):
            sub_tensor = tensor[i]
            sub_index = index[i]
            d = torch.index_select(sub_tensor, dim=dim - 1, index=sub_index)
            out.append(d)
        return torch.stack(out).to(dtype=tensor.dtype)


def batch_index_copy(
    tensor: torch.Tensor, dim: int, index: torch.LongTensor, source: torch.Tensor
) -> torch.Tensor:
    """_summary_

    Args:
        tensor (torch.Tensor): b*c*h
        dim (int):
        index (torch.LongTensor): b*d,
        source (torch.Tensor):
            b*d*h*..., if dim=1
            b*c*d*..., if dim=2

    Returns:
        torch.Tensor: b*c*d*...
    """
    if len(index.shape) == 1:
        tensor.index_copy_(dim=dim, index=index, source=source)
    else:
        index = repeat_index_to_target_size(index, tensor.shape[0])

        batch_size = tensor.shape[0]
        for b in torch.arange(batch_size):
            sub_index = index[b]
            sub_source = source[b]
            sub_tensor = tensor[b]
            sub_tensor.index_copy_(dim=dim - 1, index=sub_index, source=sub_source)
            tensor[b] = sub_tensor
    return tensor


def batch_index_fill(
    tensor: torch.Tensor,
    dim: int,
    index: torch.LongTensor,
    value: Literal[torch.Tensor, torch.float],
) -> torch.Tensor:
    """_summary_

    Args:
        tensor (torch.Tensor): b*c*h
        dim (int):
        index (torch.LongTensor): b*d,
        value (torch.Tensor): b

    Returns:
        torch.Tensor: b*c*d*...
    """
    index = repeat_index_to_target_size(index, tensor.shape[0])
    batch_size = tensor.shape[0]
    for b in torch.arange(batch_size):
        sub_index = index[b]
        sub_value = value[b] if isinstance(value, torch.Tensor) else value
        sub_tensor = tensor[b]
        sub_tensor.index_fill_(dim - 1, sub_index, sub_value)
        tensor[b] = sub_tensor
    return tensor


def adaptive_instance_normalization(
    src: torch.Tensor,
    dst: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Args:
        src (torch.Tensor): b c t h w
        dst (torch.Tensor): b c t h w
    """
    ndim = src.ndim
    if ndim == 5:
        dim = (2, 3, 4)
    elif ndim == 4:
        dim = (2, 3)
    elif ndim == 3:
        dim = 2
    else:
        raise ValueError("only support ndim in [3,4,5], but given {ndim}")
    var, mean = torch.var_mean(src, dim=dim, keepdim=True, correction=0)
    std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
    dst = align_repeat_tensor_single_dim(dst, src.shape[0], dim=0)
    mean_acc, var_acc = torch.var_mean(dst, dim=dim, keepdim=True, correction=0)
    # mean_acc = sum(mean_acc) / float(len(mean_acc))
    # var_acc = sum(var_acc) / float(len(var_acc))
    std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
    src = (((src - mean) / std) * std_acc) + mean_acc
    return src


def adaptive_instance_normalization_with_ref(
    src: torch.LongTensor,
    dst: torch.LongTensor,
    style_fidelity: float = 0.5,
    do_classifier_free_guidance: bool = True,
):
    # logger.debug(
    #     f"src={src.shape}, min={src.min()}, max={src.max()}, mean={src.mean()}, \n"
    #     f"dst={src.shape}, min={dst.min()}, max={dst.max()}, mean={dst.mean()}"
    # )
    batch_size = src.shape[0] // 2
    uc_mask = torch.Tensor([1] * batch_size + [0] * batch_size).type_as(src).bool()
    src_uc = adaptive_instance_normalization(src, dst)
    src_c = src_uc.clone()
    # TODO: 该部分默认 do_classifier_free_guidance and style_fidelity > 0 = True
    if do_classifier_free_guidance and style_fidelity > 0:
        src_c[uc_mask] = src[uc_mask]
    src = style_fidelity * src_c + (1.0 - style_fidelity) * src_uc
    return src


def batch_adain_conditioned_tensor(
    tensor: torch.Tensor,
    src_index: torch.LongTensor,
    dst_index: torch.LongTensor,
    keep_dim: bool = True,
    num_frames: int = None,
    dim: int = 2,
    style_fidelity: float = 0.5,
    do_classifier_free_guidance: bool = True,
    need_style_fidelity: bool = False,
):
    """_summary_

    Args:
        tensor (torch.Tensor): b c t h w
        src_index (torch.LongTensor): _description_
        dst_index (torch.LongTensor): _description_
        keep_dim (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    ndim = tensor.ndim
    dtype = tensor.dtype
    if ndim == 4 and num_frames is not None:
        tensor = rearrange(tensor, "(b t) c h w->  b c t h w ", t=num_frames)
    src = batch_index_select(tensor, dim=dim, index=src_index).contiguous()
    dst = batch_index_select(tensor, dim=dim, index=dst_index).contiguous()
    if need_style_fidelity:
        src = adaptive_instance_normalization_with_ref(
            src=src,
            dst=dst,
            style_fidelity=style_fidelity,
            do_classifier_free_guidance=do_classifier_free_guidance,
            need_style_fidelity=need_style_fidelity,
        )
    else:
        src = adaptive_instance_normalization(
            src=src,
            dst=dst,
        )
    if keep_dim:
        src = batch_concat_two_tensor_with_index(
            src.to(dtype=dtype),
            src_index,
            dst.to(dtype=dtype),
            dst_index,
            dim=dim,
        )

    if ndim == 4 and num_frames is not None:
        src = rearrange(tensor, "b c t h w ->(b t) c h w")
    return src


def align_repeat_tensor_single_dim(
    src: torch.Tensor,
    target_length: int,
    dim: int = 0,
    n_src_base_length: int = 1,
    src_base_index: List[int] = None,
) -> torch.Tensor:
    """沿着 dim 纬度， 补齐 src 的长度到目标 target_length。
    当 src 长度不如 target_length 时， 取其中 前 n_src_base_length 然后 repeat 到 target_length

    align length of src to target_length along dim
    when src length is less than target_length, take the first n_src_base_length and repeat to target_length

    Args:
        src (torch.Tensor): 输入 tensor, input tensor
        target_length (int): 目标长度, target_length
        dim (int, optional): 处理纬度, target dim . Defaults to 0.
        n_src_base_length (int, optional): src 的基本单元长度, basic length of src. Defaults to 1.

    Returns:
        torch.Tensor: _description_
    """
    src_dim_length = src.shape[dim]
    if target_length > src_dim_length:
        if target_length % src_dim_length == 0:
            new = src.repeat_interleave(
                repeats=target_length // src_dim_length, dim=dim
            )
        else:
            if src_base_index is None and n_src_base_length is not None:
                src_base_index = torch.arange(n_src_base_length)

            new = src.index_select(
                dim=dim,
                index=torch.LongTensor(src_base_index).to(device=src.device),
            )
            new = new.repeat_interleave(
                repeats=target_length // len(src_base_index),
                dim=dim,
            )
    elif target_length < src_dim_length:
        new = src.index_select(
            dim=dim,
            index=torch.LongTensor(torch.arange(target_length)).to(device=src.device),
        )
    else:
        new = src
    return new


def fuse_part_tensor(
    src: torch.Tensor,
    dst: torch.Tensor,
    overlap: int,
    weight: float = 0.5,
    skip_step: int = 0,
) -> torch.Tensor:
    """fuse overstep tensor with weight of src into dst
    out = src_fused_part * weight + dst * (1-weight) for overlap

    Args:
        src (torch.Tensor): b c t h w
        dst (torch.Tensor): b c t h w
        overlap (int): 1
        weight (float, optional): weight of src tensor part. Defaults to 0.5.

    Returns:
        torch.Tensor: fused tensor
    """
    if overlap == 0:
        return dst
    else:
        dst[:, :, skip_step : skip_step + overlap] = (
            weight * src[:, :, -overlap:]
            + (1 - weight) * dst[:, :, skip_step : skip_step + overlap]
        )
        return dst
