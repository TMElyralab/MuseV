# TODO: Adapted from cli
import math
from typing import Callable, List, Optional

import numpy as np

from mmcm.utils.itertools_util import generate_sample_idxs

# copy from https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/src/pipelines/context.py


def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


# TODO: closed_loop not work, to fix it
def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


def uniform_v2(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return generate_sample_idxs(
        total=num_frames,
        window_size=context_size,
        step=context_size - context_overlap,
        sample_rate=1,
        drop_last=False,
    )


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform":
        return uniform
    elif name == "uniform_v2":
        return uniform_v2
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )


def drop_last_repeat_context(contexts: List[List[int]]) -> List[List[int]]:
    """if len(contexts)>=2 and the max value the oenultimate list same as  of the last list

    Args:
        List (_type_): _description_

    Returns:
        List[List[int]]: _description_
    """
    if len(contexts) >= 2 and contexts[-1][-1] == contexts[-2][-1]:
        return contexts[:-1]
    else:
        return contexts


def prepare_global_context(
    context_schedule: str,
    num_inference_steps: int,
    time_size: int,
    context_frames: int,
    context_stride: int,
    context_overlap: int,
    context_batch_size: int,
):
    context_scheduler = get_context_scheduler(context_schedule)
    context_queue = list(
        context_scheduler(
            step=0,
            num_steps=num_inference_steps,
            num_frames=time_size,
            context_size=context_frames,
            context_stride=context_stride,
            context_overlap=context_overlap,
        )
    )
    # 如果context_queue的最后一个索引最大值和倒数第二个索引最大值相同，说明最后一个列表就是因为step带来的冗余项，可以去掉
    # remove the last context if max index of the last context is the same as the max index of the second last context
    context_queue = drop_last_repeat_context(context_queue)
    num_context_batches = math.ceil(len(context_queue) / context_batch_size)
    global_context = []
    for i_tmp in range(num_context_batches):
        global_context.append(
            context_queue[i_tmp * context_batch_size : (i_tmp + 1) * context_batch_size]
        )
    return global_context
