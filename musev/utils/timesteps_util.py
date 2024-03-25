from typing import List, Literal
import numpy as np


def generate_parameters_with_timesteps(
    start: int,
    num: int,
    stop: int = None,
    method: Literal["linear", "two_stage", "three_stage", "fix_two_stage"] = "linear",
    n_fix_start: int = 3,
) -> List[float]:
    if stop is None or start == stop:
        params = [start] * num
    else:
        if method == "linear":
            params = generate_linear_parameters(start, stop, num)
        elif method == "two_stage":
            params = generate_two_stages_parameters(start, stop, num)
        elif method == "three_stage":
            params = generate_three_stages_parameters(start, stop, num)
        elif method == "fix_two_stage":
            params = generate_fix_two_stages_parameters(start, stop, num, n_fix_start)
        else:
            raise ValueError(
                f"now only support linear, two_stage, three_stage, but given{method}"
            )
    return params


def generate_linear_parameters(start, stop, num):
    parames = list(
        np.linspace(
            start=start,
            stop=stop,
            num=num,
        )
    )
    return parames


def generate_two_stages_parameters(start, stop, num):
    num_start = num // 2
    num_end = num - num_start
    parames = [start] * num_start + [stop] * num_end
    return parames


def generate_fix_two_stages_parameters(start, stop, num, n_fix_start: int) -> List:
    num_start = n_fix_start
    num_end = num - num_start
    parames = [start] * num_start + [stop] * num_end
    return parames


def generate_three_stages_parameters(start, stop, num):
    middle = (start + stop) // 2
    num_start = num // 3
    num_middle = num_start
    num_end = num - num_start - num_middle
    parames = [start] * num_start + [middle] * num_middle + [stop] * num_end
    return parames
