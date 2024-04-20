import os
from typing import Dict, List

from .webvid import (
    generate_webvid_task_from_csv,
)
from .ucf101 import (
    generate_ucf101_tasks_from_csv,
)
from .ucb_fashion import (
    generate_ucb_task_from_csv,
)


def generate_tasks(
    path: str,
    video_dir: str,
    source: str,
    h5py_dir: str = None,
    sep: str = ",",
    **kwargs,
) -> List[Dict]:
    """将输入路径转化成任务字典列表，每个字典有任务运行的所有参数

    Args:
        path (_type_): _description_
        video_dir (_type_): _description_
        h5py_dir (_type_): _description_
        source (_type_, optional): _description_. Defaults to None.
        sep (str, optional): _description_. Defaults to ",".

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        List[Dict]:
    """
    # TODO: 待将表格或文件夹放在内部，支持多输入
    if source == "webvid":
        tasks = generate_webvid_task_from_csv(
            path, video_dir=video_dir, h5py_dir=h5py_dir, **kwargs
        )
    elif source == "ucf101":
        tasks = generate_ucf101_tasks_from_csv(
            path, video_dir=video_dir, h5py_dir=h5py_dir, sep=sep, **kwargs
        )
    elif source == "ucb_fashion":
        tasks = generate_ucb_task_from_csv(
            path, video_dir=video_dir, h5py_dir=h5py_dir, sep=sep, **kwargs
        )
    else:
        raise ValueError("source is not valid")
    return tasks
