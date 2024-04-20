import os
import re
from typing import Union, List, Tuple, Dict

import torch
import numpy as np
import pandas as pd
import h5py
import diffusers
from einops import rearrange
from PIL import Image


from .data_util import generate_tasks_of_dir
from .image_text_emb_extractor import ImageTextEmbExtractor


def convert_webvid_timestr(time_str):
    if time_str is None:
        return None
    pattern = r"PT(\d{2})H(\d{2})M(\d{2})S"
    match = re.match(pattern, time_str)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    else:
        return None


def convert_webvid_task(
    task: Dict, video_dir: str, h5py_dir: str = None, json_dir: str = None, **kwargs
) -> Dict:
    page_dir = convert_webvid_page_dir(task["page_dir"])
    task["video_path"] = os.path.join(
        video_dir, page_dir, "{}.mp4".format(task["videoid"])
    )
    if h5py_dir is not None:
        task["h5py_dir"] = os.path.join(h5py_dir, page_dir)
        task["h5py_path"] = os.path.join(
            task["h5py_dir"], "{}.h5py".format(task["videoid"])
        )
    if json_dir is not None:
        task["json_dir"] = os.path.join(json_dir, page_dir)
        task["json_path"] = os.path.join(
            task["json_dir"], "{}.json".format(task["videoid"])
        )
    task["data_type"] = "video"
    task["prompt"] = task["name"]
    task["duration_int"] = convert_webvid_timestr(task["duration"])
    if "action" not in task:
        task["action"] = "unknown"
    task.update(kwargs)
    return task


def generate_webvid_task_from_csv(
    path: Tuple[str, List[str]],
    video_dir: str,
    h5py_dir: str = None,  # 用于存储emb
    json_dir: str = None,  # 用于存储统计信息
    sep=",",
    drop_duplicates: bool = True,
    shuffle: bool = False,
    **kwargs,
) -> List[Dict]:
    if isinstance(path, str):
        path = [path]
    df = pd.concat(
        [pd.read_csv(tmp_path, sep=sep, dtype={"videoid": str}) for tmp_path in path]
    )
    if drop_duplicates:
        df = df.drop_duplicates("videoid")
    if shuffle:
        df = df.sample(1.0)
    tasks = df.to_dict(orient="records")
    tasks = [
        convert_webvid_task(
            task, video_dir=video_dir, h5py_dir=h5py_dir, json_dir=json_dir, **kwargs
        )
        for task in tasks
    ]

    return tasks


def generate_webvid_tasks_from_dir(
    path: str,
    output_dir: str = None,
    exts: Tuple[str] = ("mp4", "avi", "flv", "mpeg", "mkv"),
    same_dir_name: bool = True,
    **kwargs,
) -> List[Dict]:
    return generate_tasks_of_dir(
        path=path,
        output_dir=output_dir,
        exts=exts,
        same_dir_name=same_dir_name,
        **kwargs,
    )


def convert_webvid_page_dir(page_dir):
    if pd.isna(page_dir):
        page_dir = "pagedir_none"
    return page_dir
