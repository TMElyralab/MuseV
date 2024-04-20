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


def convert_ucb_fashion_task(
    task: Dict, video_dir: str, h5py_dir: str = None, json_dir: str = None, **kwargs
) -> Dict:
    url = task["contentUrl"]
    filename, ext = os.path.splitext(os.path.basename(url))
    task["videoid"] = filename
    task["ext"] = ext
    task["prompt"] = None
    task["video_path"] = os.path.join(video_dir, "{}.mp4".format(task["videoid"]))
    task["h5py_dir"] = h5py_dir
    task["json_dir"] = json_dir
    if h5py_dir is not None:
        task["h5py_path"] = os.path.join(h5py_dir, "{}.h5py".format(task["videoid"]))
    if json_dir is not None:
        task["json_path"] = os.path.join(json_dir, "{}.json".format(task["videoid"]))
    task.update(kwargs)
    return task


def generate_ucb_task_from_csv(
    path: Tuple[str, List[str]],
    video_dir: str,
    h5py_dir: str = None,
    json_dir: str = None,
    sep=",",
    shuffle: bool = False,
    **kwargs,
) -> List[Dict]:
    if isinstance(path, str):
        path = [path]
    df = pd.concat(
        [
            pd.read_csv(tmp_path, sep=sep, header=None, names=["contentUrl"])
            for tmp_path in path
        ]
    )
    if shuffle:
        df = df.sample(1.0)
    tasks = df.to_dict(orient="records")
    tasks = [
        convert_ucb_fashion_task(
            task, video_dir=video_dir, h5py_dir=h5py_dir, json_dir=json_dir, **kwargs
        )
        for task in tasks
    ]

    return tasks
