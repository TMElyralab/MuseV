from typing import Dict, List, Tuple
import os
import re


import pandas as pd

from .data_util import generate_tasks_of_dir


def split_words(s):
    # 使用正则表达式匹配大写字母，将其前面的部分作为一个单词
    # 例如，"ApplyLipstick" 会被分割为 ["Apply", "Lipstick"]
    if not re.search("[A-Z]", s):
        return [s]
    words = re.findall("[A-Z][^A-Z]*", s)
    # 将所有单词转换为小写
    words = [w.lower() for w in words]
    return words


def category2prompt(category):
    prompt = " ".join(split_words(category))
    return prompt


def generate_ucf101_prompt(task):
    basename_with_dir = task["basename"]
    category = os.path.basename(os.path.dirname(basename_with_dir))
    prompt = category2prompt(category)
    return prompt


def convert_ucf101_task(
    task: Dict,
    video_dir: str,
    h5py_dir: str = None,
    json_dir: str = None,
    **kwargs,
) -> Dict:
    basename_with_dir = task["basename"]
    category = os.path.basename(os.path.dirname(basename_with_dir))
    file_name_with_dir, ext = basename_with_dir.split(".")
    task["category"] = category
    task["video_path"] = os.path.join(video_dir, basename_with_dir)
    task["h5py_dir"] = h5py_dir
    task["json_dir"] = json_dir
    if h5py_dir is not None:
        task["h5py_path"] = os.path.join(h5py_dir, "{}.h5py".format(file_name_with_dir))
    if json_dir is not None:
        task["json_path"] = os.path.join(json_dir, "{}.json".format(file_name_with_dir))
    task["prompt"] = generate_ucf101_prompt(task)
    task.update(kwargs)
    return task


def generate_ucf101_tasks_from_csv(
    path: Tuple[str, List[str]],
    # class_index_path: str,
    video_dir: str,
    h5py_dir: str = None,
    json_dir: str = None,
    sep: str = "\s",
    **kwargs,
) -> List[Dict]:
    # class_index = pd.read_csv(class_index_path)
    if isinstance(path, str):
        path = [path]
    dfs = []
    for p in path:
        df = pd.read_csv(p, sep=sep)
        dfs.append(df)
    dfs = pd.concat(dfs)
    # dfs = pd.merge(dfs, class_index, on="category")
    tasks = dfs.to_dict(orient="records")
    tasks = [
        convert_ucf101_task(
            task, video_dir=video_dir, h5py_dir=h5py_dir, json_dir=json_dir, **kwargs
        )
        for task in tasks
    ]
    return tasks
