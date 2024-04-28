import multiprocessing as mp
import argparse
import os
from pprint import pprint
import tempfile
from threading import Thread
import sys
import time
from typing import List, Union, Dict
import logging

import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import torch

# logging.basicConfig(level=logging.DEBUG)

from musev.data.image_text_emb_extractor import ImageTextEmbExtractor
from musev.data.data_task import generate_tasks
from mmcm.utils.process_util import run_pipeline


class Worker:
    def __init__(self):
        self.predictor = ImageTextEmbExtractor(
            whether_extract_vae_embs=True,
            whether_extract_text_embs=True,
            # whether_extract_controlnet_embs=False,
            whether_extract_controlnet_embs=True,
            whether_extract_clip_vision_embs=True,
            text_dtype=torch.float16,
            vae_dtype=torch.float16,
            controlnet_dtype=torch.float16,
            clip_vision_dtype=torch.float16,
        )

    def do_task(self, task):
        # perform task with state
        video_path = task["video_path"]
        h5py_path = task["h5py_path"]
        target_width = task["target_width"]
        target_height = task["target_height"]
        prompt = task["prompt"]
        try:
            self.predictor.extract(
                video_path=video_path,
                output_path=h5py_path,
                prompt=prompt,
                target_width=target_width,
                target_height=target_height,
                time_size=10,
                step=10,
                text_key="tag",
                text_index=0,
            )
        except Exception as e:
            print(f"extract error {video_path}")
            logging.exception(e)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-task_path", nargs="+", help="Path to task, file path or directory"
    )
    parser.add_argument("-h5py_dir", type=str, help="Path to h5py_dir for emb")
    parser.add_argument("-video_dir", type=str, help="Path to video directory")
    parser.add_argument("-target_width", type=int, help="image width for video ")
    parser.add_argument("-target_height", type=int, help="image height for video ")
    parser.add_argument(
        "--n_tasks_codetest", type=int, default=None, help="tasks num for codetest"
    )
    parser.add_argument(
        "--n_process", type=int, default=1, help="num process of worker video "
    )
    parser.add_argument(
        "--sep", type=str, default=",", help="if input is csv, sep, default: , "
    )
    parser.add_argument(
        "-source",
        type=str,
        help="source of video ,e.g webvid, ucf101",
        choices=["webvid", "ucf101", "ucb_fashion"],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = get_args()
    pprint(args.__dict__)
    print("generate tasks")
    os.makedirs(args.h5py_dir, exist_ok=True)
    tasks = generate_tasks(
        args.task_path,
        video_dir=args.video_dir,
        h5py_dir=args.h5py_dir,
        source=args.source,
        target_width=args.target_width,
        target_height=args.target_height,
        sep=args.sep,
    )
    if args.n_tasks_codetest is not None:
        tasks = tasks[: args.n_tasks_codetest]
    total_task = len(tasks)
    print("tasks num: ", total_task)
    print("start pipeline")
    time_start = time.time()
    run_pipeline(Worker, tasks, n_process=args.n_process)
    time_end = time.time()
    total_time = time_end - time_start
    avg_time = total_time / total_task
    print(
        f"finish all tasks: cost time total_time={total_time:.3f}, avg_time={avg_time:.3f}"
    )
