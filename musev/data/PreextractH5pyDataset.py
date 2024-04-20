import random
from typing import Callable, List, Literal, Tuple, Union
import logging
from pprint import pprint, pformat

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import h5py
from diffusers.models.vae import DiagonalGaussianDistribution
from einops import rearrange, repeat
import numpy as np

from ..utils.emb_util import concat_two_text_embedding
from .data_util import align_repeat_tensor_single_dim, sample_tensor_by_idx, split_index

# for draw pose
from diffusers.image_processor import VaeImageProcessor
from controlnet_aux.dwpose import draw_pose, candidate2pose, pose2map


logger = logging.getLogger(__name__)


def mv_tail_latent2viscond(
    latents: torch.Tensor,
    vision_condition_latents: torch.Tensor,
    latent_index: torch.Tensor,
    vision_condition_latent_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 latent 的尾帧移到 vision_condition_latents，同时更新对应的 index

    Args:
        latents (torch.Tensor): c t1 h w
        vision_condition_latents (torch.Tensor):  c t2 h w
        latent_index (torch.Tensor): t1
        vision_condition_latent_index (torch.Tensor): t2

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]: 更新后
    """
    tail_index = latent_index[-1:]
    tail = latents[:, -1:, :, :]
    if vision_condition_latent_index is None:
        vision_condition_latent_index = tail_index
    else:
        vision_condition_latent_index = torch.concat(
            [vision_condition_latent_index, tail_index], dim=0
        )
    if vision_condition_latents is None:
        vision_condition_latents = tail
    else:
        vision_condition_latents = torch.concat([vision_condition_latents, tail], dim=1)
    latent_index = latent_index[:-1]
    latents = latents[:, :-1, :, :]
    return (
        latents,
        vision_condition_latents,
        latent_index,
        vision_condition_latent_index,
    )


def mv_batch_tail_latent2viscond(
    latents: torch.Tensor,
    vision_condition_latents: torch.Tensor,
    latent_index: torch.Tensor,
    vision_condition_latent_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 latent 的尾帧移到 vision_condition_latents，同时更新对应的 index

    Args:
        latents (torch.Tensor): b c t1 h w
        vision_condition_latents (torch.Tensor): b c t2 h w
        latent_index (torch.Tensor): b t1
        vision_condition_latent_index (torch.Tensor): b t2

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]: 更新后
    """
    tail_index = latent_index[:, -1:]
    tail = latents[:, :, -1:, :, :]
    if vision_condition_latent_index is None:
        vision_condition_latent_index = tail_index
    else:
        vision_condition_latent_index = torch.concat(
            [vision_condition_latent_index, tail_index], dim=1
        )
    if vision_condition_latents is None:
        vision_condition_latents = tail
    else:
        vision_condition_latents = torch.concat([vision_condition_latents, tail], dim=2)
    latent_index = latent_index[:, :-1]
    latents = latents[:, :, :-1, :, :]
    return (
        latents,
        vision_condition_latents,
        latent_index,
        vision_condition_latent_index,
    )


def mv_batch_head_latent2viscond(
    latents: torch.Tensor,
    vision_condition_latents: torch.Tensor,
    latent_index: torch.Tensor,
    vision_condition_latent_index: torch.Tensor,
    n_vision_condition: int = 0,
    max_n_vision_condition: int = 0,
    sample_n_viscond_method: Literal["union", "focus_1st"] = "union",
    first_element_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 latent 的尾帧移到 vision_condition_latents，同时更新对应的 index

    Args:
        latents (torch.Tensor): b c t1 h w
        vision_condition_latents (torch.Tensor): b c t2 h w
        latent_index (torch.Tensor): b t1
        vision_condition_latent_index (torch.Tensor):b t2

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]: 更新后
    """
    if (
        sample_n_viscond_method == "union"
        or n_vision_condition == max_n_vision_condition
    ):
        n_vision_cond = torch.randint(
            low=n_vision_condition,
            high=max_n_vision_condition + 1,
            size=(1,),
        ).tolist()[0]
    elif sample_n_viscond_method == "focus_1st":
        # 采样概率为
        # n = first_element, p=first_element_prob
        # n != first_element, p = (1-first_element_prob)/(n_elements-1)
        elements = range(n_vision_condition, max_n_vision_condition + 1)
        n_elements = len(elements)
        elements_prob = [(1 - first_element_prob) / (n_elements - 1)] * n_elements
        elements_prob[0] = first_element_prob
        n_vision_cond = np.random.choice(elements, 1, p=elements_prob)[0]
    else:
        raise ValueError(
            f"sample_n_viscond_method only support union and focus_1st, but given {sample_n_viscond_method}"
        )
    if n_vision_cond > 0:
        mv_latent_index = latent_index[:, :n_vision_cond]
        if vision_condition_latent_index is None:
            vision_condition_latent_index = mv_latent_index
        else:
            vision_condition_latent_index = torch.concat(
                [vision_condition_latent_index, mv_latent_index], dim=1
            )
        latent_index = latent_index[:, n_vision_cond:]
        mv_latents = latents[:, :, :n_vision_cond, :, :]
        if vision_condition_latents is None:
            vision_condition_latents = mv_latents
        else:
            vision_condition_latents = torch.concat(
                [vision_condition_latents, mv_latents],
                dim=2,
            )
        latents = latents[:, :, n_vision_cond:, :, :]
    return (
        latents,
        vision_condition_latents,
        latent_index,
        vision_condition_latent_index,
    )


# self.vae_scale_factor = 2 ** (vae_config_block_out_channels - 1)
vae_scale_factor = 2 ** (4 - 1)
control_image_processor = VaeImageProcessor(
    vae_scale_factor=vae_scale_factor,
    do_convert_rgb=True,
    do_normalize=False,
)


def prepare_image(
    image,  # b c t h w
    width,
    height,
):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim == 5:
        image = rearrange(image, "b c t h w-> (b t) c h w")
    if height is None:
        height = image.shape[-2]
    if width is None:
        width = image.shape[-1]
    width, height = (
        x - x % control_image_processor.vae_scale_factor for x in (width, height)
    )
    image = image / 255.0
    # image = torch.nn.functional.interpolate(image, size=(height, width))
    do_normalize = control_image_processor.config.do_normalize
    if image.min() < 0:
        import warnings

        warnings.warn(
            "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
            f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
            FutureWarning,
        )
        do_normalize = False
    if do_normalize:
        image = control_image_processor.normalize(image)
    return image


class PreextractH5pyDataset(Dataset):
    print_idx = 0

    def __init__(
        self,
        csv_path: str,
        feat_key: str,
        prompt_emb_key: str,
        prompt_text_key: str = None,
        n_sample_frames: int = 8,
        sample_start_idx: Union[int, None] = None,
        sample_frame_rate: int = 1,
        prob_uncond: float = 0.0,
        uncond_embeddings=None,
        sep: str = ",",
        text_idx: int = 0,
        h5py_key: str = "emb_path",
        n_vision_condition: int = 0,
        max_n_vision_condition: int = None,
        sample_n_viscond_method: Literal["union", "focus_1st"] = "union",
        first_element_prob: float = 0.5,
        condition_sample_method: Literal[
            "first_in_first_out", "first_in_last_out", "intertwine", "index", "random"
        ] = "first_in_last_out",
        shuffle: bool = True,
        use_text: bool = False,
        suffix_text: str = "",
        prefix_text: str = "",
        contronet_emb_keys: Union[str, List[str]] = None,
        prob_static_video: float = 0,
        static_video_prompt: str = None,
        dynamic_video_prompt: str = None,
        use_dynamic_sr: bool = False,
        max_sample_frame_rate: int = 1,
        video2emb_sr: int = 2,
        tail_frame_prob: float = 0,
        vision_clip_emb_key: str = None,
        n_refer_image: int = 0,
        dwposekps2poseimg_converter: Callable = None,
    ):
        """_summary_

        Args:
            csv_path (str): 训练列表
            feat_key (str): h5py 中 vae emb 的 key
            prompt_emb_key (str): h5py中prompt emb 的key
            prompt_text_key (str, optional): h5py中text的key. Defaults to None.
            n_sample_frames (int, optional): 用于训练的视频帧数. Defaults to 8.
            sample_start_idx (int, optional): 采样的起始位置. Defaults to 0.
            sample_frame_rate (int, optional): 从emb帧数到训练数据的默认采样率. Defaults to 1.
            prob_uncond (float, optional): uncond text 的采样概率. Defaults to 0.0.
            uncond_embeddings (_type_, optional): uncond text emb. Defaults to None.
            sep (str, optional): csv_path 的分隔符. Defaults to ",".
            text_idx (int, optional): h5py 中的 prompt_text 是个列表，这里是索引. Defaults to 0.
            h5py_key (str, optional): csv_path中的h5py key. Defaults to "emb_path".
            n_vision_condition (int, optional): 采样的视觉条件帧数量. Defaults to 0.
            max_n_vision_condition (int, optional): 采样的视觉条件帧最大数量，当大于 n_vision_condition，表示是动态采样，需要配合sampler_util.NumVisCondBatchSampler使用. Defaults to None.
            first_element_prob (float, optional): 多条件帧时首帧采样概率. Defaults to 0.0.
            sample_n_viscond_method (Literal[ &quot;union&quot;, &quot;focus_1st&quot; ], optional): 多条件帧采样的方法. Defaults to "union".
                union：表示 1 - max_n_vision_condition 均匀采样
                focus_1st： 表示概率为 first_element_prob 的重点采样1，其他平分剩余概率。
            shuffle (bool, optional): 是否打乱 csv_path表格. Defaults to True.
            use_text (bool, optional): 是否使用prefix_text和suffix_text修改prompt，表示会重新计算prompt_emb，需要在训练流程中配合使用. Defaults to False.
            suffix_text (str, optional): 加入到所有prompt的后缀. Defaults to "".
            prefix_text (str, optional): 加入到所有prompt的前缀. Defaults to "".
            contronet_emb_keys (Union[str, List[str]], optional): contronet emb 对应的 key. Defaults to None.
            prob_static_video (float, optional): 是否需要加静态视频关键词，也会重新计算 prompt_emb. Defaults to 0.
            static_video_prompt (str, optional): 静态视频采样对应的 静态视频描述字符串 . Defaults to None.
            dynamic_video_prompt (str, optional): 动态视频描述字符串, 暂时用不上 . Defaults to None.
            use_dynamic_sr (bool, optional): 是否动态采样，如果True，一个batch里的采样率就会不同，就需要使用定制的batch_sampler. Defaults to False.
            max_sample_frame_rate (int, optional): 从emb帧数到训练数据的最大采样率. Defaults to 1.
            video2emb_sr (int, optional): 从视频中到emb的采样率，目前是2帧踩一帧. Defaults to 2.
            tail_frame_prob (float, optional): 使用尾帧的概率，如果小于0，则表示不用. Defaults to 0.
            vision_clip_emb_key: str = None: 视频的 clip_vision特征
            n_refer_image: int = 0：用于referencenet的图像，
        """
        self.max_sample_frame_rate = max(max_sample_frame_rate, sample_frame_rate)
        df = pd.read_csv(csv_path, sep=sep, index_col=False)
        if shuffle:
            df = df.sample(frac=1.0)
        self.data = df.to_dict(orient="records")
        if "frames_num" in self.data[0]:
            logger.info("\n")
            logger.info(
                f"data_length before filter by frames_num {len(self.data)} ={n_sample_frames * self.max_sample_frame_rate}"
            )
            self.data = [
                d
                for d in self.data
                if d["frames_num"] > n_sample_frames * self.max_sample_frame_rate
            ]
            logger.info(f"data_length after filter by frames_num {len(self.data)}")
            logger.info("\n")

        self.feat_key = feat_key
        self.prompt_emb_key = prompt_emb_key
        self.prompt_text_key = prompt_text_key
        self.use_text = use_text
        self.suffix_text = suffix_text
        self.prefix_text = prefix_text
        self.n_sample_frames = n_sample_frames
        self.max_n_vision_condition = (
            max_n_vision_condition
            if max_n_vision_condition is not None
            else n_vision_condition
        )
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.uncond_embeddings = (
            repeat(uncond_embeddings, "n q -> t n q", t=n_sample_frames)
            if uncond_embeddings is not None
            else uncond_embeddings
        )
        self.prob_uncond = prob_uncond
        self.sep = sep
        self.text_idx = text_idx
        self.h5py_key = h5py_key
        self.n_vision_condition = n_vision_condition
        self.condition_sample_method = condition_sample_method

        self.contronet_emb_keys = contronet_emb_keys
        self.prob_static_video = prob_static_video
        self.static_video_prompt = static_video_prompt
        self.dynamic_video_prompt = dynamic_video_prompt
        self.use_dynamic_sr = use_dynamic_sr
        self.video2emb_sr = video2emb_sr
        self.tail_frame_prob = tail_frame_prob
        self.sample_n_viscond_method = sample_n_viscond_method
        self.first_element_prob = first_element_prob
        self.vision_clip_emb_key = vision_clip_emb_key
        self.n_refer_image = n_refer_image
        self.dwposekps2poseimg_converter = dwposekps2poseimg_converter
        if n_vision_condition > 0 or tail_frame_prob > 0:
            self.prepare_init_datas()
        logger.debug("PreextractH5pyDataset")
        logger.debug(
            pformat(
                {
                    k: v
                    for (k, v) in self.__dict__.items()
                    if k
                    not in [
                        "data",
                        "uncond_embeddings",
                        "n_vision_cond_lst",
                        "mv_tail_latent2viscond_flag",
                    ]
                }
            )
        )

    def prepare_init_datas(self):
        if self.n_vision_condition > 0:
            self.generate_n_vision_cond_lst()
        if self.tail_frame_prob > 0:
            self.generate_tail_prob()

    def generate_n_vision_cond_lst(self):
        """
        当返回数据所需要的n_vision_cond 是动态变化时，所需要的值由该提前生成。
        每个epoch会需要调用一次
        """
        target_length = len(self.data)
        if (
            self.sample_n_viscond_method == "union"
            or self.n_vision_condition == self.max_n_vision_condition
        ):
            self.n_vision_cond_lst = torch.randint(
                low=self.n_vision_condition,
                high=self.max_n_vision_condition + 1,
                size=(target_length,),
            ).tolist()
        elif self.sample_n_viscond_method == "focus_1st":
            # 采样概率为
            # n = first_element, p=first_element_prob
            # n != first_element, p = (1-first_element_prob)/(n_elements-1)
            elements = range(self.n_vision_condition, self.max_n_vision_condition + 1)
            n_elements = len(elements)
            elements_prob = [
                (1 - self.first_element_prob) / (n_elements - 1)
            ] * n_elements
            elements_prob[0] = self.first_element_prob
            n_vision_cond_lst = np.random.choice(
                elements, target_length, p=elements_prob
            )
            self.n_vision_cond_lst = list(n_vision_cond_lst)
        else:
            raise ValueError(
                f"now only support union, focus_1st, but given {self.sample_n_viscond_method}"
            )

    def generate_tail_prob(self):
        self.all_data_tail_probs = torch.rand(len(self.data))
        # 尾帧移动会影响条件帧的数量，需要针对性更新下
        # 目前是需要配合 sampler_util.NumVisCondBatchSampler 来更新，以便按照n_viscond 来组 batch
        self.mv_tail_latent2viscond_flag = (
            self.all_data_tail_probs <= self.tail_frame_prob
        ).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = self.data[index]["prompt"]
        prompt = None if pd.isna(prompt) else prompt
        # video_path = self.data[index]["video_path"]
        # category = self.data[index]["category"]

        # 从h5py中读取预抽取的视频的emb、prompt、emb，
        emb_path = self.data[index][self.h5py_key]
        vision_condition_latents = None
        vision_condition_latent_index = None
        vision_clip_emb = None
        controlnet_latents = {}
        with h5py.File(emb_path, "r") as f:
            moments = torch.from_numpy(f[self.feat_key][...]).float()
            if self.prompt_emb_key is not None:
                prompt_clip_emb = (
                    torch.from_numpy(f[self.prompt_emb_key][...]).float().squeeze(0)
                )
            else:
                prompt_clip_emb = None
            latents = DiagonalGaussianDistribution(moments).sample()
            if self.contronet_emb_keys is not None:
                controlnet_latents = {}
                for controlnel_condition_key in self.contronet_emb_keys:
                    controlnet_latent = f[controlnel_condition_key.h5py_key][...]
                    controlnet_latents[
                        controlnel_condition_key.name
                    ] = controlnet_latent
            if self.prompt_text_key is not None:
                prompt_text_cond = (
                    f[self.prompt_text_key][()][0].decode("utf-8").strip()
                )
            if self.vision_clip_emb_key is not None:
                vision_clip_emb = torch.from_numpy(
                    f[self.vision_clip_emb_key][...]
                ).float()
        # h5py的emb处理在 with h5py 外处理，避免长时间占据h5py文件

        if self.contronet_emb_keys is not None:
            for controlnel_condition_key in self.contronet_emb_keys:
                controlnet_latent = controlnet_latents[controlnel_condition_key.name]
                # lllyasviel_control_v11p_sd15_openpose_dwpose_w=512_h=320_pose 存储的是 controlnet_aux dwpose 关键点的输出
                # TODO: 目前使用pose关键词进行识别，需要单独转换完pose_img，待后续看如何变成更通用
                if controlnet_latent is None:
                    print("emb_path before dwposekps2poseimg_converter", emb_path)
                if (
                    "pose" in controlnel_condition_key.name
                    and self.dwposekps2poseimg_converter is not None
                ):
                    controlnet_latent = self.dwposekps2poseimg_converter(
                        controlnet_latent
                    )
                if controlnet_latent is None:
                    print("emb_path after dwposekps2poseimg_converter", emb_path)
                controlnet_latent = torch.from_numpy(controlnet_latent).float()
                controlnet_latents[controlnel_condition_key.name] = controlnet_latent

        # 采样 n_sample_frames 帧
        sample_frame_rate = self.sample_frame_rate
        if self.use_dynamic_sr:
            sample_frame_rate = random.randrange(
                sample_frame_rate, self.max_sample_frame_rate + 1
            )
        ## 动态修改采样帧率，强化采样帧率效果
        (
            latents,
            sample_index,
            sample_frame_rate,
            refer_image_latents,
            refer_image_index,
        ) = sample_tensor_by_idx(
            tensor=latents,
            n_sample=self.n_sample_frames,
            sample_rate=sample_frame_rate,
            sample_start_idx=self.sample_start_idx,
            whether_random=True,
            change_sample_rate=True,
            n_independent=self.n_refer_image,
        )
        if vision_clip_emb is not None:
            # vision_clip_emb与refer_image等是一体的，优先使用 refer_image的配套
            # n_sample * d
            if refer_image_index is not None:
                vision_clip_emb = vision_clip_emb[refer_image_index, :]
            else:
                vision_clip_emb = vision_clip_emb[sample_index, :]
            if vision_clip_emb.ndim == 2:
                vision_clip_emb = rearrange(vision_clip_emb, "t d-> t 1 d")
            # logger.debug(f"dataset vision_clip_emb={vision_clip_emb.shape}")
        # frames_idx，采样帧的索引位置，在 unet 中会转化为 femb
        frame_index = (
            torch.arange(self.n_sample_frames, dtype=torch.long)
            * sample_frame_rate
            * self.video2emb_sr
        )
        # logger.debug(
        #     f"sample rate is {sample_frame_rate}, frame_index is {frame_index}"
        # )

        latents = rearrange(latents, "f c h w -> c f h w")
        if refer_image_latents is not None:
            refer_image_latents = rearrange(refer_image_latents, "f c h w -> c f h w")
        if len(controlnet_latents) > 0:
            controlnet_latents = {
                # k: torch.index_select(v, dim=0, index=sample_index)
                k: torch.index_select(v, dim=0, index=sample_index).cpu().numpy()
                for k, v in controlnet_latents.items()
            }
            # for k, v in controlnet_latents.items():
            #     logger.debug(f"controlnet_latents:  k={k}, v={v.shape}")
        # 拆分成视觉条件帧和视频帧；
        if self.n_vision_condition == 0:
            latent_index = torch.arange(self.n_sample_frames, dtype=torch.long)
            n_vision_condition = 0
            vision_condition_latent_index = None
            vision_condition_latents = None
        else:
            # TODO: 现在是先sample_tensor_by_idx一个整体，再split成sample 和condition，多样性受限。
            # 后续可以在sample_tensor_by_idx内采样、split同时做。
            n_vision_condition = self.n_vision_cond_lst[index]
            n_sample = self.n_sample_frames - n_vision_condition
            indexs = torch.arange(self.n_sample_frames, dtype=torch.long)
            latent_index, vision_condition_latent_index = split_index(
                indexs=indexs, n_first=n_sample, method=self.condition_sample_method
            )
            vision_condition_latents = torch.index_select(
                latents, dim=1, index=vision_condition_latent_index
            )
            latents = torch.index_select(latents, dim=1, index=latent_index)

            # controlnet latents 没有拆分的必要，作为一个整体全进去即可.
            # if controlnet_latents is not None:
            #     controlnet_condition_latents = [
            #         torch.index_select(x, dim=1, index=vision_condition_latents)
            #         for x in controlnet_latents
            #     ]
            #     controlnet_latents = [
            #         torch.index_select(x, dim=1, index=latent_index)
            #         for x in controlnet_latents
            #     ]

        # 扩展 prompt_clip_emb  n q -> t n q，并修改其中的 prompt_emb，
        if self.prompt_emb_key is not None:
            prompt_clip_emb = repeat(
                prompt_clip_emb, "n q -> t n q", t=self.n_sample_frames
            )

        # 若 update_prompt_clip_emb True 时，表示更新了 video对应的prompt，将会在 训练主流程中动态更新 prompt_emb
        update_prompt_clip_emb = False
        # prompt_clip_emb_bk = prompt_clip_emb  # 备份 正确的 video prompt，以便后续使用。

        # uncond prompt
        if self.prompt_emb_key is None and self.vision_clip_emb_key is None:
            prompt_clip_emb = self.uncond_embeddings
            prompt = ""
        else:
            if torch.rand(1)[0] < self.prob_uncond and self.vision_clip_emb_key is None:
                prompt_clip_emb = self.uncond_embeddings
                prompt = ""

        if self.use_text:  # use text
            # Add suffix_text, e.g. shutterstock watermark
            if not prompt.endswith("."):
                prompt = prompt + "."
            prompt = self.prefix_text + prompt + self.suffix_text
            update_prompt_clip_emb = True

        # 随机机制，将一定百分比的 vision latents 和 Vis—cond emb 改成 静态视频帧
        # 并 在prompt emb 中增加[镜头视频]的promt_emb，以便推断的强化使用
        static_video_prompt = self.static_video_prompt
        if static_video_prompt is not None and self.prob_static_video > 0:
            if torch.rand(1)[0] < self.prob_static_video:
                update_prompt_clip_emb = True
                if not static_video_prompt.endswith(", "):
                    static_video_prompt = static_video_prompt + ", "
                prompt = static_video_prompt + prompt
                # 使用首帧生成镜头视频帧m sample_rate / motion_speed = 0
                frame_index = frame_index * 0
                # 变成纯静态视频，采样条件帧或者 视频中的一帧
                if n_vision_condition > 0:
                    static_frame_index = torch.randint(
                        0, n_vision_condition, (1,)
                    ).tolist()[0]
                    latents = align_repeat_tensor_single_dim(
                        vision_condition_latents[
                            :, static_frame_index : static_frame_index + 1, :, :
                        ],
                        latents.shape[1],
                        dim=1,
                    )
                    vision_condition_latents = align_repeat_tensor_single_dim(
                        vision_condition_latents[
                            :, static_frame_index : static_frame_index + 1, :, :
                        ],
                        vision_condition_latents.shape[1],
                        dim=1,
                    )
                else:
                    static_frame_index = torch.randint(
                        0, latents.shape[1], (1,)
                    ).tolist()[0]
                    latents = align_repeat_tensor_single_dim(
                        latents[:, static_frame_index : static_frame_index + 1, :, :],
                        latents.shape[1],
                        dim=1,
                    )
                # logger.debug("trigger prob_static_video, generate static video")
                # TODO: 该模式暂不支持controlnet
                if len(controlnet_latents) > 0:
                    raise ValueError(
                        "static video prompt sample are not compatible with controlnet latents"
                    )

        # 尾帧处理
        # 按照概率将 latents 的尾帧的 移到 vision_latents 中
        # 同时处理对应的 latents_index 和v ision_condition_latents_index
        if self.tail_frame_prob > 0 and self.mv_tail_latent2viscond_flag[index]:
            (
                latents,
                vision_condition_latents,
                latent_index,
                vision_condition_latent_index,
            ) = mv_tail_latent2viscond(
                latents,
                vision_condition_latents,
                latent_index,
                vision_condition_latent_index,
            )
            n_vision_condition += 1
            mv_tail = True
        else:
            mv_tail = False

        example = {
            "latents": latents,
            "vision_condition_latents": vision_condition_latents,
            "latent_index": latent_index,
            "vision_condition_latent_index": vision_condition_latent_index,
            "n_vision_condition": n_vision_condition,
            "frame_index": frame_index,
            "use_text": self.use_text,
            "prompt": prompt,
            "prompt_clip_emb": prompt_clip_emb,  # t n d
            "prompt_text": prompt,
            "update_prompt_clip_emb": update_prompt_clip_emb,
            "vision_clip_emb": vision_clip_emb,  # t n d
            "ids": index,
            "refer_image_latents": refer_image_latents,
        }
        example.update(controlnet_latents)
        return example


def add_text2_prompt(text: str, prompt: str, format: str = "{}, {}") -> str:
    if len(prompt) > 0:
        return format.format(text, prompt)
    else:
        return text
