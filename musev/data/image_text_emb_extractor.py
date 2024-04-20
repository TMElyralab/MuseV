import logging
import os


import torch
import numpy as np
import pandas as pd
import h5py
from einops import rearrange
from PIL import Image
from mmcm.utils.itertools_util import convert_list_flat2nest

from mmcm.vision.feature_extractor.controlnet import ControlnetFeatureExtractor


class ImageTextEmbExtractor(object):
    def __init__(
        self,
        whether_extract_vae_embs: bool,
        whether_extract_text_embs: bool,
        whether_extract_controlnet_embs: bool,
        whether_extract_clip_vision_embs: bool,
        device="cuda",
        vae_dtype=torch.float32,
        text_dtype=torch.float32,
        controlnet_dtype=torch.float32,
        clip_vision_dtype=torch.float32,
        track_gpu: bool = False,
    ) -> None:
        from mmcm.utils.gpu_util import get_gpu_status
        from mmcm.vision.feature_extractor.vae_extractor import (
            VAEFeatureExtractor,
        )
        from mmcm.text.feature_extractor.clip_text_extractor import (
            ClipTextFeatureExtractor,
        )

        if track_gpu:
            print("gpu status: init")
            print(get_gpu_status())

        # prepare video predictor
        self.whether_extract_vae_embs = whether_extract_vae_embs
        if whether_extract_vae_embs:
            self.video_predictor = VAEFeatureExtractor(
                pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
                device=device,
                name="CompVis_stable-diffusion-v1-4_vae",
                dtype=vae_dtype,
            )
            if track_gpu:
                print("\n gpu status: after VAEFeatureExtractor")
                print(get_gpu_status())

        self.whether_extract_text_embs = whether_extract_text_embs
        if whether_extract_text_embs:
            # prepare text_predictor
            self.text_predictor = ClipTextFeatureExtractor(
                pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
                device=device,
                name="CompVis_stable-diffusion-v1-4_CLIPEncoderLayer",
                dtype=text_dtype,
            )
            if track_gpu:
                print("\n gpu status: after ClipTextFeatureExtractor")
                print(get_gpu_status())

        # prepare controlnet embs
        self.whether_extract_controlnet_embs = whether_extract_controlnet_embs
        if whether_extract_controlnet_embs:
            self.controlnet_predictor = ControlnetFeatureExtractor(
                model_path="lllyasviel/control_v11p_sd15_openpose",
                # detector_name="OpenposeDetector",
                detector_name="DWposeDetector",
                detector_id="lllyasviel/Annotators",
                dtype=controlnet_dtype,
                device=device,
                name="lllyasviel_control_v11p_sd15_openpose_dwpose",
            )
            if track_gpu:
                print("\n gpu status: after controlnet")
                print(get_gpu_status())

        # prepare clip vision embs
        self.whether_extract_clip_vision_embs = whether_extract_clip_vision_embs
        if whether_extract_clip_vision_embs:
            from mmcm.vision.feature_extractor.clip_vision_extractor import (
                ImageClipVisionFeatureExtractor,
                ImageClipVisionFeatureExtractorV2,
                ImageClipVisionFeatureExtractorV3,
                ImageClipVisionFeatureExtractorV4,
            )

            # IPAdapter use hidden_state[-2]
            self.clip_vision_predictor = ImageClipVisionFeatureExtractor(
                # self.clip_vision_predictor = ImageClipVisionFeatureExtractorV2(
                # pretrained_model_name_or_path="openai/clip-vit-base-patch32",
                pretrained_model_name_or_path="/cfs-datasets/projects/VirtualIdol/models/ip_adapter/models/image_encoder",
                dtype=clip_vision_dtype,
                device=device,
                # name="CLIPVisionModelWithProjection",
                name="CLIPVisionModelWithProjection_ip_adapter",
            )

            # openai/clip-vit-large-patch14 image_embds
            # self.clip_vision_predictor = ImageClipVisionFeatureExtractor(
            #     pretrained_model_name_or_path="openai/clip-vit-large-patch14",
            #     name="openai_clip_vit_large_patch14_ModelWithProjection_image_emb",
            #     dtype=clip_vision_dtype,
            #     device=device,
            # )

            # openai/clip-vit-large-patch14 last_hidden_state
            # self.clip_vision_predictor = ImageClipVisionFeatureExtractorV3(
            #     pretrained_model_name_or_path="openai/clip-vit-large-patch14",
            #     name="openai_clip_vit_large_patch14_ModelWithProjection_last_hidden_state",
            #     dtype=clip_vision_dtype,
            #     device=device,
            # )
            if track_gpu:
                print("\n gpu status: after extract_clip_embs")
                print(get_gpu_status())

    def extract(
        self,
        output_path: str,
        video_path: str = None,
        prompt: str = None,
        sample_rate: int = 2,
        time_size: int = 10,
        step: int = 10,
        dataset_device: str = "cpu",
        drop_last: bool = False,
        text_key: str = "text",
        text_index: int = 0,
        target_width: int = None,
        target_height: int = None,
    ) -> None:
        """extract video emb, text emb, and save into h5py file

        Args:
            output_path (str): h5py file path
            video_path (str, optional): input video path. Defaults to None.
            prompt (str, optional): prompt for video. Defaults to None.
            target_width (int, optional): if not None, set target_width before model predict. Defaults to None.
            target_height (int, optional): if not None, set target_height before model predict. Defaults to None.
            sample_rate (int, optional): pick one frame per sample_rate. Defaults to 2.
            time_size (int, optional): window size for time sample, batch size for image batch data. Defaults to 10.
            step (int, optional): step for windows. Defaults to 10.
            dataset_device (str, optional): device for decord package parse video. Defaults to "cpu".
            drop_last (bool, optional): if True, the last frames < time_size would be droped. Defaults to False.
        """
        from mmcm.vision.data.video_dataset import (
            DecordVideoDataset,
        )

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_dir, exist_ok=True)
        sample_indexs = None
        has_union_sample_indexs_key = False
        # TODO: 历史原因是用的vae的关键词存的sample_index，不同vision特征往往需要同一个sample_index，但是当不需要vaeemb 需要其他emb的时候会造成混淆
        # 在某个机会下，可以看看以通用 sample_index 名 来存储会更好些。
        union_sample_indexs_key = f"CompVis_stable-diffusion-v1-4_vae_w={target_width}_h={target_height}_sample_indexs"
        if not os.path.exists(output_path):
            overwrite_vae = True
            overwrite_text = True
            overwrite_controlnet = True
            overwrite_clip_vision = True
        else:
            with h5py.File(output_path, "r") as f:
                if union_sample_indexs_key in f:
                    sample_indexs = f[union_sample_indexs_key][...]
                    has_union_sample_indexs_key = True
                if (
                    f"CompVis_stable-diffusion-v1-4_CLIPEncoderLayer_last_hidden_state_{text_index}"
                    not in f
                ):
                    overwrite_text = True
                else:
                    overwrite_text = False

                if (
                    f"CompVis_stable-diffusion-v1-4_vae_w={target_width}_h={target_height}_encoder_emb"
                    not in f
                ):
                    overwrite_vae = True
                else:
                    overwrite_vae = False

                if self.whether_extract_controlnet_embs and (
                    # f"{self.controlnet_predictor.name}_w={target_width}_h={target_height}_emb"
                    f"{self.controlnet_predictor.name}_w={target_width}_h={target_height}_pose"
                    not in f
                ):
                    overwrite_controlnet = True
                else:
                    overwrite_controlnet = False

                if self.whether_extract_clip_vision_embs and (
                    f"{self.clip_vision_predictor.name}_w={target_width}_h={target_height}_emb"
                    not in f
                ):
                    overwrite_clip_vision = True
                else:
                    overwrite_clip_vision = False

        # extract text emb
        if self.whether_extract_text_embs and overwrite_text and prompt is not None:
            try:
                self.text_predictor.extract(
                    text=prompt,
                    save_emb_path=output_path,
                    text_emb_key="last_hidden_state",
                    text_key=text_key,
                    text_index=text_index,
                    insert_name_to_key=True,
                )
                print(f"succeed text emb {prompt}, save to {output_path}")
            except Exception as e:
                logging.exception(e)
                print(f"failed text emb {prompt}")

        if (
            (
                self.whether_extract_vae_embs
                or self.whether_extract_controlnet_embs
                or self.whether_extract_clip_vision_embs
            )
            and (overwrite_vae or overwrite_controlnet or overwrite_clip_vision)
            and video_path is not None
        ):
            if not os.path.exists(video_path):
                print(f"video_path not existed: {video_path}")
                return
            try:
                if sample_indexs is not None:
                    sample_indexs = convert_list_flat2nest(
                        sample_indexs,
                        window=time_size,
                    )

                video_dataset = DecordVideoDataset(
                    path=video_path,
                    sample_rate=sample_rate,
                    time_size=time_size,
                    step=step,
                    device=dataset_device,
                    drop_last=drop_last,
                    channels_order="t h w c",
                    sample_indexs=sample_indexs,
                    data_type="rgb",
                )
                height, width = video_dataset.height, video_dataset.width
                if target_height is None:
                    target_height = height
                if target_width is None:
                    target_width = width
            except Exception as e:
                logging.exception(e)
                print(f"failed parse video {video_path}")
                return
            if self.whether_extract_vae_embs and overwrite_vae:
                try:
                    self.video_predictor.extract(
                        data=video_dataset,
                        data_type="video",
                        save_emb_path=output_path,
                        target_width=target_width,
                        target_height=target_height,
                        return_type="numpy",
                        insert_name_to_key=True,
                        emb_key=f"w={target_width}_h={target_height}_encoder_emb",
                        quant_emb_key=f"w={target_width}_h={target_height}_encoder_quant_emb",
                        sample_index_key=f"w={target_width}_h={target_height}_sample_indexs",
                        save_sample_index=not has_union_sample_indexs_key,
                        input_rgb_order="rgb",
                    )
                    print(f"succeed vae emb: {video_path}, save to {output_path}")
                    has_union_sample_indexs_key = True
                except Exception as e:
                    logging.exception(e)
                    print(f"failed vae emb {video_path}")

            if self.whether_extract_controlnet_embs and overwrite_controlnet:
                # if
                try:
                    self.controlnet_predictor.extract(
                        data=video_dataset,
                        data_type="video",
                        save_emb_path=output_path,
                        target_width=target_width,
                        target_height=target_height,
                        return_type="numpy",
                        insert_name_to_key=True,
                        # emb_key=f"w={target_width}_h={target_height}_emb",
                        emb_key=f"w={target_width}_h={target_height}_pose",
                        sample_index_key=f"w={target_width}_h={target_height}_sample_indexs",
                        processor_params={
                            "detect_resolution": min(target_height, target_width),
                            "image_resolution": min(target_height, target_width),
                            "include_body": True,
                            "hand_and_face": True,
                            "return_pose_only": True,
                        },
                        save_sample_index=not has_union_sample_indexs_key,
                        input_rgb_order="rgb",
                    )
                    has_union_sample_indexs_key = True
                    print(
                        f"succeed controlnet emb: {video_path}, save to {output_path}"
                    )
                except Exception as e:
                    logging.exception(e)
                    print(f"failed controlnet emb {video_path}")

            if self.whether_extract_clip_vision_embs and overwrite_clip_vision:
                try:
                    self.clip_vision_predictor.extract(
                        data=video_dataset,
                        data_type="video",
                        save_emb_path=output_path,
                        target_width=target_width,
                        target_height=target_height,
                        return_type="numpy",
                        insert_name_to_key=True,
                        emb_key=f"w={target_width}_h={target_height}_image_embeds",
                        # emb_key=f"w={target_width}_h={target_height}",
                        sample_index_key=f"w={target_width}_h={target_height}_sample_indexs",
                        input_rgb_order="rgb",
                        save_sample_index=not has_union_sample_indexs_key,
                    )
                    has_union_sample_indexs_key = True

                    print(
                        f"succeed clip vision emb: {video_path}, save to {output_path}"
                    )
                except Exception as e:
                    logging.exception(e)
                    print(f"failed clip vision emb {video_path}")
