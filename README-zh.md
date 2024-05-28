# MuseV [English](README.md) [中文](README-zh.md)

<font size=5>MuseV：基于视觉条件并行去噪的无限长度和高保真虚拟人视频生成。
</br>
Zhiqiang Xia <sup>\*</sup>,
Zhaokang Chen<sup>\*</sup>,
Bin Wu<sup>†</sup>,
Chao Li,
Kwok-Wai Hung,
Chao Zhan,
Yingjie He, 
Wenjiang Zhou
(<sup>*</sup>co-first author, <sup>†</sup>Corresponding Author, benbinwu@tencent.com)
</font>

**[github](https://github.com/TMElyralab/MuseV)**    **[huggingface](https://huggingface.co/TMElyralab/MuseV)**   **[HuggingfaceSpace](https://huggingface.co/spaces/AnchorFake/MuseVDemo)**  **[project](https://tmelyralab.github.io/MuseV_Page/)**    **Technical report (comming soon)**


我们在2023年3月相信扩散模型可以模拟世界，也开始基于扩散模型研发世界视觉模拟器。`MuseV`是在 2023 年 7 月左右实现的一个里程碑。受到 Sora 进展的启发，我们决定开源 MuseV。MuseV 站在开源的肩膀上成长，也希望能够借此反馈社区。接下来，我们将转向有前景的扩散+变换器方案。

我们已经发布 <a href="https://github.com/TMElyralab/MuseTalk" style="font-size:24px; color:red;">MuseTalk</a>. `MuseTalk`是一个实时高质量的唇同步模型，可与 `MuseV` 一起构建完整的`虚拟人生成解决方案`。请保持关注！

:new: 我们新发布了<a href="https://github.com/TMElyralab/MusePose" style="font-size:24px; color:red;">MusePose</a>。 MusePose是一个用于虚拟人物的图像到视频生成框架，它可以根据控制信号（姿态）生成视频。结合 MuseV 和 MuseTalk，我们希望社区能够加入我们，一起迈向一个愿景：能够端到端生成具有全身运动和交互能力的虚拟人物。

# 概述

`MuseV` 是基于扩散模型的虚拟人视频生成框架，具有以下特点：

1. 支持使用新颖的视觉条件并行去噪方案进行无限长度生成，不会再有误差累计的问题，尤其适用于固定相机位的场景。
1. 提供了基于人物类型数据集训练的虚拟人视频生成预训练模型。
1. 支持图像到视频、文本到图像到视频、视频到视频的生成。
1. 兼容 `Stable Diffusion` 文图生成生态系统，包括 `base_model`、`lora`、`controlnet` 等。
1. 支持多参考图像技术，包括 `IPAdapter`、`ReferenceOnly`、`ReferenceNet`、`IPAdapterFaceID`。
1. 我们后面也会推出训练代码。

# 重要更新
1. `musev_referencenet_pose`: `unet`, `ip_adapter` 的模型名字指定错误，请使用 `musev_referencenet_pose`而不是`musev_referencenet`，请使用最新的main分支。

# 进展
- [2024年3月27日] 发布 `MuseV` 项目和训练好的模型 `musev`、`muse_referencenet`、`muse_referencenet_pose`。
- [03/30/2024] 在 huggingface space 上新增 [gui](https://huggingface.co/spaces/AnchorFake/MuseVDemo) 交互方式来生成视频.

## 模型
### 模型结构示意图
![model_structure](./data/models/musev_structure.png)
### 并行去噪算法示意图
![parallel_denoise](./data//models/parallel_denoise.png)

## 测试用例
生成结果的所有帧直接由`MuseV`生成，没有时序超分辨、空间超分辨等任何后处理。
更多测试结果请看[MuseVPage]()

<!-- # TODO: // use youtu video link? -->
以下所有测试用例都维护在 `configs/tasks/example.yaml`，可以直接运行复现。
**[project](https://tmelyralab.github.io/)** 有更多测试用例，包括直接生成的、一两分钟的长视频。

### 输入文本、图像的视频生成
#### 人类
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%">image</td>
        <td width="45%">video </td>
        <td width="5%">prompt</td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/yongen.jpeg width="400">
    </td>
    <td >
     <video src="https://github.com/TMElyralab/MuseV/assets/163980830/732cf1fd-25e7-494e-b462-969c9425d277" width="100" controls preload></video>
    </td>
    <td>(masterpiece, best quality, highres:1),(1boy, solo:1),(eye blinks:1.8),(head wave:1.3)
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/seaside4.jpeg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/9b75a46c-f4e6-45ef-ad02-05729f091c8f" width="100" controls preload></video>
    </td>   
    <td>
    (masterpiece, best quality, highres:1), peaceful beautiful sea scene
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/seaside_girl.jpeg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/d0f3b401-09bf-4018-81c3-569ec24a4de9" width="100" controls preload></video>
    </td>
    <td>
      (masterpiece, best quality, highres:1), peaceful beautiful sea scene
    </td>
  </tr>
  <!-- guitar  -->
  <tr>
    <td>
      <img src=./data/images/boy_play_guitar.jpeg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/61bf955e-7161-44c8-a498-8811c4f4eb4f" width="100" controls preload></video>
    </td>
    <td>
       (masterpiece, best quality, highres:1), playing guitar
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/girl_play_guitar2.jpeg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/40982aa7-9f6a-4e44-8ef6-3f185d284e6a" width="100" controls preload></video>
    </td>
    <td>
      (masterpiece, best quality, highres:1), playing guitar
    </td>
  </tr>
  <!-- famous people -->
  <tr>
    <td>
      <img src=./data/images/dufu.jpeg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/28294baa-b996-420f-b1fb-046542adf87d" width="100" controls preload></video>
    </td>
    <td>
    (masterpiece, best quality, highres:1),(1man, solo:1),(eye blinks:1.8),(head wave:1.3),Chinese ink painting style
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/Mona_Lisa.jpg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/1ce11da6-14c6-4dcd-b7f9-7a5f060d71fb" width="100" controls preload></video>
    </td>   
    <td>
    (masterpiece, best quality, highres:1),(1girl, solo:1),(beautiful face,
    soft skin, costume:1),(eye blinks:{eye_blinks_factor}),(head wave:1.3)
    </td>
  </tr>
</table >

#### 场景
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td width="35%">image</td>
        <td width="50%">video</td>
        <td width="15%">prompt</td>
    </tr>

  <tr>
    <td>
      <img src=./data/images/waterfall4.jpeg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/852daeb6-6b58-4931-81f9-0dddfa1b4ea5" width="100" controls preload></video>
    </td>
    <td>
      (masterpiece, best quality, highres:1), peaceful beautiful waterfall, an
    endless waterfall
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/seaside2.jpeg width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/4a4d527a-6203-411f-afe9-31c992d26816" width="100" controls preload></video>
    </td>
    <td>(masterpiece, best quality, highres:1), peaceful beautiful sea scene
    </td>
  </tr>
</table >

### 输入视频条件的视频生成
当前生成模式下，需要参考视频的首帧条件和参考图像的首帧条件对齐，不然会破坏首帧的信息，效果会更差。所以一般生成流程是
1. 确定参考视频；
2. 用参考视频的首帧走图生图、controlnet流程，可以使用`MJ`等各种平台；
3. 拿2中的生成图、参考视频用MuseV生成视频；
4. 
**pose2video**

`duffy` 的测试用例中，视觉条件帧的姿势与控制视频的第一帧不对齐。需要`posealign` 将解决这个问题。

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td width="25%">image</td>
        <td width="65%">video</td>
        <td width="10%">prompt</td>
    </tr>
  <tr>
    <td>
      <img src=./data/images/spark_girl.png width="200">
      <img src=./data/images/cyber_girl.png width="200">
    </td>
    <td>
        <video src="https://github.com/TMElyralab/MuseV/assets/163980830/484cc69d-c316-4464-a55b-3df929780a8e" width="400" controls preload></video>
    </td>
    <td>
      (masterpiece, best quality, highres:1) , a girl is dancing, animation
    </td>
  </tr>
  <tr>   
    <td>
      <img src=./data/images/duffy.png width="400">
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/c44682e6-aafc-4730-8fc1-72825c1bacf2" width="400" controls preload></video>
    </td>
    <td>
      (masterpiece, best quality, highres:1), is dancing, animation
    </td>
  </tr>
</table >

### MuseTalk

`talk`的角色`孙昕荧`著名的网络大V，可以在 [抖音](https://www.douyin.com/user/MS4wLjABAAAAWDThbMPN_6Xmm_JgXexbOii1K-httbu2APdG8DvDyM8) 关注。

<table class="center">
    <tr style="font-weight: bolder;">
        <td width="35%">name</td>
        <td width="50%">video</td>
    </tr>

  <tr>
    <td>
       talk
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/951188d1-4731-4e7f-bf40-03cacba17f2f" width="100" controls preload></video>
    </td>
  <tr>
    <td>
       sing
    </td>
    <td>
      <video src="https://github.com/TMElyralab/MuseV/assets/163980830/50b8ffab-9307-4836-99e5-947e6ce7d112" width="100" controls preload></video>
    </td>
  </tr>
</table >


# 待办事项：
- [ ] 技术报告（即将推出）。
- [ ] 训练代码。
- [ ] 扩散变换生成框架。
- [ ] `posealign` 模块。

# 快速入门
准备 Python 环境并安装额外的包，如 `diffusers`、`controlnet_aux`、`mmcm`。

## 第三方整合版
一些第三方的整合，方便大家安装、使用，感谢第三方的工作。
同时也希望注意，我们没有对第三方的支持做验证、维护和后续更新，具体效果请以本项目为准。
### [ComfyUI](https://github.com/chaojie/ComfyUI-MuseV)
### [windows整合包](https://www.bilibili.com/video/BV1ux4y1v7pF/?vd_source=fe03b064abab17b79e22a692551405c3)
netdisk:https://www.123pan.com/s/Pf5Yjv-Bb9W3.html
code: glut

## 准备环境
建议您优先使用 `docker` 来准备 Python 环境。

### 准备 Python 环境
**注意**：我们只测试了 Docker，使用 conda 或其他环境可能会遇到问题。我们将尽力解决。但依然请优先使用 `docker`。

#### 方法 1：使用 Docker
1. 拉取 Docker 镜像
```bash
docker pull anchorxia/musev:latest
```
2. 运行 Docker 容器
```bash
docker run --gpus all -it --entrypoint /bin/bash anchorxia/musev:latest
```
docker启动后默认的 conda 环境是 `musev`。

#### 方法 2：使用 conda
从 environment.yaml 创建 conda 环境
```
conda env create --name musev --file ./environment.yml
```
#### 方法 3：使用 pip requirements
```
pip install -r requirements.txt
```
#### 准备 [openmmlab](https://openmmlab.com/) 包
如果不使用 Docker方式，还需要额外安装 mmlab 包。
```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### 准备我们开发的包
#### 下载
```bash
git clone --recursive https://github.com/TMElyralab/MuseV.git
```
#### 准备 PYTHONPATH
```bash
current_dir=$(pwd)
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/MMCM
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/diffusers/src
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/controlnet_aux/src
cd MuseV
```

1. `MMCM`：多媒体、跨模态处理包。
1. `diffusers`：基于 [diffusers](https://github.com/huggingface/diffusers) 修改的 diffusers 包。
1. `controlnet_aux`：基于 [controlnet_aux](https://github.com/TMElyralab/controlnet_aux) 修改的包。


## 下载模型
```bash
git clone https://huggingface.co/TMElyralab/MuseV ./checkpoints
```
- `motion`：多个版本的视频生成模型。使用小数据集 `ucf101` 和小 `webvid` 数据子集进行训练，约 60K 个视频文本对。GPU 内存消耗测试在 `resolution` $=512*512，`time_size=12`。
    - `musev/unet`：这个版本 仅训练 `unet` 运动模块。推断 `GPU 内存消耗` $\approx 8G$。
    - `musev_referencenet`：这个版本训练 `unet` 运动模块、`referencenet`、`IPAdapter`。推断 `GPU 内存消耗` $\approx 12G$。
        - `unet`：`motion` 模块，具有 `Attention` 层中的 `to_k`、`to_v`，参考 `IPAdapter`。
        - `referencenet`：类似于 `AnimateAnyone`。
        - `ip_adapter_image_proj.bin`：图像特征变换层，参考 `IPAdapter`。
    - `musev_referencenet_pose`：这个版本基于 `musev_referencenet`，固定 `referencenet` 和 `controlnet_pose`，训练 `unet motion` 和 `IPAdapter`。推断 `GPU 内存消耗` $\approx 12G$。
- `t2i/sd1.5`：text2image 模型，在训练运动模块时参数被冻结。
    - `majicmixRealv6Fp16`：示例，可以替换为其他 t2i 基础。从 [majicmixRealv6Fp16](https://civitai.com/models/43331/majicmix-realistic) 下载。
    - `fantasticmix_v10`: 可在 [fantasticmix_v10](https://civitai.com/models/22402?modelVersionId=26744) 下载。
- `IP-Adapter/models`：从 [IPAdapter](https://huggingface.co/h94/IP-Adapter/tree/main) 下载。
    - `image_encoder`：视觉特征抽取模型。
    - `ip-adapter_sd15.bin`：原始 IPAdapter 模型预训练权重。
    - `ip-adapter-faceid_sd15.bin`：原始 IPAdapter 模型预训练权重。

## 推理

### 准备模型路径
当使用示例推断命令运行示例任务时，可以跳过此步骤。
该模块主要是在配置文件中设置模型路径和缩写，以在推断脚本中使用简单缩写而不是完整路径。
- T2I SD：参考 `musev/configs/model/T2I_all_model.py`
- 运动 Unet：参考 `musev/configs/model/motion_model.py`
- 任务：参考 `musev/configs/tasks/example.yaml`

### musev_referencenet
#### 输入文本、图像的视频生成
```bash
python scripts/inference/text2video.py   --sd_model_name majicmixRealv6Fp16   --unet_model_name musev_referencenet --referencenet_model_name musev_referencenet --ip_adapter_model_name musev_referencenet   -test_data_path ./configs/tasks/example.yaml  --output_dir ./output  --n_batch 1  --target_datas yongen  --vision_clip_extractor_class_name ImageClipVisionFeatureExtractor --vision_clip_model_path ./checkpoints/IP-Adapter/models/image_encoder  --time_size 12 --fps 12  
```
**通用参数**：
- `test_data_path`：测试用例任务路径
- `target_datas`：如果 `test_data_path` 中的 `name` 在 `target_datas` 中，则只运行这些子任务。`sep` 是 `,`；
- `sd_model_cfg_path`：T2I sd 模型路径，模型配置路径或模型路径。
- `sd_model_name`：sd 模型名称，用于在 `sd_model_cfg_path` 中选择完整模型路径。使用 `,` 分隔的多个模型名称，或 `all`。
- `unet_model_cfg_path`：运动 unet 模型配置路径或模型路径。
- `unet_model_name`：unet 模型名称，用于获取 `unet_model_cfg_path` 中的模型路径，并在 `musev/models/unet_loader.py` 中初始化 unet 类实例。使用 `,` 分隔的多个模型名称，或 `all`。如果 `unet_model_cfg_path` 是模型路径，则 `unet_name` 必须在 `musev/models/unet_loader.py` 中支持。
- `time_size`：扩散模型每次生成一个片段，这里是一个片段的帧数。默认为 `12`。
- `n_batch`：首尾相连方式生成总片段数，$total\_frames=n\_batch * time\_size + n\_viscond$，默认为 `1`。
- `context_frames`： 并行去噪子窗口一次生成的帧数。如果 `time_size` > `context_frame`，则会启动并行去噪逻辑， `time_size` 窗口会分成多个子窗口进行并行去噪。默认为 `12`。

生成**长视频**，有两种方法，可以共同使用：
1. `视觉条件并行去噪`：设置 `n_batch=1`，`time_size` = 想要的所有帧。
2. `传统的首尾相连方式`：设置 `time_size` = `context_frames` = 一次片段的帧数 (`12`)，`context_overlap` = 0。会首尾相连方式生成`n_batch`片段数，首尾相连存在误差累计，当`n_batch`越大，最后的结果越差。


**模型参数**：
支持 `referencenet`、`IPAdapter`、`IPAdapterFaceID`、`Facein`。
- `referencenet_model_name`：`referencenet` 模型名称。
- `ImageClipVisionFeatureExtractor`：`ImageEmbExtractor` 名称，在 `IPAdapter` 中提取视觉特征。
- `vision_clip_model_path`：`ImageClipVisionFeatureExtractor` 模型路径。
- `ip_adapter_model_name`：来自 `IPAdapter` 的，它是 `ImagePromptEmbProj`，与 `ImageEmbExtractor` 一起使用。
- `ip_adapter_face_model_name`：`IPAdapterFaceID`，来自 `IPAdapter`，应该设置 `face_image_path`。

**一些影响运动范围和生成结果的参数**：
- `video_guidance_scale`：类似于 text2image，控制 cond 和 uncond 之间的影响，影响较大，默认为 `3.5`。
- `use_condition_image`：是否使用给定的第一帧进行视频生成, 默认 `True`。
- `redraw_condition_image`：是否重新绘制给定的第一帧图像。
- `video_negative_prompt`：配置文件中全 `negative_prompt` 的缩写。默认为 `V2`。


#### 输入视频的视频生成
```bash
python scripts/inference/video2video.py --sd_model_name majicmixRealv6Fp16  --unet_model_name musev_referencenet --referencenet_model_name   musev_referencenet --ip_adapter_model_name musev_referencenet    -test_data_path ./configs/tasks/example.yaml    --vision_clip_extractor_class_name ImageClipVisionFeatureExtractor --vision_clip_model_path ./checkpoints/IP-Adapter/models/image_encoder      --output_dir ./output  --n_batch 1 --controlnet_name dwpose_body_hand  --which2video "video_middle"  --target_datas dance1 --fps 12 --time_size 12
```
**一些重要参数**

大多数参数与 `musev_text2video` 相同。`video2video` 的特殊参数有：
1. 需要在 `test_data` 中设置 `video_path`。现在支持 `rgb video` 和 `controlnet_middle_video`。
- `which2video`： 参考视频类型。 如果是 `video_middle`，则只使用类似`pose`、`depth`的 `video_middle`；如果是 `video`， 视频本身也会参与视频噪声初始化，类似于`img2imge`。
- `controlnet_name`：是否使用 `controlnet condition`，例如 `dwpose,depth`， pose的话 优先建议使用`dwpose_body_hand`。
- `video_is_middle`：`video_path` 是 `rgb video` 还是 `controlnet_middle_video`。可以为 `test_data_path` 中的每个 `test_data` 设置。
- `video_has_condition`：condtion_images 是否与 video_path 的第一帧对齐。如果不是，则首先生成 `condition_images`，然后与参考视频拼接对齐。 目前仅支持参考视频是`video_is_middle=True`，可`test_data` 设置。

所有 `controlnet_names` 维护在 [mmcm](https://github.com/TMElyralab/MMCM/blob/main/mmcm/vision/feature_extractor/controlnet.py#L513)
```python
['pose', 'pose_body', 'pose_hand', 'pose_face', 'pose_hand_body', 'pose_hand_face', 'dwpose', 'dwpose_face', 'dwpose_hand', 'dwpose_body', 'dwpose_body_hand', 'canny', 'tile', 'hed', 'hed_scribble', 'depth', 'pidi', 'normal_bae', 'lineart', 'lineart_anime', 'zoe', 'sam', 'mobile_sam', 'leres', 'content', 'face_detector']
```

### musev_referencenet_pose
仅用于 `pose2video`
基于 `musev_referencenet` 训练，固定 `referencenet`、`pose-controlnet` 和 `T2I`，训练 `motion` 模块和 `IPAdapter`。
```bash
python scripts/inference/video2video.py --sd_model_name majicmixRealv6Fp16  --unet_model_name musev_referencenet_pose --referencenet_model_name   musev_referencenet --ip_adapter_model_name musev_referencenet_pose    -test_data_path ./configs/tasks/example.yaml    --vision_clip_extractor_class_name ImageClipVisionFeatureExtractor --vision_clip_model_path ./checkpoints/IP-Adapter/models/image_encoder      --output_dir ./output  --n_batch 1 --controlnet_name dwpose_body_hand  --which2video "video_middle"  --target_datas  dance1   --fps 12 --time_size 12
```

### musev
仅有动作模块，没有 referencenet，需要更少的 GPU 内存。
#### 文本到视频
```bash
python scripts/inference/text2video.py   --sd_model_name majicmixRealv6Fp16   --unet_model_name musev   -test_data_path ./configs/tasks/example.yaml  --output_dir ./output  --n_batch 1  --target_datas yongen  --time_size 12 --fps 12
```
#### 视频到视频
```bash
python scripts/inference/video2video.py --sd_model_name majicmixRealv6Fp16  --unet_model_name musev    -test_data_path ./configs/tasks/example.yaml --output_dir ./output  --n_batch 1 --controlnet_name dwpose_body_hand  --which2video "video_middle"  --target_datas  dance1   --fps 12 --time_size 12
```

### Gradio 演示
MuseV 提供 gradio 脚本，可在本地机器上生成 GUI，方便生成视频。

```bash
cd scripts/gradio
python app.py
```

# 致谢
1. MuseV 开发过程中参考学习了很多开源工作 [TuneAVideo](https://github.com/showlab/Tune-A-Video)、[diffusers](https://github.com/huggingface/diffusers)、[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/tree/master/src/pipelines)、[animatediff](https://github.com/guoyww/AnimateDiff)、[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)、[AnimateAnyone](https://arxiv.org/abs/2311.17117)、[VideoFusion](https://arxiv.org/abs/2303.08320) 和 [insightface](https://github.com/deepinsight/insightface)。
2. MuseV 基于 `ucf101` 和 `webvid` 数据集构建。

感谢开源社区的贡献！

# 限制

`MuseV` 仍然存在很多待优化项，包括：

1. 缺乏泛化能力。对视觉条件帧敏感，有些视觉条件图像表现良好，有些表现不佳。有些预训练的 t2i 模型表现良好，有些表现不佳。
1. 有限的视频生成类型和有限的动作范围，部分原因是训练数据类型有限。发布的 `MuseV` 已经在大约 6 万对分辨率为 `512*320` 的人类文本视频对上进行了训练。`MuseV` 在较低分辨率下具有更大的动作范围，但视频质量较低。`MuseV` 在高分辨率下画质很好、但动作范围较小。在更大、更高分辨率、更高质量的文本视频数据集上进行训练可能会使 `MuseV` 更好。
1. 因为使用 `webvid` 训练会有水印问题。使用没有水印的、更干净的数据集可能会解决这个问题。
1. 有限类型的长视频生成。视觉条件并行去噪可以解决视频生成的累积误差，但当前的方法只适用于相对固定的摄像机场景。
1. referencenet 和 IP-Adapter 训练不足，因为时间有限和资源有限。
1. 代码结构不够完善。`MuseV` 支持丰富而动态的功能，但代码复杂且未经过重构。熟悉需要时间。
   

<!-- # Contribution 暂时不需要组织开源共建 -->
# 引用
```bib
@article{musev,
  title={MuseV: 基于视觉条件的并行去噪的无限长度和高保真虚拟人视频生成},
  author={Xia, Zhiqiang and Chen, Zhaokang and Wu, Bin and Li, Chao and Hung, Kwok-Wai and Zhan, Chao and He, Yingjie and Zhou, Wenjiang},
  journal={arxiv},
  year={2024}
}
```
# 免责声明/许可
1. `代码`：`MuseV` 的代码采用 `MIT` 许可证发布，学术用途和商业用途都可以。
1. `模型`：训练好的模型仅供非商业研究目的使用。
1. `其他开源模型`：使用的其他开源模型必须遵守他们的许可证，如 `insightface`、`IP-Adapter`、`ft-mse-vae` 等。
1. 测试数据收集自互联网，仅供非商业研究目的使用。
1. `AIGC`：本项目旨在积极影响基于人工智能的视频生成领域。用户被授予使用此工具创建视频的自由，但他们应该遵守当地法律，并负责任地使用。开发人员不对用户可能的不当使用承担任何责任。
