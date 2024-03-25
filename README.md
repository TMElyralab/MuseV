# MuseV

<font size=5>MuseV: Infinite-length and High Fidelity Virtual Human Video Generation with Parallel Denoising
</br>
Zhiqiang Xia<sup>*</sup>,
Zhaokang Chen<sup>*</sup>,
Bin Wu<sup>†</sup>,
Chao Li,
Kwok-Wai Hung,
Chao Zhan,
Wenjiang Zhou
(<sup>*</sup>co-first author, <sup>†</sup>Corresponding Author)
</font>

**[project](comming soon)**  **Technical report (comming soon)**


We have setup the world simulator vision since March 2023, believing diffusion models can simulate the world. `MuseV` was a milestone achieved around July 2023. Amazed by the progress of Sora, we decided to opensource `MuseV`, hopefully it will benefit the community.

Our next move will switch to the promising diffusion+transformer scheme. Please stay tuned.

We will soon release `MuseTalk`, a diffusion-baesd lip sync model, which can be applied with MuseV as a complete virtual human generation solution. Please stay tuned! 

# Intro
`MuseV` is a diffusion-based virtual human video generation framework, which 
1. supports infinite length generation using a novel Parallel Denoising scheme.
2. checkpoint available for virtual human video generation trained on human dataset.
3. supports Image2Video, Text2Image2Video, Video2Video.
4. compatible with the Stable Diffusion ecosystem, including `base_model`, `lora`, `controlnet`, etc. 
5. supports multi reference image technology, including `IPAdapter`, `ReferenceOnly`, `ReferenceNet`, `IPAdapterFaceID`.
6. training codes (comming very soon).


## Model
### overview of model structure
![model_structure](./data/models/musev_structure.png)
### parallel denoise
![parallel_denoise](./data//models/parallel_denoise.png)

## Cases
All frames are generated from text2video model, without any post process.
<!-- # TODO: // use youtu video link? -->
Bellow Case could be found in `configs/tasks/example.yaml`
### Text/Image2Video

#### Human
 <!-- 2 columns, one image, one video -->
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>image</td>
        <td>video</td>
    </tr>
    
  <tr>
    <td>
      <img src=./data/images/yongen.jpeg width="250">
    </td>
    <td>
      [![Video Title](https://img.youtube.com/vi/Rsm46lDkJZE/0.jpg)](https://www.youtube.com/watch?v=Rsm46lDkJZE)
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/jinkesi2.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/jinkesi2.mp4" width="300" controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/seaside4.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/seaside4.mp4" width="300" controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/real_girl_seaside2.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/real_girl_seaside2.mp4" width="300" controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/seaside_girl.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/seaside_girl.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <!-- guitar  -->
  <tr>
    <td>
      <img src=./data/images/boy_play_guitar.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/boy_play_guitar.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/girl_play_guitar2.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/girl_play_guitar2.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/boy_play_guitar2.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/boy_play_guitar2.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/girl_play_guitar4.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/girl_play_guitar4.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <!-- famous people -->
  <tr>
    <td>
      <img src=./data/images/dufu.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/dufu.mp4" width="300" controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg width="250">
    </td>
    <td>
      <video src="./data/result_video/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/Portrait-of-Dr.-Gachet.jpg width="250">
    </td>
    <td>
      <video src="./data/result_video/Portrait-of-Dr.-Gachet.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/Self-Portrait-with-Cropped-Hair.jpg width="250">
    </td>
    <td>
      <video src="./data/result_video/Self-Portrait-with-Cropped-Hair.mp4" width="300" controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=./data/images/The-Laughing-Cavalier.jpg width="250">
    </td>
    <td>
      <video src="./data/result_video/The-Laughing-Cavalier.mp4" width="300" controls preload></video>
    </td>
  </tr>
</table >

#### scene

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>image</td>
        <td>video</td>
    </tr>

  <tr>
    <td>
      <img src=./data/images/waterfall4.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/waterfall4.mp4" width="300" controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/river.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/river.mp4" width="300" controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <img src=./data/images/seaside2.jpeg width="250">
    </td>
    <td>
      <video src="./data/result_video/seaside2.mp4" width="300" controls preload></video>
    </td>
  </tr>

</table >

### VideoMiddle2Video

### Video2Video


# News
- [03/22/2024] release `MuseV` project and trained model `musev`, `muse_referencenet`.

# TODO:
- [ ] technical report (comming soon).
- [ ] training codes.
- [ ] release pretrained unet model, which is trained with controlnet、referencenet、IPAdapter, which is better on pose2video.
- [ ] support diffusion transformer generation framework.

# Quickstart
prepare python environment and install extra package like `diffusers`, `controlnet_aux`, `mmcm`.

suggest to use `docker` primarily to prepare python environment.

## prepare environment
### docker
1. pull docker image
```bash
docker pull docker pull anchorxia/musev:latest
```
2. run docker
```bash
docker run --gpus all -it --entrypoint /bin/bash anchorxia/musev:latest
```
The default conda env is `musev`.

### conda 
create conda environment from environment.yaml
```
conda env create --name musev --file ./environment.yml
```
### requirements
```
pip install -r requirements.txt
```

**install mmlab package**
```bash
pip install--no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### custom package / modified package
```bash
git clone --recursive https://github.com/TMElyralab/MuseV.git
```
prepare PYTHONPATH
```bash
current_dir=$(pwd)
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/MMCM
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/diffusers/src
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/controlnet_aux/src
cd MuseV
```

1. `MMCM`: multi media, cross modal process package。
1. `diffusers`: modified diffusers package based on [diffusers](https://github.com/huggingface/diffusers)
1. `controlnet_aux`: modified based on [controlnet_aux](https://github.com/TMElyralab/controlnet_aux)


## download models
```bash
git clone https://huggingface.co/TMElyralab/MuseV ./checkpoints
```
- `motion`: text2video model.
    - `musev/unet`: only has and train `unet` motion module.
    - `musev_referencenet`: train `unet` module, `referencenet`, `IPAdapter`
        - `unet`: `motion` module, which has `to_k`, `to_v` in `Attention` layer refer to `IPAdapter`
        - `referencenet`: similar to `AnimateAnyone`
        - `ip_adapter_image_proj.bin`: images clip emb project layer, refer to `IPAdapter`
- `t2i/sd1.5`: text2image model, paramter are frozen when training motion module.
    - majicmixRealv6Fp16: example, could be replaced with other t2i base. download from [majicmixRealv6Fp16](https://civitai.com/models/43331/majicmix-realistic)
- `IP-Adapter/models`: download from [IPAdapter](https://huggingface.co/h94/IP-Adapter/tree/main)
    - `image_encoder`: vision clip model.
    - `ip-adapter_sd15.bin`: original IPAdapter model checkpoint.
    - `ip-adapter-faceid_sd15.bin`: original IPAdapter model checkpoint.

## Inference

### prepare model_path
skip this step when run example task with example inference command.
set model path and abbreviation in config, to use abbreviation in inference script.
- T2I SD：ref to `musev/configs/model/T2I_all_model.py`
- Motion Unet: refer to `musev/configs/model/motion_model.py`
- task: refer to `musev/configs/tasks/example.yaml`

### musev_referencenet
#### text2video
```bash
python scripts/inference/text2video.py   --sd_model_name majicmixRealv6Fp16   --unet_model_name musev_referencenet --referencenet_model_name musev_referencenet --ip_adapter_model_name musev_referencenet   -test_data_path ./configs/tasks/example.yaml  --output_dir ./output  --n_batch 1  --target_datas yongen  --vision_clip_extractor_class_name ImageClipVisionFeatureExtractor --vision_clip_model_path ./checkpoints/IP-Adapter/models/image_encoder  --time_size 12 --fps 12  
```
**common parameters**:
- `test_data_path`: task_path in yaml extention
- `target_datas`: sep is `,`, sample subtasks if `name` in `test_data_path` is in `target_datas`.
- `sd_model_cfg_path`: T2I sd models path, model config path or model path.
- `sd_model_name`: sd model name, which use to choose full model path in sd_model_cfg_path. multi model names with sep =`,`, or `all`
- `unet_model_cfg_path`: motion unet model config path or model path。
- `unet_model_name`: unet model name, use to get model path in `unet_model_cfg_path`, and init unet class instance in `musev/models/unet_loader.py`. multi model names with sep=`,`, or `all`. If `unet_model_cfg_path` is model path, `unet_name` must be supported in `musev/models/unet_loader.py`
- `time_size`: num_frames per diffusion denoise generation。default=`12`.
- `n_batch`: generation numbers. Total_frames=n_batch*time_size+n_viscond, default=`1`。
- `context_frames`: context_frames num. If `time_size` > `context_frame`，`time_size` window is split into many sub-windows for parallel denoising"。 default=`12`。

**model parameters**：
support `referencenet`, `IPAdapter`, `IPAdapterFaceID`, `Facein`.
- referencenet_model_name: `referencenet` model name.
- ImageClipVisionFeatureExtractor: `ImageEmbExtractor` name, extractor vision clip emb used in `IPAdapter`.
- vision_clip_model_path: `ImageClipVisionFeatureExtractor` model path.
- ip_adapter_model_name: from `IPAdapter`, it's `ImagePromptEmbProj`, used with `ImageEmbExtractor`。
- ip_adapter_face_model_name: `IPAdapterFaceID`, from `IPAdapter` to keep faceid，should set `face_image_path`。

**Some parameters that affect the amplitude and effect of generation**：
- `video_guidance_scale`: similar to text2image, control influence between cond and uncond，default=`3.5`
- `guidance_scale`:  The parameter ratio in the first frame image between cond and uncond, default=`3.5`
- `use_condition_image`:  Whether use the given first frame for video generation.
- `redraw_condition_image`: whether redraw the given first frame image.
- `video_negative_prompt`: abbreviation of full `negative_prompt` in config path. default=`V2`.


#### video2video
```bash
python scripts/inference/video2video.py --sd_model_name majicmixRealv6Fp16  --unet_model_name musev_referencenet --referencenet_model_name   musev_referencenet --ip_adapter_model_name musev_referencenet    -test_data_path ./configs/tasks/example.yaml    --vision_clip_extractor_class_name ImageClipVisionFeatureExtractor --vision_clip_model_path ./checkpoints/IP-Adapter/models/image_encoder      --output_dir ./output  --n_batch 1 --controlnet_name dwpose_body_hand  --which2video "video_middle"  --target_datas  bilibili_queencard   --fps 12 --time_size 12
```
**import parameters**

Most of paramters are same as `musev_text2video`. Special parameters of `video2video` are
1. need set `video_path` in `test_data`. Now support `rgb video` and `controlnet_middle_video`。
- `need_video2video`: whether `rgb` video influence initial noise.
- `controlnet_name`：whether use `controlnet condition`, such as `dwpose,depth`.
- `video_is_middle`: `video_path` is `rgb video` or  `controlnet_middle_video`. could set for every `test_data` in test_data_path.
- `video_has_condition`: whether condtion_images is aligned with the first frame of video_path. If Not, firstly generate `condition_images` and align with concatation. set in  `test_data`。

### musev
#### text2video
```bash
python scripts/inference/text2video.py   --sd_model_name majicmixRealv6Fp16   --unet_model_name musev   -test_data_path ./configs/tasks/example.yaml  --output_dir ./output  --n_batch 1  --target_datas yongen  --time_size 12 --fps 12
```
#### video2video
```bash
python scripts/inference/video2video.py --sd_model_name majicmixRealv6Fp16  --unet_model_name musev    -test_data_path ./configs/tasks/example.yaml --output_dir ./output  --n_batch 1 --controlnet_name dwpose_body_hand  --which2video "video_middle"  --target_datas  bilibili_queencard   --fps 12 --time_size 12
```


### Gradio demo
MuseV provides gradio script to generate GUI in local machine to generate video conveniently. 

```bash
cd scripts/gradio
python app.py
```

# Acknowledgements

MuseV builds on `TuneAVideo`, `diffusers`. Thanks  for open-sourcing!

<!-- # Contribution 暂时不需要组织开源共建 -->

# Citation
**paper comming soon**
```bib
@article{musev,
  title={MuseV: Infinite-length and High Fidelity Virtual Human Video Generation with Parallel Denoising},
  author={Xia, Zhiqiang and Chen, Zhaokang and Wu, Bin and Li, Chao and Hung, Kwok-Wai and Zhan, Chao and Zhou, Wenjiang},
  journal={arxiv},
  year={2024}
}
```
# Disclaimer/License
1. `code`: The code of MuseV is released under the MIT License. There is no limitation for both academic and commercial usage.
1. `model`: The trained model are available for non-commercial research purposes only.
1. `other opensource model`: Other open-source models used must comply with their license, such as `insightface`, `IP-Adapter`, `ft-mse-vae`, etc.
1. `AIGC`: This project strives to impact the domain of AI-driven video generation positively. Users are granted the freedom to create videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.
