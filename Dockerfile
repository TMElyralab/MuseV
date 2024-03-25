FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

#MAINTAINER 维护者信息
LABEL MAINTAINER="anchorxia"
LABEL Email="anchorxia@tencent.com"
LABEL Description="gpu development image, from docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
ARG DEBIAN_FRONTEND=noninteractive

USER root
# 安装必须软件
RUN apt -y update \
    && apt -y upgrade
RUN apt install -y wget git curl tmux cmake htop iotop git-lfs zip \
    && apt install -y autojump \
    && apt install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg \
    && apt clean

SHELL ["/bin/bash", "--login", "-c"]

RUN conda create -n musev python=3.10.6 -y \
    && . /opt/conda/etc/profile.d/conda.sh  \
    && echo "source activate musev" >> ~/.bashrc \
    && conda activate musev \
    && pip install tensorflow==2.12.0 tensorboard==2.12.0 \
    # && pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple -U \
    && pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install ninja==1.11.1 \
    && pip install --no-cache-dir transformers==4.33.1 bitsandbytes==0.41.1 decord==0.6.0 accelerate==0.22.0 xformers==0.0.21 omegaconf einops imageio==2.31.1 \
    && pip install --no-cache-dir pandas h5py matplotlib modelcards==0.1.6 pynvml==11.5.0 black pytest moviepy==1.0.3 torch-tb-profiler==0.4.1 scikit-learn librosa ffmpeg easydict webp mediapipe==0.10.3 \
    && pip install --no-cache-dir cython==3.0.2 easydict gdown infomap==2.7.1 insightface==0.7.3 ipython librosa==0.10.1 onnx==1.14.1 onnxruntime==1.15.1 onnxsim==0.4.33 opencv_python Pillow protobuf==3.20.3 pytube==15.0.0 PyYAML \
    && pip install --no-cache-dir requests scipy six tqdm gradio==3.43.2 cuid albumentations==1.3.1 opencv-contrib-python==4.8.0.76 imageio-ffmpeg==0.4.8 pytorch-lightning==2.0.8 test-tube==0.7.5 \
    && pip install --no-cache-dir timm addict yapf prettytable safetensors==0.3.3 basicsr fvcore pycocotools wandb==0.15.10 wget ffmpeg-python \
    && pip install --no-cache-dir streamlit webdataset kornia==0.7.0 open_clip_torch==2.20.0 streamlit-drawable-canvas==0.9.3 torchmetrics==1.1.1 \
    # 安装暗水印
    && pip install --no-cache-dir invisible-watermark==0.1.5 gdown==4.5.3 ftfy==6.1.1 modelcards==0.1.6 \
    # jupyters
    && pip install ipywidgets==8.0.3 \
    && python -m ipykernel install --user --name projectv --display-name "python(projectv)" \
    && pip install --no-cache-dir matplotlib==3.6.2 redis==4.5.1  pydantic[dotenv]==1.10.2 loguru==0.6.0 IProgress==0.4 \
    && pip install git+https://github.com/tencent-ailab/IP-Adapter.git \
    && pip install -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.1" \
    && mim install "mmdet>=3.1.0" \
    && mim install "mmpose>=1.1.0" \
    # 必须放在最后pip，避免和jupyter的不兼容
    && pip install --no-cache-dir  markupsafe==2.0.1\

USER root
