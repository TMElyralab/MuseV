FROM anchorxia/musev:1.0.0

#MAINTAINER 维护者信息
LABEL MAINTAINER="anchorxia"
LABEL Email="anchorxia@tencent.com"
LABEL Description="musev gpu runtime image, base docker is pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
ARG DEBIAN_FRONTEND=noninteractive

USER root

SHELL ["/bin/bash", "--login", "-c"]

RUN . /opt/conda/etc/profile.d/conda.sh  \
    && echo "source activate musev" >> ~/.bashrc \
    && conda activate musev \
    && conda env list \
    && pip --no-cache-dir install cuid gradio==4.12 spaces
USER root
