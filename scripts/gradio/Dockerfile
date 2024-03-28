FROM anchorxia/musev:1.0.1

#MAINTAINER 维护者信息
LABEL MAINTAINER="anchorxia"
LABEL Email="anchorxia@tencent.com, zhanchao019@foxmail.com"
LABEL Description="musev gradio image, from docker pull anchorxia/musev:latest"
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "--login", "-c"]


WORKDIR /root
RUN git clone --recursive https://github.com/TMElyralab/MuseV.git
RUN echo "export PYTHONPATH=\${PYTHONPATH}:/root/MuseV" >> ~/.bashrc \
    && echo "export PYTHONPATH=\${PYTHONPATH}:/root/MuseV/MMCM" >> ~/.bashrc \
    && echo "export PYTHONPATH=\${PYTHONPATH}:/root/MuseV/diffusers/src" >> ~/.bashrc \
    && echo "export PYTHONPATH=\${PYTHONPATH}:/root/MuseV/controlnet_aux/src" >> ~/.bashrc 


WORKDIR /root/MuseV/scripts/gradio

CMD ["python", "app.py"]