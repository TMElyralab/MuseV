FROM anchorxia/musev:latest

#MAINTAINER 维护者信息
LABEL MAINTAINER="anchorxia, zhanchao"
LABEL Email="anchorxia@tencent.com, zhanchao019@foxmail.com"
LABEL Description="musev gradio image, from docker pull anchorxia/musev:latest"


SHELL ["/bin/bash", "--login", "-c"]

USER root
RUN chown root:root -R /root


RUN which python

RUN  pwd

WORKDIR /root

RUN  pwd

RUN  ls -l
RUN whoami

RUN git clone --recursive https://github.com/TMElyralab/MuseV.git -b deploy


RUN echo "export PYTHONPATH=\${PYTHONPATH}:/root/MuseV" >> ~/.bashrc \
    && echo "export PYTHONPATH=\${PYTHONPATH}:/root/MMCM" >> ~/.bashrc \
    && echo "export PYTHONPATH=\${PYTHONPATH}:/root/diffusers/src" >> ~/.bashrc \
    && echo "export PYTHONPATH=\${PYTHONPATH}:/root/controlnet_aux/src" >> ~/.bashrc 
    

RUN chmod -R 777 /root
RUN mv /root/MuseV/* /

WORKDIR /scripts/gradio

RUN pip install cuid gradio huggingface_hub

RUN mkdir /checkpoints
RUN cat app.py
RUN  ls -l


RUN which python

EXPOSE 7860

CMD ["pwd"]
CMD ["username"]
CMD ["ls -l"]
CMD ["python", "app_test.py"]