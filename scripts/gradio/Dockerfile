FROM anchorxia/musev:latest

#MAINTAINER 维护者信息
LABEL MAINTAINER="anchorxia, zhanchao"
LABEL Email="anchorxia@tencent.com, zhanchao019@foxmail.com"
LABEL Description="musev gradio image, from docker pull anchorxia/musev:latest"

SHELL ["/bin/bash", "--login", "-c"]

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

RUN echo "docker start"\ 
  && whoami \
  && which python \
  && pwd

RUN git clone -b hg_space --recursive https://github.com/TMElyralab/MuseV.git
# RUN mkdir ./MuseV/checkpoints \
#   && ls -l ./MuseV
RUN chmod -R 777 /home/user/app/MuseV

# RUN git clone -b main https://huggingface.co/TMElyralab/MuseV /home/user/app/MuseV/checkpoints

RUN . /opt/conda/etc/profile.d/conda.sh  \
    && echo "source activate musev" >> ~/.bashrc \
    && conda activate musev \
    && conda env list

RUN echo "export PYTHONPATH=\${PYTHONPATH}:/home/user/app/MuseV:/home/user/app/MuseV/MMCM:/home/user/app/MuseV/diffusers/src:/home/user/app/MuseV/controlnet_aux/src" >> ~/.bashrc

WORKDIR /home/user/app/MuseV/scripts/gradio/

# Add entrypoint script
COPY --chown=user entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
RUN ls -l  ./

EXPOSE 7860

# CMD ["/bin/bash", "-c", "python app.py"]
CMD ["./entrypoint.sh"]