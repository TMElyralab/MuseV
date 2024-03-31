#!/bin/bash

echo "entrypoint.sh"
whoami
which python
export PYTHONPATH=${PYTHONPATH}:/home/user/app/MuseV:/home/user/app/MuseV/MMCM:/home/user/app/MuseV/diffusers/src:/home/user/app/MuseV/controlnet_aux/src
echo "pythonpath" $PYTHONPATH
# chmod 777 -R /home/user/app/MuseV
# Print the contents of the diffusers/src directory
# echo "Contents of /home/user/app/MuseV/diffusers/src:"
# Load ~/.bashrc
# source ~/.bashrc

source /opt/conda/etc/profile.d/conda.sh
conda activate musev
which python
python ap_space.py