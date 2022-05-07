##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Sun 01 May 2022 02:52:00 AM PDT
##################################################

#!/bin/bash

depth='196'
width='1'
epochs='200'
gpu_id='4'
# python run.py --out runs/vit-6-1_epoch=200 \
python run.py --out runs/wrn-"$depth"-"$width"_epoch="$epochs" \
              --arch wrn \
              --depth "$depth" \
              --width "$width" \
              --epochs "$epochs" \
              --gpu-id "$gpu_id" \
