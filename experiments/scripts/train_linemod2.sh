#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train_linemod2.py --dataset linemod\
  --dataset_root /media/q/SSD2T/1linux/Linemod2_dataset/Linemod_preprocessed