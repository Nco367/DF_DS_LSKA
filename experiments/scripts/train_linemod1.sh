#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train_linemod1.py --dataset linemod\
  --dataset_root /media/q/SSD2T/1linux/Linemod1_dataset/Linemod_preprocessed