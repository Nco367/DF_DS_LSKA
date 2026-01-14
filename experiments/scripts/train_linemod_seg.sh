#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./vanilla_segmentation/train.py --dataset linemod\
  --dataset_root /media/q/HDD3T_1.5TB/2linux/datebase/DenseFusiondatasets/linemod/Linemod_preprocessed/