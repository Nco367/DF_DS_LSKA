#!/bin/bash


export PYTHONUNBUFFERED=True
export CUDA_VISIBLE_DEVICES=0

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

/conda_envs/df6d_pvn3d/bin/python ./tools/train_linemod5.py --dataset linemod\
  --dataset_root /media/q/SSD2T/1linux/Linemod5_dataset/Linemod_preprocessed