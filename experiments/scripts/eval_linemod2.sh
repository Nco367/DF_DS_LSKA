#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod2.py --dataset_root /media/q/SSD2T/1linux/Linemod2_dataset/Linemod_preprocessed\
  --model /media/q/SSD2T/1linux/Linemod2_trained_mod/pose_model_current.pth\
  --refine_model /media/q/SSD2T/1linux/Linemod2_trained_mod/pose_refine_model_1_0.006460016133993728.pth