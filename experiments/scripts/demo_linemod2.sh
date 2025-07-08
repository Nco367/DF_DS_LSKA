#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod2.py --dataset_root /media/q/SSD2T/1linux/Linemod2_dataset/Linemod_preprocessed\
  --model /media/q/SSD2T/1linux/Linemod2_trained_mod/pose_model_2_0.008679830040346132.pth\
  --refine_model /media/q/SSD2T/1linux/Linemod2_trained_mod/pose_refine_model_10_0.003791075829140027.pth