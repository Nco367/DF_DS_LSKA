#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod.py --dataset_root /media/q/HDD3T_1.5TB/2linux/datebase/DenseFusiondatasets/linemod/Linemod_preprocessed\
  --model ./trained_models/linemod_best/pose_model_6_0.0128862230529978.pth\
  --refine_model ./trained_models/linemod_best/pose_refine_model_295_0.00516647388855369.pth