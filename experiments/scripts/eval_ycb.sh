#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d YCB_Video_toolbox-master ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi

python3 ./tools/eval_ycb.py --dataset_root /media/q/新加卷/BaiduYun/YCB_Video_Dataset\
  --model ./trained_models/linemod_best/pose_model_6_0.0128862230529978.pth\
  --refine_model ./trained_models/linemod_best/pose_refine_model_295_0.00516647388855369.pth