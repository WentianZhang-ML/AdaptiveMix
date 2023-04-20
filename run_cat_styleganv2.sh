#!/usr/bin/env bash
export gpu_device='0,1,2,3'
export output_dir='generation/experiments/AdaptiveMix_cat_StyleGAN_V2'
export data_dir='afhq-cat_5k.zip'

CUDA_VISIBLE_DEVICES=${gpu_device} python3 generation/train.py \
--outdir=${output_dir} \
--gpus=4 \
--data=${data_dir} \
--mirror=1 \
--cfg=paper256 \
--aug=noaug \
--adaptivemix=true \
--noise_std=0.05;