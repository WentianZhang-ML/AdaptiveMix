#!/usr/bin/env bash
export gpu_device='0,1,2,3'
export output_dir='generation/styleganv2/experiments/AdaptiveMix_FFHQ_70k_StyleGAN_V2'
export data_dir='ffhq_256_full.zip'

CUDA_VISIBLE_DEVICES=${gpu_device} python3 generation/train.py \
--outdir=${output_dir} \
--gpus=4 \
--data=${data_dir} \
--mirror=1 \
--cfg=paper256 \
--aug=noaug \
--adaptivemix=true \
--noise_std=0.05 ;


# --freezed=2 \
# --resume=/apdcephfs/share_1290796/waltszhang/AdaptiveMix/Generation/ffhq-res256-mirror-paper256-noaug.pkl \
# --snap=10 \