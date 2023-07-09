#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="LOG_SS_"$now"_$1"
CUDA_VISIBLE_DEVICES=0 python -u train_classification_stage2.py --name $1 --arch $2 --dataset $3 --config cfgs/config.yaml 2>&1|tee log/$log_name.log
