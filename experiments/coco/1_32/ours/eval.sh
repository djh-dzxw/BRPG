#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='eval'
ROOT=../../../..
export CUDA_VISIBLE_DEVICES=0
mkdir -p log
mkdir -p checkpoints/results
python $ROOT/eval.py \
    --config=config.yaml \
    --base_size 640 \
    --scales 1.0 \
    --model_path=checkpoints/ckpt_best.pth \
    --save_folder=checkpoints/results \
    2>&1 | tee log/val_best_$(date +"%Y%m%d_%H%M%S").txt