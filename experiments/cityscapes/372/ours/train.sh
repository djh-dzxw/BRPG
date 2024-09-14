#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='372_semi'
ROOT=../../../..

mkdir -p log
export CUDA_VISIBLE_DEVICES=0,1,2,3
# use torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$2 \
    $ROOT/train_semi.py --config=config.yaml --seed 2 --port $2 2>&1 | tee log/seg_$now.txt
