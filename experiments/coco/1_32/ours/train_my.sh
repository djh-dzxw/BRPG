
#now=$(date +"%Y%m%d_%H%M%S")
#job='1464_semi'
#ROOT=../../../..
mkdir -p log

# use torch.distributed.launch
#python -m torch.distributed.launch \
#    --nproc_per_node=4 \
#    $ROOT/train_semi.py --config=config.yaml --seed 2 2>&1 | tee log/seg_$now.txt

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 /raid/djh/Semi-supervised-segmentation/U2PL-main/train_semi.py --config config.yaml --seed 2 2>&1 | tee log/seg_20230129.txt