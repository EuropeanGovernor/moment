#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --dataset ETTh1 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 1e-3 \
    --head_lr 1e-3 \
    --max_epoch 20 \
    --lora True \
    --linear False \
    --scale_weight_lr 1e-4 \
    --pred_length 96 \
    --patience 5 \
    --note 'wolinear_ETTh1_96_epoch20_lr_1e-3_wlr_1e-4_hlr_1e-3'