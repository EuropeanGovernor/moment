#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python finetune.py \
    --dataset ETTh2 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 1e-3 \
    --max_epoch 3 \
    --scale_weight_lr 1e-3 \
    --pred_length 96 \
    --note 'ETTh2_96_epoch3_lr_1e-3_wlr_1e-3'