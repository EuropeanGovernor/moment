#!/bin/bash

python finetune.py \
    --dataset ETTh1 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 1e-3 \
    --max_epoch 3 \
    --scale_weight_lr 1e-3 \
    --pred_length 96 \
    --note '*4_ETTh1_96_epoch3_lr_1e-3_wlr_1e-3'