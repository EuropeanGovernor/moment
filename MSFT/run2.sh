#!/bin/bash

python finetune.py \
    --dataset ETTh2 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 3e-4 \
    --max_epoch 3 \
    --scale_weight_lr 1e-3 \
    --pred_length 96 \
    --note '*3_ETTh2_96_epoch3_lr_3e-4_wlr_1e-3'