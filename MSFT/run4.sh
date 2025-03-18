#!/bin/bash

python finetune.py \
    --dataset ETTm2 \
    --train_bs 8 \
    --eval_bs 8 \
    --init_lr 1e-3 \
    --max_epoch 3 \
    --scale_weight_lr 3e-4 \
    --pred_length 96 \
    --note '*ETTm2_96_epoch3_lr_1e-3_wlr_3e-4'