#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python finetune.py \
    --dataset ETTh2 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 5e-5 \
    --max_epoch 20 \
    --scale_weight_lr 5e-5 \
    --pred_length 96 \
    --patience 5 \
    --note 'ETTh2_96_epoch20_lr_5e-5_wlr_5e-5'