#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python finetune.py \
    --dataset ETTh1 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 5e-5 \
    --head_lr 1e-4 \
    --max_epoch 20 \
    --scale_weight_lr 1e-4 \
    --pred_length 96 \
    --patience 5 \
    --note 'ETTh1_96_epoch20_lr_5e-5_wlr_1e-4_hlr_1e-4'