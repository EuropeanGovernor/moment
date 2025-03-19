#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 
python finetune.py \
    --dataset ETTh1 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 3e-3 \
    --max_epoch 3 \
    --scale_weight_lr 3e-3 \
    --pred_length 96 \
    --head_dropout 0.2 \
    --note 'ETTh1_96_epoch3_lr_3e-3_wlr_3e-3_drop_0.2'