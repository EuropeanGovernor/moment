#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --dataset ETTh1 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 1e-6 \
    --head_lr 1e-4 \
    --max_epoch 50 \
    --scale_weight_lr 1e-4 \
    --head_dropout 0.10 \
    --pred_length 96 \
    --patience 5 \
    --weight_decay 1e-4 \
    --note 'ETTh1_96_lr_1e-6_hlr_1e-4_wlr_1e-4_wdc_1e-4'