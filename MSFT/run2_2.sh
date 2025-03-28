#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python finetune.py \
    --dataset ETTh2 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 1e-4 \
    --head_lr 1e-4 \
    --max_epoch 20 \
    --scale_weight_lr 5e-5 \
    --weight_decay 1e-5 \
    --head_dropout 0.15 \
    --pred_length 96 \
    --patience 5 \
    --note 'ETTh2_96_lr_1e-4_hlr_1e-4_wlr_5e-5_drop_0.15_wd_1e-5'