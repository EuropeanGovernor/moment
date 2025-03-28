#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --dataset ETTh1 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 3e-7 \
    --head_lr 1e-4 \
    --max_epoch 20 \
    --scale_weight_lr 1e-4 \
    --head_dropout 0.10 \
    --pred_length 96 \
    --patience 5 \
    --weight_decay 1e-4 \
    --note 'ETTh1_96_lr_3e-7_hlr_1e-4_wlr_1e-4_wdc_1e-4'