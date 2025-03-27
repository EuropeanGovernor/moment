#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --version small \
    --dataset ETTh1 \
    --train_bs 32 \
    --eval_bs 32 \
    --init_lr 1e-3 \
    --head_lr 1e-4 \
    --max_epoch 50 \
    --scale_weight_lr 1e-4 \
    --head_dropout 0.10 \
    --pred_length 96 \
    --patience 5 \
    --weight_decay 1e-4 \
    --note 'Small_ETTh1_96_lr_1e-3_hlr_1e-4_wlr_1e-4_wdc_1e-4'