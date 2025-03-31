#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --version small \
    --dataset ETTh1 \
    --train_bs 8 \
    --eval_bs 8 \
    --init_lr 5e-7 \
    --head_lr 1e-4 \
    --max_epoch 10 \
    --scale_weight_lr 1e-3 \
    --head_dropout 0.10 \
    --pred_length 96 \
    --patience 5 \
    --weight_decay 1e-4 \
    --note 'Small_ETTh1_96_lr_5e-7_hlr_1e-4_wlr_1e-3_wdc_1e-4'