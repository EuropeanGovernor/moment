#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --dataset ETTh2 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 3e-3 \
    --head_lr 3e-3 \
    --max_epoch 20 \
    --scale_weight_lr 1e-4 \
    --weight_decay 1e-3 \
    --pred_length 96 \
    --patience 5 \
    --note 'ETTh2_96_lr_3e-3_hlr_3e-3_wlr_1e-4_wd_1e-3(optim changed)'