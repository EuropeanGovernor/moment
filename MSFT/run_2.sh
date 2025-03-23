#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --dataset ETTh1 \
    --train_bs 4 \
    --eval_bs 4 \
    --init_lr 3e-3 \
    --head_lr 1e-3 \
    --max_epoch 20 \
    --scale_weight_lr 1e-2 \
    --head_dropout 0.10 \
    --pred_length 96 \
    --patience 5 \
    --note 'ETTh1_96_lr_3e-3_hlr_1e-3_wlr_1e-2(optim changed)'