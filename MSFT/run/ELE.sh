#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

version=small
bs=2
lr=5e-7
head_lr=5e-5
scale_weight_lr=1e-3
max_epoch=5
num_new_scales=3

for pl in 96 192 336 720; do
  note="${version}_ELE_${pl}_NS${num_new_scales}_lr${lr}_hlr${head_lr}_wlr${scale_weight_lr}_bs${bs}_ep${max_epoch}"
  python finetune.py \
      --version "${version}" \
      --dataset electricity \
      --train_bs "${bs}" \
      --eval_bs "${bs}" \
      --init_lr "${lr}" \
      --head_lr "${head_lr}" \
      --max_epoch "${max_epoch}" \
      --scale_weight_lr "${scale_weight_lr}" \
      --pred_length "${pl}" \
      --num_new_scales "${num_new_scales}" \
      --lora true \
      --linear true \
      --pred_mask_tokens false \
      --note "${note}"
done