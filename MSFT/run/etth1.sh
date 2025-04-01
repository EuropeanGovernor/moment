#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

version=small
bs=32
lr=5e-7
head_lr=1e-4
scale_weight_lr=1e-3
max_epoch=20
num_new_scales=3
lora=true
linear=true
mask=false

for pl in 96; do  #  192 336 720
  note="${pl}_NS${num_new_scales}_lr${lr}_hlr${head_lr}_wlr${scale_weight_lr}_bs${bs}_ep${max_epoch}"

  if [ "$lora" = true ]; then
    note="${note}_lora"
  fi

  if [ "$linear" = true ]; then
    note="${note}_linear"
  fi

  if [ "$mask" = true ]; then
    note="${note}_mask"
  fi

  python finetune.py \
      --version "${version}" \
      --dataset ETTh1 \
      --train_bs "${bs}" \
      --eval_bs "${bs}" \
      --init_lr "${lr}" \
      --head_lr "${head_lr}" \
      --max_epoch "${max_epoch}" \
      --scale_weight_lr "${scale_weight_lr}" \
      --pred_length "${pl}" \
      --num_new_scales "${num_new_scales}" \
      --lora "${lora}" \
      --linear "${linear}" \
      --pred_mask_tokens "${mask}" \
      --note "${note}"
done