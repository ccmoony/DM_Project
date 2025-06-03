#!/bin/bash

#SBATCH -p RTX3090
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres gpu:2

DATASET=game


accelerate launch --config_file accelerate_config_ddp.yaml main.py \
    --config ./config/${DATASET}.yaml \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --cycle=2 \
    --eval_step=2 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003 \
    --interest_fusion=true \
    # --debug