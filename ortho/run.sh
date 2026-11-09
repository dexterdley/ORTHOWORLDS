#!/usr/bin/env bash

NUM_GPU=4
GPUS=0,1,2,3
TORCHRUN="torchrun --standalone --nproc_per_node=$NUM_GPU"

# Shared by every arm, single-GPU and DDP
COMMON="--epochs=20 --steps_per_epoch=200 \
        --teacher_forcing_prob=0.5 \
        --lr=3e-4 --eval_steps=16 \
        --model_id=./Wan2.1-T2V-1.3B-Diffusers/"

DIST="--rollout_steps=4096 --batch_size=4"

for ENVS in 12; do
  for SEED in 42; do
    # CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN ./ortho/train_single_dist.py $COMMON $DIST --seed=$SEED --num_envs=$ENVS   # FPV-Single (DDP)
    # CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN ./ortho/train_ortho_dist.py  $COMMON $DIST --seed=$SEED --num_envs=$ENVS   # Ortho-Base (DDP)
    CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN ./ortho/train_ortho_project_dist.py $COMMON $DIST --seed=$SEED --num_envs=$ENVS   # Ortho-Project (DDP)
  done
done

# Single-GPU
# SINGLE="--rollout_steps=4096 --batch_size=16"
# python ./ortho/train_single_wan.py    $COMMON $SINGLE --seed=$S   # FPV-Single
# python ./ortho/train_ortho_base.py    $COMMON $SINGLE --seed=$S   # Ortho-Base
# python ./ortho/train_ortho_project.py $COMMON $SINGLE --seed=$S   # Ortho-Project