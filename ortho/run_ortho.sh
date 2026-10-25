# Orthographic Multi-View World Models — experiment commands
# Run from the ORTHOWORLDS/ root. Copy-paste one block at a time.
# Each training run saves to the same checkpoint folder, so move it right after the run finishes.
mkdir -p checkpoints/runs

# Every training run uses COMMON, so the FPV-only and multi-view arms differ only in --views
# (plus ORTHO_FLAGS, which only affect the multi-view arm).
COMMON="--rollout_steps=2048 --epochs=20 --batch_size=16 --teacher_forcing_prob=1.0 --steps_per_epoch=200 --lr=3e-4"
ORTHO_FLAGS="--target_order=same"   # set from the diagnostics in section 0

# =============================================================================
# 0. DIAGNOSTICS — short runs to choose ORTHO_FLAGS before the main comparison
# =============================================================================
# Compare Train/Loss_3D_FPV at equal steps and Validation/*/PSNR/FPV in TensorBoard (runs/). Choose from these,
# never from the held-out eval in section 1. --teacher_forcing_prob=0 keeps self-predicted contexts out of it.
DIAG="--rollout_steps=2048 --epochs=4 --batch_size=16 --teacher_forcing_prob=0 --steps_per_epoch=200 --lr=3e-4 --num_envs=12 --seed=0 --save_every=1000"
CUDA_VISIBLE_DEVICES=0 python ./ortho/new_train_ortho_wan.py $DIAG --views=fpv                        # reference
CUDA_VISIBLE_DEVICES=0 python ./ortho/new_train_ortho_wan.py $DIAG --views=all                        # current layout
CUDA_VISIBLE_DEVICES=0 python ./ortho/new_train_ortho_wan.py $DIAG --views=all --target_order=mirror  # FPV target next to FPV context

# =============================================================================
# 1. MAIN: FPV-only vs multi-view (Tab. 1, Tab. 3, rollout figure)
# =============================================================================
# Same script, data, steps and eval for both arms; for a given seed both arms train on the same buffer.
# Matched in steps and data, not compute: the multi-view arm processes 4x the tokens per step.
for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 python ./ortho/new_train_ortho_wan.py $COMMON --num_envs=12 --seed=$S --views=fpv
    mv checkpoints/wan_single_cam_12_lora_final checkpoints/runs/fpv_s$S
    CUDA_VISIBLE_DEVICES=0 python ./ortho/new_train_ortho_wan.py $COMMON --num_envs=12 --seed=$S --views=all $ORTHO_FLAGS
    mv checkpoints/wan_ortho_cam_12_lora_final checkpoints/runs/ortho_s$S
done

# Eval: all training + held-out environments, 32-step rollouts; report mean +- std over the seed rows of summary.csv
CUDA_VISIBLE_DEVICES=0 python ./ortho/eval_longrollout.py --tag main --envs all --rollout_steps 32 --n_rollouts 20 \
    --multi ./checkpoints/runs/ortho_s0 ./checkpoints/runs/ortho_s1 ./checkpoints/runs/ortho_s2 \
    --single ./checkpoints/runs/fpv_s0 ./checkpoints/runs/fpv_s1 ./checkpoints/runs/fpv_s2

# =============================================================================
# 2. ENVIRONMENT SCALING (scaling figure) — N=12 is ortho_s0 from section 1
# =============================================================================

for N in 1 3 6; do
    CUDA_VISIBLE_DEVICES=0 python ./ortho/new_train_ortho_wan.py $COMMON --num_envs=$N --seed=0 --views=all $ORTHO_FLAGS
    mv checkpoints/wan_ortho_cam_${N}_lora_final checkpoints/runs/envs_$N
done

# Eval: held-out environments (with and without obstacles)
CUDA_VISIBLE_DEVICES=0 python ./ortho/eval_longrollout.py --tag scaling --envs heldout --rollout_steps 32 --n_rollouts 20 --multi ./checkpoints/runs/envs_1 ./checkpoints/runs/envs_3 ./checkpoints/runs/envs_6 ./checkpoints/runs/ortho_s0 --single
