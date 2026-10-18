# Orthographic Multi-View World Models — experiment commands
# Run from the ORTHOWORLDS/ root. Copy-paste one block at a time.
# Each training run saves to the same checkpoint folder, so move it right after the run finishes.
mkdir -p checkpoints/runs

# =============================================================================
# 1. MAIN (Tab. 1, Tab. 3, rollout figure)
# =============================================================================

# Ours (FPV + supporting views, projection loss lambda=1)
CUDA_VISIBLE_DEVICES=0 python ./ortho/train_voxel_wan.py --rollout_steps=2048 --epochs=20 --batch_size=16 --env "all" --teacher_forcing_prob=1.0 --steps_per_epoch=200 --num_envs=12 --lr=3e-4 --cons_weight=1.0
mv checkpoints/wan_ortho_voxel_cam_12_lora_final checkpoints/runs/ours

# Ours without projection loss (lambda=0)
CUDA_VISIBLE_DEVICES=0 python ./ortho/train_voxel_wan.py --rollout_steps=2048 --epochs=20 --batch_size=16 --env "all" --teacher_forcing_prob=1.0 --steps_per_epoch=200 --num_envs=12 --lr=3e-4 --cons_weight=0
mv checkpoints/wan_ortho_voxel_cam_12_lora_final checkpoints/runs/ours_lambda0

# FPV-only baseline
CUDA_VISIBLE_DEVICES=0 python ./ortho/train_single_wan.py --rollout_steps=2048 --epochs=20 --batch_size=16 --env "all" --teacher_forcing_prob=1.0 --steps_per_epoch=200 --num_envs=12 --lr=3e-4
mv checkpoints/wan_single_cam_12_lora_final checkpoints/runs/fpv_only

# Eval: all training + held-out environments, 32-step rollouts
CUDA_VISIBLE_DEVICES=0 python ./ortho/eval_longrollout.py --tag main --envs all --rollout_steps 32 --n_rollouts 20 --multi ./checkpoints/runs/ours ./checkpoints/runs/ours_lambda0 --single ./checkpoints/runs/fpv_only

# =============================================================================
# 2. ABLATIONS (Tab. 4)
# =============================================================================
# "No projection loss" = ours_lambda0 and "FPV only" = fpv_only, both from main expts.
# "Jointly trained probe" and "FPV + top only" need new train_voxel_wan.py options (not added yet).

# =============================================================================
# 3. LAMBDA SWEEP (lambda figure) — lambda=0 and lambda=1 come from main expts
# =============================================================================

for L in 0.1 0.3 3; do
    CUDA_VISIBLE_DEVICES=0 python ./ortho/train_voxel_wan.py --rollout_steps=2048 --epochs=20 --batch_size=16 --env "all" --teacher_forcing_prob=1.0 --steps_per_epoch=200 --num_envs=12 --lr=3e-4 --cons_weight=$L
    mv checkpoints/wan_ortho_voxel_cam_12_lora_final checkpoints/runs/lambda_$L
done

# Eval: ablations + lambda sweep, training environments, 16-step rollouts (Tab. 4 and lambda figure)
CUDA_VISIBLE_DEVICES=0 python ./ortho/eval_longrollout.py --tag ablations --envs train --rollout_steps 16 --n_rollouts 20 --multi ./checkpoints/runs/ours ./checkpoints/runs/ours_lambda0 ./checkpoints/runs/lambda_0.1 ./checkpoints/runs/lambda_0.3 ./checkpoints/runs/lambda_3 --single ./checkpoints/runs/fpv_only

# =============================================================================
# 4. ENVIRONMENT SCALING (scaling figure) — N=12 is "ours"
# =============================================================================

for N in 1 3 6; do
    CUDA_VISIBLE_DEVICES=0 python ./ortho/train_voxel_wan.py --rollout_steps=2048 --epochs=20 --batch_size=16 --env "all" --teacher_forcing_prob=1.0 --steps_per_epoch=200 --num_envs=$N --lr=3e-4
    mv checkpoints/wan_ortho_voxel_cam_${N}_lora_final checkpoints/runs/envs_$N
done

# Eval: held-out environments (with and without obstacles)
CUDA_VISIBLE_DEVICES=0 python ./ortho/eval_longrollout.py --tag scaling --envs heldout --rollout_steps 32 --n_rollouts 20 --multi ./checkpoints/runs/envs_1 ./checkpoints/runs/envs_3 ./checkpoints/runs/envs_6 ./checkpoints/runs/ours --single
