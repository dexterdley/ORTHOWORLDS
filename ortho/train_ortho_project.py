"""
Fine-tuning Wan2.1-1.3B with LoRA for Orthographic Multi-View World Modeling.

Modelling Process

Linear projector: Ortho-Base plus a sketch frame

frame:     0     1     2     3      4      |   5      6      7      8
         Top   Side  Rear   FPV   Sketch   |  Top    Side   Rear   FPV
          |     |     |             ^      |
          +-----+-----+             |      |
                |                   |      |
                v                   |      |
        [ Linear Proj. ] -----------+      |
         ---------- clean context -------- | ---- views_{t+1} (noisy at tau) ----
loss:                                      |   yes    yes    yes    yes

Usage:
CUDA_VISIBLE_DEVICES=3 python ./ortho/train_ortho_project.py --rollout_steps=4096 --epochs=20 --batch_size=32 --teacher_forcing_prob=0.5 --steps_per_epoch=200 --num_envs=3 --lr=3e-4 --model_id=./Wan2.1-T2V-1.3B-Diffusers/
"""

import os
import math
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# HuggingFace & Diffusers imports
from utils import _build_registry, load_model, compose_player_prompts, set_active_env, set_seed
from buffer import OrthoTransitionBuffer
from models import _get_wan_latent_stats, encode_views_to_latents, LinearProjector, SoftZProjector, CrossViewProjector, cam_feats
from data_rollout import collect_all_envs_rollouts
from flow import predict_next_view_latents_sketch as predict_next_view_latents
from validation import evaluate_and_log_videos

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
IMG_H, IMG_W = 96,96

# ==========================================
# MULTI-STEP TRAJECTORY FLOW MATCHING TRAINING
# ==========================================
def train_ortho_flow_step(
    transformer,
    latents_t,
    latents_next,
    prompt_embeds,
    cam,
    shift=3.0,
    ctx_noise_max=0.15,
):
    """
    Single transition Flow-Matching step (used inside multi-step trajectory rollout).
    Applies mild context noise to `latents_t` to prevent autoregressive drift,
    and optionally returns the model's reconstructed clean latent `pred_x0` at t+1.
    """
    b, c, num_views, h, w = latents_next.shape

    # 1. Sample shifted logit-normal flow-matching timestep tau in (0, 1)
    u = torch.randn(b, device=latents_next.device, dtype=torch.float32)
    tau = torch.sigmoid(u)
    tau = (shift * tau) / (1.0 + (shift - 1.0) * tau)
    tau = tau.clamp(1e-4, 1.0 - 1e-4)
    tau_exp = tau.view(b, 1, 1, 1, 1)

    # 2. Context noise augmentation on conditioning views `latents_t`
    latents_t_fp32 = latents_t.float()
    tau_ctx = torch.rand((b, 1, 1, 1, 1), device=latents_t.device, dtype=torch.float32) * ctx_noise_max
    ctx_noise = torch.randn_like(latents_t_fp32)
    cond_latents_t = ((1.0 - tau_ctx) * latents_t_fp32 + tau_ctx * ctx_noise).to(dtype=transformer.dtype)

    sketch_latents_t = transformer.sketch_proj(cond_latents_t[:, :, :3], cam).to(dtype=transformer.dtype)

    # 3. Noise all 4 target views at t+1 simultaneously
    latents_next_fp32 = latents_next.float()
    noise = torch.randn_like(latents_next_fp32)
    noise[:, :, 3:4] += sketch_latents_t.detach().float()
    noisy_latents_next_fp32 = (1.0 - tau_exp) * latents_next_fp32 + tau_exp * noise
    noisy_latents_next = noisy_latents_next_fp32.to(dtype=transformer.dtype)

    # Leave-one-out renders at t+1: each view from the other three (aux loss only)
    sketch_latents_a = transformer.sketch_proj(latents_next_fp32, cam, tgt=0)   # Top  <- Side, Rear, FPV
    sketch_latents_b = transformer.sketch_proj(latents_next_fp32, cam, tgt=1)   # Side <- Top, Rear, FPV
    sketch_latents_c = transformer.sketch_proj(latents_next_fp32, cam, tgt=2)   # Rear <- Top, Side, FPV
    sketch_latents_d = transformer.sketch_proj(latents_next_fp32, cam, tgt=3)   # FPV  <- Top, Side, Rear

    combined_latents = torch.cat([cond_latents_t, sketch_latents_t, noisy_latents_next], dim=2)

    # 5. Ground truth velocity: v_target = noise - x_0
    target_velocity = noise - latents_next_fp32

    # 6. DiT Forward pass
    timesteps = (tau * 1000.0).to(dtype=transformer.dtype)
    model_pred = transformer(
        hidden_states=combined_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
    ).sample

    # 7. Supervise all 4 target views at t+1 (indices 4:8)
    pred_velocity = model_pred[:, :, num_views + 1 :].float()
    per_view_losses = [
        F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
        for idx in range(num_views)
    ]
    loss = sum(per_view_losses)
    # aux_loss = F.mse(sketch_latents_t.float(), latents_next_fp32[:, :, 3:4])
    aux_loss_a = F.l1_loss(sketch_latents_a, latents_next_fp32[:, :, 0:1])
    aux_loss_b = F.l1_loss(sketch_latents_b, latents_next_fp32[:, :, 1:2])
    aux_loss_c = F.l1_loss(sketch_latents_c, latents_next_fp32[:, :, 2:3])
    aux_loss_d = F.l1_loss(sketch_latents_d, latents_next_fp32[:, :, 3:4])
    aux_loss = (aux_loss_a + aux_loss_b + aux_loss_c + aux_loss_d)/4

    loss = loss + aux_loss
    # Reconstruct 1-step denoised prediction: x_0_hat = x_tau - tau * v_pred
    with torch.no_grad():
        pred_x0 = (noisy_latents_next_fp32 - tau_exp * pred_velocity.detach()).clamp(-6.0, 6.0)
        rollout_latent = pred_x0.to(dtype=transformer.dtype)
    return loss, [l.item() for l in per_view_losses], aux_loss.item(), rollout_latent

def train_ortho_trajectory_step(
    transformer,
    latents_t0,
    latents_next_seq,
    prompt_embeds_seq,
    cam_seq,
    shift=3.0,
    teacher_forcing_prob=0.5,
    ctx_noise_max=0.15,
):
    """
    Unrolls a multi-step trajectory of length `seq_len` during training.
    - At step s=0, conditions on ground-truth `latents_t0`.
    - At step s>0, with probability `teacher_forcing_prob`, conditions on the model's own predicted
      latent from step s-1 (student forcing) so the model learns to recover from its
      own rollout errors over multi-step horizons.
    - Backpropagates `(loss_s / seq_len)` at each step so peak VRAM remains constant.
    """
    seq_len = len(latents_next_seq)
    num_views = latents_t0.shape[2]

    curr_latents = latents_t0
    total_loss = 0.0
    avg_per_view = [0.0] * num_views
    avg_sketch = 0.0

    for s in range(seq_len):
        target_latents = latents_next_seq[s]
        prompt_embeds = prompt_embeds_seq[s]
        cam = cam_seq[s]
        is_last_step = (s == seq_len - 1)

        if not is_last_step:
            step_loss, step_view_losses, step_sketch, pred_next_latents = train_ortho_flow_step(
                transformer=transformer,
                latents_t=curr_latents,
                latents_next=target_latents,
                prompt_embeds=prompt_embeds,
                cam=cam,
                shift=shift,
                ctx_noise_max=ctx_noise_max,
            )
        else:
            step_loss, step_view_losses, step_sketch, _ = train_ortho_flow_step(
                transformer=transformer,
                latents_t=curr_latents,
                latents_next=target_latents,
                prompt_embeds=prompt_embeds,
                cam=cam,
                shift=shift,
                ctx_noise_max=ctx_noise_max,
            )

        # Normalize loss across trajectory length and backpropagate immediately to save VRAM
        scaled_loss = step_loss / float(seq_len)
        scaled_loss.backward()

        total_loss += step_loss.item() / float(seq_len)
        avg_sketch += step_sketch / float(seq_len)
        for v_idx in range(num_views):
            avg_per_view[v_idx] += step_view_losses[v_idx] / float(seq_len)

        # Prepare context for step s + 1 (Student Forcing vs. Teacher Forcing per sample)
        if not is_last_step:
            if teacher_forcing_prob > 0.0:
                b = curr_latents.shape[0]
                use_ar_mask = (torch.rand((b, 1, 1, 1, 1), device=curr_latents.device) < teacher_forcing_prob).to(dtype=transformer.dtype)
                #curr_latents = use_ar_mask * pred_next_latents + (1.0 - use_ar_mask) * target_latents
                with torch.no_grad():
                    sampled = predict_next_view_latents(transformer, curr_latents, prompt_embeds, cam, steps=4)
                curr_latents = use_ar_mask * sampled + (1.0 - use_ar_mask) * target_latents
            else:
                curr_latents = target_latents

    return total_loss, avg_per_view, avg_sketch

# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
def main(args):
    set_seed(args.seed)
    print("=" * 60)
    print("  Orthographic Multi-View World Model (LoRA)")
    print(f"  Multi-Step Trajectory Training (seq_len={args.train_seq_len}, teacher_forcing_prob={args.teacher_forcing_prob})")
    print("=" * 60)

    # 1. Environments & Replay Buffer
    registry = _build_registry()
    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(
        capacity=args.buffer_capacity, img_h=IMG_H, img_w=IMG_W, max_action_len=8
    )

    active_envs, agents = collect_all_envs_rollouts(
        registry, buffer, rollout_steps=args.rollout_steps, num_envs=args.num_envs,
        device=device,
    )

    # 2. Model & LoRA Setup
    model, vae, text_encoder, tokenizer = load_model(
        model_id=args.model_id, lora_rank=args.lora_rank
    )
    # model.sketch_proj = LinearProjector(h_lat=IMG_H // 8, w_lat=IMG_W // 8).to(device)
    # model.sketch_proj = SoftZProjector(h_lat=IMG_H // 8, w_lat=IMG_W // 8).to(device)
    model.sketch_proj = CrossViewProjector(h_lat=IMG_H // 8, w_lat=IMG_W // 8).to(device)

    proj_params = list(model.sketch_proj.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("sketch_proj.")]
    trainable_params = lora_params + proj_params
    optimizer = optim.AdamW(
        [{"params": lora_params}, {"params": proj_params, "lr": args.proj_lr}],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    log_dir = f"./runs/wan_project_multienv_cam_{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    view_names = ["Top", "Side", "Rear", "3D_FPV"]

    # 3. Multi-Step Trajectory Training Loop
    print("\nStarting Multi-Step Trajectory LoRA Fine-Tuning ...")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        # Ramp up autoregressive student-forcing probability across epochs
        progress = epoch / max(1, args.epochs - 1)
        curr_teacher_forcing_prob = args.teacher_forcing_prob * min(1.0, progress * 1.5)

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} (Self-pred Prob.={curr_teacher_forcing_prob:.2f})")

        for _ in pbar:
            # Sample batch of contiguous trajectories of length `args.train_seq_len`
            # views_t:       (B, 4, 3, H, W) — anchor frame at t=0
            # views_next_seq: list[T] of (B, 4, 3, H, W) — target frames t=1..T
            views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq = buffer.sample_trajectory_batch(
                batch_size=args.batch_size,
                seq_len=args.train_seq_len,
            )

            with torch.no_grad():
                latents_t = encode_views_to_latents(vae, views_t, out_dtype=model.dtype)

                # Encode target views and text prompts for each step along the trajectory
                latents_next_seq = []
                prompt_embeds_seq = []
                cam_seq = []

                for s in range(args.train_seq_len):
                    latents_s = encode_views_to_latents(vae, views_next_seq[s], out_dtype=model.dtype)
                    latents_next_seq.append(latents_s)

                    prompts_s = compose_player_prompts(
                        actions_seq[s],
                        env_name=env_names_seq[s],
                        camera_pose=camera_poses_seq[s],
                    )

                    text_inputs = tokenizer(
                        prompts_s, padding="max_length", max_length=128, truncation=True, return_tensors="pt"
                    ).to(device)
                    embeds_s = text_encoder(**text_inputs).last_hidden_state.to(dtype=model.dtype)
                    prompt_embeds_seq.append(embeds_s)
                    cam_seq.append(cam_feats(camera_poses_seq[s]))

            optimizer.zero_grad(set_to_none=True)

            # Unroll multi-step trajectory with student forcing & context noise
            loss, per_view_losses, sketch_loss = train_ortho_trajectory_step(
                transformer=model,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                teacher_forcing_prob=curr_teacher_forcing_prob,
                ctx_noise_max=args.ctx_noise_max,
                cam_seq=cam_seq
            )

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss
            global_step += 1

            writer.add_scalar("Train/Loss", loss, global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("Train/Teacher_Student_Prob", curr_teacher_forcing_prob, global_step)
            for v_idx, v_name in enumerate(view_names):
                writer.add_scalar(f"Train/Loss_{v_name}", per_view_losses[v_idx], global_step)
            writer.add_scalar("Train/Loss_Sketch", sketch_loss, global_step)   # aux MSE(sketch, FPV_t+1): should fall below FPV's variance

            env_summary = "+".join(list(dict.fromkeys(env_names_seq[0]))[:2])
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",
                "FPV": f"{per_view_losses[-1]:.3f}",
                "Envs": env_summary,
            })

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"  [Epoch {epoch+1}] Average Trajectory Flow Loss ({args.train_seq_len} steps): {avg_loss:.4f}")

        # --- VALIDATION: AUTOREGRESSIVE VIDEO ROLLOUTS TO TENSORBOARD ---
        if (epoch + 1) % args.eval_every == 0:
            evaluate_and_log_videos(
                transformer=model,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                writer=writer,
                epoch=epoch + 1,
                agents=agents,
                rollout_steps=args.rollout_video_steps,
                steps=args.eval_steps,
                fps=args.video_fps,
                seed=args.seed + 1000,
                img_hw=(IMG_H, IMG_W),
                sketch=True,
                val_envs=["Bipedal Walker", "Lunar Lander", "MultiCar Racing"],
                log_gt=args.log_gt,
            )

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = f"./checkpoints/wan_project_lora_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            torch.save(model.sketch_proj.state_dict(), os.path.join(ckpt_dir, "sketch_proj.pt"))
            print(f"  [SAVED] LoRA checkpoint to {ckpt_dir}")

    print("\nTraining Complete! Saving final LoRA weights...")
    final_dir = f"./checkpoints/wan_project_cam_{args.num_envs}_lora_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    torch.save(model.sketch_proj.state_dict(), os.path.join(final_dir, "sketch_proj.pt"))
    writer.close()

    for _, env in active_envs:
        try:
            env.close()
        except Exception:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA across multiple Single Game Views")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout steps when training on a single environment")
    parser.add_argument("--buffer_capacity", type=int, default=40000, help="Capacity of multi-environment transition buffer")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per gradient step")
    parser.add_argument("--train_seq_len", type=int, default=16, help="Contiguous trajectory steps unrolled per training update")
    parser.add_argument("--teacher_forcing_prob", type=float, default=1.0, help="Max prob of using self-predicted latents onto next trajectory step")
    parser.add_argument("--ctx_noise_max", type=float, default=0.15, help="Max Gaussian noise added to conditioning context latents to prevent drift")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--proj_lr", type=float, default=3e-3,
                        help="Learning rate for the sketch projector (at the LoRA lr it barely leaves its copy-Rear init)")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Generate and evaluate views every N epochs")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of Euler flow matching sampling steps per frame")
    parser.add_argument("--rollout_video_steps", type=int, default=16, help="Autoregressive rollout steps for validation videos")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for TensorBoard video logging")
    parser.add_argument("--log_gt", action="store_true", help="Also log the ground-truth validation videos (identical every epoch)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_envs", type=int, default=12, help="Number of training envs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)