"""
Pretraining the 4 per-view Ortho projectors on ground-truth Wan2.1 VAE latents.

Each projector predicts one view from the other 3 views + the FPV camera pose and env (same timestep):
  top_projector(fpv,  side, rear) -> top   (OrthoProjectorAttn: pose-dependent cross-attention)
  side_projector(top, rear, fpv)  -> side
  rear_projector(top, side, fpv)  -> rear
  fpv_projector(top,  side, rear) -> fpv   (OrthoProjectorAttn: pose-dependent cross-attention)

The pretrained projectors are meant to be frozen afterwards and used to enforce cross-view
consistency in `train_ortho_project_wan.py` (saved in the same `projectors.pt` format).

Uses:
  - Frozen Wan2.1 VAE only (no DiT / text encoder needed)
  - Same buffer, rollout collection, latent encoding and OrthoProjector as the world model
"""

import os
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from diffusers import AutoencoderKLWan

from utils import _build_registry, set_seed
from buffer import OrthoTransitionBuffer
from train_ortho_project_wan import (
    OrthoProjector,
    OrthoProjectorAttn,
    encode_camera_poses,
    encode_views_to_latents,
    decode_latents_to_views,
    collect_all_envs_rollouts,
    device,
    dtype,
)

# ==========================================
# PROJECTOR TRAINING STEP
# ==========================================
def train_projectors_step(
    fpv_projector,
    top_projector,
    rear_projector,
    side_projector,
    latents_next,
    camera_poses,
    env_names=None,
    noise_std=0.1,
):
    """
    One supervised step for the 4 projectors on GT latents at a single timestep.
    latents_next: (B, C, 4, h, w) GT view latents (0: Top, 1: Side, 2: Rear, 3: FPV)
    camera_poses: list of B camera-state dicts (FPV pose of the same frames)
    env_names: list of B env names (env embedding in the pose vector)
    Inputs get independent Gaussian noise per view (robustness to imperfect predicted views);
    targets stay clean.
    Returns: loss, per_view_losses [top, side, rear, fpv], zero_baseline, proj_latents (B, C, 4, h, w)
    """
    gt_next = latents_next.float()
    gt_top_next, gt_side_next, gt_rear_next, gt_fpv_next = gt_next[:, :, 0], gt_next[:, :, 1], gt_next[:, :, 2], gt_next[:, :, 3]
    pose_next = encode_camera_poses(camera_poses, env_names=env_names, target_device=gt_next.device)

    noisy_next = gt_next + torch.randn_like(gt_next) * noise_std
    noisy_top, noisy_side, noisy_rear, noisy_fpv = noisy_next[:, :, 0], noisy_next[:, :, 1], noisy_next[:, :, 2], noisy_next[:, :, 3]

    proj_fpv_next = fpv_projector(noisy_top, noisy_side, noisy_rear, pose_next)
    proj_top_next = top_projector(noisy_fpv, noisy_side, noisy_rear, pose_next)
    proj_rear_next = rear_projector(noisy_top, noisy_side, noisy_fpv, pose_next)
    proj_side_next = side_projector(noisy_top, noisy_rear, noisy_fpv, pose_next)

    per_view_losses = [
        F.mse_loss(proj_top_next, gt_top_next),
        F.mse_loss(proj_side_next, gt_side_next),
        F.mse_loss(proj_rear_next, gt_rear_next),
        F.mse_loss(proj_fpv_next, gt_fpv_next),
    ]
    loss = sum(per_view_losses)

    # Zero-prediction baseline summed over the 4 views, comparable to loss (sum of 4 view MSEs)
    zero_baseline = gt_next.pow(2).mean(dim=(0, 1, 3, 4)).sum().item()

    proj_latents = torch.stack([proj_top_next, proj_side_next, proj_rear_next, proj_fpv_next], dim=2).detach()
    return loss, [l.item() for l in per_view_losses], zero_baseline, proj_latents

@torch.no_grad()
def log_projection_images(vae, latents_next, proj_latents, writer, epoch, view_names, num_images=8):
    """
    Decodes GT and projected latents for the first `num_images` samples in a single VAE call
    and writes, per view, a GT row followed by a projector row to TensorBoard.
    """
    n = min(num_images, latents_next.shape[0])
    gt_latents = latents_next[:n].float()
    pr_latents = proj_latents[:n].float()

    decoded = decode_latents_to_views(vae, torch.cat([gt_latents, pr_latents], dim=0))  # (2n, 4, 3, H, W) in [0, 1]
    decoded_gt, decoded_proj = decoded[:n], decoded[n:]

    for v_idx, v_name in enumerate(view_names):
        rows = torch.cat([decoded_gt[:, v_idx], decoded_proj[:, v_idx]], dim=0).cpu()  # (2n, 3, H, W)
        writer.add_images(f"Projector/{v_name} (top: GT, bottom: Proj)", rows, epoch)

# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
def main(args):
    set_seed(args.seed)
    print("=" * 60)
    print("  Ortho Projector Pretraining (Top / Side / Rear / FPV)")
    print("=" * 60)

    # 1. Environments & Replay Buffer
    registry = _build_registry()
    buffer = OrthoTransitionBuffer(
        capacity=args.buffer_capacity, img_h=96, img_w=96, max_action_len=8
    )

    active_envs = collect_all_envs_rollouts(
        registry, buffer, rollout_steps=args.rollout_steps, num_envs=args.num_envs
    )

    # 2. Frozen Wan VAE (latent space shared with the world model)
    print(f"\n[INFO] Loading Wan2.1 VAE from: {args.model_id}")
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # 3. Projectors & Optimizer
    top_projector = OrthoProjectorAttn(hidden=args.proj_hidden, hw=96 // 8).to(device)  # 96x96 views -> 12x12 Wan latents; pose-dependent inverse-perspective mapping from the FPV
    side_projector = OrthoProjector(hidden=args.proj_hidden, hw=96 // 8).to(device)
    rear_projector = OrthoProjector(hidden=args.proj_hidden, hw=96 // 8).to(device)
    fpv_projector = OrthoProjectorAttn(hidden=args.proj_hidden, hw=96 // 8).to(device)  # pose-dependent spatial mapping for the orbiting FPV

    projectors = {"top": top_projector, "side": side_projector, "rear": rear_projector, "fpv": fpv_projector}
    proj_params = [p for proj in projectors.values() for p in proj.parameters()]
    print(f"[INFO] Projector params (4 total): {sum(p.numel() for p in proj_params) / 1e6:.2f}M")

    optimizer = optim.AdamW(proj_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    log_dir = f"./runs/ortho_projectors_{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    view_names = ["Top", "Side", "Rear", "3D_FPV"]

    # 4. Training Loop
    print("\nStarting Projector Pretraining ...")
    global_step = 0

    for epoch in range(args.epochs):
        top_projector.train()
        side_projector.train()
        rear_projector.train()
        fpv_projector.train()

        epoch_loss = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")

        for _ in pbar:
            # Single-step batch: views at t+1, their env and the FPV camera pose they were rendered with
            _, views_next_seq, _, env_names_seq, camera_poses_seq = buffer.sample_trajectory_batch(
                batch_size=args.batch_size,
                seq_len=1,
            )

            with torch.no_grad():
                latents_next = encode_views_to_latents(vae, views_next_seq[0], out_dtype=dtype)  # (B, C, 4, h, w)

            optimizer.zero_grad(set_to_none=True)

            loss, per_view_losses, zero_baseline, proj_latents = train_projectors_step(
                fpv_projector=fpv_projector,
                top_projector=top_projector,
                rear_projector=rear_projector,
                side_projector=side_projector,
                latents_next=latents_next,
                camera_poses=camera_poses_seq[0],
                env_names=env_names_seq[0],
                noise_std=args.noise_std,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(proj_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            writer.add_scalar("Train/Proj_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Proj_Zero_Baseline", zero_baseline, global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            for v_idx, v_name in enumerate(view_names):
                writer.add_scalar(f"Train/Proj_Loss_{v_name}", per_view_losses[v_idx], global_step)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Zero": f"{zero_baseline:.4f}",
                "FPV": f"{per_view_losses[3]:.3f}",
            })

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"  [Epoch {epoch+1}] Average Projector Loss: {avg_loss:.4f} (zero baseline ~{zero_baseline:.4f})")

        # --- VALIDATION: GT vs PROJECTED VIEWS TO TENSORBOARD ---
        log_projection_images(vae, latents_next, proj_latents, writer, epoch + 1, view_names, num_images=args.log_images)

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = f"./checkpoints/ortho_projectors_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({name: p.state_dict() for name, p in projectors.items()}, os.path.join(ckpt_dir, "projectors.pt"))
            print(f"  [SAVED] Projectors checkpoint to {ckpt_dir}")

    print("\nTraining Complete! Saving final projector weights...")
    os.makedirs("./checkpoints/ortho_projectors_final", exist_ok=True)
    torch.save({name: p.state_dict() for name, p in projectors.items()},
               "./checkpoints/ortho_projectors_final/projectors.pt")
    writer.close()

    for _, env in active_envs:
        try:
            env.close()
        except Exception:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain the 4 per-view Ortho projectors on GT Wan latents")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout steps collected per environment")
    parser.add_argument("--buffer_capacity", type=int, default=40000, help="Capacity of multi-environment transition buffer")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID (VAE subfolder is used)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per gradient step")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the projectors")
    parser.add_argument("--proj_hidden", type=int, default=128, help="Hidden channels of each projector (must match the world-model script)")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Std of independent Gaussian noise added to each input view latent")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--log_images", type=int, default=8, help="Samples per view written to TensorBoard each epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_envs", type=int, default=12, help="Number of training envs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
