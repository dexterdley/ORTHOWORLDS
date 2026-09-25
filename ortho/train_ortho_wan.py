"""
Fine-tuning Wan2.1-1.3B with LoRA for Orthographic Multi-View World Modeling.

Learns conditional cross-view distributions:
  P(top_{t+1}  | rear_{t+1}, side_{t+1}, 3D_{t+1}, views_t)
  P(rear_{t+1} | top_{t+1},  side_{t+1}, 3D_{t+1}, views_t)
  P(side_{t+1} | top_{t+1},  rear_{t+1}, 3D_{t+1}, views_t)
  P(3D_{t+1}   | top_{t+1},  rear_{t+1}, side_{t+1}, views_t)

Uses:
  - Wan2.1-T2V-1.3B-Diffusers pipeline
  - Frozen 3D VAE (AutoencoderKLWan)
  - Frozen UMT5 text encoder
  - Trainable PEFT LoRA on WanTransformer3DModel
"""

import os
import math
import random
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# HuggingFace & Diffusers imports
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from peft import LoraConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# ==========================================
# 1. ENVIRONMENT REGISTRY
# ==========================================
def _build_registry():
    """Returns list of (display_name, factory_fn) for orthographic environments."""
    registry = []

    def _try(name, factory):
        try:
            env = factory()
            registry.append((name, factory))
            env.close()
        except Exception as e:
            print(f"[WARN] Could not register {name}: {e}")

    try:
        import gymnasium
        from game_envs.multicar_racing_orthographic import MultiCarOrthographicWrapper
        _try("MultiCar Racing",
             lambda: MultiCarOrthographicWrapper(
                 gymnasium.make("CarRacing-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] MultiCar Racing unavailable: {e}")

    try:
        from game_envs.drone_dogfight_orthographic import (
            DroneDogfightEnv, DroneDogfightOrthographicWrapper)
        _try("Drone Dogfight",
             lambda: DroneDogfightOrthographicWrapper(DroneDogfightEnv()))
    except Exception as e:
        print(f"[WARN] Drone Dogfight unavailable: {e}")

    try:
        from game_envs.mario_orthographic import (
            MarioEscapeEnv, MarioOrthographicWrapper)
        _try("Mario Escape",
             lambda: MarioOrthographicWrapper(MarioEscapeEnv()))
    except Exception as e:
        print(f"[WARN] Mario Escape unavailable: {e}")

    return registry


# ==========================================
# 2. MULTI-VIEW TRANSITION REPLAY BUFFER
# ==========================================
class OrthoTransitionBuffer:
    """Stores paired orthographic views (t and t+1) with player actions."""
    def __init__(self, capacity=2000, img_h=128, img_w=128, action_len=3):
        self.capacity = capacity
        self.img_h = img_h
        self.img_w = img_w
        self.action_len = action_len
        self.ptr = 0
        self.size = 0

        # 4 Views at time t (top, side, rear, fpv/3D)
        self.views_t = np.zeros((capacity, 4, 3, img_h, img_w), dtype=np.uint8)
        # 4 Views at time t+1
        self.views_next = np.zeros((capacity, 4, 3, img_h, img_w), dtype=np.uint8)
        # Actions & Dones
        self.actions = np.zeros((capacity, action_len), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

    def _resize(self, img):
        if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
            return cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        return img

    def push(self, top_t, side_t, rear_t, fpv_t,
             top_next, side_next, rear_next, fpv_next,
             action, done):
        
        vt = [self._resize(x) for x in [top_t, side_t, rear_t, fpv_t]]
        vn = [self._resize(x) for x in [top_next, side_next, rear_next, fpv_next]]

        # Convert HWC uint8 -> CHW uint8
        self.views_t[self.ptr] = np.stack([np.transpose(v, (2, 0, 1)) for v in vt])
        self.views_next[self.ptr] = np.stack([np.transpose(v, (2, 0, 1)) for v in vn])
        self.actions[self.ptr] = action
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size, target_device=device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # Convert uint8 [0, 255] -> float32 [-1, 1]
        vt = torch.from_numpy(self.views_t[idxs]).float().to(target_device) / 127.5 - 1.0
        vn = torch.from_numpy(self.views_next[idxs]).float().to(target_device) / 127.5 - 1.0
        act = torch.from_numpy(self.actions[idxs]).float().to(target_device)

        return vt, vn, act


def collect_rollouts(env, buffer, n_steps=200):
    """Collects multi-camera rollouts from the gym environment into buffer."""
    print(f"Collecting {n_steps} rollout steps from environment...")
    obs, _ = env.reset()
    for _ in tqdm(range(n_steps), desc="Rollout Buffer"):
        top_t, rear_t, side_t, fpv_t = env.render()
        action = env.action_space.sample()

        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        top_next, rear_next, side_next, fpv_next = env.render()

        # Ordering convention: 0:Top, 1:Side, 2:Rear, 3:3D/FPV
        buffer.push(
            top_t=top_t, side_t=side_t, rear_t=rear_t, fpv_t=fpv_t,
            top_next=top_next, side_next=side_next, rear_next=rear_next, fpv_next=fpv_next,
            action=action, done=done
        )

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    print(f"Buffer populated with {buffer.size} transitions.")


# ==========================================
# 3. TEXT PROMPT COMPOSER
# ==========================================
def compose_player_prompts(action_batch):
    """
    Composes a multi-agent text prompt for Wan2.1 text encoder.
    Example: 'Multi-view orthographic racing. Player 1: steer -0.35, gas 0.80. Player 2: follow curve.'
    """
    prompts = []
    for a in action_batch:
        if a.numel() >= 3:
            steer, gas, brake = a[0].item(), a[1].item(), a[2].item()
            p1_str = f"Player 1: steer {steer:+.2f}, throttle {gas:.2f}, brake {brake:.2f}"
        elif a.numel() == 2:
            p1_str = f"Player 1: steer {a[0].item():+.2f}, throttle {a[1].item():+.2f}"
        else:
            p1_str = f"Player 1 action: {a[0].item():.2f}"
        
        p2_str = "Player 2: maintaining track position"
        full_prompt = (
            f"Orthographic game views: Top-down, Side, Rear, and 3D FPV cameras. "
            f"[P1]: {p1_str} | [P2]: {p2_str}"
        )
        prompts.append(full_prompt)
    return prompts


# ==========================================
# 4. WAN2.1 MODEL SETUP WITH LORA
# ==========================================
def setup_wan_model(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", lora_rank=16):
    """
    Loads Wan2.1-1.3B, freezes VAE and Text Encoder, applies LoRA to DiT Transformer.
    """
    print(f"\n[INFO] Loading Wan2.1 components from: {model_id}")
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # 1. Freeze VAE and Text Encoder
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()

    text_encoder = pipe.text_encoder
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    tokenizer = pipe.tokenizer

    # 2. Setup LoRA on Wan Transformer DiT
    transformer = pipe.transformer
    transformer.requires_grad_(False)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj"],
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Move to device
    vae.to(device)
    text_encoder.to(device)
    transformer.to(device)

    return transformer, vae, text_encoder, tokenizer


# ==========================================
# 5. MULTI-VIEW FLOW MATCHING TRAINING STEP
# ==========================================
def encode_views_to_latents(vae, views_tensor):
    """
    Encodes 4 orthographic views into Wan 3D VAE latents.
    views_tensor: (B, 4, 3, H, W) -> Latents: (B, 16, 4, h, w)
    """
    b, num_views, c, h, w = views_tensor.shape
    # Flatten views into batch to encode as 2D frames
    flat_views = views_tensor.view(b * num_views, c, 1, h, w)

    with torch.no_grad():
        # Encode with Wan VAE
        latent_dist = vae.encode(flat_views.to(dtype=vae.dtype)).latent_dist
        latents = latent_dist.sample() * 0.18215  # Scaling factor
        
    _, c_lat, _, h_lat, w_lat = latents.shape
    # Reshape back to (B, C_lat, 4, h_lat, w_lat) where 4 is the temporal view axis
    latents = latents.view(b, num_views, c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4)
    return latents


def train_ortho_flow_step(transformer, latents_t, latents_next, prompt_embeds, target_view_idx=None):
    """
    Performs one Flow-Matching update for conditional multi-view generation.
    
    Target View index mapping at t+1:
      0: Top
      1: Side
      2: Rear
      3: 3D / FPV
    
    Conditioning:
      - All 4 views at time t (latents_t)
      - The other 3 views at time t+1 (latents_next[:, :, other_views])
    Target:
      - The selected view at time t+1 (latents_next[:, :, target_view_idx])
    """
    b, c, num_views, h, w = latents_next.shape

    # If target view not specified, pick randomly to train all conditional paths
    if target_view_idx is None:
        target_view_idx = random.randint(0, num_views - 1)

    # 1. Sample flow-matching timestep tau in [0, 1]
    tau = torch.rand(b, device=device, dtype=transformer.dtype)

    # 2. Sample Gaussian noise for the target view
    noise = torch.randn_like(latents_next)

    # 3. Create view mask: 1 for clean conditioning views, 0 for target view to denoise
    mask = torch.ones((b, 1, num_views, 1, 1), device=device, dtype=transformer.dtype)
    mask[:, :, target_view_idx] = 0.0

    # 4. Flow Matching linear interpolation: x_tau = (1 - tau) * x_0 + tau * noise
    tau_exp = tau.view(b, 1, 1, 1, 1)
    noisy_latents = (1.0 - tau_exp) * latents_next + tau_exp * noise

    # 5. Composite next-step latents: condition views are clean (tau=0), target view is noisy
    input_latents_next = mask * latents_next + (1.0 - mask) * noisy_latents

    # 6. Concatenate past context (t) and next context (t+1) along temporal sequence axis
    # Total sequence length = 8 frames: [top_t, side_t, rear_t, 3d_t, top_t+1, side_t+1, rear_t+1, 3d_t+1]
    combined_latents = torch.cat([latents_t, input_latents_next], dim=2)  # (B, C, 8, h, w)

    # 7. Ground truth flow velocity v_target = noise - target_clean
    target_velocity = noise[:, :, target_view_idx] - latents_next[:, :, target_view_idx]

    # 8. DiT Forward pass: scaled timestep in [0, 1000]
    timesteps = tau * 1000.0

    model_pred = transformer(
        hidden_states=combined_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
    ).sample

    # Predicted velocity for target view (index target_view_idx in next-step chunk, i.e. 4 + target_view_idx)
    pred_target_velocity = model_pred[:, :, 4 + target_view_idx]

    # 9. Flow Matching MSE loss strictly on the target view
    loss = F.mse_loss(pred_target_velocity.float(), target_velocity.float())
    return loss, target_view_idx


# ==========================================
# 6. CONDITIONAL SAMPLING (INFERENCE)
# ==========================================
@torch.no_grad()
def sample_conditional_view(transformer, vae, latents_t, clean_other_views_next, target_view_idx, prompt_embeds, steps=15):
    """
    Denoises a specific view P(target | other views_next, views_t) via reverse Euler Flow Matching.
    Integrates backwards from pure noise tau=1.0 down to clean data tau=0.0.
    """
    b, c, num_views, h, w = clean_other_views_next.shape
    # Initialize target view from standard normal noise at tau=1.0
    x_tau = torch.randn((b, c, 1, h, w), device=device, dtype=transformer.dtype)

    dt = 1.0 / float(steps)

    for step_i in range(steps):
        # Current flow time from 1.0 (noise) down to 0.0 (data)
        current_tau = 1.0 - (step_i / float(steps))
        t_tensor = torch.full((b,), current_tau * 1000.0, device=device, dtype=transformer.dtype)

        # Assemble next-views tensor with current noisy target view
        input_next = clean_other_views_next.clone()
        input_next[:, :, target_view_idx:target_view_idx + 1] = x_tau

        # Concatenate past views (t) and candidate next views (t+1) along temporal sequence
        combined = torch.cat([latents_t, input_next], dim=2)

        pred_velocity = transformer(
            hidden_states=combined,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, 4 + target_view_idx:5 + target_view_idx]

        # Reverse Euler step: dx/dtau = v -> x_{tau - dt} = x_tau - dt * v
        x_tau = x_tau - dt * pred_velocity

    # Decode denoised target latent back to image: x_tau at tau=0 is clean latent
    latents_clean = (x_tau / 0.18215).to(dtype=vae.dtype)
    pred_video = vae.decode(latents_clean).sample  # (B, 3, 1, H, W)
    pred_img = pred_video.squeeze(2)              # (B, 3, H, W)
    pred_img = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
    return pred_img


@torch.no_grad()
def evaluate_and_log_images(transformer, vae, buffer, text_encoder, tokenizer, writer, epoch, save_dir="./eval_images", steps=15):
    """
    Evaluates cross-view conditional generation:
      Generates P(Top | others), P(Side | others), P(Rear | others), P(3D | others)
    Computes PSNR/MSE metrics, saves a visual comparison grid to disk, and logs to TensorBoard.
    """
    transformer.eval()
    os.makedirs(save_dir, exist_ok=True)
    view_names = ["Top", "Side", "Rear", "3D_FPV"]

    # 1. Sample validation pair
    vt, vn, act = buffer.sample_batch(batch_size=1)
    prompts = compose_player_prompts(act)

    text_inputs = tokenizer(
        prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    ).to(device)
    prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)

    latents_t = encode_views_to_latents(vae, vt)
    latents_next = encode_views_to_latents(vae, vn)

    # 2. Generate each view conditionally given the other clean views
    pred_views = []
    print(f"\n[EVAL] Generating all 4 orthographic views for epoch {epoch + 1}...")

    for target_idx in range(4):
        pred_img = sample_conditional_view(
            transformer=transformer,
            vae=vae,
            latents_t=latents_t,
            clean_other_views_next=latents_next,
            target_view_idx=target_idx,
            prompt_embeds=prompt_embeds,
            steps=steps
        )
        pred_views.append(pred_img[0].cpu())  # (3, H, W) in [0, 1]

    # Ground truth next views & context views (rescaled from [-1, 1] to [0, 1])
    gt_views = [torch.clamp((vn[0, i].cpu() + 1.0) / 2.0, 0.0, 1.0) for i in range(4)]
    ctx_views = [torch.clamp((vt[0, i].cpu() + 1.0) / 2.0, 0.0, 1.0) for i in range(4)]

    # 3. Compute PSNR & MSE per view
    val_psnr_list = []
    for i, name in enumerate(view_names):
        mse = F.mse_loss(pred_views[i], gt_views[i]).item()
        psnr = 20.0 * math.log10(1.0) - 10.0 * math.log10(max(mse, 1e-8))
        val_psnr_list.append(psnr)
        writer.add_scalar(f"Val/PSNR_{name}", psnr, epoch + 1)
        writer.add_scalar(f"Val/MSE_{name}", mse, epoch + 1)

    mean_psnr = sum(val_psnr_list) / len(val_psnr_list)
    writer.add_scalar("Val/Mean_PSNR", mean_psnr, epoch + 1)
    print(f"  [Epoch {epoch + 1}] Val Mean PSNR: {mean_psnr:.2f} dB "
          f"({', '.join([f'{view_names[i]}: {val_psnr_list[i]:.2f}dB' for i in range(4)])})")

    # 4. Create comparison grid:
    # Row 1: Context views at time t       [Top, Side, Rear, 3D]
    # Row 2: Ground Truth views at t+1    [Top, Side, Rear, 3D]
    # Row 3: Model Generated views at t+1  [Top, Side, Rear, 3D]
    comparison_imgs = ctx_views + gt_views + pred_views  # 12 images
    grid = make_grid(torch.stack(comparison_imgs), nrow=4, padding=4, normalize=False)

    # 5. Save to disk
    save_path = os.path.join(save_dir, f"epoch_{epoch + 1:03d}_views.png")
    save_image(grid, save_path)
    print(f"  [SAVED] Generated views comparison image saved to: {save_path}")

    # 6. Log to TensorBoard
    writer.add_image("Val/Views_Comparison_Grid", grid, epoch + 1)
    for i, name in enumerate(view_names):
        writer.add_image(f"Val_Generated/{name}", pred_views[i], epoch + 1)
        writer.add_image(f"Val_GroundTruth/{name}", gt_views[i], epoch + 1)

    transformer.train()


# ==========================================
# 7. MAIN TRAINING SCRIPT
# ==========================================
def main(args):
    print("=" * 60)
    print("  Wan2.1 Orthographic Multi-View World Model (LoRA)")
    print("=" * 60)

    # 1. Environments
    registry = _build_registry()
    if not registry:
        print("[ERROR] No orthographic gym environments found in game_envs. Exiting.")
        return

    env_name, make_env = registry[0]
    print(f"Loading environment: {env_name}")
    env = make_env()

    # 2. Buffer & Rollouts
    action_len = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    buffer = OrthoTransitionBuffer(capacity=1000, img_h=128, img_w=128, action_len=action_len)
    collect_rollouts(env, buffer, n_steps=args.rollout_steps)

    # 3. Model & LoRA Setup
    transformer, vae, text_encoder, tokenizer = setup_wan_model(
        model_id=args.model_id, lora_rank=args.lora_rank
    )

    optimizer = optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2
    )

    log_dir = f"./runs/wan_ortho_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    eval_dir = os.path.join(log_dir, "eval_images")

    # View name labels
    view_names = ["Top", "Side", "Rear", "3D_FPV"]

    # 4. Training Loop
    print("\nStarting LoRA Fine-Tuning on Wan DiT...")
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")

        for _ in pbar:
            # Sample batch of orthographic views
            vt, vn, act = buffer.sample_batch(args.batch_size)
            
            # Compose player text prompts and extract text embeddings
            prompts = compose_player_prompts(act)
            with torch.no_grad():
                text_inputs = tokenizer(
                    prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
                ).to(device)
                prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)

            # Encode views into 3D latents
            latents_t = encode_views_to_latents(vae, vt)
            latents_next = encode_views_to_latents(vae, vn)

            # Flow matching loss on a random conditional target view
            loss, target_idx = train_ortho_flow_step(
                transformer, latents_t, latents_next, prompt_embeds
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            writer.add_scalar("Train/Loss", loss_val, global_step)
            writer.add_scalar(f"Train/Loss_{view_names[target_idx]}", loss_val, global_step)
            pbar.set_postfix({"Loss": f"{loss_val:.4f}", "TargetView": view_names[target_idx]})

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"  [Epoch {epoch+1}] Average Flow Matching Loss: {avg_loss:.4f}")

        # --- VALIDATION: GENERATE AND LOG ORTHOGRAPHIC IMAGES ---
        if (epoch + 1) % args.eval_every == 0:
            evaluate_and_log_images(
                transformer=transformer,
                vae=vae,
                buffer=buffer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                writer=writer,
                epoch=epoch,
                save_dir=eval_dir,
                steps=args.eval_steps,
            )

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = f"./checkpoints/wan_ortho_lora_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            transformer.save_pretrained(ckpt_dir)
            print(f"  [SAVED] LoRA checkpoint to {ckpt_dir}")

    print("\nTraining Complete! Saving final LoRA weights...")
    os.makedirs("./checkpoints/wan_ortho_lora_final", exist_ok=True)
    transformer.save_pretrained("./checkpoints/wan_ortho_lora_final")
    writer.close()
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA on Orthographic Game Views")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per gradient step")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Training steps per epoch")
    parser.add_argument("--rollout_steps", type=int, default=300, help="Steps to collect from gym env")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Generate and evaluate views every N epochs")
    parser.add_argument("--eval_steps", type=int, default=15, help="Number of Euler flow matching sampling steps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

