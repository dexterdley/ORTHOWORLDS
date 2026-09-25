"""
Fine-tuning Wan2.1-1.3B with LoRA for Orthographic Multi-View World Modeling.

Learns conditional cross-view distributions:
  P(top_{t+1}  | rear_{t+1}, side_{t+1}, 3D_{t+1}, views_t)
  P(rear_{t+1} | top_{t+1},  side_{t+1}, 3D_{t+1}, views_t)
  P(side_{t+1} | top_{t+1},  rear_{t+1}, 3D_{t+1}, views_t)
  P(3D_{t+1}   | top_{t+1},  rear_{t+1}, side_{t+1}, views_t)

Uses:
  - Wan2.1-T2V-1.3B-Diffusers pipeline
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
from diffusers.utils import export_to_video
from utils import _build_registry, load_model, compose_player_prompts, set_active_env

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# ==========================================
# A. MULTI-VIEW TRANSITION REPLAY BUFFER
# ==========================================
class OrthoTransitionBuffer:
    """Stores paired orthographic views (t and t+1) with actions across multiple environments."""
    def __init__(self, capacity=4000, img_h=128, img_w=128, max_action_len=8, env_name="Bipedal Walker"):
        self.capacity = capacity
        self.img_h = img_h
        self.img_w = img_w
        self.max_action_len = max_action_len
        self.default_env_name = env_name
        self.ptr = 0
        self.size = 0

        # 4 Views at time t (top, side, rear, fpv/3D)
        self.views_t = np.zeros((capacity, 4, 3, img_h, img_w), dtype=np.uint8)
        # 4 Views at time t+1
        self.views_next = np.zeros((capacity, 4, 3, img_h, img_w), dtype=np.uint8)
        # Actions & Dones
        self.actions = np.zeros((capacity, max_action_len), dtype=np.float32)
        self.action_lens = np.zeros(capacity, dtype=np.int32)
        self.dones = np.zeros(capacity, dtype=bool)
        # Environment name per transition
        self.env_names = [None] * capacity

    def _resize(self, img):
        if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
            return cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        return img

    def push(self, top_t, side_t, rear_t, fpv_t,
             top_next, side_next, rear_next, fpv_next,
             action, done, env_name=None):
        
        vt = [self._resize(x) for x in [top_t, side_t, rear_t, fpv_t]]
        vn = [self._resize(x) for x in [top_next, side_next, rear_next, fpv_next]]

        # Convert HWC uint8 -> CHW uint8
        self.views_t[self.ptr] = np.stack([np.transpose(v, (2, 0, 1)) for v in vt])
        self.views_next[self.ptr] = np.stack([np.transpose(v, (2, 0, 1)) for v in vn])
        
        # Safely assign action vector or scalar
        act_arr = np.array(action, dtype=np.float32)
        if act_arr.ndim == 0:
            self.actions[self.ptr, :] = 0.0
            self.actions[self.ptr, 0] = float(act_arr)
            self.action_lens[self.ptr] = 1
        else:
            length = min(len(act_arr), self.max_action_len)
            self.actions[self.ptr, :] = 0.0
            self.actions[self.ptr, :length] = act_arr[:length]
            self.action_lens[self.ptr] = length

        self.dones[self.ptr] = done
        self.env_names[self.ptr] = env_name or self.default_env_name

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size, target_device=device, env_filter=None):
        if env_filter is not None:
            valid_idxs = [i for i in range(self.size) if self.env_names[i] == env_filter]
            if not valid_idxs:
                valid_idxs = list(range(self.size))
            idxs = np.random.choice(valid_idxs, size=batch_size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        
        # Convert uint8 [0, 255] -> float32 [-1, 1]
        vt = torch.from_numpy(self.views_t[idxs]).float().to(target_device) / 127.5 - 1.0
        vn = torch.from_numpy(self.views_next[idxs]).float().to(target_device) / 127.5 - 1.0
        
        # Slice each action to its true dimension
        acts = [
            torch.from_numpy(self.actions[i, :self.action_lens[i]]).float().to(target_device)
            for i in idxs
        ]
        sample_envs = [self.env_names[i] for i in idxs]

        return vt, vn, acts, sample_envs

    def sample_sequence(self, seq_len=16, env_filter=None):
        """Samples a contiguous sequence of transitions for autoregressive validation."""
        candidates = []
        for i in range(max(0, self.size - seq_len)):
            if env_filter is None or self.env_names[i] == env_filter:
                if all(self.env_names[i + k] == (env_filter or self.env_names[i]) for k in range(seq_len)):
                    candidates.append(i)
        
        if not candidates:
            start_idx = random.randint(0, max(0, self.size - seq_len)) if self.size > seq_len else 0
        else:
            start_idx = random.choice(candidates)

        actual_len = min(seq_len, max(1, self.size - start_idx)) if self.size > 0 else 0
        idxs = list(range(start_idx, start_idx + actual_len))
        vt_seq = self.views_t[idxs]
        vn_seq = self.views_next[idxs]
        act_seq = [self.actions[i, :self.action_lens[i]] for i in idxs]
        env_seq = [self.env_names[i] for i in idxs]

        return vt_seq, vn_seq, act_seq, env_seq


def collect_rollouts(env, buffer, n_steps=200, env_name="Bipedal Walker"):
    """Collects multi-camera rollouts from a gym environment into buffer."""
    print(f"Collecting {n_steps} rollout steps from '{env_name}'...")
    try:
        reset_res = env.reset()
        obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
    except Exception as e:
        print(f"[WARN] Failed to reset {env_name}: {e}")
        return

    collected = 0
    for _ in tqdm(range(n_steps), desc=f"Rollouts [{env_name}]"):
        try:
            top_t, rear_t, side_t, fpv_t = env.render()
            action = env.action_space.sample()

            step_res = env.step(action)
            if len(step_res) == 5:
                next_obs, reward, term, trunc, _ = step_res
                done = term or trunc
            else:
                next_obs, reward, done, _ = step_res

            top_next, rear_next, side_next, fpv_next = env.render()

            if any(v is None for v in [top_t, rear_t, side_t, fpv_t, top_next, rear_next, side_next, fpv_next]):
                continue

            # Ordering convention: 0:Top, 1:Side, 2:Rear, 3:3D/FPV
            buffer.push(
                top_t=top_t, side_t=side_t, rear_t=rear_t, fpv_t=fpv_t,
                top_next=top_next, side_next=side_next, rear_next=rear_next, fpv_next=fpv_next,
                action=action, done=done, env_name=env_name
            )
            collected += 1

            if done:
                reset_res = env.reset()
                obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
            else:
                obs = next_obs
        except Exception as e:
            print(f"[WARN] Error during rollout step in {env_name}: {e}")
            break


def collect_all_envs_rollouts(registry, buffer, steps_per_env=150):
    """Loops through all registered environments and populates the multi-env buffer."""
    print("=" * 60)
    print(f"Collecting rollouts across all {len(registry)} environments...")
    print("=" * 60)
    
    active_envs = []
    for idx, (name, factory) in enumerate(registry):
        print(f"\n[{idx + 1}/{len(registry)}] Initializing environment: {name}")
        try:
            env = factory()
            collect_rollouts(env, buffer, n_steps=steps_per_env, env_name=name)
            active_envs.append((name, env))
        except Exception as e:
            print(f"[WARN] Could not load or collect from '{name}': {e}")
            
    unique_recorded = list(dict.fromkeys([e for e in buffer.env_names[:buffer.size] if e is not None]))
    print(f"\n[INFO] Buffer populated with {buffer.size} transitions across {len(unique_recorded)} environments: {', '.join(unique_recorded)}")
    return active_envs



# ==========================================
# TEXT PROMPT COMPOSER
# ==========================================
# (Note: compose_player_prompts is imported from utils and also defined here for direct access)
# Examples:
#   Bipedal Walker: 'Orthographic game views: Top-down, Side, Rear, and 3D FPV cameras. Bipedal Walker 2D robot locomotion: leg torques [hip1: +0.20, knee1: -0.45, hip2: +0.10, knee2: +0.80], balancing across terrain.'
#   Mario Escape:   'Orthographic game views: Top-down, Side, Rear, and 3D FPV cameras. Mario Escape platformer: horizontal move +0.75, jump impulse active, navigating obstacles towards flagpole.'
#   MultiCar:       'Orthographic game views: Top-down, Side, Rear, and 3D FPV cameras. MultiCar circuit racing: steer -0.35, throttle 0.80, brake 0.00, high-speed track cornering.'
#   Drone Dogfight: 'Orthographic game views: Top-down, Side, Rear, and 3D FPV cameras. Drone Dogfight aerial combat: thrust (+0.40, -0.60), firing weapon, aerial engagement against opponent.'
#   Excavator:      'Orthographic game views: Top-down, Side, Rear, and 3D FPV cameras. Excavator heavy machinery: drive +0.50, boom -0.20, dipper +0.80, bucket -0.10, scooping rock payload into bin.'



# ==========================================
# MULTI-VIEW FLOW MATCHING TRAINING STEP
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
# AUTOREGRESSIVE VIDEO ROLLOUT & SAMPLING
# ==========================================
@torch.no_grad()
def sample_next_views_latents(transformer, latents_t, prompt_embeds, steps=10):
    """
    Simultaneously denoises all 4 orthographic views at t+1 via reverse Euler Flow Matching.
    Input:
      latents_t: (B, C, 4, h, w) - past views at time t
    Output:
      latents_next: (B, C, 4, h, w) - predicted views at time t+1
    """
    b, c, num_views, h, w = latents_t.shape
    x_tau = torch.randn((b, c, num_views, h, w), device=device, dtype=transformer.dtype)
    dt = 1.0 / float(steps)

    for step_i in range(steps):
        current_tau = 1.0 - (step_i / float(steps))
        t_tensor = torch.full((b,), current_tau * 1000.0, device=device, dtype=transformer.dtype)
        combined = torch.cat([latents_t, x_tau], dim=2)  # (B, C, 8, h, w)

        pred_velocity = transformer(
            hidden_states=combined,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, 4:8]  # (B, C, 4, h, w)

        x_tau = x_tau - dt * pred_velocity

    return x_tau


def decode_latents_to_views(vae, latents):
    """
    Decodes 4 orthographic views latents back into image tensors.
    latents: (B, C_lat, 4, h_lat, w_lat) -> (B, 4, 3, H, W) in [0, 1]
    """
    b, c_lat, num_views, h_lat, w_lat = latents.shape
    flat_latents = latents.permute(0, 2, 1, 3, 4).reshape(b * num_views, c_lat, 1, h_lat, w_lat)
    flat_latents = (flat_latents / 0.18215).to(dtype=vae.dtype)
    with torch.no_grad():
        decoded = vae.decode(flat_latents).sample  # (B * num_views, 3, 1, H, W)
    decoded = decoded.squeeze(2)  # (B * num_views, 3, H, W)
    views = decoded.view(b, num_views, 3, decoded.shape[-2], decoded.shape[-1])
    views = torch.clamp((views + 1.0) / 2.0, 0.0, 1.0)
    return views


@torch.no_grad()
def autoregressive_rollout(transformer, vae, text_encoder, tokenizer, buffer, env_name=None, rollout_steps=16, steps=10):
    """
    Autoregressive video rollout for all 4 orthographic views.
    Returns:
      gen_views: (vid_top, vid_rear, vid_side, vid_fpv) each (1, T, 3, H, W) in [0, 1]
      gt_views:  (gt_top,  gt_rear,  gt_side,  gt_fpv)  each (1, T, 3, H, W) in [0, 1]
    """
    transformer.eval()

    # 1. Grab continuous sequence from buffer
    vt_seq, vn_seq, act_seq, env_seq = buffer.sample_sequence(seq_len=rollout_steps, env_filter=env_name)
    actual_steps = len(act_seq)

    # Initial frame at t=0
    curr_views = torch.from_numpy(vt_seq[0:1]).float().to(device) / 127.5 - 1.0
    curr_latents = encode_views_to_latents(vae, curr_views)

    # Frame storage (0: Top, 1: Side, 2: Rear, 3: FPV)
    gen_top,  gen_side,  gen_rear,  gen_fpv  = [], [], [], []
    gt_top,   gt_side,   gt_rear,   gt_fpv   = [], [], [], []

    print(f"Starting {actual_steps}-step Autoregressive Rollout for [{env_name or 'Default'}]...")

    for step_i in range(actual_steps):
        act = act_seq[step_i]
        sample_env = env_seq[step_i] or env_name

        prompt = compose_player_prompts([act], env_name=sample_env)
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to(device)
        prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)

        next_latents = sample_next_views_latents(
            transformer=transformer,
            latents_t=curr_latents,
            prompt_embeds=prompt_embeds,
            steps=steps
        )

        decoded_views = decode_latents_to_views(vae, next_latents)  # (1, 4, 3, H, W) in [0, 1]

        # Store generated frames (0: Top, 1: Side, 2: Rear, 3: FPV)
        gen_top.append(decoded_views[0, 0].cpu())
        gen_side.append(decoded_views[0, 1].cpu())
        gen_rear.append(decoded_views[0, 2].cpu())
        gen_fpv.append(decoded_views[0, 3].cpu())

        # Ground truth next views
        gt_vn_01 = torch.from_numpy(vn_seq[step_i]).float() / 255.0  # (4, 3, H, W)
        gt_top.append(gt_vn_01[0])
        gt_side.append(gt_vn_01[1])
        gt_rear.append(gt_vn_01[2])
        gt_fpv.append(gt_vn_01[3])

        # Autoregressive feedback: predicted next latents become current context
        curr_latents = next_latents

    vid_gen_top  = torch.stack(gen_top,  dim=0).unsqueeze(0)
    vid_gen_side = torch.stack(gen_side, dim=0).unsqueeze(0)
    vid_gen_rear = torch.stack(gen_rear, dim=0).unsqueeze(0)
    vid_gen_fpv  = torch.stack(gen_fpv,  dim=0).unsqueeze(0)

    vid_gt_top   = torch.stack(gt_top,   dim=0).unsqueeze(0)
    vid_gt_side  = torch.stack(gt_side,  dim=0).unsqueeze(0)
    vid_gt_rear  = torch.stack(gt_rear,  dim=0).unsqueeze(0)
    vid_gt_fpv   = torch.stack(gt_fpv,   dim=0).unsqueeze(0)

    transformer.train()
    return (vid_gen_top, vid_gen_rear, vid_gen_side, vid_gen_fpv,
            vid_gt_top,  vid_gt_rear,  vid_gt_side,  vid_gt_fpv)


@torch.no_grad()
def evaluate_and_log_videos(transformer, vae, buffer, text_encoder, tokenizer, writer, epoch, rollout_steps=16, steps=10, fps=15, env_names=None):
    """
    Performs autoregressive video rollouts for each view and writes the videos to TensorBoard.
    Matches train_ortho_diffusion.py video logging (no static images).
    """
    transformer.eval()
    
    if env_names is None:
        unique_envs = list(dict.fromkeys([e for e in buffer.env_names[:buffer.size] if e is not None]))
    elif isinstance(env_names, str):
        unique_envs = [env_names]
    else:
        unique_envs = list(env_names)

    if not unique_envs:
        unique_envs = ["Bipedal Walker"]

    print(f"\n[VALIDATION] Running autoregressive video rollouts for epoch {epoch} across envs: {', '.join(unique_envs[:2])}...")

    for env_idx, env_name in enumerate(unique_envs[:2]):
        (vid_top_down_01, vid_rear_01, vid_side_01, vid_fpv_01,
         vid_gt_top_down_01, vid_gt_rear_01, vid_gt_side_01, vid_gt_fpv_01) = autoregressive_rollout(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            buffer=buffer,
            env_name=env_name,
            rollout_steps=rollout_steps,
            steps=steps
        )

        clean_tag = env_name.replace(" ", "_").lower()

        # Compute PSNR metrics
        def _psnr(pred, gt):
            mse = F.mse_loss(pred, gt).item()
            return 20.0 * math.log10(1.0) - 10.0 * math.log10(max(mse, 1e-8))

        psnr_top_down = _psnr(vid_top_down_01, vid_gt_top_down_01)
        psnr_rear     = _psnr(vid_rear_01, vid_gt_rear_01)
        psnr_side     = _psnr(vid_side_01, vid_gt_side_01)
        psnr_fpv      = _psnr(vid_fpv_01, vid_gt_fpv_01)
        psnr_mean     = (psnr_top_down + psnr_rear + psnr_side + psnr_fpv) / 4.0

        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Top Down", psnr_top_down, epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Rear",     psnr_rear,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Side",     psnr_side,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/FPV",      psnr_fpv,      epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Mean",     psnr_mean,     epoch)

        print(f"  [Epoch {epoch}] [{env_name}] Autoregressive Video PSNR: {psnr_mean:.2f} dB "
              f"(Top-Down: {psnr_top_down:.2f}dB, Rear: {psnr_rear:.2f}dB, Side: {psnr_side:.2f}dB, FPV: {psnr_fpv:.2f}dB)")

        # Write videos to TensorBoard matching train_ortho_diffusion.py:
        # writer.add_video("Test/Top-Down", vid_top_down_01, epoch, fps=15)
        # writer.add_video("Test/Rear",     vid_rear_01,     epoch, fps=15)
        # writer.add_video("Test/Side",     vid_side_01,     epoch, fps=15)
        # writer.add_video("Test/FPV",      vid_fpv_01,      epoch, fps=15)
        if env_idx == 0:
            writer.add_video("Test/Top-Down", vid_top_down_01, epoch, fps=fps)
            writer.add_video("Test/Rear",     vid_rear_01,     epoch, fps=fps)
            writer.add_video("Test/Side",     vid_side_01,     epoch, fps=fps)
            writer.add_video("Test/FPV",      vid_fpv_01,      epoch, fps=fps)

        if len(unique_envs) > 1:
            writer.add_video(f"Test/{clean_tag}/Top-Down", vid_top_down_01, epoch, fps=fps)
            writer.add_video(f"Test/{clean_tag}/Rear",     vid_rear_01,     epoch, fps=fps)
            writer.add_video(f"Test/{clean_tag}/Side",     vid_side_01,     epoch, fps=fps)
            writer.add_video(f"Test/{clean_tag}/FPV",      vid_fpv_01,      epoch, fps=fps)

    transformer.train()



# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
def main(args):
    print("=" * 60)
    print("  Wan2.1 Orthographic Multi-View World Model (LoRA)")
    print("  Multi-Environment Cross-Domain Generalization")
    print("=" * 60)

    # 1. Environments & Replay Buffer
    registry = _build_registry()
    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(
        capacity=args.buffer_capacity, img_h=128, img_w=128, max_action_len=8
    )

    active_envs = []
    if args.env.lower() == "all":
        # Loop through ALL environments and collect rollouts into one unified buffer
        active_envs = collect_all_envs_rollouts(
            registry, buffer, steps_per_env=args.steps_per_env
        )
    else:
        # Match single environment
        matched_name = next((k for k in reg_dict if k.lower() == args.env.lower()), None)
        if matched_name is None:
            matched_name = registry[0][0]
        print(f"Loading single environment: {matched_name}")
        env = reg_dict[matched_name]()
        collect_rollouts(env, buffer, n_steps=args.rollout_steps, env_name=matched_name)
        active_envs.append((matched_name, env))

    if buffer.size == 0:
        raise RuntimeError("Buffer is empty! Could not collect rollout transitions from environments.")

    # 2. Model & LoRA Setup (One shared model across all environments)
    transformer, vae, text_encoder, tokenizer = load_model(
        model_id=args.model_id, lora_rank=args.lora_rank
    )

    optimizer = optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5
    )

    log_dir = f"./runs/wan_ortho_multienv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # View name labels
    view_names = ["Top", "Side", "Rear", "3D_FPV"]

    # 3. Multi-Environment Training Loop
    print("\nStarting Cross-Environment LoRA Fine-Tuning on Wan DiT...")
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")

        for _ in pbar:
            # Sample mixed batch of orthographic views across environments
            vt, vn, acts, sample_envs = buffer.sample_batch(args.batch_size)
            
            # Compose text prompts tailored to each sample's environment
            prompts = compose_player_prompts(acts, env_name=sample_envs)

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
            
            env_summary = "+".join(list(dict.fromkeys(sample_envs))[:2])
            pbar.set_postfix({"Loss": f"{loss_val:.4f}", "TargetView": view_names[target_idx], "Envs": env_summary})

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"  [Epoch {epoch+1}] Average Flow Matching Loss: {avg_loss:.4f}")

        # --- VALIDATION: AUTOREGRESSIVE VIDEO ROLLOUTS TO TENSORBOARD ---
        if (epoch + 1) % args.eval_every == 0:
            evaluate_and_log_videos(
                transformer=transformer,
                vae=vae,
                buffer=buffer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                writer=writer,
                epoch=epoch + 1,
                rollout_steps=args.rollout_video_steps,
                steps=args.eval_steps,
                fps=args.video_fps,
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
    for _, env in active_envs:
        try:
            env.close()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA across multiple Orthographic Game Views")
    parser.add_argument("--env", type=str, default="all", help="Environment to train on: 'all' to train across all envs, or a specific name like 'Bipedal Walker'")
    parser.add_argument("--steps_per_env", type=int, default=150, help="Rollout steps to collect per environment when env='all'")
    parser.add_argument("--rollout_steps", type=int, default=300, help="Rollout steps when training on a single environment")
    parser.add_argument("--buffer_capacity", type=int, default=4000, help="Capacity of multi-environment transition buffer")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per gradient step")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for LoRA parameters")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Generate and evaluate views every N epochs")
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of Euler flow matching sampling steps per frame")
    parser.add_argument("--rollout_video_steps", type=int, default=16, help="Autoregressive rollout steps for validation videos")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for TensorBoard video logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

