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

# HuggingFace & Diffusers imports
from utils import _build_registry, load_model, compose_player_prompts, set_active_env, set_seed, load_agent, _annotate_frame_pose, _psnr
from buffer import OrthoTransitionBuffer

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
IMG_H, IMG_W = 96,96

# ==========================================
# WAN 2.1 HELPERS
# ==========================================
def _get_wan_latent_stats(vae, target_device, target_dtype):
    """Retrieves per-channel mean and std from AutoencoderKLWan config."""
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        mean = torch.tensor(vae.config.latents_mean, device=target_device, dtype=target_dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=target_device, dtype=target_dtype).view(1, -1, 1, 1, 1)
    return mean, std

def encode_views_to_latents(vae, views_tensor, out_dtype=None):
    """
    Encodes 4 orthographic views into normalized Wan 3D VAE latents ~ N(0, I).
    views_tensor: (B, 4, 3, H, W) in [-1, 1] -> Latents: (B, 16, 4, h, w)
    """
    b, num_views, c, h, w = views_tensor.shape
    flat_views = views_tensor.view(b * num_views, c, 1, h, w)

    with torch.no_grad():
        latent_dist = vae.encode(flat_views.to(dtype=vae.dtype)).latent_dist
        latents = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()
        mean, std = _get_wan_latent_stats(vae, latents.device, latents.dtype)
        latents = (latents - mean) / std

    _, c_lat, _, h_lat, w_lat = latents.shape
    latents = latents.view(b, num_views, c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4)
    if out_dtype is not None:
        latents = latents.to(dtype=out_dtype)
    return latents

def decode_latents_to_views(vae, latents):
    """
    Unnormalizes and decodes 4 orthographic views latents back into image tensors.
    latents: (B, C_lat, 4, h_lat, w_lat) -> (B, 4, 3, H, W) in [0, 1]
    """
    b, c_lat, num_views, h_lat, w_lat = latents.shape
    flat_latents = latents.permute(0, 2, 1, 3, 4).reshape(b * num_views, c_lat, 1, h_lat, w_lat)
    flat_latents = flat_latents.to(dtype=vae.dtype)

    mean, std = _get_wan_latent_stats(vae, flat_latents.device, flat_latents.dtype)
    flat_latents = flat_latents * std + mean

    with torch.no_grad():
        decoded = vae.decode(flat_latents).sample  # (B * num_views, 3, 1, H, W)

    decoded = decoded.squeeze(2)  # (B * num_views, 3, H, W)
    views = decoded.view(b, num_views, 3, decoded.shape[-2], decoded.shape[-1])
    views = torch.clamp((views + 1.0) / 2.0, 0.0, 1.0)
    return views

class LinearProjector(nn.Module):
    def __init__(self, c_lat=16, h_lat=IMG_H // 8, w_lat=IMG_W // 8, n_src=3):
        super().__init__()
        self.h_lat, self.w_lat = h_lat, w_lat
        hw = h_lat * w_lat
        # Per view: a spatial map (hw ortho tokens -> hw sketch tokens), then its own channel mix; views are summed.
        # One channel mix shared across views caps the sketch well above the best linear predictor (toy: 0.72 vs 0.59).
        self.spatial = nn.Parameter(torch.eye(hw).repeat(n_src, 1, 1))   # (3, hw_out, hw_in)
        self.channel = nn.Linear(n_src * c_lat, c_lat)
        self.bias = nn.Parameter(torch.zeros(c_lat, hw))
        with torch.no_grad():                      # init = copy Rear, so frame 4 starts sensible
            self.channel.weight.zero_(); self.channel.weight[:, (n_src - 1) * c_lat:] = torch.eye(c_lat)
            self.channel.bias.zero_()

    def forward(self, latents_ortho):              # (B,C,3,h,w) -> (B,C,1,h,w)
        b, c, v = latents_ortho.shape[:3]
        x = torch.einsum("bcvp,vqp->bvcq", latents_ortho.float().reshape(b, c, v, -1), self.spatial)  # (B,3,C,hw)
        x = self.channel(x.reshape(b, v * c, -1).transpose(1, 2)).transpose(1, 2) + self.bias          # (B,C,hw)
        return x.view(b, c, 1, self.h_lat, self.w_lat)

# ==========================================
# MULTI-STEP TRAJECTORY FLOW MATCHING TRAINING
# ==========================================
def train_ortho_flow_step(
    transformer,
    latents_t,
    latents_next,
    prompt_embeds,
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

    sketch_latents_t = transformer.sketch_proj(cond_latents_t[:, :, :3]).to(dtype=transformer.dtype)

    # 3. Noise all 4 target views at t+1 simultaneously
    latents_next_fp32 = latents_next.float()
    noise = torch.randn_like(latents_next_fp32)
    noise[:, :, 3:4] += sketch_latents_t.detach().float()
    noisy_latents_next_fp32 = (1.0 - tau_exp) * latents_next_fp32 + tau_exp * noise
    noisy_latents_next = noisy_latents_next_fp32.to(dtype=transformer.dtype)

    
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
    aux_loss = F.mse_loss(sketch_latents_t.float(), latents_next_fp32[:, :, 3:4])

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
        is_last_step = (s == seq_len - 1)

        if not is_last_step:
            step_loss, step_view_losses, step_sketch, pred_next_latents = train_ortho_flow_step(
                transformer=transformer,
                latents_t=curr_latents,
                latents_next=target_latents,
                prompt_embeds=prompt_embeds,
                shift=shift,
                ctx_noise_max=ctx_noise_max,
            )
        else:
            step_loss, step_view_losses, step_sketch, _ = train_ortho_flow_step(
                transformer=transformer,
                latents_t=curr_latents,
                latents_next=target_latents,
                prompt_embeds=prompt_embeds,
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
                    sampled = predict_next_view_latents(transformer, curr_latents, prompt_embeds, steps=4)
                curr_latents = use_ar_mask * sampled + (1.0 - use_ar_mask) * target_latents
            else:
                curr_latents = target_latents

    return total_loss, avg_per_view, avg_sketch

# ==========================================
# AUTOREGRESSIVE VIDEO ROLLOUT & SAMPLING
# ==========================================
@torch.no_grad()
def predict_next_view_latents(transformer, latents_t, prompt_embeds, steps=20, shift=3.0):
    """
    Simultaneously denoises all 4 orthographic views at t+1 via reverse Euler Flow Matching.
    Input:
      latents_t: (B, C, 4, h, w) - clean past views at time t
    Output:
      latents_next: (B, C, 4, h, w) - predicted views at time t+1
    """
    b, c, num_views, h, w = latents_t.shape
    latents_t = latents_t.to(dtype=transformer.dtype)
    sketch_t = transformer.sketch_proj(latents_t[:, :, :3]).to(dtype=transformer.dtype)

    # Keep Euler state in float32 to prevent bfloat16 rounding stagnation
    x_tau = torch.randn((b, c, num_views, h, w), device=latents_t.device, dtype=torch.float32)
    x_tau[:, :, 3:4] += sketch_t.float()

    # Shifted timestep schedule from 1.0 -> 0.0
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=latents_t.device, dtype=torch.float32)
    sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)

    for step_i in range(steps):
        sigma_curr = sigmas[step_i]
        sigma_next = sigmas[step_i + 1]
        dt = sigma_curr - sigma_next

        t_tensor = torch.full((b,), sigma_curr * 1000.0, device=latents_t.device, dtype=transformer.dtype)
        combined = torch.cat([latents_t, sketch_t, x_tau.to(dtype=transformer.dtype)], dim=2)  # (B, C, 9, h, w)

        pred_velocity = transformer(
            hidden_states=combined,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, num_views + 1 :].float()  # (B, C, 4, h, w)
        x_tau = x_tau - dt * pred_velocity

    return x_tau.to(dtype=transformer.dtype)

@torch.no_grad()
def autoregressive_rollout(transformer, vae, text_encoder, tokenizer, env_name, env, agent=None,
                           rollout_steps=16, steps=20, camera_command="Right", seed=42):
    """
    Autoregressive video rollout for all 4 orthographic views.
    Returns:
      gen_views: (vid_top, vid_rear, vid_side, vid_fpv) each (1, T, 3, H, W) in [0, 1]
      gt_views:  (gt_top,  gt_rear,  gt_side,  gt_fpv)  each (1, T, 3, H, W) in [0, 1]
      vis:       (gen_fpv_vis, gt_fpv_vis) pose-annotated FPV videos for logging
    """
    transformer.eval()

    obs, _ = env.reset(seed=seed)
    obs = _skip_intro(env, env_name, obs)
    env.action_space.seed(seed)

    # Same first render as collect_rollouts: "Center" is a no-op right after reset (yaw = pitch = 0)
    img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")

    vt = [cv2.resize(x, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA) for x in [img_top, im_side, im_rear, im_fpv]]
    vt = np.stack([np.transpose(v, (2, 0, 1)) for v in vt])
    vt = np.expand_dims(vt, axis=0)

    curr_views = torch.from_numpy(vt).float().to(device) / 127.5 - 1.0
    curr_latents = encode_views_to_latents(vae, curr_views, out_dtype=transformer.dtype)

    # Frame storage (0: Top, 1: Side, 2: Rear, 3: FPV)
    gen_top, gen_side, gen_rear, gen_fpv = [], [], [], []
    gt_top,  gt_side,  gt_rear,  gt_fpv  = [], [], [], []
    gen_fpv_vis, gt_fpv_vis = [], []  # pose-annotated copies, for logged videos only
    gen_sketch = []  # decoded sketch (the FPV's starting point), for logging only

    print(f"Starting {rollout_steps}-step Autoregressive Rollout for [{env_name}] (camera {camera_command})...")

    for step_i in range(rollout_steps):
        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        # Render the GT target before predicting: training prompts use the camera state read right after this render
        next_top, next_rear, next_side, next_fpv = env.render(camera_pose=camera_command)
        cam_state = _camera_state(env, camera_command)

        # Same float32 action vector the buffer stores (a scalar discrete action becomes length 1)
        act = np.atleast_1d(np.asarray(action, dtype=np.float32))
        prompt = compose_player_prompts(
            [act],
            env_name=env_name,
            camera_pose=cam_state,
        )

        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt"
        ).to(device)
        
        prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)

        # Same sketch the sampler starts the FPV from; decoded for logging only
        sketch_views = decode_latents_to_views(vae, transformer.sketch_proj(curr_latents[:, :, :3]))  # (1, 1, 3, H, W)
        gen_sketch.append(sketch_views[0, 0].cpu())

        next_latents = predict_next_view_latents(
            transformer=transformer,
            latents_t=curr_latents,
            prompt_embeds=prompt_embeds,
            steps=steps,
        )

        decoded_views = decode_latents_to_views(vae, next_latents)  # (1, 4, 3, H, W) in [0, 1]

        # Store generated frames (0: Top, 1: Side, 2: Rear, 3: FPV)
        gen_top.append(decoded_views[0, 0].cpu())
        gen_side.append(decoded_views[0, 1].cpu())
        gen_rear.append(decoded_views[0, 2].cpu())

        vn = [cv2.resize(x, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA) for x in [next_top, next_side, next_rear, next_fpv]]
        vn = np.stack([np.transpose(v, (2, 0, 1)) for v in vn])

        gt_vn = torch.from_numpy(vn).float() / 255.0  # (4, 3, H, W)
        gt_top.append(gt_vn[0])
        gt_side.append(gt_vn[1])
        gt_rear.append(gt_vn[2])

        # Raw FPV frames are used for PSNR; the pose overlay only goes on the logged videos
        curr_gen_fpv = decoded_views[0, 3].cpu()
        curr_gt_fpv = gt_vn[3]

        gen_fpv.append(curr_gen_fpv)
        gt_fpv.append(curr_gt_fpv)
        gen_fpv_vis.append(_annotate_frame_pose(curr_gen_fpv, camera_command))
        gt_fpv_vis.append(_annotate_frame_pose(curr_gt_fpv, camera_command))

        # Re-encode decoded RGB frame back through VAE to stay on the clean VAE manifold
        next_views_norm = decoded_views * 2.0 - 1.0
        curr_latents = encode_views_to_latents(vae, next_views_norm, out_dtype=transformer.dtype)

        # Training windows never cross an episode reset, so stop instead of jumping to a new episode
        if terminated or truncated:
            break

    vid_gen_top  = torch.stack(gen_top,  dim=0).unsqueeze(0)
    vid_gen_side = torch.stack(gen_side, dim=0).unsqueeze(0)
    vid_gen_rear = torch.stack(gen_rear, dim=0).unsqueeze(0)
    vid_gen_fpv  = torch.stack(gen_fpv,  dim=0).unsqueeze(0)

    vid_gt_top   = torch.stack(gt_top,   dim=0).unsqueeze(0)
    vid_gt_side  = torch.stack(gt_side,  dim=0).unsqueeze(0)
    vid_gt_rear  = torch.stack(gt_rear,  dim=0).unsqueeze(0)
    vid_gt_fpv   = torch.stack(gt_fpv,   dim=0).unsqueeze(0)
    vid_gen_fpv_vis = torch.stack(gen_fpv_vis, dim=0).unsqueeze(0)
    vid_gt_fpv_vis  = torch.stack(gt_fpv_vis,  dim=0).unsqueeze(0)
    vid_gen_sketch  = torch.stack(gen_sketch,  dim=0).unsqueeze(0)

    transformer.train()
    env.close()
    return (vid_gen_top, vid_gen_rear, vid_gen_side, vid_gen_fpv,
            vid_gt_top,  vid_gt_rear,  vid_gt_side,  vid_gt_fpv,
            vid_gen_fpv_vis, vid_gt_fpv_vis, vid_gen_sketch)


@torch.no_grad()
def evaluate_and_log_videos(transformer, vae, text_encoder, tokenizer, writer, epoch, agents=None, rollout_steps=16, steps=20, fps=15, seed=42):
    """
    Performs autoregressive video rollouts for each view and writes the videos to TensorBoard.
    `agents` maps env_name -> the policy used during data collection (random actions if missing).
    """
    agents = agents or {}
    transformer.eval()
    unique_envs = ['Bipedal Walker', 'Lunar Lander', 'MultiCar Racing']
    reg_dict = dict(_build_registry())
    
    print(f"\n[VALIDATION] Running autoregressive video rollouts for Epoch {epoch} across envs: {', '.join(unique_envs)}...")

    for env_idx, env_name in enumerate(unique_envs):
        env = reg_dict[env_name]()

        (vid_top_down, vid_rear, vid_side, vid_fpv,
         vid_gt_top_down, vid_gt_rear, vid_gt_side, vid_gt_fpv,
         vid_fpv_vis, vid_gt_fpv_vis, vid_sketch) = autoregressive_rollout(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            env_name=env_name,
            env=env,
            agent=agents.get(env_name),
            rollout_steps=rollout_steps,
            steps=steps,
            seed=seed,
        )

        clean_tag = env_name.replace(" ", "_").lower()

        psnr_top_down = _psnr(vid_top_down, vid_gt_top_down)
        psnr_rear     = _psnr(vid_rear, vid_gt_rear)
        psnr_side     = _psnr(vid_side, vid_gt_side)
        psnr_fpv      = _psnr(vid_fpv, vid_gt_fpv)
        psnr_mean     = (psnr_top_down + psnr_rear + psnr_side + psnr_fpv) / 4.0

        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Top Down", psnr_top_down, epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Rear",     psnr_rear,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Side",     psnr_side,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/FPV",      psnr_fpv,      epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Sketch",   _psnr(vid_sketch, vid_gt_fpv), epoch)  # sketch vs GT FPV_{t+1}
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Mean",     psnr_mean,     epoch)

        print(f"  [Epoch {epoch}] [{env_name}] Autoregressive Video PSNR: {psnr_mean:.2f} dB "
              f"(Top-Down: {psnr_top_down:.2f}dB, Rear: {psnr_rear:.2f}dB, Side: {psnr_side:.2f}dB, FPV: {psnr_fpv:.2f}dB)")

        # Always write per-env tagged videos; also alias as generic Test/ for the first env
        writer.add_video(f"Test/{clean_tag}/Top-Down", vid_top_down, epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Rear",     vid_rear,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Side",     vid_side,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/FPV",      vid_fpv_vis,  epoch, fps=fps)

        writer.add_video(f"GT/{clean_tag}/Top-Down", vid_gt_top_down, epoch, fps=fps)
        writer.add_video(f"GT/{clean_tag}/Rear",     vid_gt_rear,     epoch, fps=fps)
        writer.add_video(f"GT/{clean_tag}/Side",     vid_gt_side,     epoch, fps=fps)
        writer.add_video(f"GT/{clean_tag}/FPV",      vid_gt_fpv_vis,  epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Sketch",  vid_sketch,      epoch, fps=fps)

    transformer.train()

def _skip_intro(env, env_name, obs):
    """Steps CarRacing past its zoom-in intro with no-op actions so those frames never reach the buffer."""
    if env_name == "MultiCar Racing":
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.zeros(3))
    return obs

def _camera_state(env, command):
    """Camera state stored with each transition and fed to the prompt; read right after env.render(command)."""
    return {
        "command": command,
        "x": getattr(env, "cam_x", 0.0),
        "y": getattr(env, "cam_y", 0.0),
        "z": getattr(env, "cam_z", 0.0),
        "roll": getattr(env, "cam_roll", 0.0),
        "yaw": getattr(env, "cam_yaw", 0.0),
        "pitch": getattr(env, "cam_pitch", 0.0),
    }

def collect_rollouts(env, agent, buffer, n_steps, seed, env_name="Bipedal Walker"):
    """Collects camera rollouts storing stateful camera poses."""
    obs, _ = env.reset(seed=seed)
    obs = _skip_intro(env, env_name, obs)
    env.action_space.seed(seed)
    state = torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)

    pose_hold = 0

    # Render once per env step (each render advances the camera), so views_next[k] == views_t[k+1]
    # and every transition is one camera increment, matching the validation rollout.
    # "Center" is a no-op right after reset (yaw = pitch = 0).
    img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")

    for _ in tqdm(range(n_steps), desc=f"Rollouts [{env_name}]"):
        if pose_hold <= 0:
          current_pose = random.choice(["Center", "Up", "Down", "Left", "Right"])
          pose_hold = random.randint(8, 16)
        else:
          pose_hold -= 1

        with torch.inference_mode():
          if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
          else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_top, next_rear, next_side, next_fpv = env.render(
            camera_pose=current_pose
        )

        # Extract wrapper's stateful angles if available
        cam_state = _camera_state(env, current_pose)

        buffer.push(
            top_t=img_top,
            side_t=im_side,
            rear_t=im_rear,
            fpv_t=im_fpv,
            top_next=next_top,
            side_next=next_side,
            rear_next=next_rear,
            fpv_next=next_fpv,
            action=action,
            done=done,
            env_name=env_name,
            camera_state=cam_state,
        )

        if done:
          obs, _ = env.reset()
          obs = _skip_intro(env, env_name, obs)
          pose_hold = 0
          img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")
        else:
          obs = next_obs
          img_top, im_rear, im_side, im_fpv = next_top, next_rear, next_side, next_fpv
        state = torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
    env.close()

def collect_all_envs_rollouts(registry, buffer, rollout_steps=150, num_envs=3):
    """Loops through all registered environments and populates the multi-env buffer.
    Returns (active_envs, agents) where agents maps env_name -> the collection policy (reused by validation)."""
    print("=" * 60)
    print(f"Collecting rollouts across all {min(num_envs, len(registry))} environments...")
    print("=" * 60)

    active_envs = []
    agents = {}
    for idx, (env_name, factory) in enumerate(registry[:num_envs]):
        print(f"\n[{idx + 1}/{len(registry)}] Initializing environment: {env_name}")
        env = factory()

        obs, info = env.reset()
        state_dim = int(np.prod(obs.shape if hasattr(obs, 'shape') else env.observation_space.shape))
        clean_env_name = "".join(c for c in env_name if c.isalnum() or c in ('_', '-')).lower()
        ckpt_path = os.path.join("checkpoints", f"ppo_{clean_env_name}.pt")
        
        if env_name == "Lunar Lander":
            checkpoint = load_from_hub(
                repo_id="Adilbai/ppo-LunarLander-v2",
                filename="ppo-LunarLander-v2.zip"
            )
            agent = PPO.load(checkpoint)

        elif env_name == "MultiCar Racing":
            checkpoint = load_from_hub(
                repo_id="igpaub/ppo-CarRacing-v2",
                filename="ppo-CarRacing-v2.zip"
            )
            agent = PPO.load(checkpoint)

        else:
            print(f"Loading PPO agent from: {ckpt_path}")
            agent = load_agent(ckpt_path, state_dim, env.action_space, device)
        
        collect_rollouts(env, agent, buffer, n_steps=rollout_steps, seed=42, env_name=env_name)
        active_envs.append((env_name, env))
        agents[env_name] = agent

    unique_recorded = list(dict.fromkeys([e for e in buffer.env_names[:buffer.size] if e is not None]))
    print(f"\n[INFO] Buffer populated with {buffer.size} transitions across {len(unique_recorded)} environments: {', '.join(unique_recorded)}")
    return active_envs, agents


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
        registry, buffer, rollout_steps=args.rollout_steps, num_envs=args.num_envs
    )

    # 2. Model & LoRA Setup
    model, vae, text_encoder, tokenizer = load_model(
        model_id=args.model_id, lora_rank=args.lora_rank
    )
    model.sketch_proj = LinearProjector().to(device)

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

            optimizer.zero_grad(set_to_none=True)

            # Unroll multi-step trajectory with student forcing & context noise
            loss, per_view_losses, sketch_loss = train_ortho_trajectory_step(
                transformer=model,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                teacher_forcing_prob=curr_teacher_forcing_prob,
                ctx_noise_max=args.ctx_noise_max,
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_envs", type=int, default=12, help="Number of training envs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)