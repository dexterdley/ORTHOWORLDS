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

# HuggingFace & Diffusers imports
from utils import _build_registry, load_model, compose_player_prompts, set_active_env, set_seed, load_agent,  _annotate_frame_pose, _psnr
from buffer import OrthoTransitionBuffer

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

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

# ==========================================
# ORTHO -> FPV PROJECTOR (camera-pose conditioned)
# ==========================================
POSE_COMMANDS = ["Center", "Up", "Down", "Left", "Right"]
# Fixed env list (same names as `_build_registry`), so env indices don't shift when an env fails to register
ENV_NAMES = [
    "Bipedal Walker", "Lunar Lander", "MultiCar Racing", "Mario Escape", "Drone Dogfight",
    "Excavator", "Tank Duel", "Catapult War", "Robot Sumo", "Mountain Goat",
    "Wrecking Ball", "Toxic Gas Escape", "Rescue Ambulance", "Parkour Runner", "Snowball Fight",
]
# command one-hot + sin/cos of (yaw, pitch, roll) + signed-log (x, y, z) + env one-hot
POSE_DIM = len(POSE_COMMANDS) + 6 + 3 + len(ENV_NAMES)

def encode_camera_poses(camera_poses, env_names=None, target_device=device):
    """
    List of B camera-state dicts (from the buffer) [+ B env names] -> (B, POSE_DIM) float tensor.
    Angles are radians and yaw accumulates without bound, so they are encoded as sin/cos.
    Camera x/y/z are unbounded world coordinates, so they are squashed with sign(p) * log(1 + |p|).
    The env one-hot goes through the projector's pose MLP, i.e. it acts as a learned env embedding
    (each env has its own FPV fov / orbit pivot / ortho scale); unknown or missing envs are all-zero.
    """
    if env_names is None:
        env_names = [None] * len(camera_poses)
    feats = []
    for cam, env_name in zip(camera_poses, env_names):
        cam = cam or {}
        one_hot = [1.0 if cam.get("command", "Center") == c else 0.0 for c in POSE_COMMANDS]
        angles = [float(cam.get(k, 0.0)) for k in ("yaw", "pitch", "roll")]
        position = [float(cam.get(k, 0.0)) for k in ("x", "y", "z")]
        position = [math.copysign(math.log1p(abs(p)), p) for p in position]
        env_one_hot = [1.0 if env_name == e else 0.0 for e in ENV_NAMES]
        feats.append(one_hot + [math.sin(a) for a in angles] + [math.cos(a) for a in angles] + position + env_one_hot)
    return torch.tensor(feats, dtype=torch.float32, device=target_device)

class OrthoProjector(nn.Module):
    """
    (top, side, rear) latents (B,16,h,w) each + FPV camera pose (B,POSE_DIM) -> fpv latent (B,16,h,w).
    - Pose enters via FiLM on every hidden conv, so the projector knows where the FPV camera looks
      (without it, MSE regresses to a blur of all camera poses).
    - One global spatial-mixing layer lets every output cell read every input cell, since the
      ortho -> perspective mapping is non-local (an FPV pixel is a ray across the top view).
    """
    def __init__(self, c_lat=16, hidden=128, pose_dim=POSE_DIM, hw=12, num_layers=3):
        super().__init__()
        self.inp = nn.Conv2d(3 * c_lat, hidden, 3, padding=1)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.convs = nn.ModuleList([nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(num_layers)])
        self.films = nn.ModuleList([nn.Linear(hidden, 2 * hidden) for _ in range(num_layers)])
        self.spatial_mix = nn.Linear(hw * hw, hw * hw)
        self.out = nn.Conv2d(hidden, c_lat, 3, padding=1)
        self.act = nn.SiLU()

        # FiLM starts as identity (scale=1, shift=0)
        for film in self.films:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def forward(self, top, side, rear, pose):
        x = torch.cat([top, side, rear], dim=1).float()  # (B, 48, h, w)
        x = self.act(self.inp(x))
        p = self.pose_mlp(pose.float())

        for i, (conv, film) in enumerate(zip(self.convs, self.films)):
            scale, shift = film(p).chunk(2, dim=-1)
            x = self.act(conv(x) * (1.0 + scale[..., None, None]) + shift[..., None, None])
            if i == 0:
                b, c, h, w = x.shape
                x = x + self.spatial_mix(x.flatten(2)).view(b, c, h, w)
        return self.out(x)

class OrthoProjectorAttn(nn.Module):
    """
    (a, b, c) latents (B,16,h,w) each + FPV camera pose (B,POSE_DIM) -> target latent (B,16,h,w).
    Same call signature as OrthoProjector, but the spatial mapping itself is pose-dependent:
    - Each input cell of the 3 views becomes a token (+ learned position/view embedding).
    - Each output cell is a learned query + pose embedding that cross-attends over all 3*h*w input
      tokens, so the camera pose decides *where* each output cell reads from (needed for the
      orbiting perspective FPV; FiLM in OrthoProjector can only change *what*, not *where*).
    """
    def __init__(self, c_lat=16, hidden=128, pose_dim=POSE_DIM, hw=12, num_layers=2, num_heads=4):
        super().__init__()
        n = hw * hw
        self.inp = nn.Linear(c_lat, hidden)
        self.in_pos = nn.Parameter(torch.randn(1, 3 * n, hidden) * 0.02)  # per input view & position
        self.queries = nn.Parameter(torch.randn(1, n, hidden) * 0.02)     # one per output position
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerDecoderLayer(hidden, num_heads, 4 * hidden, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(hidden, c_lat)

    def forward(self, a, b, c, pose):
        bsz, c_lat, h, w = a.shape
        tokens = torch.cat([v.float().flatten(2).transpose(1, 2) for v in (a, b, c)], dim=1)  # (B, 3*h*w, C)
        memory = self.inp(tokens) + self.in_pos
        queries = self.queries.expand(bsz, -1, -1) + self.pose_mlp(pose.float()).unsqueeze(1)  # (B, h*w, hidden)
        y = self.out(self.decoder(queries, memory))  # (B, h*w, C)
        return y.transpose(1, 2).reshape(bsz, c_lat, h, w)

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
    Returns: loss, per_view_losses, rollout_latent, pred_x0 (detached, fp32)
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

    # 3. Noise all 4 target views at t+1 simultaneously
    latents_next_fp32 = latents_next.float()
    noise = torch.randn_like(latents_next_fp32)
    noisy_latents_next_fp32 = (1.0 - tau_exp) * latents_next_fp32 + tau_exp * noise
    noisy_latents_next = noisy_latents_next_fp32.to(dtype=transformer.dtype)

    # 4. Concatenate conditioning views (frames 0:4) and noisy target views (frames 4:8)
    combined_latents = torch.cat([cond_latents_t, noisy_latents_next], dim=2)

    # 5. Ground truth velocity: v_target = noise - x_0
    target_velocity = noise - latents_next_fp32

    # 6. DiT Forward pass
    timesteps = (tau * 1000.0).to(dtype=transformer.dtype)
    model_pred = transformer(
        hidden_states=combined_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
    ).sample

    # 7. Supervise target views at t+1 (indices num_views:2*num_views)
    pred_velocity = model_pred[:, :, num_views : 2 * num_views].float()

    per_view_losses = [
        F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
        for idx in range(num_views)
    ]
    loss = sum(per_view_losses) / float(num_views)

    # Reconstruct 1-step denoised prediction: x_0_hat = x_tau - tau * v_pred
    with torch.no_grad():
        pred_x0 = (noisy_latents_next_fp32 - tau_exp * pred_velocity.detach()).clamp(-6.0, 6.0)
        confidence = (1.0 - tau_exp).clamp(0.2, 1.0)
        rollout_latent = (confidence * pred_x0 + (1.0 - confidence) * latents_next_fp32).to(dtype=transformer.dtype)
    return loss, [l.item() for l in per_view_losses], rollout_latent, pred_x0

def train_ortho_trajectory_step(
    transformer,
    fpv_projector,
    top_projector,
    rear_projector,
    side_projector,
    latents_t0,
    latents_next_seq,
    prompt_embeds_seq,
    camera_poses_seq,
    env_names_seq=None,
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
    - Jointly trains the 4 per-view projectors to match inference, where they see the DiT's predicted
      `next_latents`: inputs are the DiT's own x0 estimate `pred_x0` at t+1, targets are the GT views
      at t+1, conditioned on the FPV camera pose + env at t+1. `pred_x0` is detached, so no gradient reaches the DiT.
    """
    seq_len = len(latents_next_seq)
    num_views = latents_t0.shape[2]

    curr_latents = latents_t0
    total_loss = 0.0
    avg_per_view = [0.0] * num_views
    avg_proj_loss = 0.0
    avg_proj_zero_baseline = 0.0

    for s in range(seq_len):
        target_latents = latents_next_seq[s]
        prompt_embeds = prompt_embeds_seq[s]
        is_last_step = (s == seq_len - 1)

        if not is_last_step:
            step_loss, step_view_losses, pred_next_latents, pred_x0 = train_ortho_flow_step(
                transformer=transformer,
                latents_t=curr_latents,
                latents_next=target_latents,
                prompt_embeds=prompt_embeds,
                shift=shift,
                ctx_noise_max=ctx_noise_max,
            )
        else:
            step_loss, step_view_losses, _, pred_x0 = train_ortho_flow_step(
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

        # Projectors: DiT's predicted views at t+1 (like `next_latents` at inference) + FPV pose at t+1 -> GT views at t+1
        gt_next = target_latents.detach().float()
        gt_top_next, gt_side_next, gt_rear_next, gt_fpv_next = gt_next[:, :, 0], gt_next[:, :, 1], gt_next[:, :, 2], gt_next[:, :, 3]
        env_names_next = env_names_seq[s] if env_names_seq is not None else None
        pose_next = encode_camera_poses(camera_poses_seq[s], env_names=env_names_next, target_device=gt_next.device)

        pred_next = pred_x0.detach().float()
        pred_top, pred_side, pred_rear, pred_fpv = pred_next[:, :, 0], pred_next[:, :, 1], pred_next[:, :, 2], pred_next[:, :, 3]

        proj_fpv_next = fpv_projector(pred_top, pred_side, pred_rear, pose_next)
        proj_top_next = top_projector(pred_fpv, pred_side, pred_rear, pose_next)
        proj_rear_next = rear_projector(pred_top, pred_side, pred_fpv, pose_next)
        proj_side_next = side_projector(pred_top, pred_rear, pred_fpv, pose_next)

        proj_loss = F.mse_loss(proj_fpv_next, gt_fpv_next) + \
                    F.mse_loss(proj_top_next, gt_top_next) + \
                    F.mse_loss(proj_rear_next, gt_rear_next) + \
                    F.mse_loss(proj_side_next, gt_side_next)

        (proj_loss / float(seq_len)).backward()

        total_loss += step_loss.item() / float(seq_len)
        for v_idx in range(num_views):
            avg_per_view[v_idx] += step_view_losses[v_idx] / float(seq_len)
        avg_proj_loss += proj_loss.item() / float(seq_len)
        # Zero-prediction baseline summed over the 4 views, comparable to proj_loss (sum of 4 view MSEs)
        avg_proj_zero_baseline += gt_next.pow(2).mean(dim=(0, 1, 3, 4)).sum().item() / float(seq_len)

        # Prepare context for step s + 1 (Student Forcing vs. Teacher Forcing per sample)
        if not is_last_step:
            if teacher_forcing_prob > 0.0:
                b = curr_latents.shape[0]
                use_ar_mask = (torch.rand((b, 1, 1, 1, 1), device=curr_latents.device) < teacher_forcing_prob).to(dtype=transformer.dtype)
                curr_latents = use_ar_mask * pred_next_latents + (1.0 - use_ar_mask) * target_latents
            else:
                curr_latents = target_latents

    return total_loss, avg_per_view, avg_proj_loss, avg_proj_zero_baseline

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

    # Keep Euler state in float32 to prevent bfloat16 rounding stagnation
    x_tau = torch.randn((b, c, num_views, h, w), device=latents_t.device, dtype=torch.float32)

    # Shifted timestep schedule from 1.0 -> 0.0
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=latents_t.device, dtype=torch.float32)
    sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)

    for step_i in range(steps):
        sigma_curr = sigmas[step_i]
        sigma_next = sigmas[step_i + 1]
        dt = sigma_curr - sigma_next

        t_tensor = torch.full((b,), sigma_curr * 1000.0, device=latents_t.device, dtype=transformer.dtype)
        combined = torch.cat([latents_t, x_tau.to(dtype=transformer.dtype)], dim=2)  # (B, C, 8, h, w)

        pred_velocity = transformer(
            hidden_states=combined,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, num_views : 2 * num_views].float()
        x_tau = x_tau - dt * pred_velocity

    return x_tau.to(dtype=transformer.dtype)


@torch.no_grad()
def autoregressive_rollout(
    transformer,
    fpv_projector,
    top_projector,
    rear_projector,
    side_projector,
    vae, text_encoder, tokenizer, buffer, env_name=None, env=None, rollout_steps=16, steps=20,
):
    """
    Autoregressive video rollout for all 4 orthographic views.
    Returns:
      gen_views:  (vid_top, vid_rear, vid_side, vid_fpv) each (1, T, 3, H, W) in [0, 1]
      gt_views:   (gt_top,  gt_rear,  gt_side,  gt_fpv)  each (1, T, 3, H, W) in [0, 1]
      proj_views: (proj_top, proj_rear, proj_side, proj_fpv) each (1, T, 3, H, W) in [0, 1] - each
                  projector applied to the DiT's 3 *predicted* other views (same input order as training).
                  Report-only: the DiT's own views are still what feed the next rollout step.
    """
    transformer.eval()
    fpv_projector.eval()
    top_projector.eval()
    rear_projector.eval()
    side_projector.eval()
    proj_top, proj_side, proj_rear, proj_fpv = [], [], [], []

    # 1. Grab continuous sequence from buffer (without episode boundaries)
    vt_seq, vn_seq, act_seq, env_seq, cam_seq = buffer.sample_sequence(seq_len=rollout_steps, env_filter=env_name, return_camera_states=True)

    obs, _ = env.reset()
    if env_name == "MultiCar Racing":
        for _ in tqdm(range(50)): # skip first 50 frames
            obs, reward, terminated, truncated, info = env.step(np.zeros(3))

    img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Right")

    # All 4 views jointly as one sample: (1, 4, 3, H, W) -> latents (1, C, 4, h, w)
    vt = [cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA) for x in [img_top, im_side, im_rear, im_fpv]]
    vt = np.stack([np.transpose(v, (2, 0, 1)) for v in vt])
    vt = np.expand_dims(vt, axis=0)

    curr_views = torch.from_numpy(vt).float().to(device) / 127.5 - 1.0
    curr_latents = encode_views_to_latents(vae, curr_views, out_dtype=transformer.dtype)

    # Frame storage (0: Top, 1: Side, 2: Rear, 3: FPV)
    gen_top, gen_side, gen_rear, gen_fpv = [], [], [], []
    gt_top,  gt_side,  gt_rear,  gt_fpv  = [], [], [], []
    
    print(f"Starting {rollout_steps}-step Autoregressive Rollout for [{env_name or 'Default'}]...")

    for step_i in range(rollout_steps):
        act = act_seq[step_i]
        
        # Adapt action for env.step (Discrete vs Box)
        step_act = act
        if hasattr(env, "action_space"):
            if hasattr(env.action_space, "n"):  # Discrete (e.g. Lunar Lander)
                if isinstance(step_act, (np.ndarray, list)):
                    step_act = int(round(float(step_act[0])))
                else:
                    step_act = int(round(float(step_act)))
                step_act = max(0, min(step_act, env.action_space.n - 1))
            elif hasattr(env.action_space, "shape"):  # Box (e.g. Bipedal Walker, Car Racing)
                step_act = np.array(step_act, dtype=np.float32).flatten()
                target_dim = env.action_space.shape[0]
                if step_act.size >= target_dim:
                    step_act = step_act[:target_dim]
                else:
                    step_act = env.action_space.sample()
                step_act = np.clip(step_act, env.action_space.low, env.action_space.high)

        sample_env = env_seq[step_i] or env_name
        sample_cam = dict(cam_seq[step_i])  # copy so the buffer's stored dict isn't mutated

        sample_cam["command"] = "Right"
        sample_cam["x"] += 1.0

        # Compose prompt with stateful context
        prompt = compose_player_prompts(
            [act],
            env_name=sample_env,
            camera_pose=sample_cam,
        )

        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt"
        ).to(device)
        
        prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)

        next_latents = predict_next_view_latents(
            transformer=transformer,
            latents_t=curr_latents,
            prompt_embeds=prompt_embeds,
            steps=steps,
        )

        next_obs, reward, terminated, truncated, _ = env.step(step_act)
        next_top, next_rear, next_side, next_fpv = env.render(
            camera_pose=sample_cam["command"]
        )

        # FPV pose from the env's actual camera after this render (same source as the buffer at
        # collection); read before a possible reset so it belongs to the rendered GT frame
        pose_next = encode_camera_poses([{
            "command": sample_cam["command"],
            "x": getattr(env, "cam_x", 0.0),
            "y": getattr(env, "cam_y", 0.0),
            "z": getattr(env, "cam_z", 0.0),
            "yaw": getattr(env, "cam_yaw", 0.0),
            "pitch": getattr(env, "cam_pitch", 0.0),
            "roll": getattr(env, "cam_roll", 0.0),
        }], env_names=[sample_env])

        if terminated or truncated:
            env.reset()
        vn = [cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA) for x in [next_top, next_side, next_rear, next_fpv]]
        vn = np.stack([np.transpose(v, (2, 0, 1)) for v in vn])

        gt_vn = torch.from_numpy(vn).float() / 255.0  # (4, 3, H, W)
        gt_top.append(gt_vn[0])
        gt_side.append(gt_vn[1])
        gt_rear.append(gt_vn[2])

        # Projectors on the DiT's predicted views at t+1
        pred_top_next, pred_side_next, pred_rear_next, pred_fpv_next = next_latents[:, :, 0], next_latents[:, :, 1], next_latents[:, :, 2], next_latents[:, :, 3]

        proj_fpv_next = fpv_projector(pred_top_next, pred_side_next, pred_rear_next, pose_next)
        proj_top_next = top_projector(pred_fpv_next, pred_side_next, pred_rear_next, pose_next)
        proj_rear_next = rear_projector(pred_top_next, pred_side_next, pred_fpv_next, pose_next)
        proj_side_next = side_projector(pred_top_next, pred_rear_next, pred_fpv_next, pose_next)

        # Single VAE decode for DiT + projector views; view axis 0-3 = DiT, 4-7 = projectors,
        # each block in (0: Top, 1: Side, 2: Rear, 3: FPV) order
        proj_latents = torch.stack([proj_top_next, proj_side_next, proj_rear_next, proj_fpv_next], dim=2)  # (1, C, 4, h, w)
        decoded_all = decode_latents_to_views(vae, torch.cat([next_latents.float(), proj_latents], dim=2))  # (1, 8, 3, H, W) in [0, 1]
        decoded_views, decoded_proj = decoded_all[:, :4], decoded_all[:, 4:]

        # Store generated frames (0: Top, 1: Side, 2: Rear, 3: FPV)
        gen_top.append(decoded_views[0, 0].cpu())
        gen_side.append(decoded_views[0, 1].cpu())
        gen_rear.append(decoded_views[0, 2].cpu())

        proj_top.append(decoded_proj[0, 0].cpu())
        proj_side.append(decoded_proj[0, 1].cpu())
        proj_rear.append(decoded_proj[0, 2].cpu())
        proj_fpv.append(_annotate_frame_pose(decoded_proj[0, 3].cpu(), sample_cam["command"]))  # GT FPV carries the same overlay

        # Add the input pose overlay to the generated and ground truth FPV frames
        curr_gen_fpv = decoded_views[0, 3].cpu()
        curr_gt_fpv = gt_vn[3]

        curr_gen_fpv = _annotate_frame_pose(curr_gen_fpv, sample_cam["command"])
        curr_gt_fpv = _annotate_frame_pose(curr_gt_fpv, sample_cam["command"])

        gen_fpv.append(curr_gen_fpv)
        gt_fpv.append(curr_gt_fpv)

        # Re-encode decoded RGB frame back through VAE to stay on the clean VAE manifold
        next_views_norm = decoded_views * 2.0 - 1.0
        curr_latents = encode_views_to_latents(vae, next_views_norm, out_dtype=transformer.dtype)

    vid_gen_top  = torch.stack(gen_top,  dim=0).unsqueeze(0)
    vid_gen_side = torch.stack(gen_side, dim=0).unsqueeze(0)
    vid_gen_rear = torch.stack(gen_rear, dim=0).unsqueeze(0)
    vid_gen_fpv  = torch.stack(gen_fpv,  dim=0).unsqueeze(0)

    vid_gt_top   = torch.stack(gt_top,   dim=0).unsqueeze(0)
    vid_gt_side  = torch.stack(gt_side,  dim=0).unsqueeze(0)
    vid_gt_rear  = torch.stack(gt_rear,  dim=0).unsqueeze(0)
    vid_gt_fpv   = torch.stack(gt_fpv,   dim=0).unsqueeze(0)

    vid_proj_top  = torch.stack(proj_top,  dim=0).unsqueeze(0)
    vid_proj_side = torch.stack(proj_side, dim=0).unsqueeze(0)
    vid_proj_rear = torch.stack(proj_rear, dim=0).unsqueeze(0)
    vid_proj_fpv  = torch.stack(proj_fpv,  dim=0).unsqueeze(0)

    fpv_projector.train()
    top_projector.train()
    rear_projector.train()
    side_projector.train()
    transformer.train()
    env.close()
    return (vid_gen_top,  vid_gen_rear,  vid_gen_side,  vid_gen_fpv,
            vid_gt_top,   vid_gt_rear,   vid_gt_side,   vid_gt_fpv,
            vid_proj_top, vid_proj_rear, vid_proj_side, vid_proj_fpv)



@torch.no_grad()
def evaluate_and_log_videos(
    transformer,
    fpv_projector,
    top_projector,
    rear_projector,
    side_projector,
    vae, buffer, text_encoder, tokenizer, writer, epoch, rollout_steps=16, steps=20, fps=15,
):
    """
    Performs autoregressive video rollouts for each view and writes the videos to TensorBoard.
    """
    transformer.eval()
    unique_envs = ['Bipedal Walker', 'Lunar Lander', 'MultiCar Racing']
    reg_dict = dict(_build_registry())
    
    print(f"\n[VALIDATION] Running autoregressive video rollouts for Epoch {epoch} across envs: {', '.join(unique_envs)}...")

    for env_idx, env_name in enumerate(unique_envs):
        env = reg_dict[env_name]()
        env.reset()

        (vid_top_down, vid_rear, vid_side, vid_fpv,
         vid_gt_top_down, vid_gt_rear, vid_gt_side, vid_gt_fpv,
         vid_proj_top_down, vid_proj_rear, vid_proj_side, vid_proj_fpv) = autoregressive_rollout(
            transformer=transformer,
            fpv_projector=fpv_projector,
            top_projector=top_projector,
            rear_projector=rear_projector,
            side_projector=side_projector,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            buffer=buffer,
            env_name=env_name,
            env=env,
            rollout_steps=rollout_steps,
            steps=steps,
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
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Mean",     psnr_mean,     epoch)

        print(f"  [Epoch {epoch}] [{env_name}] Autoregressive Video PSNR: {psnr_mean:.2f} dB "
              f"(Top-Down: {psnr_top_down:.2f}dB, Rear: {psnr_rear:.2f}dB, Side: {psnr_side:.2f}dB, FPV: {psnr_fpv:.2f}dB)")

        # Always write per-env tagged videos; also alias as generic Test/ for the first env
        writer.add_video(f"Test/{clean_tag}/Top-Down", vid_top_down, epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Rear",     vid_rear,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Side",     vid_side,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/FPV",      vid_fpv,      epoch, fps=fps)

        # Projectors applied to the DiT's predicted views, compared against the same GT
        psnr_proj_top_down = _psnr(vid_proj_top_down, vid_gt_top_down)
        psnr_proj_rear     = _psnr(vid_proj_rear,     vid_gt_rear)
        psnr_proj_side     = _psnr(vid_proj_side,     vid_gt_side)
        psnr_proj_fpv      = _psnr(vid_proj_fpv,      vid_gt_fpv)

        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Proj_Top Down", psnr_proj_top_down, epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Proj_Rear",     psnr_proj_rear,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Proj_Side",     psnr_proj_side,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Proj_FPV",      psnr_proj_fpv,      epoch)

        print(f"  [Epoch {epoch}] [{env_name}] Projector PSNR (DiT -> Proj): "
              f"Top-Down: {psnr_top_down:.2f} -> {psnr_proj_top_down:.2f}dB, Rear: {psnr_rear:.2f} -> {psnr_proj_rear:.2f}dB, "
              f"Side: {psnr_side:.2f} -> {psnr_proj_side:.2f}dB, FPV: {psnr_fpv:.2f} -> {psnr_proj_fpv:.2f}dB")

        writer.add_video(f"Test/{clean_tag}/Proj_Top-Down", vid_proj_top_down, epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Proj_Rear",     vid_proj_rear,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Proj_Side",     vid_proj_side,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Proj_FPV",      vid_proj_fpv,      epoch, fps=fps)

    transformer.train()

def collect_rollouts(env, agent, buffer, n_steps, seed, env_name="Bipedal Walker"):
    """Collects camera rollouts storing stateful camera poses."""
    obs, _ = env.reset(seed=seed)
    state = torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)

    pose_hold = 0

    for _ in tqdm(range(n_steps), desc=f"Rollouts [{env_name}]"):
        if pose_hold <= 0:
          current_pose = random.choice(["Center", "Up", "Down", "Left", "Right"])
          pose_hold = random.randint(8, 16)
        else:
          pose_hold -= 1

        # Render pass returns orthographic views + FPV
        img_top, im_rear, im_side, im_fpv = env.render(camera_pose=current_pose)

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
        cam_state = {
            "command": current_pose,
            "x": getattr(env, "cam_x", 0.0),
            "y": getattr(env, "cam_y", 0.0),
            "z": getattr(env, "cam_z", 0.0),
            "roll": getattr(env, "cam_roll", 0.0),
            "yaw": getattr(env, "cam_yaw", 0.0),
            "pitch": getattr(env, "cam_pitch", 0.0),
        }

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
          current_pose = "Center"
          pose_hold = 0
        else:
          obs = next_obs
        state = torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
    env.close()

def collect_all_envs_rollouts(registry, buffer, rollout_steps=150, num_envs=3):
    """Loops through all registered environments and populates the multi-env buffer."""
    print("=" * 60)
    print(f"Collecting rollouts across all {min(num_envs, len(registry))} environments...")
    print("=" * 60)

    active_envs = []
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
            for _ in tqdm(range(50)): # skip first 50 frames
                obs, reward, terminated, truncated, info = env.step(np.zeros(3))

        else:
            print(f"Loading PPO agent from: {ckpt_path}")
            agent = load_agent(ckpt_path, state_dim, env.action_space, device)
        
        collect_rollouts(env, agent, buffer, n_steps=rollout_steps, seed=42, env_name=env_name)
        active_envs.append((env_name, env))

    unique_recorded = list(dict.fromkeys([e for e in buffer.env_names[:buffer.size] if e is not None]))
    print(f"\n[INFO] Buffer populated with {buffer.size} transitions across {len(unique_recorded)} environments: {', '.join(unique_recorded)}")
    return active_envs


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
        capacity=args.buffer_capacity, img_h=96, img_w=96, max_action_len=8
    )

    active_envs = []
    active_envs = collect_all_envs_rollouts(
        registry, buffer, rollout_steps=args.rollout_steps, num_envs=args.num_envs
    )

    # 2. Model & LoRA Setup
    model, vae, text_encoder, tokenizer = load_model(
        model_id=args.model_id, lora_rank=args.lora_rank
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Ortho (top, side, rear) -> FPV projector
    top_projector = OrthoProjectorAttn(hidden=args.proj_hidden, hw=96 // 8).to(device)  # pose-dependent inverse-perspective mapping from the FPV
    side_projector = OrthoProjector(hidden=args.proj_hidden, hw=96 // 8).to(device)
    rear_projector = OrthoProjector(hidden=args.proj_hidden, hw=96 // 8).to(device)
    fpv_projector = OrthoProjectorAttn(hidden=args.proj_hidden, hw=96 // 8).to(device)  # pose-dependent spatial mapping for the orbiting FPV

    projectors = {"top": top_projector, "side": side_projector, "rear": rear_projector, "fpv": fpv_projector}
    proj_params = [p for proj in projectors.values() for p in proj.parameters()]
    print(f"[INFO] Ortho-FPV projector params: {sum(p.numel() for p in proj_params) / 1e6:.2f}M")

    optimizer = optim.AdamW(
        [
            {"params": trainable_params, "lr": args.lr},
            {"params": proj_params, "lr": args.proj_lr, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    log_dir = f"./runs/wan_ortho_multienv_cam_{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    view_names = ["Top", "Side", "Rear", "3D_FPV"]

    # 3. Multi-Step Trajectory Training Loop
    print("\nStarting Multi-Step Trajectory LoRA Fine-Tuning ...")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        top_projector.train()
        side_projector.train()
        rear_projector.train()
        fpv_projector.train()

        epoch_loss = 0.0
        # Ramp up autoregressive student-forcing probability across epochs
        progress = epoch / max(1, args.epochs - 1)
        curr_teacher_forcing_prob = args.teacher_forcing_prob * min(1.0, progress * 1.5)

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} (Teacher Prob.={curr_teacher_forcing_prob:.2f})")

        for _ in pbar:
            # Sample batch of contiguous trajectories of length `args.train_seq_len`
            # views_t:       (B, 4, 3, H, W) — anchor frame at t=0
            # views_next_seq: list[T] of (B, 4, 3, H, W) — target frames t=1..T
            views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq = buffer.sample_trajectory_batch(
                batch_size=args.batch_size,
                seq_len=args.train_seq_len,
            )

            with torch.no_grad():
                # Encode views at t=0
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
            loss, per_view_losses, proj_loss, proj_zero_baseline = train_ortho_trajectory_step(
                transformer=model,
                top_projector=top_projector,
                side_projector=side_projector,
                rear_projector=rear_projector,
                fpv_projector=fpv_projector,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                camera_poses_seq=camera_poses_seq,
                env_names_seq=env_names_seq,
                teacher_forcing_prob=curr_teacher_forcing_prob,
                ctx_noise_max=args.ctx_noise_max,
            )

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(proj_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss
            global_step += 1

            writer.add_scalar("Train/Loss", loss, global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("Train/Teacher_Student_Prob", curr_teacher_forcing_prob, global_step)
            for v_idx, v_name in enumerate(view_names):
                writer.add_scalar(f"Train/Loss_{v_name}", per_view_losses[v_idx], global_step)
            writer.add_scalar("Train/Proj_Loss", proj_loss, global_step)
            writer.add_scalar("Train/Proj_Zero_Baseline", proj_zero_baseline, global_step)
            env_summary = "+".join(list(dict.fromkeys(env_names_seq[0]))[:2])
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",
                "Top": f"{per_view_losses[0]:.3f}",
                "FPV": f"{per_view_losses[3]:.3f}",
                "Envs": env_summary,
            })

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"  [Epoch {epoch+1}] Average Trajectory Flow Loss ({args.train_seq_len} steps): {avg_loss:.4f}")

        # --- VALIDATION: AUTOREGRESSIVE VIDEO ROLLOUTS TO TENSORBOARD ---
        if (epoch + 1) % args.eval_every == 0:
            evaluate_and_log_videos(
                transformer=model,
                fpv_projector=fpv_projector,
                top_projector=top_projector,
                rear_projector=rear_projector,
                side_projector=side_projector,
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
            model.save_pretrained(ckpt_dir)
            torch.save({name: p.state_dict() for name, p in projectors.items()}, os.path.join(ckpt_dir, "projectors.pt"))
            print(f"  [SAVED] LoRA + projectors checkpoint to {ckpt_dir}")

    print("\nTraining Complete! Saving final LoRA weights...")
    os.makedirs(f"./checkpoints/wan_ortho_cam_{args.num_envs}_lora_final", exist_ok=True)
    model.save_pretrained(f"./checkpoints/wan_ortho_cam_{args.num_envs}_lora_final")
    torch.save({name: p.state_dict() for name, p in projectors.items()},
               f"./checkpoints/wan_ortho_cam_{args.num_envs}_lora_final/projectors.pt")
    writer.close()

    for _, env in active_envs:
        try:
            env.close()
        except Exception:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA across multiple Orthographic Game Views")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout steps when training on a single environment")
    parser.add_argument("--buffer_capacity", type=int, default=40000, help="Capacity of multi-environment transition buffer")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per gradient step")
    parser.add_argument("--train_seq_len", type=int, default=16, help="Contiguous trajectory steps unrolled per training update")
    parser.add_argument("--teacher_forcing_prob", type=float, default=1.0, help="Max prob of using self-predicted latents onto next trajectory step")
    parser.add_argument("--ctx_noise_max", type=float, default=0.15, help="Max Gaussian noise added to conditioning context latents to prevent drift")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--proj_lr", type=float, default=1e-3, help="Learning rate for the ortho->FPV projector")
    parser.add_argument("--proj_hidden", type=int, default=128, help="Hidden channels of the ortho->FPV projector")
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