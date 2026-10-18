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
from utils import _build_registry, load_model, compose_player_prompts, set_active_env, set_seed, load_agent,  _annotate_frame_pose
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
        # Weight confidence by (1 - tau) so highly noisy steps blend smoothly with GT
        confidence = (1.0 - tau_exp).clamp(0.2, 1.0)
        rollout_latent = (confidence * pred_x0 + (1.0 - confidence) * latents_next_fp32).to(dtype=transformer.dtype)
    return loss, [l.item() for l in per_view_losses], rollout_latent, pred_x0

# ==========================================
# CROSS-VIEW PROJECTOR
# ==========================================
# View order everywhere (buffer, training, rollout): 0: Top, 1: Side, 2: Rear, 3: 3D FPV
PROJECT_VIEW_ORDER = [3, 0, 2, 1]  # 3D, Top, Rear, Side

class CrossViewProjector(nn.Module):
    """
    Small residual projector (~47k params) predicting a correction for one target view at t+1.
    Every view has a fixed input slot (0: Top, 1: Side, 2: Rear, 3: 3D), so slot k always means view k.
    Inputs:
      views:    (B, C, V, h, w) current t+1 guesses; non-given views (other than the target) are zeroed
      self_t:   (B, C, h, w)    target view at t
      given:    (B, V)          1 = view is given as conditioning, 0 = not given (target or dropped)
      target_id: int            view being predicted (one-hot planes)
    Zero-initialised output so it starts as the identity.
    Per-view gate sigmoid(gate[v]) in [0, 1] scales the correction, so training can learn how much
    each view (e.g. the perspective FPV, hard to fix from orthographic views) should be edited.
    """
    def __init__(self, c=16, hidden=32, num_views=4, dilations=(1, 2, 4, 1)):
        super().__init__()
        self.num_views = num_views
        # view slots (as-is + transposed) + self_t + given mask + target one-hot
        in_ch = 2 * c * num_views + c + 2 * num_views
        self.inp = nn.Conv2d(in_ch, hidden, 1)
        # Dilated 3x3 residual blocks: receptive field 17 covers the full 12x12 latent
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(),
                          nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d))
            for d in dilations
        ])
        self.out = nn.Conv2d(hidden, c, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # sigmoid(0) = 0.5: the zero-init output already gives the identity, and an open-ish gate
        # keeps gradients to `out` from being squashed at the start
        self.gate = nn.Parameter(torch.zeros(num_views))

    def forward(self, views, self_t, given, target_id):
        b, c, v, h, w = views.shape
        keep = given.clone()
        keep[:, target_id] = 1.0  # target slot keeps its own current guess
        x_views = views.float() * keep.view(b, 1, v, 1, 1)
        # Views have permuted axes (Top[y,x], Side[z,x], Rear[z,y]): Top<->Rear share y on different
        # image axes, which a translation-equivariant conv can't align. Transposed copies line them up.
        x_views = torch.cat([x_views.reshape(b, c * v, h, w),
                             x_views.transpose(-1, -2).reshape(b, c * v, w, h)], dim=1)

        given_planes = given.view(b, v, 1, 1).expand(b, v, h, w)
        target_planes = torch.zeros((b, v, h, w), device=views.device)
        target_planes[:, target_id] = 1.0

        x = torch.cat([x_views, self_t.float(), given_planes, target_planes], dim=1)
        hdn = self.inp(x)
        for blk in self.blocks:
            hdn = hdn + blk(hdn)
        return torch.sigmoid(self.gate[target_id]) * self.out(hdn)

class CrossViewAttnProjector(nn.Module):
    """
    Attention variant of CrossViewProjector with the same inputs/outputs.
    Views have permuted image axes (Top[y,x], Side[z,x], Rear[z,y]) and FPV is a perspective
    view, so the pixel correspondence between views is not translation-equivariant. Learned
    per-view positional embeddings + attention over all views' tokens can learn any correspondence.
    """
    def __init__(self, c=16, dim=64, depth=3, heads=4, num_views=4, hw=12):
        super().__init__()
        self.num_views, self.hw = num_views, hw
        # per token: own latent + self_t (target view only) + given flag + target flag
        self.inp = nn.Linear(2 * c + 2, dim)
        self.pos = nn.Parameter(torch.randn(1, num_views * hw * hw, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4 * dim, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, c)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.gate = nn.Parameter(torch.zeros(num_views))

    def forward(self, views, self_t, given, target_id):
        b, c, v, h, w = views.shape
        keep = given.clone()
        keep[:, target_id] = 1.0
        x_views = views.float() * keep.view(b, 1, v, 1, 1)                      # (B, C, V, h, w)
        st = torch.zeros_like(x_views)
        st[:, :, target_id] = self_t.float()
        flags = torch.zeros((b, 2, v, h, w), device=views.device)
        flags[:, 0] = given.view(b, v, 1, 1)
        flags[:, 1, target_id] = 1.0

        x = torch.cat([x_views, st, flags], dim=1)                               # (B, 2C+2, V, h, w)
        x = x.permute(0, 2, 3, 4, 1).reshape(b, v * h * w, -1)                   # (B, V*h*w, 2C+2)
        hdn = self.norm(self.blocks(self.inp(x) + self.pos))
        hdn = hdn.view(b, v, h * w, -1)[:, target_id]                            # target view tokens
        delta = self.out(hdn).transpose(1, 2).reshape(b, c, h, w)
        return torch.sigmoid(self.gate[target_id]) * delta

def project_views(projector, latents_t, init_next, sweeps=2, view_drop_prob=0.0):
    """
    Loops across views, re-predicting each t+1 view from the other three (Gauss-Seidel:
    later views in a sweep see the already-updated ones).
    `view_drop_prob` randomly hides given views (training only) so the projector learns to
    use whichever views it is told are given.
    latents_t, init_next: (B, C, 4, h, w) -> (B, C, 4, h, w) float32
    """
    latents_t = latents_t.float()
    preds = list(init_next.float().unbind(dim=2))  # 4 x (B, C, h, w)
    b, num_views = init_next.shape[0], init_next.shape[2]
    for _ in range(sweeps):
        for v in PROJECT_VIEW_ORDER:
            given = torch.ones((b, num_views), device=init_next.device)
            given[:, v] = 0.0
            if view_drop_prob > 0.0:
                given = given * (torch.rand_like(given) >= view_drop_prob).float()
            delta = projector(torch.stack(preds, dim=2), self_t=latents_t[:, :, v], given=given, target_id=v)
            preds[v] = preds[v] + delta
    return torch.stack(preds, dim=2)

def sample_project_init(latents_t, latents_next, pred_x0, p_pred=0.5, p_noisy_gt=0.3, max_sigma=0.5):
    """
    Per-sample mix of initial t+1 guesses for the projector:
      p_pred:     DiT one-step pred_x0 (the model's real error distribution)
      p_noisy_gt: GT latents_next + sigma * noise, sigma ~ U(0, max_sigma)
      rest:       latents_t (previous frame as the guess)
    """
    b = latents_next.shape[0]
    latents_t, latents_next, pred_x0 = latents_t.float(), latents_next.float(), pred_x0.float()
    sigma = torch.rand((b, 1, 1, 1, 1), device=latents_next.device) * max_sigma
    noisy_gt = latents_next + sigma * torch.randn_like(latents_next)

    u = torch.rand((b, 1, 1, 1, 1), device=latents_next.device)
    use_pred = (u < p_pred).float()
    use_noisy = ((u >= p_pred) & (u < p_pred + p_noisy_gt)).float()
    use_prev = 1.0 - use_pred - use_noisy
    return use_pred * pred_x0 + use_noisy * noisy_gt + use_prev * latents_t

def train_ortho_trajectory_step(
    transformer,
    latents_t0,
    latents_next_seq,
    prompt_embeds_seq,
    projector=None,
    lambda_proj=0.5,
    project_sweeps=2,
    proj_view_drop=0.15,
    p_sampled_init=0.25,
    sample_init_steps=8,
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
    - If `projector` is given, also trains the cross-view projector at each step with
      MSE(project_views(init_next), GT t+1), weighted by `lambda_proj`. The DiT's pred_x0
      is detached, so the projector loss does not backprop into the DiT.
    - With probability `p_sampled_init` per step, the projector's initial guess is a real
      `sample_init_steps`-step Euler sample (no grad), matching what it corrects at eval.
    """
    seq_len = len(latents_next_seq)
    num_views = latents_t0.shape[2]

    curr_latents = latents_t0
    total_loss = 0.0
    total_proj_loss = 0.0
    avg_per_view = [0.0] * num_views
    proj_gain = [0.0] * num_views

    for s in range(seq_len):
        target_latents = latents_next_seq[s]
        prompt_embeds = prompt_embeds_seq[s]
        is_last_step = (s == seq_len - 1)

        step_loss, step_view_losses, pred_next_latents, pred_x0 = train_ortho_flow_step(
            transformer=transformer,
            latents_t=curr_latents,
            latents_next=target_latents,
            prompt_embeds=prompt_embeds,
            shift=shift,
            ctx_noise_max=ctx_noise_max,
        )

        # Normalize loss across trajectory length and backpropagate immediately to save VRAM
        scaled_loss = step_loss / float(seq_len)

        if projector is not None:
            if random.random() < p_sampled_init:
                init_next = predict_next_view_latents(
                    transformer, curr_latents, prompt_embeds, steps=sample_init_steps, shift=shift
                ).float()
            else:
                init_next = sample_project_init(curr_latents, target_latents, pred_x0)
            projected = project_views(projector, curr_latents, init_next, sweeps=project_sweeps,
                                      view_drop_prob=proj_view_drop)
            proj_loss = F.mse_loss(projected, target_latents.float())
            scaled_loss = scaled_loss + lambda_proj * proj_loss / float(seq_len)
            total_proj_loss += proj_loss.item() / float(seq_len)

            # Diagnostics: per-view latent MSE before vs after projection (gain > 0 = projector helps)
            with torch.no_grad():
                tgt = target_latents.float()
                for v_idx in range(num_views):
                    mse_init = F.mse_loss(init_next[:, :, v_idx], tgt[:, :, v_idx]).item()
                    mse_proj = F.mse_loss(projected[:, :, v_idx].detach(), tgt[:, :, v_idx]).item()
                    proj_gain[v_idx] += (mse_init - mse_proj) / float(seq_len)

        scaled_loss.backward()

        total_loss += step_loss.item() / float(seq_len)
        for v_idx in range(num_views):
            avg_per_view[v_idx] += step_view_losses[v_idx] / float(seq_len)

        # Prepare context for step s + 1 (Student Forcing vs. Teacher Forcing per sample)
        if not is_last_step:
            if teacher_forcing_prob > 0.0:
                b = curr_latents.shape[0]
                use_ar_mask = (torch.rand((b, 1, 1, 1, 1), device=curr_latents.device) < teacher_forcing_prob).to(dtype=transformer.dtype)
                curr_latents = use_ar_mask * pred_next_latents + (1.0 - use_ar_mask) * target_latents
            else:
                curr_latents = target_latents

    return total_loss, avg_per_view, total_proj_loss, proj_gain

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
def autoregressive_rollout(transformer, vae, text_encoder, tokenizer, buffer, env_name=None, env=None, rollout_steps=16, steps=20,
                           projector=None, project_sweeps=2):
    """
    Autoregressive video rollout for all 4 orthographic views.
    Returns:
      gen_views: (vid_top, vid_rear, vid_side, vid_fpv) each (1, T, 3, H, W) in [0, 1]
      gt_views:  (gt_top,  gt_rear,  gt_side,  gt_fpv)  each (1, T, 3, H, W) in [0, 1]
    """
    transformer.eval()

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

        # Cross-view projection: loop across views so each t+1 view agrees with the other three
        if projector is not None:
            next_latents = project_views(
                projector, curr_latents, next_latents, sweeps=project_sweeps
            ).to(dtype=transformer.dtype)

        decoded_views = decode_latents_to_views(vae, next_latents)  # (1, 4, 3, H, W) in [0, 1]

        # Store generated frames (0: Top, 1: Side, 2: Rear, 3: FPV)
        gen_top.append(decoded_views[0, 0].cpu())
        gen_side.append(decoded_views[0, 1].cpu())
        gen_rear.append(decoded_views[0, 2].cpu())
        
        next_obs, reward, terminated, truncated, _ = env.step(step_act)
        next_top, next_rear, next_side, next_fpv = env.render(
            camera_pose=sample_cam["command"]
        )
        if terminated or truncated:
            env.reset()
        vn = [cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA) for x in [next_top, next_side, next_rear, next_fpv]]
        vn = np.stack([np.transpose(v, (2, 0, 1)) for v in vn])

        gt_vn = torch.from_numpy(vn).float() / 255.0  # (4, 3, H, W)
        gt_top.append(gt_vn[0])
        gt_side.append(gt_vn[1])
        gt_rear.append(gt_vn[2])

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

    transformer.train()
    env.close()
    return (vid_gen_top, vid_gen_rear, vid_gen_side, vid_gen_fpv,
            vid_gt_top,  vid_gt_rear,  vid_gt_side,  vid_gt_fpv)


@torch.no_grad()
def evaluate_and_log_videos(transformer, vae, buffer, text_encoder, tokenizer, writer, epoch, rollout_steps=16, steps=20, fps=15,
                            projector=None, project_sweeps=2, compare_no_projector=True):
    """
    Performs autoregressive video rollouts for each view and writes the videos to TensorBoard.
    """
    transformer.eval()
    unique_envs = ['Bipedal Walker', 'Lunar Lander', 'MultiCar Racing']
    reg_dict = dict(_build_registry())
    
    print(f"\n[VALIDATION] Running autoregressive video rollouts for epoch {epoch} across envs: {', '.join(unique_envs)}...")

    def _psnr(pred, gt):
        mse = F.mse_loss(pred, gt).item()
        return -10.0 * math.log10(max(mse, 1e-8))

    # With a projector, also roll out without it from the same RNG state (same buffer sequence,
    # same sampling noise) so PSNR/<view> vs PSNR_NoProj/<view> isolates the projector's effect
    modes = [("PSNR", projector)]
    if projector is not None and compare_no_projector:
        modes.append(("PSNR_NoProj", None))

    for env_idx, env_name in enumerate(unique_envs):
        clean_tag = env_name.replace(" ", "_").lower()
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

        for mode_idx, (metric, mode_projector) in enumerate(modes):
            random.setstate(rng_state[0]); np.random.set_state(rng_state[1]); torch.set_rng_state(rng_state[2])
            if rng_state[3] is not None:
                torch.cuda.set_rng_state_all(rng_state[3])

            env = reg_dict[env_name]()
            env.reset()

            (vid_top_down, vid_rear, vid_side, vid_fpv,
             vid_gt_top_down, vid_gt_rear, vid_gt_side, vid_gt_fpv) = autoregressive_rollout(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                buffer=buffer,
                env_name=env_name,
                env=env,
                rollout_steps=rollout_steps,
                steps=steps,
                projector=mode_projector,
                project_sweeps=project_sweeps,
            )

            psnr_top_down = _psnr(vid_top_down, vid_gt_top_down)
            psnr_rear     = _psnr(vid_rear, vid_gt_rear)
            psnr_side     = _psnr(vid_side, vid_gt_side)
            psnr_fpv      = _psnr(vid_fpv, vid_gt_fpv)
            psnr_mean     = (psnr_top_down + psnr_rear + psnr_side + psnr_fpv) / 4.0

            writer.add_scalar(f"Validation/{clean_tag}/{metric}/Top Down", psnr_top_down, epoch)
            writer.add_scalar(f"Validation/{clean_tag}/{metric}/Rear",     psnr_rear,     epoch)
            writer.add_scalar(f"Validation/{clean_tag}/{metric}/Side",     psnr_side,     epoch)
            writer.add_scalar(f"Validation/{clean_tag}/{metric}/FPV",      psnr_fpv,      epoch)
            writer.add_scalar(f"Validation/{clean_tag}/{metric}/Mean",     psnr_mean,     epoch)

            print(f"  [Epoch {epoch}] [{env_name}] [{metric}] Autoregressive Video PSNR: {psnr_mean:.2f} dB "
                  f"(Top-Down: {psnr_top_down:.2f}dB, Rear: {psnr_rear:.2f}dB, Side: {psnr_side:.2f}dB, FPV: {psnr_fpv:.2f}dB)")

            if mode_idx == 0:
                writer.add_video(f"Test/{clean_tag}/Top-Down", vid_top_down, epoch, fps=fps)
                writer.add_video(f"Test/{clean_tag}/Rear",     vid_rear,     epoch, fps=fps)
                writer.add_video(f"Test/{clean_tag}/Side",     vid_side,     epoch, fps=fps)
                writer.add_video(f"Test/{clean_tag}/FPV",      vid_fpv,      epoch, fps=fps)
            else:
                writer.add_video(f"Test_NoProj/{clean_tag}/FPV", vid_fpv, epoch, fps=fps)

    transformer.train()

def collect_rollouts(env, agent, buffer, n_steps, seed, env_name="Bipedal Walker"):
    """Collects camera rollouts storing stateful camera poses."""
    obs, _ = env.reset(seed=seed)
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
          pose_hold = 0
          img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")
        else:
          obs = next_obs
          img_top, im_rear, im_side, im_fpv = next_top, next_rear, next_side, next_fpv
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

    if args.proj_type == "attn":
        projector = CrossViewAttnProjector(hw=96 // 8).to(device)  # 96x96 views -> 12x12 Wan latents
    else:
        projector = CrossViewProjector(hidden=args.proj_hidden).to(device)
    print(f"[INFO] Cross-view projector params: {sum(p.numel() for p in projector.parameters()) / 1e6:.2f}M")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    proj_params = list(projector.parameters())
    optimizer = optim.AdamW(
        [
            {"params": trainable_params},
            {"params": proj_params, "lr": args.lr * args.proj_lr_mult},  # trained from scratch
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
            loss, per_view_losses, proj_loss, proj_gain = train_ortho_trajectory_step(
                transformer=model,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                projector=projector,
                lambda_proj=args.lambda_proj,
                project_sweeps=args.project_sweeps,
                proj_view_drop=args.proj_view_drop,
                p_sampled_init=args.p_sampled_init,
                sample_init_steps=args.sample_init_steps,
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
            writer.add_scalar("Train/Loss_Projector", proj_loss, global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("Train/Teacher_Student_Prob", curr_teacher_forcing_prob, global_step)
            for v_idx, v_name in enumerate(view_names):
                writer.add_scalar(f"Train/Loss_{v_name}", per_view_losses[v_idx], global_step)
                writer.add_scalar(f"Train/Proj_Gate_{v_name}", torch.sigmoid(projector.gate[v_idx]).item(), global_step)
                writer.add_scalar(f"Train/Proj_Gain_{v_name}", proj_gain[v_idx], global_step)

            env_summary = "+".join(list(dict.fromkeys(env_names_seq[0]))[:2])
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",
                "Proj": f"{proj_loss:.4f}",
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
                vae=vae,
                buffer=buffer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                writer=writer,
                epoch=epoch + 1,
                rollout_steps=args.rollout_video_steps,
                steps=args.eval_steps,
                fps=args.video_fps,
                projector=projector,
                project_sweeps=args.project_sweeps,
                compare_no_projector=not args.no_proj_compare,
            )

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = f"./checkpoints/wan_ortho_lora_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            torch.save(projector.state_dict(), os.path.join(ckpt_dir, "projector.pt"))
            print(f"  [SAVED] LoRA + projector checkpoint to {ckpt_dir}")

    print("\nTraining Complete! Saving final LoRA weights...")
    os.makedirs(f"./checkpoints/wan_ortho_cam_{args.num_envs}_lora_final", exist_ok=True)
    model.save_pretrained(f"./checkpoints/wan_ortho_cam_{args.num_envs}_lora_final")
    torch.save(projector.state_dict(), f"./checkpoints/wan_ortho_cam_{args.num_envs}_lora_final/projector.pt")
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
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Generate and evaluate views every N epochs")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of Euler flow matching sampling steps per frame")
    parser.add_argument("--rollout_video_steps", type=int, default=16, help="Autoregressive rollout steps for validation videos")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for TensorBoard video logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_envs", type=int, default=12, help="Number of training envs")
    parser.add_argument("--lambda_proj", type=float, default=0.5, help="Weight of the cross-view projector MSE loss")
    parser.add_argument("--project_sweeps", type=int, default=2, help="Number of loops across the 4 views in project_views")
    parser.add_argument("--proj_type", type=str, default="conv", choices=["conv", "attn"], help="Cross-view projector: dilated conv or attention over all views' tokens")
    parser.add_argument("--no_proj_compare", action="store_true", help="Skip the extra validation rollout without the projector")
    parser.add_argument("--proj_hidden", type=int, default=32, help="Hidden channels of the cross-view projector")
    parser.add_argument("--proj_view_drop", type=float, default=0.15, help="Prob. of hiding each given view from the projector during training")
    parser.add_argument("--proj_lr_mult", type=float, default=5.0, help="Projector LR multiplier relative to --lr")
    parser.add_argument("--p_sampled_init", type=float, default=0.25, help="Per-step prob. of training the projector on a real Euler sample instead of pred_x0/noisy GT/prev")
    parser.add_argument("--sample_init_steps", type=int, default=8, help="Euler steps for the projector's sampled initial guess during training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)