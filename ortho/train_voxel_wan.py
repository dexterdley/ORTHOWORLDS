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
from PIL import Image

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
import matplotlib.pyplot as plt

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
# VOXEL BUILDERS
# ==========================================
def mask(image, thresh=0.5):
    """RGB uint8 array (H,W,3) -> boolean silhouette (N,N)."""
    im = Image.fromarray(image.astype(np.uint8)).convert("L")
    im = np.asarray(im, dtype=np.float32) / 255.
    return im < thresh

def first_hit(V, axis, from_high):
    """Voxels that are the first occupied cell along `axis`, seen from the high or low end."""
    W = np.flip(V, axis) if from_high else V
    hit = W & (np.cumsum(W, axis=axis) == 1)
    return np.flip(hit, axis) if from_high else hit

def build_voxel(im_top_down, im_side, im_rear, N=64):
    top  = mask(im_top_down)   # rows = y, cols = x
    side = mask(im_side)       # rows = z (row 0 = top), cols = x
    rear = mask(im_rear)       # rows = z (row 0 = top), cols = y

    T = top.T                  # T[x, y]
    S = side[::-1].T           # S[x, z]
    R = rear[::-1].T           # R[y, z]

    V = T[:, :, None] & S[:, None, :] & R[None, :, :]
    return V

def build_rgb_voxel(V, im_top_down, im_side, im_rear, N=96):
    Tc = im_top_down.transpose(1, 0, 2)          # Tc[x, y]
    Sc = im_side[::-1].transpose(1, 0, 2)        # Sc[x, z]
    Rc = im_rear[::-1].transpose(1, 0, 2)        # Rc[y, z]

    # Broadcast each view's colour into 3D: (N,N,N,3)
    C_top  = np.broadcast_to(Tc[:, :, None, :], (N, N, N, 3))
    C_side = np.broadcast_to(Sc[:, None, :, :], (N, N, N, 3))
    C_rear = np.broadcast_to(Rc[None, :, :, :], (N, N, N, 3))

    vis_top  = first_hit(V, axis=2, from_high=True)    # looking down from +z
    vis_side = first_hit(V, axis=1, from_high=True)    # camera on +y (left) side
    vis_rear = first_hit(V, axis=0, from_high=False)   # camera behind, at low x

    w = np.stack([vis_top, vis_side, vis_rear], -1).astype(np.float32)[..., None]   # (N,N,N,3,1)
    C = np.stack([C_top, C_side, C_rear], -2)                                       # (N,N,N,3,3)
    wsum = w.sum(-2)
    V_rgb = np.where(wsum > 0, (w * C).sum(-2) / np.maximum(wsum, 1), C.mean(-2))   # visible -> seen colour
    V_rgb = V_rgb * V[..., None]                                                    # zero outside the hull

    V_rgba = np.concatenate([V_rgb, V[..., None].astype(np.float32)], -1)
    return V_rgba

def build_voxel_batch_numpy(views):
    """
    Reference implementation: one voxel per sample with the numpy builders (CPU, slow).
    views: (B, 4, 3, H, W) in [-1, 1] ordered (top, side, rear, fpv)
    Returns: (B, 4, N, N, N) float in [0, 1] -> channel 0 = occupancy, channels 1:4 = RGB.
    """
    voxel_t_stack = []
    for item in views:
        vt = [((v.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8) for v in item[:3]]
        voxel_t = build_voxel(vt[0], vt[1], vt[2])
        voxel_t_rgba = build_rgb_voxel(voxel_t, vt[0], vt[1], vt[2], N=voxel_t.shape[0])
        voxel_t_rgba[..., :3] /= 255.0
        occ_rgb = np.concatenate([voxel_t_rgba[..., 3:], voxel_t_rgba[..., :3]], -1)   # (N,N,N,4)
        voxel_t_stack.append(occ_rgb.transpose(3, 0, 1, 2))                            # (4,N,N,N)
    return torch.from_numpy(np.stack(voxel_t_stack)).float().to(views.device)

def _first_hit_batch(V, dim, from_high):
    """Batched first_hit: voxels that are the first occupied cell along `dim`."""
    W = V.flip(dim) if from_high else V
    hit = W & (W.cumsum(dim=dim) == 1)
    return hit.flip(dim) if from_high else hit

@torch.no_grad()
def build_voxel_batch(views, thresh=0.5):
    """
    Batched torch version of build_voxel + build_rgb_voxel (runs on the views' device, e.g. GPU).
    Same masks, axes and visibility-weighted colours as the numpy builders.
    views: (B, 4, 3, H, W) in [-1, 1] ordered (top, side, rear, fpv)
    Returns: (B, 4, N, N, N) float in [0, 1] -> channel 0 = occupancy, channels 1:4 = RGB.
    """
    img = ((views[:, :3].float() + 1.0) * 127.5).clamp(0, 255).floor() / 255.0   # (B, 3 views, 3, N, N), uint8-quantised
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]     # (B, 3 views, N, N)
    m = gray < thresh

    T = m[:, 0].transpose(1, 2)                 # T[x, y]
    S = m[:, 1].flip(1).transpose(1, 2)         # S[x, z]
    R = m[:, 2].flip(1).transpose(1, 2)         # R[y, z]
    V = T[:, :, :, None] & S[:, :, None, :] & R[:, None, :, :]                    # (B, N, N, N)

    Tc = img[:, 0].transpose(2, 3)[:, :, :, :, None]           # (B, 3, x, y, 1)
    Sc = img[:, 1].flip(2).transpose(2, 3)[:, :, :, None, :]   # (B, 3, x, 1, z)
    Rc = img[:, 2].flip(2).transpose(2, 3)[:, :, None, :, :]   # (B, 3, 1, y, z)

    w_top  = _first_hit_batch(V, dim=3, from_high=True)[:, None].float()    # looking down from +z
    w_side = _first_hit_batch(V, dim=2, from_high=True)[:, None].float()    # camera on +y side
    w_rear = _first_hit_batch(V, dim=1, from_high=False)[:, None].float()   # camera behind, at low x

    wsum = w_top + w_side + w_rear
    seen = (w_top * Tc + w_side * Sc + w_rear * Rc) / wsum.clamp(min=1)
    rgb = torch.where(wsum > 0, seen, (Tc + Sc + Rc) / 3.0) * V[:, None]  # visible -> seen colour

    return torch.cat([V[:, None].float(), rgb], dim=1)                      # (B, 4, N, N, N)

def render_voxel_image(voxel, angle=30.0, elev=20.0, scale=2, bg=1.0):
    """
    Fast oblique render of one voxel for TensorBoard (vectorised z-buffer, no matplotlib).
    voxel: (4, N, N, N) [occ, r, g, b] indexed [x, y, z], z up -> (3, S, S) float in [0, 1].
    """
    v = voxel.float().cpu().numpy() if torch.is_tensor(voxel) else np.asarray(voxel, dtype=np.float32)
    occ = v[0] > 0.5
    N = occ.shape[0]
    S = int(np.ceil(N * np.sqrt(3))) * scale
    img = np.full((S * S, 3), bg, dtype=np.float32)

    idx = np.argwhere(occ)
    if len(idx) == 0:
        return img.reshape(S, S, 3).transpose(2, 0, 1)

    # Rotate around z by `angle`, tilt by `elev`; centre the grid at the origin
    p = idx - (N - 1) / 2.0
    a, e = np.radians(angle), np.radians(elev)
    x = p[:, 0] * np.cos(a) - p[:, 1] * np.sin(a)
    y = p[:, 0] * np.sin(a) + p[:, 1] * np.cos(a)
    z = p[:, 2]
    up    = z * np.cos(e) + y * np.sin(e)
    depth = y * np.cos(e) - z * np.sin(e)                    # larger = farther

    u_px = ((x + S / scale / 2) * scale).astype(int)
    v_px = ((up + S / scale / 2) * scale).astype(int)
    shade = 1.0 - 0.4 * (depth - depth.min()) / max(np.ptp(depth), 1e-6)   # darker = farther
    cols = v[1:, idx[:, 0], idx[:, 1], idx[:, 2]].T * shade[:, None]

    # Splat each voxel as a scale x scale block, keep the nearest voxel per pixel
    du, dv = np.meshgrid(np.arange(scale), np.arange(scale))
    U = (u_px[:, None] + du.ravel()[None]).ravel()
    V = (v_px[:, None] + dv.ravel()[None]).ravel()
    D = np.repeat(depth, scale * scale)
    C = np.repeat(cols, scale * scale, axis=0)
    ok = (U >= 0) & (U < S) & (V >= 0) & (V < S)
    flat, D, C = (V * S + U)[ok], D[ok], C[ok]

    order = np.argsort(D, kind="stable")                     # nearest first
    _, first = np.unique(flat[order], return_index=True)     # first hit per pixel = nearest
    sel = order[first]
    img[flat[sel]] = C[sel]

    img = img.reshape(S, S, 3)[::-1]                         # image rows go down, "up" goes up
    return np.ascontiguousarray(img).transpose(2, 0, 1).clip(0.0, 1.0)

def voxel_iou(vox_a, vox_b):
    """Occupancy IoU between two (4, N, N, N) voxels."""
    a, b = vox_a[0] > 0.5, vox_b[0] > 0.5
    return ((a & b).sum() / (a | b).sum().clamp(min=1)).item()

# ==========================================
# PROJECTION CONSISTENCY
# ==========================================
def mask_border(img, thresh=0.1):
    """img (B, V, 3, H, W) in [0, 1] -> (B, V, H, W) bool. Background = median border colour per image."""
    border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., :, 0], img[..., :, -1]], dim=-1)
    bg = border.median(dim=-1).values[..., None, None]
    return (img - bg).abs().sum(dim=2) > thresh

@torch.no_grad()
def silhouettes_from_views(views, thresh=0.1):
    """views (B, 4, 3, H, W) in [-1, 1] ordered (top, side, rear, fpv) -> (B, 3, H, W) float masks (top, side, rear)."""
    img = (views[:, :3].float() + 1.0) / 2.0
    return mask_border(img, thresh).float()

class SilhouetteProbe(nn.Module):
    """
    Latent -> silhouette decoder, pre-trained on GT latents and then FROZEN, so the consistency loss
    can only be lowered by the DiT (a trainable head could absorb the error instead).
    in:  (B, lat_ch, V, h, w) latents -> out: (B, V, 8h, 8w) silhouette logits
    """
    def __init__(self, lat_ch=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(lat_ch, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, 64, kernel_size=1), nn.PixelShuffle(8),   # 64 = 1 x 8 x 8 -> (1, 8h, 8w)
        )

    def forward(self, latents):
        b, c, v, h, w = latents.shape
        flat = latents.float().permute(0, 2, 1, 3, 4).reshape(b * v, c, h, w)
        return self.net(flat).view(b, v, 8 * h, 8 * w)

def projection_error(masks):
    """
    Per-sample disagreement of the extents two views share. Same axes as build_voxel_batch:
    top rows=y cols=x, side rows=z cols=x, rear rows=z cols=y (rows of side/rear share the same z flip).
    masks: (B, 3, H, W) soft silhouettes (top, side, rear) -> (B,) mean L1 over the 3 shared axes.
    """
    top, side, rear = masks.unbind(1)
    err_x = (top.amax(1) - side.amax(1)).abs().mean(-1)   # x: column extents of top vs side
    err_y = (top.amax(2) - rear.amax(1)).abs().mean(-1)   # y: row extents of top vs column extents of rear
    err_z = (side.amax(2) - rear.amax(2)).abs().mean(-1)  # z: row extents of side vs rear
    return (err_x + err_y + err_z) / 3.0

def projection_consistency_loss(pred_masks, gt_masks):
    """
    Hinge on the GT floor: only penalise inconsistency beyond what the GT masks themselves have
    (renders/masks are never perfectly consistent, e.g. ground planes, so don't fight that).
    """
    return F.relu(projection_error(pred_masks) - projection_error(gt_masks))

def pretrain_silhouette_probe(probe, vae, buffer, steps=500, batch_size=32, lr=1e-3, noise_max=0.3, writer=None):
    """Fits the probe on GT latents (with mild noise so it tolerates imperfect x0 estimates), then freezes it."""
    opt = optim.AdamW(probe.parameters(), lr=lr)
    pbar = tqdm(range(steps), desc="Pre-training silhouette probe")
    for it in pbar:
        views, *_ = buffer.sample_trajectory_batch(batch_size=batch_size, seq_len=1)
        with torch.no_grad():
            lat = encode_views_to_latents(vae, views)[:, :, :3].float()
            s = torch.rand(lat.shape[0], 1, 1, 1, 1, device=lat.device) * noise_max
            lat = (1.0 - s) * lat + s * torch.randn_like(lat)
            target = silhouettes_from_views(views)

        logits = probe(lat)
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = 1.0 - (2.0 * (p * target).sum() + 1.0) / (p.sum() + target.sum() + 1.0)
        loss = bce + dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        iou = (((p > 0.5) & (target > 0.5)).sum() / ((p > 0.5) | (target > 0.5)).sum().clamp(min=1)).item()
        pbar.set_postfix({"Loss": f"{loss.item():.3f}", "IoU": f"{iou:.3f}"})
        if writer is not None:
            writer.add_scalar("Probe/Loss", loss.item(), it)
            writer.add_scalar("Probe/IoU", iou, it)

    probe.eval().requires_grad_(False)
    return probe

# ==========================================
# MULTI-STEP TRAJECTORY FLOW MATCHING TRAINING
# ==========================================
def train_voxel_ortho_flow_step(
    transformer,
    latents_t,
    latents_next,
    prompt_embeds,
    sil_probe=None,
    gt_masks_next=None,
    shift=3.0,
    ctx_noise_max=0.15,
    cons_weight=0.0,
):
    """
    Single transition Flow-Matching step (used inside multi-step trajectory rollout).
    Applies mild context noise to `latents_t` to prevent autoregressive drift,
    and optionally returns the model's reconstructed clean latent `pred_x0` at t+1.
    Auxiliary projection consistency loss: the frozen `sil_probe` reads top/side/rear silhouettes off the
    model's clean estimate of the t+1 latents, whose shared-axis extents must agree (GT masks `gt_masks_next` set the floor).
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

    # 4. Concatenate conditioning views (frames 0:4), noisy target views (frames 4:8)
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

    # 8. Projection consistency: the frozen probe reads silhouettes off pred_x0 and the shared-axis
    #    extents of top/side/rear must agree. No trainable params, so only the DiT can reduce it.
    #    pred_x0 keeps its graph so the loss trains the DiT; (1 - tau) since x0 is unreliable at high noise.
    cons_loss = torch.zeros((), device=latents_next.device)
    if cons_weight > 0.0 and sil_probe is not None:
        pred_x0_grad = noisy_latents_next_fp32 - tau_exp * pred_velocity
        pred_masks = torch.sigmoid(sil_probe(pred_x0_grad[:, :, :3]))   # top, side, rear
        cons_loss = ((1.0 - tau) * projection_consistency_loss(pred_masks, gt_masks_next)).mean()
        loss = loss + cons_weight * cons_loss

    # Reconstruct 1-step denoised prediction: x_0_hat = x_tau - tau * v_pred
    with torch.no_grad():
        pred_x0 = (noisy_latents_next_fp32 - tau_exp * pred_velocity.detach()).clamp(-6.0, 6.0)
        # Weight confidence by (1 - tau) so highly noisy steps blend smoothly with GT
        confidence = (1.0 - tau_exp).clamp(0.2, 1.0)
        rollout_latent = (confidence * pred_x0 + (1.0 - confidence) * latents_next_fp32).to(dtype=transformer.dtype)
    return loss, [l.item() for l in per_view_losses], rollout_latent, cons_loss.item()

def train_ortho_trajectory_step(
    transformer,
    latents_t0,
    latents_next_seq,
    views_next_seq,
    prompt_embeds_seq,
    sil_probe=None,
    shift=3.0,
    teacher_forcing_prob=0.5,
    ctx_noise_max=0.15,
    cons_weight=0.0,
):
    """
    Unrolls a multi-step trajectory of length `seq_len` during training.
    - At step s=0, conditions on ground-truth `latents_t0`.
    - At step s>0, with probability `teacher_forcing_prob`, conditions on the model's own predicted
      latent from step s-1 (student forcing) so the model learns to recover from its
      own rollout errors over multi-step horizons.
    - Each step's projection consistency floor comes from the GT masks of that step's target views (t+1).
    - Backpropagates `(loss_s / seq_len)` at each step so peak VRAM remains constant.
    """
    seq_len = len(latents_next_seq)
    num_views = latents_t0.shape[2]

    curr_latents = latents_t0
    total_loss = 0.0
    avg_cons_loss = 0.0
    avg_per_view = [0.0] * num_views

    for s in range(seq_len):
        target_latents = latents_next_seq[s]
        prompt_embeds = prompt_embeds_seq[s]
        is_last_step = (s == seq_len - 1)

        # GT silhouettes of this step's target frame (t+1), built lazily
        gt_masks_next = silhouettes_from_views(views_next_seq[s]) if cons_weight > 0.0 else None

        step_loss, step_view_losses, pred_next_latents, step_cons_loss = train_voxel_ortho_flow_step(
            transformer=transformer,
            latents_t=curr_latents,
            latents_next=target_latents,
            prompt_embeds=prompt_embeds,
            sil_probe=sil_probe,
            gt_masks_next=gt_masks_next,
            shift=shift,
            ctx_noise_max=ctx_noise_max,
            cons_weight=cons_weight,
        )

        # Normalize loss across trajectory length and backpropagate immediately to save VRAM
        scaled_loss = step_loss / float(seq_len)
        scaled_loss.backward()

        total_loss += step_loss.item() / float(seq_len)
        avg_cons_loss += step_cons_loss / float(seq_len)
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

    return total_loss, avg_per_view, avg_cons_loss

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
def autoregressive_rollout(transformer, vae, text_encoder, tokenizer, buffer, env_name=None, env=None, rollout_steps=16, steps=20):
    """
    Autoregressive video rollout for all 4 orthographic views.
    Returns:
      gen_views: (vid_top, vid_rear, vid_side, vid_fpv) each (1, T, 3, H, W) in [0, 1]
      gt_views:  (gt_top,  gt_rear,  gt_side,  gt_fpv)  each (1, T, 3, H, W) in [0, 1]
      vid_gen_vox, vid_gt_vox: (1, T, 3, S, S) renders of voxels built from generated / GT views at t+1
      vox_iou: mean occupancy IoU between generated-view and GT voxels over the rollout
      proj_err, proj_err_gt: mean projection error of the generated / GT views over the rollout
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
    gen_vox, gt_vox, vox_ious = [], [], []
    proj_errs, proj_errs_gt = [], []

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

        # Voxels at t+1: built from the GENERATED views and from the GT views
        voxel = build_voxel_batch(next_views_norm)
        voxel_gt = build_voxel_batch((gt_vn * 2.0 - 1.0).unsqueeze(0).to(device))
        gen_vox.append(torch.from_numpy(render_voxel_image(voxel[0])))
        gt_vox.append(torch.from_numpy(render_voxel_image(voxel_gt[0])))
        vox_ious.append(voxel_iou(voxel[0], voxel_gt[0]))

        # Pixel-space projection error of the generated views (no probe), and the GT floor
        proj_errs.append(projection_error(silhouettes_from_views(next_views_norm)).item())
        proj_errs_gt.append(projection_error(silhouettes_from_views((gt_vn * 2.0 - 1.0).unsqueeze(0).to(device))).item())

    vid_gen_top  = torch.stack(gen_top,  dim=0).unsqueeze(0)
    vid_gen_side = torch.stack(gen_side, dim=0).unsqueeze(0)
    vid_gen_rear = torch.stack(gen_rear, dim=0).unsqueeze(0)
    vid_gen_fpv  = torch.stack(gen_fpv,  dim=0).unsqueeze(0)

    vid_gt_top   = torch.stack(gt_top,   dim=0).unsqueeze(0)
    vid_gt_side  = torch.stack(gt_side,  dim=0).unsqueeze(0)
    vid_gt_rear  = torch.stack(gt_rear,  dim=0).unsqueeze(0)
    vid_gt_fpv   = torch.stack(gt_fpv,   dim=0).unsqueeze(0)

    vid_gen_vox  = torch.stack(gen_vox,  dim=0).unsqueeze(0)
    vid_gt_vox   = torch.stack(gt_vox,   dim=0).unsqueeze(0)
    vox_iou      = float(np.mean(vox_ious)) if vox_ious else 0.0
    proj_err     = float(np.mean(proj_errs)) if proj_errs else 0.0
    proj_err_gt  = float(np.mean(proj_errs_gt)) if proj_errs_gt else 0.0

    transformer.train()
    env.close()
    return (vid_gen_top, vid_gen_rear, vid_gen_side, vid_gen_fpv,
            vid_gt_top,  vid_gt_rear,  vid_gt_side,  vid_gt_fpv,
            vid_gen_vox, vid_gt_vox,   vox_iou,      proj_err,     proj_err_gt)


@torch.no_grad()
def evaluate_and_log_videos(transformer, vae, buffer, text_encoder, tokenizer, writer, epoch, rollout_steps=16, steps=20, fps=15):
    """
    Performs autoregressive video rollouts for each view and writes the videos to TensorBoard.
    """
    transformer.eval()
    unique_envs = ['Bipedal Walker', 'Lunar Lander', 'MultiCar Racing']
    reg_dict = dict(_build_registry())
    
    print(f"\n[VALIDATION] Running autoregressive video rollouts for epoch {epoch} across envs: {', '.join(unique_envs)}...")

    for env_idx, env_name in enumerate(unique_envs):
        env = reg_dict[env_name]()
        env.reset()

        (vid_top_down, vid_rear, vid_side, vid_fpv,
         vid_gt_top_down, vid_gt_rear, vid_gt_side, vid_gt_fpv,
         vid_gen_vox, vid_gt_vox, vox_iou, proj_err, proj_err_gt) = autoregressive_rollout(
            transformer=transformer,
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

        def _psnr(pred, gt):
            mse = F.mse_loss(pred, gt).item()
            return 20.0 * math.log10(1.0) - 10.0 * math.log10(max(mse, 1e-8))

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
        writer.add_scalar(f"Validation/{clean_tag}/Voxel_IoU",     vox_iou,       epoch)
        writer.add_scalar(f"Validation/{clean_tag}/Proj_Err",    proj_err,      epoch)
        writer.add_scalar(f"Validation/{clean_tag}/Proj_Err_GT", proj_err_gt,   epoch)

        print(f"  [Epoch {epoch}] [{env_name}] Autoregressive Video PSNR: {psnr_mean:.2f} dB "
              f"(Top-Down: {psnr_top_down:.2f}dB, Rear: {psnr_rear:.2f}dB, Side: {psnr_side:.2f}dB, FPV: {psnr_fpv:.2f}dB)")

        # Always write per-env tagged videos; also alias as generic Test/ for the first env
        writer.add_video(f"Test/{clean_tag}/Top-Down", vid_top_down, epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Rear",     vid_rear,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Side",     vid_side,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/FPV",      vid_fpv,      epoch, fps=fps)
        # Voxels built from the generated (vs GT) top/side/rear views at each step
        writer.add_video(f"Test/{clean_tag}/Voxel_Generated", vid_gen_vox, epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Voxel_GT",        vid_gt_vox,  epoch, fps=fps)

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
    print(" Voxel Constrained Orthographic Multi-View World Model (LoRA)")
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
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    log_dir = f"./runs/wan_ortho_multienv_voxel_cam_{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # Frozen silhouette probe for the projection consistency loss (fit on GT latents before LoRA training)
    sil_probe = None
    if args.cons_weight > 0.0:
        sil_probe = SilhouetteProbe(lat_ch=getattr(vae.config, "z_dim", 16)).to(device)
        pretrain_silhouette_probe(sil_probe, vae, buffer, steps=args.probe_steps, writer=writer)

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
            loss, per_view_losses, cons_loss = train_ortho_trajectory_step(
                transformer=model,
                sil_probe=sil_probe,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                views_next_seq=views_next_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                teacher_forcing_prob=curr_teacher_forcing_prob,
                ctx_noise_max=args.ctx_noise_max,
                cons_weight=args.cons_weight,
            )

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss
            global_step += 1

            writer.add_scalar("Train/Loss", loss, global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("Train/Teacher_Student_Prob", curr_teacher_forcing_prob, global_step)
            writer.add_scalar("Train/Loss_Proj", cons_loss, global_step)
            for v_idx, v_name in enumerate(view_names):
                writer.add_scalar(f"Train/Loss_{v_name}", per_view_losses[v_idx], global_step)

            env_summary = "+".join(list(dict.fromkeys(env_names_seq[0]))[:2])
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",
                "Top": f"{per_view_losses[0]:.3f}",
                "FPV": f"{per_view_losses[3]:.3f}",
                "Proj": f"{cons_loss:.3f}",
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
            )

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = f"./checkpoints/wan_ortho_lora_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            if sil_probe is not None:
                torch.save(sil_probe.state_dict(), os.path.join(ckpt_dir, "sil_probe.pt"))
            print(f"  [SAVED] LoRA + silhouette probe checkpoint to {ckpt_dir}")

    print("\nTraining Complete! Saving final LoRA weights...")
    final_dir = f"./checkpoints/wan_ortho_voxel_cam_{args.num_envs}_lora_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    if sil_probe is not None:
        torch.save(sil_probe.state_dict(), os.path.join(final_dir, "sil_probe.pt"))
    writer.close()

    for _, env in active_envs:
        try:
            env.close()
        except Exception:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA across multiple Orthographic Game Views")
    parser.add_argument("--env", type=str, default="all", help="Environment to train on: 'all' to train across all envs, or a specific name like 'Bipedal Walker'")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout steps when training on a single environment")
    parser.add_argument("--buffer_capacity", type=int, default=40000, help="Capacity of multi-environment transition buffer")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per gradient step")
    parser.add_argument("--train_seq_len", type=int, default=16, help="Contiguous trajectory steps unrolled per training update")
    parser.add_argument("--teacher_forcing_prob", type=float, default=1.0, help="Max prob of using self-predicted latents onto next trajectory step")
    parser.add_argument("--cons_weight", type=float, default=1.0, help="Weight of the projection consistency loss (0 disables it and the probe)")
    parser.add_argument("--probe_steps", type=int, default=500, help="Pre-training steps for the frozen silhouette probe")
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)