"""
Fine-tuning Wan2.1-1.3B with LoRA for Orthographic Multi-View World Modeling.

All four views at t+1 (Top, Side, Rear, FPV) are denoised jointly; `--arm` picks what the model sees:

  --arm fpv_base      FPV-BaseSingle: sees FPV_t only
      FPV_t | Top' Side' Rear' FPV'
  --arm ortho_base    Ortho-Base: sees all four views at t
      Top_t Side_t Rear_t FPV_t | Top' Side' Rear' FPV'
  --arm ortho_memory  Ortho-Memory: Ortho-Base plus the ortho views of earlier steps as memory
      [Top Side Rear]_{t-8} [Top Side Rear]_{t-4} | Top_t Side_t Rear_t FPV_t | Top' Side' Rear' FPV'

Ortho-Memory only stores ortho views: they don't depend on the camera, so they stay valid however the camera
moves, and FPV can be re-rendered from them. Memory slots (`--memory_offsets`) are blank (all zeros) when the
step is before the rollout start or dropped (`--mem_dropout`), and get more context noise than views_t
(`--mem_noise_max`), since at inference they are the model's own generated views. During training, student
forcing also writes the model's own predictions into later memory slots.

Every arm uses the same code, data and settings; at inference only generated frames are fed back.

Uses:
  - Wan2.1-T2V-1.3B-Diffusers pipeline
  - Frozen UMT5 text encoder
  - Trainable PEFT LoRA on WanTransformer3DModel
"""

import os
import math
import random
import argparse
import json
from datetime import datetime
from tqdm import tqdm

import numpy as np
import cv2
import torch
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


"""
Ideas to try
1. Ortho as memory: consistency when you revisit a place
Video world models forget what's off-screen: pan 360° and the scene behind you has changed when you come back. The Top/Rear views keep the whole surrounding state in view at every step, so Ortho-Base should remember what FPV can't see.
- Experiment: rollouts that pan away and back (Right ×N, then Left ×N). Score FPV on return against the frame it saw before leaving (LPIPS/PSNR), for FPV-Mono, FPV-History (more past frames), and Ortho-Base.
- Claim: "Explicit ortho state gives a world model persistent memory without long context." That's a known weakness of current game world models (Oasis, GameNGen-style), so it would be a real contribution.

2. Ortho as a control interface: edit the map, and the FPV follows
Edit the generated Top view (add an obstacle, move a car, erase a wall), feed it back as context, and see whether FPV renders the change consistently from then on. FPV-Mono has no equivalent handle.
- Experiment: scripted edits on the Top view. Measure whether the edit appears in FPV (detector or GT render) and persists over k steps.
- Claim: "Ortho views turn a world model into an editable game state": level design, counterfactuals, debugging. It also fits your game-engine framing.

Show that orthographic views (Top, Side, Rear) make a first-person (FPV) world model better. Supporting clean frames are context; noisy frames are what the model denoises and is scored on.
Usage the model has to become a Game Engine, so we cant keep feeding it real frames during inference, only x_0 

Naming Convention

A branch's name is what it sees plus how it's trained:
- Sees: FPV (only FPV at $t$) or Ortho (all four views at $t$)

Frame numbers follow a full 8-frame layout, so a view always has the same number.

FPV-MonoSingle: the Baseline (Typical in Diffusion World models)
frame:     3   |   7
          FPV  |  FPV
        views_t| views_{t+1} (noisy at tau)
loss:          |  yes

FPV-BaseSingle: also predicts the ortho views (In train_single_wan.py, very shaky but obvious becuase the other views arent given and only FPV is decent)
frame:     3   |   4      5      6      7
          FPV  |  Top    Side   Rear   FPV
        views_t| ---- views_{t+1} (noisy at tau) ----
loss:          |   yes    yes    yes    yes

Ortho-Link: the current code (The most obvious, during rollouts, we feed 4 real multiview frames, issues also since model gotta learn four views, the loss for FPV vs single is higher)
frame:     0  <->   1  <->    2   <->   3   |   4      5      6      7
         Top   Side  Rear  FPV   |  Top    Side   Rear   FPV
         ---- views_t (clean) -- | ---- views_{t+1} (noisy at tau) ----
loss:                            |   yes    yes    yes    yes

"""

# View order everywhere (buffer, renders, latents): 0 Top, 1 Side, 2 Rear, 3 FPV. FPV is always the last modelled view.
VIEW_NAMES = ["Top", "Side", "Rear", "3D_FPV"]
FPV_IDX = 3
ALL_VIEWS = (0, 1, 2, 3)
# TensorBoard names per view (scalar tag, video tag), unchanged from the earlier scripts so old and new runs line up
VIEW_TAGS = {"Top": ("Top Down", "Top-Down"), "Side": ("Side", "Side"), "Rear": ("Rear", "Rear"), "3D_FPV": ("FPV", "FPV")}
# Views stored in Ortho-Memory slots (camera-independent, so valid under any camera motion)
MEMORY_VIEWS = (0, 1, 2)
# Experiment arms: views seen at t (context) and views denoised at t+1 (targets). FPV is last in both.
ARMS = {
    "fpv_base":     {"context": (FPV_IDX,), "targets": ALL_VIEWS, "memory": False},
    "ortho_base":   {"context": ALL_VIEWS,  "targets": ALL_VIEWS, "memory": False},
    "ortho_memory": {"context": ALL_VIEWS,  "targets": ALL_VIEWS, "memory": True},
}

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
    memory=None,
    memory_valid=None,
    mem_noise_max=0.3,
):
    """
    Single transition Flow-Matching step (used inside multi-step trajectory rollout).
    Applies mild context noise to `latents_t` to prevent autoregressive drift,
    and optionally returns the model's reconstructed clean latent `pred_x0` at t+1.
      latents_t:    (B, C, V_ctx, h, w) context views at t (V_ctx = 1 for fpv_base)
      latents_next: (B, C, V, h, w) target views at t+1; the targets are always the last V frames
      memory:       optional (B, C, M, h, w) Ortho-Memory frames, prepended before `latents_t`, with
                    `memory_valid` (B, M) bool; invalid (blank) frames stay all-zero, as at inference
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

    # 3b. Ortho-Memory: larger context noise (memory is older generated output at inference); blank slots stay zero
    if memory is not None:
        memory_fp32 = memory.float()
        tau_mem = torch.rand((b, 1, 1, 1, 1), device=memory.device, dtype=torch.float32) * mem_noise_max
        noisy_memory = (1.0 - tau_mem) * memory_fp32 + tau_mem * torch.randn_like(memory_fp32)
        noisy_memory = noisy_memory * memory_valid.view(b, 1, -1, 1, 1).float()
        cond_latents_t = torch.cat([noisy_memory.to(dtype=transformer.dtype), cond_latents_t], dim=2)

    # 4. Concatenate conditioning frames ([memory,] views_t) and noisy target views (the last V frames)
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

    # 7. Supervise target views at t+1 (the last num_views frames)
    pred_velocity = model_pred[:, :, -num_views:].float()

    per_view_losses = [
        F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
        for idx in range(num_views)
    ]
    # Every view's loss has weight 1, so the FPV term is the same as in an FPV-only run
    loss = sum(per_view_losses)

    # Reconstruct 1-step denoised prediction: x_0_hat = x_tau - tau * v_pred
    with torch.no_grad():
        pred_x0 = (noisy_latents_next_fp32 - tau_exp * pred_velocity.detach()).clamp(-6.0, 6.0)
        confidence = (1.0 - tau_exp).clamp(0.2, 1.0)
        rollout_latent = (confidence * pred_x0 + (1.0 - confidence) * latents_next_fp32).to(dtype=transformer.dtype)
    return loss, [l.item() for l in per_view_losses], rollout_latent

def gather_memory(states, s, memory_offsets, drop_prob=0.0):
    """
    Ortho-Memory frames for predicting state s+1 from state s: Top/Side/Rear of states[s - o] for each offset o
    (`memory_offsets` sorted oldest first). A slot is blank (all zeros, invalid) if s - o is before the rollout
    start or, in training, dropped with probability `drop_prob`, so the model also learns to run with no memory.
      states: list of context latents (B, C, 4, h, w), states[k] = the views the model saw at step k
    Returns memory (B, C, 3 * len(memory_offsets), h, w) and memory_valid (B, 3 * len(memory_offsets)) bool.
    """
    ref = states[s][:, :, list(MEMORY_VIEWS)]
    b = ref.shape[0]
    slots, valid = [], []
    for o in memory_offsets:
        if s - o >= 0:
            keep = torch.rand(b, device=ref.device) >= drop_prob
            slots.append(states[s - o][:, :, list(MEMORY_VIEWS)] * keep.view(b, 1, 1, 1, 1).to(ref.dtype))
        else:
            keep = torch.zeros(b, dtype=torch.bool, device=ref.device)
            slots.append(torch.zeros_like(ref))
        valid.append(keep.view(b, 1).expand(b, len(MEMORY_VIEWS)))
    return torch.cat(slots, dim=2), torch.cat(valid, dim=1)

def train_ortho_trajectory_step(
    transformer,
    latents_t0,
    latents_next_seq,
    prompt_embeds_seq,
    ctx_in_tgt,
    shift=3.0,
    teacher_forcing_prob=0.5,
    ctx_noise_max=0.15,
    memory_offsets=(),
    mem_noise_max=0.3,
    mem_dropout=0.3,
):
    """
    Unrolls a multi-step trajectory during training.
    - At the first trained step, conditions on ground-truth context.
    - At later steps, with probability `teacher_forcing_prob`, conditions on the model's own predicted
      latent from the step before (student forcing) so the model learns to recover from its
      own rollout errors over multi-step horizons.
    - `ctx_in_tgt` picks the context views out of the target views (fpv_base: only FPV goes back in).
    - Ortho-Memory (`memory_offsets` non-empty): the first max(memory_offsets) transitions are ground-truth
      warm-up that only fills the memory (no loss); student-forced predictions also land in later memory slots.
    - Backpropagates `(loss_s / num_trained_steps)` at each step so peak VRAM remains constant.
    """
    seq_len = len(latents_next_seq)
    warmup = max(memory_offsets, default=0)
    num_trained = seq_len - warmup

    states = [latents_t0]  # context latents seen at each step
    total_loss = 0.0
    avg_per_view = [0.0] * latents_next_seq[0].shape[2]

    for s in range(seq_len):
        target_latents = latents_next_seq[s]
        target_ctx = target_latents[:, :, ctx_in_tgt]
        if s < warmup:
            states.append(target_ctx)
            continue

        memory, memory_valid = None, None
        if memory_offsets:
            memory, memory_valid = gather_memory(states, s, memory_offsets, drop_prob=mem_dropout)

        step_loss, step_view_losses, pred_next_latents = train_ortho_flow_step(
            transformer=transformer,
            latents_t=states[s],
            latents_next=target_latents,
            prompt_embeds=prompt_embeds_seq[s],
            shift=shift,
            ctx_noise_max=ctx_noise_max,
            memory=memory,
            memory_valid=memory_valid,
            mem_noise_max=mem_noise_max,
        )

        # Normalize loss across trained steps and backpropagate immediately to save VRAM
        scaled_loss = step_loss / float(num_trained)
        scaled_loss.backward()

        total_loss += step_loss.item() / float(num_trained)
        for v_idx in range(len(avg_per_view)):
            avg_per_view[v_idx] += step_view_losses[v_idx] / float(num_trained)

        # Prepare context for step s + 1 (Student Forcing vs. Teacher Forcing per sample)
        if s < seq_len - 1:
            pred_ctx = pred_next_latents[:, :, ctx_in_tgt]
            if teacher_forcing_prob > 0.0:
                b = target_ctx.shape[0]
                use_ar_mask = (torch.rand((b, 1, 1, 1, 1), device=target_ctx.device) < teacher_forcing_prob).to(dtype=transformer.dtype)
                states.append(use_ar_mask * pred_ctx + (1.0 - use_ar_mask) * target_ctx)
            else:
                states.append(target_ctx)

    return total_loss, avg_per_view

# ==========================================
# AUTOREGRESSIVE VIDEO ROLLOUT & SAMPLING
# ==========================================
@torch.no_grad()
def predict_next_view_latents(transformer, latents_t, prompt_embeds, steps=20, shift=3.0, num_targets=None, memory=None):
    """
    Jointly denoises the target views at t+1 via reverse Euler Flow Matching.
    Input:
      latents_t: (B, C, V_ctx, h, w) - clean past views at time t
      num_targets: number of views at t+1 (default V_ctx; 4 for fpv_base)
      memory: optional (B, C, M, h, w) Ortho-Memory frames (blank slots all-zero), prepended to the context
    Output:
      latents_next: (B, C, num_targets, h, w) - predicted views at time t+1
    """
    num_views = num_targets or latents_t.shape[2]
    if memory is not None:
        latents_t = torch.cat([memory.to(dtype=latents_t.dtype), latents_t], dim=2)
    b, c, _, h, w = latents_t.shape
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
        combined = torch.cat([latents_t, x_tau.to(dtype=transformer.dtype)], dim=2)  # (B, C, F_ctx + V, h, w)

        pred_velocity = transformer(
            hidden_states=combined,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, -num_views:].float()
        x_tau = x_tau - dt * pred_velocity

    return x_tau.to(dtype=transformer.dtype)

@torch.no_grad()
def autoregressive_rollout(transformer, vae, text_encoder, tokenizer, env_name, env, agent=None,
                           rollout_steps=16, steps=20, camera_command="Right", seed=42,
                           arm="ortho_base", memory_offsets=()):
    """
    Autoregressive video rollout of the arm's target views (predicted jointly), holding `camera_command`
    every step (e.g. a Right pan). The env is driven exactly like `collect_rollouts` (same policy, same render
    order), so every prompt carries the env's real camera state for its target frame, as in training.
    Only the first context is real; after that the arm's context views (and Ortho-Memory slots) are the
    model's own decoded and re-encoded outputs.
    Returns:
      gen: {view name: (1, T, 3, H, W) in [0, 1]} for each rolled-out view
      gt:  {view name: (1, T, 3, H, W) in [0, 1]} ground truth for the same views
      vis: (gen_fpv_vis, gt_fpv_vis) pose-annotated FPV videos for logging
    """
    transformer.eval()
    context_idx, target_idx = ARMS[arm]["context"], ARMS[arm]["targets"]
    ctx_in_tgt = [target_idx.index(i) for i in context_idx]

    obs, _ = env.reset(seed=seed)
    obs = _skip_intro(env, env_name, obs)
    env.action_space.seed(seed)

    # Same first render as collect_rollouts: "Center" is a no-op right after reset (yaw = pitch = 0)
    img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")

    # Context views as one sample: (1, V_ctx, 3, H, W) -> latents (1, C, V_ctx, h, w)
    vt = [cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA) for x in [img_top, im_side, im_rear, im_fpv]]
    vt = np.stack([np.transpose(vt[i], (2, 0, 1)) for i in context_idx])
    vt = np.expand_dims(vt, axis=0)

    curr_views = torch.from_numpy(vt).float().to(device) / 127.5 - 1.0
    curr_latents = encode_views_to_latents(vae, curr_views, out_dtype=transformer.dtype)
    states = [curr_latents]  # context latents seen at each step, for Ortho-Memory

    gen = {VIEW_NAMES[i]: [] for i in target_idx}
    gt = {VIEW_NAMES[i]: [] for i in target_idx}
    gen_fpv_vis, gt_fpv_vis = [], []  # pose-annotated copies, for logged videos only

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

        memory = gather_memory(states, step_i, memory_offsets)[0] if memory_offsets else None
        next_latents = predict_next_view_latents(
            transformer=transformer,
            latents_t=curr_latents,
            prompt_embeds=prompt_embeds,
            steps=steps,
            num_targets=len(target_idx),
            memory=memory,
        )

        decoded_views = decode_latents_to_views(vae, next_latents)  # (1, V, 3, H, W) in [0, 1]

        vn = [cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA) for x in [next_top, next_side, next_rear, next_fpv]]
        gt_vn = torch.from_numpy(np.stack([np.transpose(v, (2, 0, 1)) for v in vn])).float() / 255.0  # (4, 3, H, W)

        # Raw frames are used for PSNR; the pose overlay only goes on the logged FPV videos (FPV is the last view)
        for k, i in enumerate(target_idx):
            gen[VIEW_NAMES[i]].append(decoded_views[0, k].cpu())
            gt[VIEW_NAMES[i]].append(gt_vn[i])
        gen_fpv_vis.append(_annotate_frame_pose(decoded_views[0, -1].cpu(), camera_command))
        gt_fpv_vis.append(_annotate_frame_pose(gt_vn[FPV_IDX], camera_command))

        # Re-encode the decoded context views back through the VAE to stay on the clean VAE manifold
        next_views_norm = decoded_views[:, ctx_in_tgt] * 2.0 - 1.0
        curr_latents = encode_views_to_latents(vae, next_views_norm, out_dtype=transformer.dtype)
        states.append(curr_latents)

        # Training windows never cross an episode reset, so stop instead of jumping to a new episode
        if terminated or truncated:
            break

    transformer.train()
    env.close()
    gen = {name: torch.stack(frames, dim=0).unsqueeze(0) for name, frames in gen.items()}
    gt = {name: torch.stack(frames, dim=0).unsqueeze(0) for name, frames in gt.items()}
    vis = (torch.stack(gen_fpv_vis, dim=0).unsqueeze(0), torch.stack(gt_fpv_vis, dim=0).unsqueeze(0))
    return gen, gt, vis


@torch.no_grad()
def evaluate_and_log_videos(transformer, vae, text_encoder, tokenizer, writer, epoch, agents=None, rollout_steps=16, steps=20,
                            fps=15, seed=42, arm="ortho_base", memory_offsets=()):
    """
    Performs autoregressive video rollouts and writes per-view PSNR and videos to TensorBoard.
    Validation/<env>/PSNR/FPV is the number to compare between arms.
    `agents` maps env_name -> the policy used during data collection (random actions if missing).
    """
    agents = agents or {}
    transformer.eval()
    unique_envs = ['Bipedal Walker', 'Lunar Lander', 'MultiCar Racing']
    reg_dict = dict(_build_registry())

    print(f"\n[VALIDATION] Running autoregressive video rollouts for Epoch {epoch} across envs: {', '.join(unique_envs)}...")

    for env_name in unique_envs:
        env = reg_dict[env_name]()

        gen, gt, (vid_fpv_vis, vid_gt_fpv_vis) = autoregressive_rollout(
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
            arm=arm,
            memory_offsets=memory_offsets,
        )

        clean_tag = env_name.replace(" ", "_").lower()

        psnrs = {name: _psnr(gen[name], gt[name]) for name in gen}
        for name, value in psnrs.items():
            writer.add_scalar(f"Validation/{clean_tag}/PSNR/{VIEW_TAGS[name][0]}", value, epoch)
        if len(psnrs) > 1:
            writer.add_scalar(f"Validation/{clean_tag}/PSNR/Mean", sum(psnrs.values()) / len(psnrs), epoch)

        print(f"  [Epoch {epoch}] [{env_name}] Autoregressive Video PSNR: "
              + ", ".join(f"{VIEW_TAGS[name][0]}: {value:.2f}dB" for name, value in psnrs.items()))

        for name in gen:
            is_fpv = name == VIEW_NAMES[FPV_IDX]
            writer.add_video(f"Test/{clean_tag}/{VIEW_TAGS[name][1]}", vid_fpv_vis if is_fpv else gen[name], epoch, fps=fps)
            writer.add_video(f"GT/{clean_tag}/{VIEW_TAGS[name][1]}", vid_gt_fpv_vis if is_fpv else gt[name], epoch, fps=fps)

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
    env.action_space.seed(seed)  # random-action envs: every run (and every --arm) gets the same data
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

def save_checkpoint(model, ckpt_dir, args):
    """Saves the LoRA plus the training flags (train_args.json, incl. --arm and --memory_offsets)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    with open(os.path.join(ckpt_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  [SAVED] LoRA checkpoint to {ckpt_dir}")


# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
def main(args):
    set_seed(args.seed)
    run_tag = args.arm
    context_idx, target_idx = ARMS[args.arm]["context"], ARMS[args.arm]["targets"]
    ctx_in_tgt = [target_idx.index(i) for i in context_idx]
    # Ortho-Memory slots, oldest first; the trajectory gets max(offsets) extra ground-truth warm-up steps
    memory_offsets = tuple(sorted(set(args.memory_offsets), reverse=True)) if ARMS[args.arm]["memory"] else ()
    warmup = max(memory_offsets, default=0)
    view_names = [VIEW_NAMES[i] for i in target_idx]
    print("=" * 60)
    print(f"  Orthographic Multi-View World Model (LoRA), arm: {args.arm}")
    print(f"  Context: {', '.join(VIEW_NAMES[i] for i in context_idx)} -> Targets: {', '.join(view_names)}")
    if memory_offsets:
        print(f"  Ortho-Memory: Top/Side/Rear at t-{', t-'.join(map(str, memory_offsets))} "
              f"(dropout={args.mem_dropout}, noise<={args.mem_noise_max})")
    print(f"  Multi-Step Trajectory Training (seq_len={args.train_seq_len}, teacher_forcing_prob={args.teacher_forcing_prob})")
    print("=" * 60)

    # 1. Environments & Replay Buffer
    registry = _build_registry()
    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(
        capacity=args.buffer_capacity, img_h=96, img_w=96, max_action_len=8
    )

    active_envs, agents = collect_all_envs_rollouts(
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

    log_dir = f"./runs/wan_{run_tag}_multienv_cam_{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

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
            # Sample batch of contiguous trajectories (Ortho-Memory: plus `warmup` steps that only fill the memory)
            # views_t:       (B, 4, 3, H, W) — anchor frame at t=0
            # views_next_seq: list[T] of (B, 4, 3, H, W) — target frames t=1..T
            seq_len = args.train_seq_len + warmup
            views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq = buffer.sample_trajectory_batch(
                batch_size=args.batch_size,
                seq_len=seq_len,
            )
            # The model sees the arm's context views at t and denoises its target views at t+1 (FPV last in both)
            views_t = views_t[:, list(context_idx)]
            views_next_seq = [v[:, list(target_idx)] for v in views_next_seq]

            with torch.no_grad():
                # Encode views at t=0
                latents_t = encode_views_to_latents(vae, views_t, out_dtype=model.dtype)

                # Encode target views and text prompts for each step along the trajectory
                latents_next_seq = []
                prompt_embeds_seq = []

                for s in range(seq_len):
                    latents_s = encode_views_to_latents(vae, views_next_seq[s], out_dtype=model.dtype)
                    latents_next_seq.append(latents_s)
                    if s < warmup:
                        prompt_embeds_seq.append(None)  # warm-up steps only fill the memory: no prediction
                        continue

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
            loss, per_view_losses = train_ortho_trajectory_step(
                transformer=model,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                ctx_in_tgt=ctx_in_tgt,
                teacher_forcing_prob=curr_teacher_forcing_prob,
                ctx_noise_max=args.ctx_noise_max,
                memory_offsets=memory_offsets,
                mem_noise_max=args.mem_noise_max,
                mem_dropout=args.mem_dropout,
            )

            # Pre-clip norm; values above 1.0 mean clipping is shrinking the update
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss
            global_step += 1

            writer.add_scalar("Train/Loss", loss, global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("Train/Teacher_Student_Prob", curr_teacher_forcing_prob, global_step)
            writer.add_scalar("Train/GradNorm", grad_norm.item(), global_step)
            for v_idx, v_name in enumerate(view_names):
                writer.add_scalar(f"Train/Loss_{v_name}", per_view_losses[v_idx], global_step)

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
                seed=args.seed,
                arm=args.arm,
                memory_offsets=memory_offsets,
            )

        # Checkpointing
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, f"./checkpoints/wan_{run_tag}_lora_epoch_{epoch+1}", args)

    print("\nTraining Complete! Saving final LoRA weights...")
    save_checkpoint(model, f"./checkpoints/wan_{run_tag}_cam_{args.num_envs}_lora_final", args)
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
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank dimension")
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
    parser.add_argument("--arm", type=str, default="ortho_base", choices=list(ARMS),
                        help="fpv_base: FPV_t -> all 4 views; ortho_base: all 4 views at t -> all 4; "
                             "ortho_memory: ortho_base plus past Top/Side/Rear as memory")
    parser.add_argument("--memory_offsets", type=int, nargs="+", default=[4, 8],
                        help="ortho_memory: steps back for each memory slot (Top/Side/Rear at t-k)")
    parser.add_argument("--mem_dropout", type=float, default=0.3,
                        help="ortho_memory: prob. a memory slot is blanked in training (inference starts with blank memory)")
    parser.add_argument("--mem_noise_max", type=float, default=0.3,
                        help="ortho_memory: max context noise on memory frames (views_t use --ctx_noise_max)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)