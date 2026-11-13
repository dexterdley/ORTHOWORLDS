"""
Fine-tuning Wan2.1-1.3B with LoRA for Orthographic Multi-View World Modeling.
FPV-Single + ortho memory guide: train_single_dist.py, plus one clean context frame per step, the "guide":
a sparse-voxel memory of everything the Top/Side/Rear views have shown so far in the episode, drawn from the
FPV camera of the target frame.

Modelling Process
frame:     0      1       |   2
          FPV_t  guide_t+1 |  FPV_t+1 (noisy at tau)
loss:                      |   yes

The guide is built during rollout collection (GymBox envs) and cached in the buffer with every transition:
  1. Depth of every Top/Side/Rear pixel: the env's render() is traced to grab its locals (the 3D polygons it
     draws), and its painter pass is re-run with polygon ids instead of colours (_paint), so each pixel knows
     which polygon it shows and how deep that polygon is there. Each ortho view is an affine map
     A: world (metres) -> (u, v, depth), so every pixel lifts back to a 3D point.
  2. Memory (_remember): those points with their colours, one per VOXEL cell (latest write wins). The agent and
     cars (painter layer DYNAMIC) are never stored.
  3. Guide (_draw_guide): the memory up to step t, drawn from the FPV camera at t+1 as squares of each voxel's
     projected size (nearest wins), at buffer resolution; black where nothing is remembered.
Only Bipedal Walker, Lunar Lander and MultiCar Racing (the first three registry envs) are supported: --num_envs <= 3.
Validation replays a fixed clip per env from the buffer with its cached guides, and logs the guide videos.

Usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./ortho/train_single_dist_memory.py --rollout_steps=2048 --epochs=20 --batch_size=16 --teacher_forcing_prob=0.5 --steps_per_epoch=200 --num_envs=3 --lr=3e-4 --model_id=./Wan2.1-T2V-1.3B-Diffusers/

"""

import os
import sys
import random
import argparse
import contextlib
from datetime import datetime, timedelta
from tqdm import tqdm

import numpy as np
import pygame
from pygame import gfxdraw
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from gymnasium.envs.box2d import car_racing

# HuggingFace & Diffusers imports
from utils import load_model, compose_player_prompts, set_seed, is_main, load_agent, _unwrap, _psnr, _annotate_frame_pose
from buffer import OrthoTransitionBuffer
from models import encode_views_to_latents, decode_latents_to_views
from data_rollout import get_registries, _skip_intro, _camera_state
from validation import VAL_ENVS


# torchrun sets LOCAL_RANK; set the CUDA device before anything touches "cuda" (utils/buffer use the bare "cuda" device)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
IMG_H, IMG_W = 128, 128

GUIDE_ENVS = ("Bipedal Walker", "Lunar Lander", "MultiCar Racing")
VOXEL = 0.25    # memory cell size, metres
DYNAMIC = 2     # painter layer of the agent / cars in the GUIDE_ENVS: never stored in the memory
SPLAT = 3       # largest half-size of a voxel's square in the guide, pixels
EMPTY = (np.zeros((0, 3)), np.zeros((0, 3), np.uint8))   # memory = (points (N, 3) in metres, colours (N, 3))


# ==========================================
# ORTHO MEMORY -> FPV GUIDE (GymBox)
# ==========================================
def _paint(polys, A, size, min_d=-np.inf):
    """Re-runs a GymBox painter pass for one ortho view with polygon ids as colours: same projection A
    (world -> (u, v, d)), same (layer, -mean depth) order, same pygame rasteriser. Polygons with a vertex
    shallower than min_d are dropped (the env's own clip). Returns depth (H, W) along the view axis
    (inf = background) and the painter layer of the polygon each pixel shows (-1 = background)."""
    W, H = size
    items = []
    for verts, layer in polys:
        uvd = verts @ A[:, :3].T + A[:, 3]
        if (uvd[:, 2].min() < min_d or uvd[:, 0].max() < 0 or uvd[:, 0].min() >= W
                or uvd[:, 1].max() < 0 or uvd[:, 1].min() >= H):
            continue
        items.append((layer, -uvd[:, 2].mean(), uvd))
    items.sort(key=lambda it: it[:2])
    surf = pygame.Surface((W, H))                       # black = id 0 = background
    planes = np.zeros((len(items) + 1, 3))              # per polygon: d = a*u + b*v + c
    layers = np.full(len(items) + 1, -1)
    for i, (layer, _, uvd) in enumerate(items, 1):
        try:
            gfxdraw.filled_polygon(surf, [(int(round(u)), int(round(v))) for u, v, _ in uvd],
                                   (i & 255, (i >> 8) & 255, i >> 16))
        except Exception:
            pass
        m = uvd.mean(0)
        ab = np.linalg.lstsq(uvd[:, :2] - m[:2], uvd[:, 2] - m[2], rcond=1e-9)[0]
        planes[i] = (ab[0], ab[1], m[2] - ab @ m[:2])
        layers[i] = layer
    rgb = pygame.surfarray.array3d(surf).transpose(1, 0, 2).astype(np.int64)
    ids = rgb[..., 0] | rgb[..., 1] << 8 | rgb[..., 2] << 16
    v, u = np.mgrid[0:H, 0:W] + 0.5
    depth = planes[ids, 0] * u + planes[ids, 1] * v + planes[ids, 2]
    return np.where(ids > 0, depth, np.inf), layers[ids]


def _gym_aov(env_name, L, frames):
    """Rebuilds what the memory needs from the locals L of the env's render(): for each ortho view
    (depth, painter layer, A (3, 4): metres -> (u, v, depth) in frame pixels), and the FPV camera in metres.
    The A rows are the env's own projection formulas (game_envs/*_orthographic.py render())."""
    s, (fh, fw) = L["self"], frames[0].shape[:2]
    if env_name == "MultiCar Racing":
        polys = [(np.array([(p[0], p[1], p[2] if len(p) == 3 else 0.0) for p in v]), layer) for v, _, layer in L["polys"]]
        W, H, k, c, sn = s.WINDOW_W, s.WINDOW_H, s.scale_factor, L["cos_a"], L["sin_a"]
        x0, y0, ch = L["car_x"], L["car_y"], s.camera_height
        A = {"rear": ([[k * c, -k * sn, 0, -k * (c * x0 - sn * y0) + W / 2], [0, 0, -k, k * ch + H / 2],
                       [sn, c, 0, -(sn * x0 + c * y0) + s.camera_dist]], 2.5),
             "side": ([[k * sn, k * c, 0, -k * (sn * x0 + c * y0) + W / 2], [0, 0, -k, k * ch + H / 2],
                       [-c, sn, 0, c * x0 - sn * y0]], -np.inf)}
        upm, near = 1.0, 1.0
    else:   # Bipedal Walker / Lunar Lander: world in pixels, SCALE px per metre
        polys = [(np.array(v, float), layer) for v, _, layer in L["polys_3d"]]
        W, H, cx, cy = s.VIEWPORT_W, s.VIEWPORT_H, L["cx"], L["cy"]
        if env_name == "Bipedal Walker":
            o = -L["scroll_x"]
            A = {"top": [[1, 0, 0, o], [0, 0, 1, cy], [0, -1, 0, 0]],
                 "rear": [[0, 0, 1, cx], [0, -1, 0, H], [1, 0, 0, o]],
                 "side": [[1, 0, 0, o], [0, -1, 0, H], [0, 0, 1, 0]]}
        else:
            lx = L["cam_ortho_x"]
            A = {"top": [[1, 0, 0, cx - lx], [0, 0, 1, cy], [0, -1, 0, 0]],
                 "rear": [[1, 0, 0, cx - lx], [0, -1, 0, H], [0, 0, -1, 0]],
                 "side": [[0, 0, 1, cx], [0, -1, 0, H], [1, 0, 0, -lx]]}
        A = {v: (a, -np.inf) for v, a in A.items()}
        upm, near = s.SCALE, 10.0

    aov = {}
    for view, (a, min_d) in A.items():
        a = np.diag([fw / W, fh / H, 1.0]) @ np.array(a, float)   # the env draws at W x H, render() returns fw x fh
        aov[view] = (*_paint(polys, a, (fw, fh), min_d), a)
    if env_name == "MultiCar Racing":
        # Top is CarRacing's own render: zoom, rotate, translate, flip vertically, scale to the video size.
        # Everything on it is ground (height 0) except the car; its HUD strip is the bottom 5/40 of the frame.
        z = (0.1 * car_racing.SCALE * max(1 - L["base_env"].t, 0) + car_racing.ZOOM * car_racing.SCALE * min(L["base_env"].t, 1))
        WW, WH = car_racing.WINDOW_W, car_racing.WINDOW_H
        a = np.array([[fw / WW * z * c, -fw / WW * z * sn, 0, fw / WW * (-z * (c * x0 - sn * y0) + WW / 2)],
                      [-fh / WH * z * sn, -fh / WH * z * c, 0, fh / WH * (WH * 3 / 4 + z * (sn * x0 + c * y0))],
                      [0, 0, -1, 0]])
        layer = _paint([p for p in polys if p[1] == DYNAMIC], a, (fw, fh))[1]
        depth = np.zeros((fh, fw))
        depth[fh * 35 // 40:] = np.inf
        aov["top"] = (depth, layer, a)
    for view, (depth, layer, a) in aov.items():      # world units -> metres
        a = a.copy()
        a[:, :3] *= upm
        aov[view] = (depth, layer, a)
    cam = dict(pos=np.array([s.cam_x, s.cam_y, s.cam_z]) / upm, f=s.fov, W=W, H=H, near=near / upm,
               right=np.array([L["right_x"], L["right_y"], L["right_z"]]),
               up=np.array([L["up_x"], L["up_y"], L["up_z"]]),
               fwd=np.array([L["fwd_x"], L["fwd_y"], L["fwd_z"]]))
    return aov, cam


def _render(env, env_name, pose):
    """env.render(camera_pose=pose) -> (frames, ortho AOV, FPV camera). The env's render() is traced only to
    read its locals when it returns; the frames are exactly what it draws."""
    code, L = env.render.__func__.__code__, {}

    def trace_render(frame, event, arg):
        if event == "return":
            L.update(frame.f_locals)
        return trace_render

    def trace_calls(frame, event, arg):
        if frame.f_code is code:
            frame.f_trace_lines = False
            return trace_render

    sys.settrace(trace_calls)
    try:
        frames = env.render(camera_pose=pose)
    finally:
        sys.settrace(None)
    return (frames, *_gym_aov(env_name, L, frames))


def _remember(mem, frames, aov):
    """Adds the static surfaces Top/Side/Rear show in this render to the memory; latest write wins per VOXEL cell."""
    frames = dict(zip(("top", "rear", "side", "fpv"), frames))   # render() order
    X, rgb = [mem[0]], [mem[1]]
    for view, (depth, layer, A) in aov.items():
        r, c = np.nonzero(np.isfinite(depth) & (layer != DYNAMIC))
        X.append((np.c_[c + 0.5, r + 0.5, depth[r, c]] - A[:, 3]) @ np.linalg.inv(A[:, :3]).T)
        rgb.append(frames[view][r, c])
    X, rgb = np.concatenate(X)[::-1], np.concatenate(rgb)[::-1]   # newest first: np.unique keeps the first
    k = np.floor(X / VOXEL).astype(np.int64) + (1 << 20)
    _, keep = np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2], return_index=True)
    return X[keep], rgb[keep]


def _draw_guide(mem, cam, hw):
    """Memory -> FPV guide (hw, hw, 3) uint8 at buffer resolution: each voxel a square of its projected size,
    nearest wins; black where nothing is remembered."""
    X, rgb = mem
    d = X - cam["pos"]
    z = d @ cam["fwd"]
    front = z > cam["near"]
    d, z, rgb = d[front], z[front], rgb[front]
    sx, sy = hw / cam["W"], hw / cam["H"]
    u = np.floor((cam["f"] * (d @ cam["right"]) / z + cam["W"] / 2) * sx).astype(int)
    v = np.floor((cam["H"] / 2 - cam["f"] * (d @ cam["up"]) / z) * sy).astype(int)
    r = np.minimum(0.5 * VOXEL * cam["f"] / z * max(sx, sy), SPLAT).astype(int)
    pix, depth, col = [], [], []
    for dy in range(-SPLAT, SPLAT + 1):
        for dx in range(-SPLAT, SPLAT + 1):
            m = (r >= max(abs(dx), abs(dy))) & (u + dx >= 0) & (u + dx < hw) & (v + dy >= 0) & (v + dy < hw)
            pix.append((v[m] + dy) * hw + u[m] + dx)
            depth.append(z[m])
            col.append(rgb[m])
    order = np.argsort(np.concatenate(depth))
    pix, col = np.concatenate(pix)[order], np.concatenate(col)[order]
    _, first = np.unique(pix, return_index=True)
    img = np.zeros((hw * hw, 3), np.uint8)
    img[pix[first]] = col[first]
    return img.reshape(hw, hw, 3)


# ==========================================
# ROLLOUTS + BUFFER WITH GUIDES
# ==========================================
class GuidedBuffer(OrthoTransitionBuffer):
    """OrthoTransitionBuffer plus guides_next: the FPV guide of every transition, aligned with views_next."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guides_next = np.zeros((self.capacity, 3, self.img_h, self.img_w), dtype=np.uint8)
        self._ARRAYS += ("guides_next",)

    def push(self, guide_next, **transition):
        self.guides_next[self.ptr] = np.transpose(guide_next, (2, 0, 1))
        super().push(**transition)

    def sample_trajectory_batch(self, batch_size, seq_len, target_device=device):
        """OrthoTransitionBuffer.sample_trajectory_batch (camera states included) + guides_seq: list of seq_len
        tensors (B, 3, H, W) in [-1, 1], the guide of each target frame."""
        valid_starts = self._get_valid_trajectory_starts(seq_len)
        if valid_starts:
            start_idxs = np.random.choice(valid_starts, size=batch_size)
        else:
            start_idxs = np.random.randint(0, max(1, self.size - seq_len), size=batch_size)
        to = lambda a: torch.from_numpy(a).to(target_device).float() / 127.5 - 1.0
        views_t = to(self.views_t[start_idxs])
        views_next_seq, actions_seq, env_names_seq, camera_states_seq, guides_seq = [], [], [], [], []
        for s in range(seq_len):
            idxs = np.minimum(start_idxs + s, self.size - 1)
            views_next_seq.append(to(self.views_next[idxs]))
            guides_seq.append(to(self.guides_next[idxs]))
            actions_seq.append([torch.from_numpy(self.actions[i, :self.action_lens[i]]).float().to(target_device) for i in idxs])
            env_names_seq.append([self.env_names[i] for i in idxs])
            camera_states_seq.append([self.camera_states[i] for i in idxs])
        return views_t, views_next_seq, actions_seq, env_names_seq, camera_states_seq, guides_seq


def collect_rollouts(env, agent, buffer, n_steps, seed, env_name):
    """data_rollout.collect_rollouts plus the guide: guide_{t+1} = memory of Top/Side/Rear up to step t,
    drawn from the FPV camera at t+1."""
    obs, _ = env.reset(seed=seed)
    obs = _skip_intro(env, env_name, obs)
    env.action_space.seed(seed)
    pose_hold = 0

    # Render once per env step (each render advances the camera), so views_next[k] == views_t[k+1]
    frames, aov, _ = _render(env, env_name, "Center")
    mem = _remember(EMPTY, frames, aov)

    for _ in tqdm(range(n_steps), desc=f"Rollouts [{env_name}]", disable=not is_main()):
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

        next_frames, aov, cam = _render(env, env_name, current_pose)
        guide_next = _draw_guide(mem, cam, buffer.img_h)     # memory of Top/Side/Rear up to step t
        mem = _remember(mem, next_frames, aov)               # now up to t+1

        (top, rear, side, fpv), (n_top, n_rear, n_side, n_fpv) = frames, next_frames
        buffer.push(
            guide_next=guide_next,
            top_t=top, side_t=side, rear_t=rear, fpv_t=fpv,
            top_next=n_top, side_next=n_side, rear_next=n_rear, fpv_next=n_fpv,
            action=action, done=done, env_name=env_name,
            camera_state=_camera_state(env, current_pose),
        )

        if done:
            obs, _ = env.reset()
            obs = _skip_intro(env, env_name, obs)
            pose_hold = 0
            frames, aov, _ = _render(env, env_name, "Center")
            mem = _remember(EMPTY, frames, aov)
        else:
            obs, frames = next_obs, next_frames
    env.close()


def collect_all_envs_rollouts(registry, buffer, rollout_steps, num_envs, seed, device):
    """data_rollout.collect_all_envs_rollouts with the guided collect_rollouts. Validation replays the buffer,
    so no envs or agents are kept."""
    for env_name, factory in registry[:num_envs]:
        assert env_name in GUIDE_ENVS, f"no guide geometry for {env_name}: use --num_envs <= {len(GUIDE_ENVS)}"
        print(f"\nInitializing environment: {env_name}")
        env = factory()
        obs, _ = env.reset()
        if env_name == "Lunar Lander":
            agent = PPO.load(load_from_hub(repo_id="Adilbai/ppo-LunarLander-v2", filename="ppo-LunarLander-v2.zip"))
        elif env_name == "MultiCar Racing":
            agent = PPO.load(load_from_hub(repo_id="igpaub/ppo-CarRacing-v2", filename="ppo-CarRacing-v2.zip"))
        else:
            state_dim = int(np.prod(obs.shape if hasattr(obs, "shape") else env.observation_space.shape))
            clean_env_name = "".join(c for c in env_name if c.isalnum() or c in ("_", "-")).lower()
            agent = load_agent(os.path.join("checkpoints", f"ppo_{clean_env_name}.pt"), state_dim, env.action_space, device)
        collect_rollouts(env, agent, buffer, rollout_steps, seed, env_name)
    print(f"\n[INFO] Buffer populated with {buffer.size} transitions")


# ==========================================
# FLOW MATCHING WITH THE GUIDE FRAME
# ==========================================
def train_guided_flow_step(transformer, latents_t, guide, latents_next, prompt_embeds, shift=3.0, ctx_noise_max=0.15):
    """flow.train_ortho_flow_step with the guide as a clean frame between context and target:
    [FPV_t (context-noised), guide_{t+1}, noisy FPV_{t+1}]; the loss is on FPV_{t+1} only."""
    b, c, num_views, h, w = latents_next.shape
    model_dtype = _unwrap(transformer).dtype

    # Shifted logit-normal flow-matching timestep tau in (0, 1)
    tau = torch.sigmoid(torch.randn(b, device=latents_next.device, dtype=torch.float32))
    tau = ((shift * tau) / (1.0 + (shift - 1.0) * tau)).clamp(1e-4, 1.0 - 1e-4)
    tau_exp = tau.view(b, 1, 1, 1, 1)

    # Context noise augmentation on FPV_t (the guide stays clean)
    latents_t_fp32 = latents_t.float()
    tau_ctx = torch.rand((b, 1, 1, 1, 1), device=latents_t.device, dtype=torch.float32) * ctx_noise_max
    cond_latents_t = ((1.0 - tau_ctx) * latents_t_fp32 + tau_ctx * torch.randn_like(latents_t_fp32)).to(dtype=model_dtype)

    latents_next_fp32 = latents_next.float()
    noise = torch.randn_like(latents_next_fp32)
    noisy_latents_next = ((1.0 - tau_exp) * latents_next_fp32 + tau_exp * noise).to(dtype=model_dtype)

    model_pred = transformer(
        hidden_states=torch.cat([cond_latents_t, guide, noisy_latents_next], dim=2),
        timestep=(tau * 1000.0).to(dtype=model_dtype),
        encoder_hidden_states=prompt_embeds,
    ).sample
    pred_velocity = model_pred[:, :, -num_views:].float()
    target_velocity = noise - latents_next_fp32
    per_view_losses = [F.mse_loss(pred_velocity[:, :, i], target_velocity[:, :, i]) for i in range(num_views)]
    return sum(per_view_losses), [l.item() for l in per_view_losses]


def train_guided_trajectory_step(transformer, latents_t0, latents_next_seq, guide_seq, prompt_embeds_seq,
                                 teacher_forcing_prob=0.5, ctx_noise_max=0.15, sf_steps=16):
    """flow.train_ortho_trajectory_step with guide_seq[s] (the guide of target s) in every step, including the
    student-forced samples. Backpropagates each step's loss / seq_len immediately to keep VRAM constant."""
    seq_len = len(latents_next_seq)
    curr_latents = latents_t0
    total_loss, avg_per_view = 0.0, [0.0] * latents_t0.shape[2]

    for s in range(seq_len):
        is_last_step = s == seq_len - 1
        # Skip the gradient all-reduce until the last step of the trajectory
        sync_ctx = transformer.no_sync() if (isinstance(transformer, DDP) and not is_last_step) else contextlib.nullcontext()
        with sync_ctx:
            step_loss, step_view_losses = train_guided_flow_step(
                transformer, curr_latents, guide_seq[s], latents_next_seq[s], prompt_embeds_seq[s], ctx_noise_max=ctx_noise_max)
            (step_loss / float(seq_len)).backward()
        total_loss += step_loss.item() / float(seq_len)
        avg_per_view = [a + l / float(seq_len) for a, l in zip(avg_per_view, step_view_losses)]

        # Context for step s + 1: ground truth, or the model's own sample (student forcing) per sample
        if not is_last_step:
            next_latents = latents_next_seq[s].clone()
            use_ar = torch.rand(curr_latents.shape[0], device=curr_latents.device) < teacher_forcing_prob
            if use_ar.any():
                next_latents[use_ar] = predict_next_fpv(
                    _unwrap(transformer), curr_latents[use_ar], guide_seq[s][use_ar], prompt_embeds_seq[s][use_ar], steps=sf_steps)
            curr_latents = next_latents

    return total_loss, avg_per_view


@torch.no_grad()
def predict_next_fpv(transformer, latents_t, guide, prompt_embeds, steps=20, shift=3.0):
    """flow.predict_next_view_latents with the guide frame: Euler-samples FPV_{t+1} from [FPV_t, guide_{t+1}, x_tau]."""
    b, c, num_views, h, w = latents_t.shape
    latents_t = latents_t.to(dtype=transformer.dtype)
    # Keep Euler state in float32 to prevent bfloat16 rounding stagnation
    x_tau = torch.randn((b, c, num_views, h, w), device=latents_t.device, dtype=torch.float32)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=latents_t.device, dtype=torch.float32)
    sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)
    for i in range(steps):
        t_tensor = torch.full((b,), sigmas[i] * 1000.0, device=latents_t.device, dtype=transformer.dtype)
        pred_velocity = transformer(
            hidden_states=torch.cat([latents_t, guide, x_tau.to(dtype=transformer.dtype)], dim=2),
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, -num_views:].float()
        x_tau = x_tau - (sigmas[i] - sigmas[i + 1]) * pred_velocity
    return x_tau.to(dtype=transformer.dtype)


# ==========================================
# VALIDATION: GUIDED FPV ROLLOUTS + GUIDE VIDEOS
# ==========================================
@torch.no_grad()
def evaluate_and_log_videos(transformer, vae, text_encoder, tokenizer, writer, epoch, buffer, rollout_steps=16,
                            steps=20, fps=15, log_gt=False):
    """Autoregressive FPV rollout on a fixed cached clip per validation env (the middle valid clip, as
    data_rollout.ReplayEnv picks), each step guided by its cached guide. Logs PSNR (prediction and, as a
    reference, the guide alone vs GT FPV) and the videos: predicted FPV, the guide, and guide | predicted | GT."""
    transformer.eval()
    to = lambda a: torch.from_numpy(np.asarray(a)).float().to(device)[None, None] / 127.5 - 1.0   # (3, H, W) uint8 -> (1, 1, 3, H, W)
    frames = lambda a: torch.from_numpy(np.asarray(a)).float() / 255.0
    video = lambda f: torch.stack(f, dim=0).unsqueeze(0)                                     # (1, T, 3, H, W)

    for env_name in VAL_ENVS:
        starts = buffer._get_valid_trajectory_starts(rollout_steps, env_filter=env_name)
        if not starts:
            print(f"  [skip] {env_name} not in the cached buffer")
            continue
        i0 = starts[len(starts) // 2]   # same clip every epoch
        curr_latents = encode_views_to_latents(vae, to(buffer.views_t[i0, 3]), out_dtype=transformer.dtype)
        gen, gen_vis, gt, guides = [], [], [], []

        for i in range(i0, i0 + rollout_steps):
            prompt = compose_player_prompts([buffer.actions[i, :buffer.action_lens[i]]], env_name=env_name,
                                            camera_pose=buffer.camera_states[i])
            text_inputs = tokenizer(prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
            prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)
            guide = encode_views_to_latents(vae, to(buffer.guides_next[i]), out_dtype=transformer.dtype)

            next_latents = predict_next_fpv(transformer, curr_latents, guide, prompt_embeds, steps=steps)
            decoded = decode_latents_to_views(vae, next_latents)   # (1, 1, 3, H, W) in [0, 1]
            # Re-encode the decoded frame to stay on the clean VAE manifold
            curr_latents = encode_views_to_latents(vae, decoded * 2.0 - 1.0, out_dtype=transformer.dtype)

            gen.append(decoded[0, 0].cpu())
            gen_vis.append(_annotate_frame_pose(gen[-1], buffer.camera_states[i]["command"]))
            gt.append(frames(buffer.views_next[i, 3]))
            guides.append(frames(buffer.guides_next[i]))

        tag = env_name.replace(" ", "_").lower()
        psnr_fpv, psnr_guide = _psnr(video(gen), video(gt)), _psnr(video(guides), video(gt))
        writer.add_scalar(f"Validation/{tag}/PSNR/FPV", psnr_fpv, epoch)
        writer.add_scalar(f"Validation/{tag}/PSNR/Guide", psnr_guide, epoch)
        writer.add_video(f"Test/{tag}/FPV", video(gen_vis), epoch, fps=fps)
        writer.add_video(f"Test/{tag}/Guide", video(guides), epoch, fps=fps)
        writer.add_video(f"Test/{tag}/Guide_Pred_GT", torch.cat([video(guides), video(gen), video(gt)], dim=-1), epoch, fps=fps)
        if log_gt:
            writer.add_video(f"GT/{tag}/FPV", video(gt), epoch, fps=fps)
        print(f"  [Epoch {epoch}] [{env_name}] FPV PSNR {psnr_fpv:.2f} dB (guide alone {psnr_guide:.2f} dB)")

    transformer.train()


# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
def main(args):
    # Long timeout: non-zero ranks wait at a barrier while rank 0 runs validation rollouts
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Different seed per rank so each rank collects different rollouts and samples different batches
    set_seed(args.seed + rank)
    if is_main():
        print("=" * 60)
        print("  FPV-Single + Ortho Memory Guide (LoRA) [DDP]")
        print(f"  Multi-Step Trajectory Training (seq_len={args.train_seq_len}, teacher_forcing_prob={args.teacher_forcing_prob})")
        print(f"  World size: {world_size} | Per-GPU batch: {args.batch_size} | Effective batch: {args.batch_size * world_size}")
        print("=" * 60)

    # 1. Environments & Replay Buffer (one buffer per rank)
    registry, _ = get_registries()
    buffer = GuidedBuffer(
        capacity=args.rollout_steps * min(args.num_envs, len(registry)), img_h=IMG_H, img_w=IMG_W, max_action_len=8
    )

    # Rollout cache: the buffer depends on the RNG seed (camera-pose picks), the env seed, rank, envs, steps and image size
    cache_dir = (f"./buffer_cache/s{args.seed}_c{args.seed + rank}_r{rank}"
                 f"_e{args.num_envs}_n{args.rollout_steps}_{IMG_H}px_guide")

    if os.path.exists(os.path.join(cache_dir, "meta.pkl")):
        buffer.load(cache_dir)
        if is_main():
            print(f"[INFO] Loaded cached rollouts from {cache_dir} ({buffer.size} transitions)")
    else:
        collect_all_envs_rollouts(registry, buffer, rollout_steps=args.rollout_steps, num_envs=args.num_envs,
                                  seed=args.seed + rank, device=device)
        buffer.save(cache_dir)
    set_seed(args.seed + rank)   # same RNG state from here on, whether collected or loaded

    # 2. Model & LoRA Setup
    model, vae, text_encoder, tokenizer = load_model(
        model_id=args.model_id, lora_rank=args.lora_rank
    )

    # DDP broadcasts rank 0's weights (incl. the randomly initialised LoRA) to all ranks at construction
    ddp_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    writer = None
    if is_main():
        log_dir = f"./runs/wan_single_dist_memory_{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    ckpt_base = f"./checkpoints/wan_single_dist_memory_{args.num_envs}"
    view_names = ["3D_FPV"]

    # 3. Multi-Step Trajectory Training Loop
    if is_main():
        print("\nStarting Multi-Step Trajectory LoRA Fine-Tuning ...")
    global_step = 0

    for epoch in range(args.epochs):
        ddp_model.train()
        epoch_loss = 0.0
        # Ramp up autoregressive student-forcing probability across epochs
        progress = epoch / max(1, args.epochs - 1)
        curr_teacher_forcing_prob = args.teacher_forcing_prob * min(1.0, progress * 1.5)

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} (Self-pred Prob.={curr_teacher_forcing_prob:.2f})", disable=not is_main())

        for step_i in pbar:
            # Sample batch of contiguous trajectories of length `args.train_seq_len`
            # views_t:        (B, 4, 3, H, W) — anchor frame at t=0
            # views_next_seq: list[T] of (B, 4, 3, H, W) — target frames t=1..T
            # guides_seq:     list[T] of (B, 3, H, W) — the guide of each target frame
            views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq, guides_seq = buffer.sample_trajectory_batch(
                batch_size=args.batch_size,
                seq_len=args.train_seq_len,
            )
            # Take only 3D FPV
            views_t = views_t[:,-1 : :]
            views_next_seq = [v[:,-1 : :] for v in views_next_seq]

            with torch.no_grad():
                latents_t = encode_views_to_latents(vae, views_t, out_dtype=model.dtype)

                # Encode target views, their guides and text prompts for each step along the trajectory
                latents_next_seq = []
                guide_latents_seq = []
                prompt_embeds_seq = []

                for s in range(args.train_seq_len):
                    latents_next_seq.append(encode_views_to_latents(vae, views_next_seq[s], out_dtype=model.dtype))
                    guide_latents_seq.append(encode_views_to_latents(vae, guides_seq[s][:, None], out_dtype=model.dtype))

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

            # The guides this epoch's first batch trains with, next to their FPV targets (first trajectory step)
            if is_main() and step_i == 0:
                writer.add_images("Train/Guide_vs_FPV",
                                  torch.cat([guides_seq[0][:4], views_next_seq[0][:4, 0]], dim=-1) * 0.5 + 0.5, global_step)

            optimizer.zero_grad(set_to_none=True)

            # Unroll multi-step trajectory with student forcing & context noise
            loss, per_view_losses = train_guided_trajectory_step(
                transformer=ddp_model,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                guide_seq=guide_latents_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                teacher_forcing_prob=curr_teacher_forcing_prob,
                ctx_noise_max=args.ctx_noise_max,
                sf_steps=16,
            )

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Average logged losses across ranks
            loss_stats = torch.tensor([loss] + list(per_view_losses), device=device, dtype=torch.float32)
            dist.all_reduce(loss_stats, op=dist.ReduceOp.AVG)
            loss, per_view_losses = loss_stats[0].item(), loss_stats[1:].tolist()

            epoch_loss += loss
            global_step += 1

            if is_main():
                writer.add_scalar("Train/Loss", loss, global_step)
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("Train/Teacher_Student_Prob", curr_teacher_forcing_prob, global_step)
                for v_idx, v_name in enumerate(view_names):
                    writer.add_scalar(f"Train/Loss_{v_name}", per_view_losses[v_idx], global_step)

                env_summary = "+".join(list(dict.fromkeys(env_names_seq[0]))[:2])
                pbar.set_postfix({
                    "Loss": f"{loss:.4f}",
                    "FPV": f"{per_view_losses[0]:.3f}",
                    "Envs": env_summary,
                })

        avg_loss = epoch_loss / args.steps_per_epoch
        if is_main():
            print(f"  [Epoch {epoch+1}] Average Trajectory Flow Loss ({args.train_seq_len} steps): {avg_loss:.4f}")

        # --- VALIDATION: GUIDED AUTOREGRESSIVE FPV ROLLOUTS TO TENSORBOARD (rank 0 only, unwrapped model) ---
        if (epoch + 1) % args.eval_every == 0 and is_main():
            evaluate_and_log_videos(
                transformer=model,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                writer=writer,
                epoch=epoch + 1,
                buffer=buffer,
                rollout_steps=args.rollout_video_steps,
                steps=args.eval_steps,
                fps=args.video_fps,
                log_gt=args.log_gt,
            )

        # Checkpointing
        if (epoch + 1) % args.save_every == 0 and is_main():
            ckpt_dir = f"{ckpt_base}_lora_epoch_{epoch+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"  [SAVED] LoRA checkpoint to {ckpt_dir}")

        # Keep ranks in step while rank 0 validates / saves
        dist.barrier()

    if is_main():
        print("\nTraining Complete! Saving final LoRA weights...")
        final_dir = f"{ckpt_base}_lora_final"
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        writer.close()

    dist.barrier()
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA on FPV-Single + the ortho memory guide (DDP)")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout steps per environment, collected independently on each rank")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size (effective batch = batch_size * world_size)")
    parser.add_argument("--train_seq_len", type=int, default=16, help="Contiguous trajectory steps unrolled per training update")
    parser.add_argument("--teacher_forcing_prob", type=float, default=1.0, help="Max prob of using self-predicted latents onto next trajectory step")
    parser.add_argument("--ctx_noise_max", type=float, default=0.15, help="Max Gaussian noise added to conditioning context latents to prevent drift")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Generate and evaluate views every N epochs")
    parser.add_argument("--eval_steps", type=int, default=8, help="Number of Euler flow matching sampling steps per frame")
    parser.add_argument("--rollout_video_steps", type=int, default=16, help="Autoregressive rollout steps for validation videos")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for TensorBoard video logging")
    parser.add_argument("--log_gt", action="store_true", help="Also log the ground-truth validation videos (identical every epoch)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_envs", type=int, default=3, help="Number of training envs (guide geometry exists for the first 3)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
