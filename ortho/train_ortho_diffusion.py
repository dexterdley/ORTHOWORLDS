# Standard library imports
import math
import random
import os
import time
import argparse

# Third-party imports
import cv2
import gnwrapper
import gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame import gfxdraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as ckpt_fn
from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import lpips
from torchvision.models.video import r3d_18, R3D_18_Weights
from scipy import linalg

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# EVALUATION METRICS
# ==========================================

# Lazy-initialised LPIPS network (VGG backbone, loaded once on first call)
_lpips_fn = None

def _get_lpips_fn():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    return _lpips_fn


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR between two batches of images in [-1, 1].
    
    Args:
        pred: (B, 3, H, W) predicted frames in [-1, 1]
        gt:   (B, 3, H, W) ground-truth frames in [-1, 1]
    Returns:
        Mean PSNR (dB) over the batch.
    """
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float('inf')
    # Pixel range is [-1, 1] → max_val = 2.0  → 20*log10(2 / sqrt(mse))
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(mse)


@torch.no_grad()
def compute_lpips(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean LPIPS (VGG) between predicted and ground-truth frames.
    
    Args:
        pred: (B, 3, H, W) in [-1, 1]
        gt:   (B, 3, H, W) in [-1, 1]
    Returns:
        Mean LPIPS score (lower = more perceptually similar).
    """
    fn = _get_lpips_fn()
    return fn(pred, gt).mean().item()


# Lazy-initialised R3D-18 feature extractor for FVD
_r3d_model = None

def _get_r3d_model():
    """Return a pretrained R3D-18 with the final FC layer removed, used as a
    video feature extractor (I3D proxy) for FVD computation."""
    global _r3d_model
    if _r3d_model is None:
        weights = R3D_18_Weights.DEFAULT
        m = r3d_18(weights=weights)
        # Strip classifier head — keep everything up to the pooling layer
        m.fc = nn.Identity()
        _r3d_model = m.to(device).eval()
    return _r3d_model


@torch.no_grad()
def _extract_video_features(videos: torch.Tensor) -> np.ndarray:
    """Extract R3D-18 features from a batch of videos.
    
    Args:
        videos: (N, T, 3, H, W) float32 in [0, 1], H/W >= 112 recommended.
                If H/W < 112, frames are upsampled automatically.
    Returns:
        features: (N, 512) numpy array.
    """
    model = _get_r3d_model()
    N, T, C, H, W = videos.shape
    
    # R3D expects (N, C, T, H, W) and input in approx. ImageNet normalisation
    # We normalise [0,1] → ImageNet mean/std
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.22803, 0.22145,  0.216989], device=device).view(1, 3, 1, 1, 1)
    
    v = videos.to(device).float()          # (N, T, C, H, W)
    v = v.permute(0, 2, 1, 3, 4)          # (N, C, T, H, W)
    
    # Upsample spatial dims to at least 112 x 112 if needed
    if H < 112 or W < 112:
        v = F.interpolate(v.view(N * C, T, H, W).unsqueeze(1),
                          size=(T, 112, 112), mode='trilinear',
                          align_corners=False).squeeze(1).view(N, C, T, 112, 112)
    
    v = (v - mean) / std
    
    feats = model(v)                       # (N, 512)
    return feats.cpu().numpy()


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    # Product of covariances — use scipy for numerically stable sqrt
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


@torch.no_grad()
def compute_fvd(real_videos: torch.Tensor, fake_videos: torch.Tensor) -> float:
    """Fréchet Video Distance between real and generated video batches.

    Uses a pretrained R3D-18 as the video feature backbone (I3D proxy).
    The more clips you provide, the more reliable the estimate.

    Args:
        real_videos: (N, T, 3, H, W) float32 in [0, 1]
        fake_videos: (N, T, 3, H, W) float32 in [0, 1]
    Returns:
        FVD scalar (lower = more realistic videos).
    """
    real_feats = _extract_video_features(real_videos)   # (N, 512)
    fake_feats = _extract_video_features(fake_videos)   # (N, 512)

    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    # Guard against rank-deficient covariance when N is small
    eps = 1e-6
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_f += np.eye(sigma_f.shape[0]) * eps

    return _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


def set_seed(seed):
    """Set random seed for reproducibility across random, numpy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _build_registry():
    """Return a list of (display_name, factory_fn) tuples.
    Each factory_fn must return a fully-wrapped env that exposes:
        env.reset()  -> (obs, info)
        env.step(a)  -> (obs, reward, term, trunc, info)
        env.render() -> (im_top_down, im_rear, im_side, im_fpv)  [H x W x 3 uint8]
        env.action_space.sample()
        env.close()
    """
    registry = []

    def _try(name, factory):
        registry.append((name, factory))
    # ------------------------------------------------------------------
    # Bipedal Walker (wraps gymnasium's BipedalWalker-v3)
    try:
        import gymnasium
        from game_envs.bipedal_orthographic import BipedalOrthographicWrapper
        _try("Bipedal Walker",
             lambda: BipedalOrthographicWrapper(
                 gymnasium.make("BipedalWalker-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register Bipedal Walker: {e}")

    # ------------------------------------------------------------------
    # Lunar Lander (wraps gymnasium's LunarLander-v3)
    try:
        import gymnasium
        from game_envs.lunar_orthographic import LunarOrthographicWrapper
        _try("Lunar Lander",
             lambda: LunarOrthographicWrapper(
                 gymnasium.make("LunarLander-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register Lunar Lander: {e}")
        
    # ------------------------------------------------------------------
    # Mario Escape
    try:
        from game_envs.mario_orthographic import (
            MarioEscapeEnv, MarioOrthographicWrapper)
        _try("Mario Escape",
             lambda: MarioOrthographicWrapper(MarioEscapeEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Mario Escape: {e}")

    # ------------------------------------------------------------------
    # MultiCar Racing (wraps gymnasium's CarRacing-v3)
    try:
        import gymnasium
        from game_envs.multicar_racing_orthographic import MultiCarOrthographicWrapper
        _try("MultiCar Racing",
             lambda: MultiCarOrthographicWrapper(
                 gymnasium.make("CarRacing-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register MultiCar Racing: {e}")

    # ------------------------------------------------------------------
    # Drone Dogfight
    try:
        from game_envs.drone_dogfight_orthographic import (
            DroneDogfightEnv, DroneDogfightOrthographicWrapper)
        _try("Drone Dogfight",
             lambda: DroneDogfightOrthographicWrapper(DroneDogfightEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Drone Dogfight: {e}")

    # ------------------------------------------------------------------
    # Excavator
    try:
        from game_envs.excavator_orthographic import (
            ExcavatorEnv, ExcavatorOrthographicWrapper)
        _try("Excavator",
             lambda: ExcavatorOrthographicWrapper(ExcavatorEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Excavator: {e}")
        
    return registry

class TransitionSequenceBuffer:
    def __init__(self, capacity, img_h, img_w, action_len):
        self.capacity = capacity
        self.img_h = img_h
        self.img_w = img_w
        self.ptr = 0
        self.size = 0
        
        # --- Images ---
        self.im_top_down_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_rear_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_side_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_fpv_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        
        self.im_top_down_next_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_rear_next_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_side_next_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_fpv_next_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)

        # --- Actions ---
        self.action_t = np.zeros((capacity, action_len), dtype=np.float32)
        
        # --- Environment ---
        self.done_t = np.zeros(capacity, dtype=bool)

    def push(self,
            im_top_down, # Top-Down
            im_rear, # Rear View
            im_side, # Side View
            im_fpv, # First Person View
            next_im_top_down, # Top-Down
            next_im_rear, # Rear View
            next_im_side, # Side View
            next_im_fpv, # First Person View
            action,
            done):
        
        # Resize if image doesn't match buffer resolution
        h, w = im_top_down.shape[:2]
        if h != self.img_h or w != self.img_w:
            im_top_down = cv2.resize(im_top_down, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            im_rear = cv2.resize(im_rear, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            im_side = cv2.resize(im_side, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            im_fpv = cv2.resize(im_fpv, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            next_im_top_down = cv2.resize(next_im_top_down, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            next_im_rear = cv2.resize(next_im_rear, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            next_im_side = cv2.resize(next_im_side, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            next_im_fpv = cv2.resize(next_im_fpv, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        
        # Images
        self.im_top_down_t[self.ptr] = np.transpose(im_top_down, (2, 0, 1))
        self.im_rear_t[self.ptr] = np.transpose(im_rear, (2, 0, 1))
        self.im_side_t[self.ptr] = np.transpose(im_side, (2, 0, 1))
        self.im_fpv_t[self.ptr] = np.transpose(im_fpv, (2, 0, 1))
        
        self.im_top_down_next_t[self.ptr] = np.transpose(next_im_top_down, (2, 0, 1))
        self.im_rear_next_t[self.ptr] = np.transpose(next_im_rear, (2, 0, 1))
        self.im_side_next_t[self.ptr] = np.transpose(next_im_side, (2, 0, 1))
        self.im_fpv_next_t[self.ptr] = np.transpose(next_im_fpv, (2, 0, 1))
        
        # Actions
        self.action_t[self.ptr] = action

        # Environment
        self.done_t[self.ptr] = done
        
        # Advance pointers
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample_sequence(self, batch_size, seq_len, device="cuda"):
        """Samples a contiguous sequence of length seq_len."""
        idxs = np.zeros(batch_size, dtype=np.int32)
        
        # Ensure sequences don't cross episode boundaries OR the ring buffer pointer
        for i in range(batch_size):
            valid = False
            while not valid:
                idx = np.random.randint(0, self.size - seq_len)
                
                # 1. Prevent crossing the ring buffer wrap-around point
                if self.size == self.capacity and idx < self.ptr <= idx + seq_len:
                    continue
                    
                # 2. Prevent crossing episode boundaries (dones)
                if self.done_t[idx : idx + seq_len - 1].any():
                    continue
                    
                valid = True
                idxs[i] = idx
                
        # Slice sequences
        im_top_down_seq = np.stack([self.im_top_down_t[idx : idx + seq_len] for idx in idxs])
        im_rear_seq = np.stack([self.im_rear_t[idx : idx + seq_len] for idx in idxs])
        im_side_seq = np.stack([self.im_side_t[idx : idx + seq_len] for idx in idxs])
        im_fpv_seq = np.stack([self.im_fpv_t[idx : idx + seq_len] for idx in idxs])
        
        im_top_down_next_seq = np.stack([self.im_top_down_next_t[idx + seq_len - 1] for idx in idxs])
        im_rear_next_seq = np.stack([self.im_rear_next_t[idx + seq_len - 1] for idx in idxs])
        im_side_next_seq = np.stack([self.im_side_next_t[idx + seq_len - 1] for idx in idxs])
        im_fpv_next_seq = np.stack([self.im_fpv_next_t[idx + seq_len - 1] for idx in idxs])
        
        a_seq = np.stack([self.action_t[idx : idx + seq_len] for idx in idxs])

        # Convert to Tensors 
        im_top_down_seq_t = (torch.from_numpy(im_top_down_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        im_rear_seq_t = (torch.from_numpy(im_rear_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        im_side_seq_t = (torch.from_numpy(im_side_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        im_fpv_seq_t = (torch.from_numpy(im_fpv_seq).to(device, non_blocking=True).float() / 127.5) - 1.0

        im_top_down_next_seq_t = (torch.from_numpy(im_top_down_next_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        im_rear_next_seq_t = (torch.from_numpy(im_rear_next_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        im_side_next_seq_t = (torch.from_numpy(im_side_next_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        im_fpv_next_seq_t = (torch.from_numpy(im_fpv_next_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        
        a_seq_t = torch.from_numpy(a_seq).to(device, non_blocking=True)
        
        return (im_top_down_seq_t, im_rear_seq_t, im_side_seq_t, im_fpv_seq_t,
                im_top_down_next_seq_t, im_rear_next_seq_t, im_side_next_seq_t, im_fpv_next_seq_t,
                a_seq_t)

def collect_rollouts(env, agent, buffer, n_steps, seed):
    obs, _ = env.reset(seed=seed)
    state = torch.tensor(obs, dtype=torch.float32, device=device)

    for _ in tqdm(range(n_steps)):

        img_top_down, im_rear, im_side, im_fpv = env.render()
        # action = env.action_space.sample()
        with torch.inference_mode():
            features = agent.feature_net(state)
            if agent.is_discrete:
                logits = agent.actor_head(features)
                action = torch.argmax(logits, dim=-1).item()
            else:
                mean = torch.tanh(agent.actor_mean(features))
                action = mean.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_img_top_down, next_im_rear, next_im_side, next_im_fpv = env.render()
                
        # --- PUSH TO BUFFER ---
        buffer.push(
            img_top_down,
            im_rear,
            im_side,
            im_fpv,
            next_img_top_down,
            next_im_rear,
            next_im_side,
            next_im_fpv,
            action,
            done
        )
        
        obs = next_obs
        
        if done:
            obs, _ = env.reset()

    env.close()

# ==========================================
# 2. COSINE NOISE SCHEDULE
# ==========================================
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in Nichol & Dhariwal 2021."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space, hidden_dim=256):
        super().__init__()
        self.is_discrete = hasattr(action_space, 'n')
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Value head (Critic)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Policy head (Actor)
        if self.is_discrete:
            self.action_dim = action_space.n
            self.actor_head = nn.Linear(hidden_dim, self.action_dim)
        else:
            self.action_dim = action_space.shape[0]
            self.actor_mean = nn.Linear(hidden_dim, self.action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, state):
        features = self.feature_net(state)
        value = self.value_head(features)
        
        if self.is_discrete:
            logits = self.actor_head(features)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = torch.tanh(self.actor_mean(features))
            std = torch.exp(self.actor_log_std.expand_as(mean))
            dist = torch.distributions.Normal(mean, std)
            
        return dist, value

    def get_value(self, state):
        features = self.feature_net(state)
        return self.value_head(features)

def load_agent(ckpt_path, state_dim, action_space, device):
    """Instantiate ActorCritic and load state dict from checkpoint."""
    ac = ActorCritic(state_dim, action_space).to(device)
    if os.path.exists(ckpt_path):
        ac.load_state_dict(torch.load(ckpt_path, map_location=device))
        print((f"  [LOADED] PPO Agent weights from {ckpt_path}"))
    else:
        print((f"  [WARN] PPO Agent Checkpoint '{ckpt_path}' not found!"))
    ac.eval()
    return ac

# ==========================================
# 3. DIFFUSION U-NET WITH ATTENTION & GRU WORLD MODEL
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    """Self-attention layer for spatial feature maps."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = self.norm(x).view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        return x + attn_out.permute(0, 2, 1).view(b, c, h, w)

class ConditionalUNet(nn.Module):
    def __init__(self, cond_dim=256, base_ch=32, in_channels=6, out_channels=3):
        super().__init__()
        ch1, ch2, ch3, ch4, ch5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        self.noise_aug_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, 256)
        )

        self.view_emb = nn.Embedding(4, 256)  # 0: Top, 1: Rear, 2: Side, 3: FPV
        
        # UPDATED: Use in_channels (12)
        self.inc = ResBlock(in_channels, ch1, emb_dim=256)
        
        self.down1 = nn.MaxPool2d(2)
        self.res_down1 = ResBlock(ch1, ch2, emb_dim=256)
        self.down2 = nn.MaxPool2d(2)
        self.res_down2 = ResBlock(ch2, ch3, emb_dim=256)
        self.down3 = nn.MaxPool2d(2)
        self.res_down3 = ResBlock(ch3, ch4, emb_dim=256)
        self.attn_down3 = SelfAttention(ch4)
        self.down4 = nn.MaxPool2d(2)
        self.res_down4 = ResBlock(ch4, ch5, emb_dim=256)
        
        self.bot1 = ResBlock(ch5, ch5, emb_dim=256)
        self.bot2 = ResBlock(ch5, ch5, emb_dim=256)
        
        self.res_up1 = ResBlock(ch5 + ch4, ch4, emb_dim=256)
        self.attn_up1 = SelfAttention(ch4)
        self.res_up2 = ResBlock(ch4 + ch3, ch3, emb_dim=256)
        self.res_up3 = ResBlock(ch3 + ch2, ch2, emb_dim=256)
        self.res_up4 = ResBlock(ch2 + ch1, ch1, emb_dim=256)
        
        # UPDATED: Use out_channels (6)
        self.outc = nn.Conv2d(ch1, out_channels, kernel_size=3, padding=1)

    def _encoder_block(self, x, emb):
        x1 = self.inc(x, emb)
        x2 = self.res_down1(self.down1(x1), emb)
        x3 = self.res_down2(self.down2(x2), emb)
        x4 = self.attn_down3(self.res_down3(self.down3(x3), emb))
        x5 = self.res_down4(self.down4(x4), emb)
        return x1, x2, x3, x4, x5
    
    def _decoder_block(self, x5, x4, x3, x2, x1, emb):
        x5 = self.bot1(x5, emb)
        x5 = self.bot2(x5, emb)
        up5 = F.interpolate(x5, size=x4.shape[2:], mode='nearest')
        x = self.attn_up1(self.res_up1(torch.cat([up5, x4], dim=1), emb))
        up4 = F.interpolate(x, size=x3.shape[2:], mode='nearest')
        x = self.res_up2(torch.cat([up4, x3], dim=1), emb)
        up3 = F.interpolate(x, size=x2.shape[2:], mode='nearest')
        x = self.res_up3(torch.cat([up3, x2], dim=1), emb)
        up2 = F.interpolate(x, size=x1.shape[2:], mode='nearest')
        x = self.res_up4(torch.cat([up2, x1], dim=1), emb)
        return self.outc(x)

    def forward(self, x_noisy, t, x_cond, h_t, x_anchor=None, noise_aug_level=None):
        emb = self.time_mlp(t) + self.cond_mlp(h_t)
        
        if noise_aug_level is not None:
            emb = emb + self.noise_aug_mlp(noise_aug_level.float())
        
        # x_anchor: the already-denoised/noisy previous view in the chain.
        # Zeros for the first (anchor) view; noisy sibling for all subsequent views.
        if x_anchor is None:
            x_anchor = torch.zeros_like(x_noisy)
        
        x = torch.cat([x_noisy, x_cond, x_anchor], dim=1)  # (B, 9, H, W)
        
        if self.training:
            x1, x2, x3, x4, x5 = ckpt_fn(self._encoder_block, x, emb, use_reentrant=False)
            return ckpt_fn(self._decoder_block, x5, x4, x3, x2, x1, emb, use_reentrant=False)
        else:
            x1, x2, x3, x4, x5 = self._encoder_block(x, emb)
            return self._decoder_block(x5, x4, x3, x2, x1, emb)

class RecurrentDiffusionWorldModel(nn.Module):
    def __init__(self, timesteps=200, hidden_dim=256, action_dim=4, noise_aug_max=20, cond_drop_prob=0.1, base_ch=32, batch_size=64, seq_len=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_aug_max = noise_aug_max
        self.cond_drop_prob = cond_drop_prob
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.unet = ConditionalUNet(cond_dim=hidden_dim, base_ch=base_ch, in_channels=9, out_channels=3)

        # Keep the vision encoder at 12 channels for the RNN state
        enc_dim = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, enc_dim, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        
        gru_input_dim = enc_dim + action_dim
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)
        

        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def forward(self, im_top_down_seq, im_rear_seq, im_side_seq, im_fpv_seq, im_top_down_next_seq, im_rear_next_seq, im_side_next_seq, im_fpv_next_seq, a_seq):

        h = torch.zeros(self.batch_size, self.hidden_dim, device=device)        
        combined_img_seq = torch.cat([im_top_down_seq, im_rear_seq, im_side_seq, im_fpv_seq], dim=2)
        
        for t in range(self.seq_len):
            z_t = self.encoder(combined_img_seq[:, t])
            vec_in = torch.cat([z_t, a_seq[:, t]], dim=1)
            h = self.gru(vec_in, h)
        
        t_diff = torch.randint(0, self.timesteps, (self.batch_size, ), device=device).long()
            
        # 1. Add noise to all views independently
        noise_top_down = torch.randn_like(im_top_down_next_seq)
        noise_rear = torch.randn_like(im_rear_next_seq)
        noise_side = torch.randn_like(im_side_next_seq)
        noise_fpv = torch.randn_like(im_fpv_next_seq)
            
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
            
        noisy_top_down = sqrt_alpha_t * im_top_down_next_seq + sqrt_one_minus_alpha_t * noise_top_down
        noisy_rear = sqrt_alpha_t * im_rear_next_seq + sqrt_one_minus_alpha_t * noise_rear
        noisy_side = sqrt_alpha_t * im_side_next_seq + sqrt_one_minus_alpha_t * noise_side
        noisy_fpv = sqrt_alpha_t * im_fpv_next_seq + sqrt_one_minus_alpha_t * noise_fpv
            
        # 2. Process conditions and noise augmentations
        cond_top_down = im_top_down_seq[:, -1]
        cond_rear = im_rear_seq[:, -1]
        cond_side = im_side_seq[:, -1]
        cond_fpv = im_fpv_seq[:, -1]
        noise_aug_level = torch.zeros(self.batch_size, device=device, dtype=torch.long)
        h_unet = h.clone()
            
        if self.training:
            noise_aug_level = torch.randint(0, self.noise_aug_max, (self.batch_size, ), device=device).long()

            # Apply augmentation independently
            aug_n_top_down = torch.randn_like(cond_top_down)
            aug_n_rear = torch.randn_like(cond_rear)
            aug_n_side = torch.randn_like(cond_side)
            aug_n_fpv = torch.randn_like(cond_fpv)
            
            sq_a_aug = self.sqrt_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
            sq_one_m_aug = self.sqrt_one_minus_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
                
            cond_top_down = sq_a_aug * cond_top_down + sq_one_m_aug * aug_n_top_down
            cond_rear = sq_a_aug * cond_rear + sq_one_m_aug * aug_n_rear
            cond_side = sq_a_aug * cond_side + sq_one_m_aug * aug_n_side
            cond_fpv = sq_a_aug * cond_fpv + sq_one_m_aug * aug_n_fpv
                
            drop_mask = torch.rand(self.batch_size, device=device) < self.cond_drop_prob
            if drop_mask.any():
                h_unet[drop_mask] = 0.0
                cond_top_down[drop_mask] = 0.0
                cond_rear[drop_mask] = 0.0
                cond_side[drop_mask] = 0.0
                cond_fpv[drop_mask] = 0.0
            
            # ----------------------------------------------------------------
            # 3. CROSS-VIEW SEQUENTIAL FORWARD PASS
            # Anchor chain: top_down → side → rear → FPV
            # Each view receives the *noisy* predecessor (same t) as x_anchor,
            # teaching the model to be robust to the anchor's noise level.
            # ----------------------------------------------------------------

            # Pass 1 — Top-down: no cross-view anchor (first in chain)
            pred_noise_top_down = self.unet(
                noisy_top_down, t_diff, cond_top_down, h_unet,
                x_anchor=None,
                noise_aug_level=noise_aug_level
            )

            # Pass 2 — Side: anchored on noisy top-down
            pred_noise_side = self.unet(
                noisy_side, t_diff, cond_side, h_unet,
                x_anchor=noisy_top_down,
                noise_aug_level=noise_aug_level
            )

            # Pass 3 — Rear: anchored on noisy side
            pred_noise_rear = self.unet(
                noisy_rear, t_diff, cond_rear, h_unet,
                x_anchor=noisy_side,
                noise_aug_level=noise_aug_level
            )

            # Pass 4 — FPV: anchored on noisy rear
            pred_noise_fpv = self.unet(
                noisy_fpv, t_diff, cond_fpv, h_unet,
                x_anchor=noisy_rear,
                noise_aug_level=noise_aug_level
            )
            
            return pred_noise_top_down, pred_noise_rear, pred_noise_side, pred_noise_fpv, noise_top_down, noise_rear, noise_side, noise_fpv
        
        return pred_noise_top_down, pred_noise_rear, pred_noise_side, pred_noise_fpv

    @torch.no_grad()
    def sample(self, im_top_down_seq, im_rear_seq, 
                     im_side_seq, im_fpv_seq,
                     a_seq
                     ):
        """Runs the diffusion sampling loop with cross-view conditioning.
        
        Anchor chain per denoising step: top_down → side → rear → FPV.
        After each view is denoised at step i, its result is passed as
        x_anchor to the next view in the chain, propagating geometric
        context without requiring all views to start from random noise.
        """
        self.eval() 
        batch_size, seq_len, _, h_img, w_img = im_top_down_seq.shape

        h = torch.zeros(batch_size, self.hidden_dim, device=device)        
        combined_img_seq = torch.cat([im_top_down_seq, im_rear_seq, im_side_seq, im_fpv_seq], dim=2)
        
        for t in range(seq_len):
            z_t = self.encoder(combined_img_seq[:, t])
            vec_in = torch.cat([z_t, a_seq[:, t]], dim=1)
            h = self.gru(vec_in, h)
        
        # All views start from independent Gaussian noise.
        # top_down is the true anchor — it has no cross-view input.
        # side/rear/FPV will be warm-guided each step by their predecessor.
        x_t_top_down = torch.randn((batch_size, 3, h_img, w_img), device=device)
        x_t_side     = torch.randn((batch_size, 3, h_img, w_img), device=device)
        x_t_rear     = torch.randn((batch_size, 3, h_img, w_img), device=device)
        x_t_fpv      = torch.randn((batch_size, 3, h_img, w_img), device=device)

        # Previous-frame conditions (fixed across all denoising steps)
        cond_top_down = im_top_down_seq[:, -1]
        cond_side     = im_side_seq[:, -1]
        cond_rear     = im_rear_seq[:, -1]
        cond_fpv      = im_fpv_seq[:, -1]
        
        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            alpha_t       = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t        = self.betas[i]
            stochastic    = i > 0  # no noise on the final step

            # ---- helper: one DDPM reverse step ----
            def _ddpm_step(x_t, pred_noise):
                mean = (1 / torch.sqrt(alpha_t)) * (
                    x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
                )
                if stochastic:
                    return mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
                return mean

            # ---- Pass 1: Top-down — no cross-view anchor ----
            pred_noise_top_down = self.unet(
                x_noisy=x_t_top_down, t=t_tensor,
                x_cond=cond_top_down, h_t=h,
                x_anchor=None
            )
            x_t_top_down = _ddpm_step(x_t_top_down, pred_noise_top_down)

            # ---- Pass 2: Side — anchored on just-denoised top-down ----
            pred_noise_side = self.unet(
                x_noisy=x_t_side, t=t_tensor,
                x_cond=cond_side, h_t=h,
                x_anchor=x_t_top_down
            )
            x_t_side = _ddpm_step(x_t_side, pred_noise_side)

            # ---- Pass 3: Rear — anchored on just-denoised side ----
            pred_noise_rear = self.unet(
                x_noisy=x_t_rear, t=t_tensor,
                x_cond=cond_rear, h_t=h,
                x_anchor=x_t_side
            )
            x_t_rear = _ddpm_step(x_t_rear, pred_noise_rear)

            # ---- Pass 4: FPV — anchored on just-denoised rear ----
            pred_noise_fpv = self.unet(
                x_noisy=x_t_fpv, t=t_tensor,
                x_cond=cond_fpv, h_t=h,
                x_anchor=x_t_rear
            )
            x_t_fpv = _ddpm_step(x_t_fpv, pred_noise_fpv)
            
        pred_top_down = torch.clamp(x_t_top_down, -1.0, 1.0)
        pred_rear     = torch.clamp(x_t_rear,     -1.0, 1.0)
        pred_side     = torch.clamp(x_t_side,     -1.0, 1.0)
        pred_fpv      = torch.clamp(x_t_fpv,      -1.0, 1.0)
        
        self.train() 
        return pred_top_down, pred_rear, pred_side, pred_fpv

@torch.no_grad()
def autoregressive_rollout(model, buffer, device, rollout_steps=50, seq_len=8):
    """Autoregressive rollout. Returns generated frames and the corresponding
    ground-truth next frames (needed for FVD computation)."""
    model.eval()
    
    # 1. Grab a continuous chunk of data and actions
    (val_im_top_down_seq_t, val_im_rear_seq_t, val_im_side_seq_t, val_im_fpv_seq_t,
    val_im_top_down_next_seq_t, val_im_rear_next_seq_t, val_im_side_next_seq_t, val_im_fpv_next_seq_t,
    val_a_seq_t) = buffer.sample_sequence(batch_size=1, seq_len=rollout_steps + seq_len, device=device)

    # Initialize the Sliding Windows with the "Burn-in" sequence (first frames)
    curr_top_down_seq = val_im_top_down_seq_t[:, :seq_len].clone()
    curr_rear_seq = val_im_rear_seq_t[:, :seq_len].clone()
    curr_side_seq = val_im_side_seq_t[:, :seq_len].clone()
    curr_fpv_seq = val_im_fpv_seq_t[:, :seq_len].clone()

    # Storage for generated and ground-truth frames
    generated_top_down_imgs, generated_rear_imgs = [], []
    generated_side_imgs,     generated_fpv_imgs  = [], []
    gt_top_down_imgs, gt_rear_imgs = [], []
    gt_side_imgs,     gt_fpv_imgs  = [], []
    
    print(f"Starting {rollout_steps}-step Autoregressive Rollout...")

    # The Autoregressive Loop
    for step in range(rollout_steps):
        curr_action_seq = val_a_seq_t[:, step : step + seq_len]

        val_pred_top_down, val_pred_rear, val_pred_side, val_pred_fpv = model.sample(
            curr_top_down_seq, curr_rear_seq,
            curr_side_seq, curr_fpv_seq,
            curr_action_seq
        )

        # Store generated frames
        generated_top_down_imgs.append(val_pred_top_down)
        generated_rear_imgs.append(val_pred_rear)
        generated_side_imgs.append(val_pred_side)
        generated_fpv_imgs.append(val_pred_fpv)

        # Store the corresponding ground-truth next frame
        gt_top_down_imgs.append(val_im_top_down_next_seq_t[0:1])
        gt_rear_imgs.append(val_im_rear_next_seq_t[0:1])
        gt_side_imgs.append(val_im_side_next_seq_t[0:1])
        gt_fpv_imgs.append(val_im_fpv_next_seq_t[0:1])
    
        # SLIDING WINDOW UPDATE
        curr_top_down_seq = torch.cat([curr_top_down_seq[:, 1:], val_pred_top_down.unsqueeze(1)], dim=1)
        curr_rear_seq = torch.cat([curr_rear_seq[:, 1:], val_pred_rear.unsqueeze(1)], dim=1)
        curr_side_seq = torch.cat([curr_side_seq[:, 1:], val_pred_side.unsqueeze(1)], dim=1)
        curr_fpv_seq = torch.cat([curr_fpv_seq[:, 1:], val_pred_fpv.unsqueeze(1)], dim=1)

    return (generated_top_down_imgs, generated_rear_imgs, generated_side_imgs, generated_fpv_imgs,
            gt_top_down_imgs, gt_rear_imgs, gt_side_imgs, gt_fpv_imgs)

def main(args):
    set_seed(args.seed)
    SEQ_LEN = 32

    print(f"Using device: {device}")
    print(f"Hyperparameters: total_timesteps={args.total_timesteps}, rollout_steps={args.rollout_steps}, batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, seed={args.seed}")

    registry = _build_registry()
    if not registry:
        print("No environments could be loaded. Exiting.")
        return

    for idx, (env_name, make_env) in enumerate(registry):
        print(f"\n[{idx+1}/{len(registry)}] Initializing training for {env_name}...")
        print("  Loading env ...", end=" ", flush=True)
        env = make_env()
        obs, info = env.reset()

        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(args.seed)

        if hasattr(env.action_space, 'n'):
            action_len = env.action_space.n
        else:
            action_len = env.action_space.shape[0]

        experiment_name = "experiment_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"./runs/{experiment_name}/{env_name}")

        model = RecurrentDiffusionWorldModel(
            timesteps=200,  
            hidden_dim=128,
            noise_aug_max=20,
            cond_drop_prob=0.1,
            base_ch=32,
            batch_size=args.batch_size,
            seq_len=SEQ_LEN, 
            action_dim=action_len
        ).to(device)
        
        state_dim = int(np.prod(obs.shape if hasattr(obs, 'shape') else env.observation_space.shape))
        clean_name = "".join(c for c in env_name if c.isalnum() or c in ('_', '-')).lower()
        ckpt_path = os.path.join("checkpoints", f"ppo_{clean_name}.pt")
        agent = load_agent(ckpt_path, state_dim, env.action_space, device)

        buffer = TransitionSequenceBuffer(capacity=5000, img_h=64, img_w=64, action_len=action_len)
        collect_rollouts(env, agent, buffer, args.rollout_steps, args.seed)

        # --- TRAINING LOOP ---
        print(f"Starting Training on {env_name} using {device}...")

        losses = []
        top_down_losses = []
        rear_losses = []
        side_losses = []
        fpv_losses = []

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            model.train()
            epoch_loss_total, epoch_loss_img_top_down, epoch_loss_img_rear, epoch_loss_img_side, epoch_loss_img_fpv = 0.0, 0.0, 0.0, 0.0, 0.0
            
            pbar = tqdm(range(args.rollout_steps), desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for step in pbar:
                # 1. Sample Batch
                (im_top_down_seq_t, im_rear_seq_t, im_side_seq_t, im_fpv_seq_t,
                im_top_down_next_seq_t, im_rear_next_seq_t, im_side_next_seq_t, im_fpv_next_seq_t,
                a_seq_t) = buffer.sample_sequence(args.batch_size, SEQ_LEN, device)

                # 2. Forward Pass
                pred_noise_top_down, pred_noise_rear, pred_noise_side, pred_noise_fpv, noise_top_down, noise_rear, noise_side, noise_fpv = model(
                    im_top_down_seq_t, im_rear_seq_t, im_side_seq_t, im_fpv_seq_t,
                    im_top_down_next_seq_t, im_rear_next_seq_t, im_side_next_seq_t, im_fpv_next_seq_t,
                    a_seq_t
                )

                # 3. Compute Losses
                loss_top_down = F.mse_loss(
                    pred_noise_top_down, noise_top_down
                )
                loss_rear = F.mse_loss(
                    pred_noise_rear, noise_rear
                )
                loss_side = F.mse_loss(
                    pred_noise_side, noise_side
                )
                loss_fpv = F.mse_loss(
                    pred_noise_fpv, noise_fpv
                )
                loss = loss_top_down + loss_rear + loss_side + loss_fpv

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Log losses
                epoch_loss_total += loss.item()
                epoch_loss_img_top_down += loss_top_down.item()
                epoch_loss_img_rear += loss_rear.item()
                epoch_loss_img_side += loss_side.item()
                epoch_loss_img_fpv += loss_fpv.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss_total / args.rollout_steps
            avg_img_top_down = epoch_loss_img_top_down / args.rollout_steps
            avg_img_rear = epoch_loss_img_rear / args.rollout_steps
            avg_img_side = epoch_loss_img_side / args.rollout_steps
            avg_img_fpv = epoch_loss_img_fpv / args.rollout_steps

            losses.append(avg_loss)
            top_down_losses.append(avg_img_top_down)
            rear_losses.append(avg_img_rear)
            side_losses.append(avg_img_side)
            fpv_losses.append(avg_img_fpv)
            
            writer.add_scalar("Training/Total loss", avg_loss, epoch)
            writer.add_scalar("Training/Top Down loss", avg_img_top_down, epoch)
            writer.add_scalar("Training/Rear loss", avg_img_rear, epoch)
            writer.add_scalar("Training/Side loss", avg_img_side, epoch)
            writer.add_scalar("Training/FPV loss", avg_img_fpv, epoch)

            if (epoch + 1) % 10 == 0:
                os.makedirs("./checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"./checkpoints/{env_name}_world_model_epoch_{epoch+1}.pth")

            if epoch % 1 == 0: # Validation, only first image is real
                val_pred_top_down, val_pred_rear, val_pred_side, val_pred_fpv = model.sample(
                    im_top_down_seq_t[0:1], im_rear_seq_t[0:1], 
                    im_side_seq_t[0:1], im_fpv_seq_t[0:1], 
                    a_seq_t[0:1]
                )

                # --- Per-view MSE (kept for backwards compatibility) ---
                val_img_top_down = F.mse_loss(val_pred_top_down, im_top_down_next_seq_t[0:1])
                val_img_rear = F.mse_loss(val_pred_rear, im_rear_next_seq_t[0:1])
                val_img_side = F.mse_loss(val_pred_side, im_side_next_seq_t[0:1])
                val_img_fpv = F.mse_loss(val_pred_fpv, im_fpv_next_seq_t[0:1])
                
                writer.add_scalar("Validation/MSE/Top Down", val_img_top_down, epoch)
                writer.add_scalar("Validation/MSE/Rear",     val_img_rear,      epoch)
                writer.add_scalar("Validation/MSE/Side",     val_img_side,      epoch)
                writer.add_scalar("Validation/MSE/FPV",      val_img_fpv,       epoch)

                # --- Per-view PSNR ---
                psnr_top_down = compute_psnr(val_pred_top_down, im_top_down_next_seq_t[0:1])
                psnr_rear     = compute_psnr(val_pred_rear,     im_rear_next_seq_t[0:1])
                psnr_side     = compute_psnr(val_pred_side,     im_side_next_seq_t[0:1])
                psnr_fpv      = compute_psnr(val_pred_fpv,      im_fpv_next_seq_t[0:1])
                psnr_mean     = (psnr_top_down + psnr_rear + psnr_side + psnr_fpv) / 4.0

                writer.add_scalar("Validation/PSNR/Top Down", psnr_top_down, epoch)
                writer.add_scalar("Validation/PSNR/Rear",     psnr_rear,     epoch)
                writer.add_scalar("Validation/PSNR/Side",     psnr_side,     epoch)
                writer.add_scalar("Validation/PSNR/FPV",      psnr_fpv,      epoch)
                writer.add_scalar("Validation/PSNR/Mean",     psnr_mean,     epoch)

                # --- Per-view LPIPS ---
                lpips_top_down = compute_lpips(val_pred_top_down, im_top_down_next_seq_t[0:1])
                lpips_rear     = compute_lpips(val_pred_rear,     im_rear_next_seq_t[0:1])
                lpips_side     = compute_lpips(val_pred_side,     im_side_next_seq_t[0:1])
                lpips_fpv      = compute_lpips(val_pred_fpv,      im_fpv_next_seq_t[0:1])
                lpips_mean     = (lpips_top_down + lpips_rear + lpips_side + lpips_fpv) / 4.0

                writer.add_scalar("Validation/LPIPS/Top Down", lpips_top_down, epoch)
                writer.add_scalar("Validation/LPIPS/Rear",     lpips_rear,     epoch)
                writer.add_scalar("Validation/LPIPS/Side",     lpips_side,     epoch)
                writer.add_scalar("Validation/LPIPS/FPV",      lpips_fpv,      epoch)
                writer.add_scalar("Validation/LPIPS/Mean",     lpips_mean,     epoch)

                print(f"  [Epoch {epoch}] PSNR: {psnr_mean:.2f} dB | LPIPS: {lpips_mean:.4f}")

                # --- AUTOREGRESSIVE VIDEO ROLLOUT FOR TENSORBOARD + FVD ---
                (gen_top_down, gen_rear, gen_side, gen_fpv,
                 gt_top_down,  gt_rear,  gt_side,  gt_fpv) = autoregressive_rollout(
                    model, buffer, device, rollout_steps=50, seq_len=SEQ_LEN
                )

                # Stacking along dim=1 turns it into (1, T, C, H, W)
                vid_top_down_tensor = torch.stack(gen_top_down, dim=1)
                vid_rear_tensor     = torch.stack(gen_rear,     dim=1)
                vid_side_tensor     = torch.stack(gen_side,     dim=1)
                vid_fpv_tensor      = torch.stack(gen_fpv,      dim=1)

                vid_gt_top_down = torch.stack(gt_top_down, dim=1)
                vid_gt_rear     = torch.stack(gt_rear,     dim=1)
                vid_gt_side     = torch.stack(gt_side,     dim=1)
                vid_gt_fpv      = torch.stack(gt_fpv,      dim=1)

                # Normalise to [0, 1] for TensorBoard + FVD feature extractor
                def _to_01(v):
                    return torch.clamp((v + 1.0) / 2.0, 0.0, 1.0)

                vid_top_down_01 = _to_01(vid_top_down_tensor)
                vid_rear_01     = _to_01(vid_rear_tensor)
                vid_side_01     = _to_01(vid_side_tensor)
                vid_fpv_01      = _to_01(vid_fpv_tensor)

                vid_gt_top_down_01 = _to_01(vid_gt_top_down)
                vid_gt_rear_01     = _to_01(vid_gt_rear)
                vid_gt_side_01     = _to_01(vid_gt_side)
                vid_gt_fpv_01      = _to_01(vid_gt_fpv)

                # Write videos to TensorBoard
                writer.add_video("Test/Top-Down", vid_top_down_01, epoch, fps=15)
                writer.add_video("Test/Rear",     vid_rear_01,     epoch, fps=15)
                writer.add_video("Test/Side",     vid_side_01,     epoch, fps=15)
                writer.add_video("Test/FPV",      vid_fpv_01,      epoch, fps=15)

                # --- FVD: computed on each view independently ---
                # FVD needs (N, T, C, H, W); here N=1 so FVD is approximate
                # but still tracks distributional drift over training.
                fvd_top_down = compute_fvd(vid_gt_top_down_01, vid_top_down_01)
                fvd_rear     = compute_fvd(vid_gt_rear_01,     vid_rear_01)
                fvd_side     = compute_fvd(vid_gt_side_01,     vid_side_01)
                fvd_fpv      = compute_fvd(vid_gt_fpv_01,      vid_fpv_01)
                fvd_mean     = (fvd_top_down + fvd_rear + fvd_side + fvd_fpv) / 4.0

                writer.add_scalar("Validation/FVD/Top Down", fvd_top_down, epoch)
                writer.add_scalar("Validation/FVD/Rear",     fvd_rear,     epoch)
                writer.add_scalar("Validation/FVD/Side",     fvd_side,     epoch)
                writer.add_scalar("Validation/FVD/FPV",      fvd_fpv,      epoch)
                writer.add_scalar("Validation/FVD/Mean",     fvd_mean,     epoch)

                print(f"  [Epoch {epoch}] FVD mean: {fvd_mean:.2f}")

            writer.flush()

        writer.close()
        env.close()
    print("Training Complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on orthographic game environments.")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps per environment")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout steps per batch")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of PPO update epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)