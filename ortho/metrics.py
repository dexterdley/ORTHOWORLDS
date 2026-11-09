"""
Evaluation metrics shared by the eval scripts: PSNR, LPIPS (VGG) and FVD (R3D-18 features).
Moved unchanged from the old train_ortho_diffusion.py.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from scipy import linalg
from torchvision.models.video import r3d_18, R3D_18_Weights

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
def _extract_video_features(videos: torch.Tensor, batch_size: int = 16) -> np.ndarray:
    """Extract R3D-18 features from a batch of videos.

    Clips are moved to the GPU `batch_size` at a time (the model is in eval mode, so the features are
    identical to a single full-batch pass); pooled FVD over many envs would otherwise not fit in memory.

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

    feats = []
    for i in range(0, N, batch_size):
        v = videos[i:i + batch_size].to(device).float()   # (n, T, C, H, W)
        n = v.shape[0]
        v = v.permute(0, 2, 1, 3, 4)                      # (n, C, T, H, W)

        # Upsample spatial dims to at least 112 x 112 if needed
        if H < 112 or W < 112:
            v = F.interpolate(v.reshape(n * C, T, H, W).unsqueeze(1),
                              size=(T, 112, 112), mode='trilinear',
                              align_corners=False).squeeze(1).view(n, C, T, 112, 112)

        v = (v - mean) / std
        feats.append(model(v).cpu())                      # (n, 512)
    return torch.cat(feats, dim=0).numpy()


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
