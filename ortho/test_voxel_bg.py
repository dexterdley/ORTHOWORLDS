"""
test_voxel_bg.py — Visual check of the voxel foreground mask across all envs.

For each registered env: run a short random rollout, build the voxel from the top/side/rear views with
  (a) the current grayscale mask  (dark objects on a light background), and
  (b) a border-median background mask (background = median edge colour, foreground = pixels that differ),
then play an animation comparing inputs, masks and rendered voxels, and save it as a GIF.

Usage (from ortho/):
  python test_voxel_bg.py                          # all envs, 16 steps, play + save GIFs
  python test_voxel_bg.py --envs "Lunar Lander"    # one env
  python test_voxel_bg.py --no_show                # headless (e.g. WSL): only save GIFs
"""

import os
import argparse

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import _build_registry, _build_heldout_registry
from train_voxel_wan import _first_hit_batch, render_voxel_image

N = 96
VIEW_NAMES = ["Top", "Side", "Rear"]


# ==========================================
# MASKS: img (B, 3 views, 3, N, N) in [0, 1] -> (B, 3 views, N, N) bool
# ==========================================
def mask_gray(img, thresh=0.5):
    """Current mask: dark-on-light grayscale threshold (fails on dark backgrounds, e.g. Lunar Lander)."""
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return gray < thresh

def mask_border(img, thresh=0.1):
    """Proposed mask: background = median border colour per image; foreground = pixels that differ from it."""
    border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., :, 0], img[..., :, -1]], dim=-1)
    bg = border.median(dim=-1).values[..., None, None]
    return (img - bg).abs().sum(dim=2) > thresh


# ==========================================
# VOXEL (same hull + visibility colouring as build_voxel_batch, with a pluggable mask)
# ==========================================
@torch.no_grad()
def voxel_from_views(views, mask_fn):
    """views: (B, 4, 3, N, N) in [-1, 1] -> voxel (B, 4, N, N, N) [occ, rgb], masks (B, 3, N, N)."""
    img = ((views[:, :3].float() + 1.0) * 127.5).clamp(0, 255).floor() / 255.0
    m = mask_fn(img)

    T = m[:, 0].transpose(1, 2)                 # T[x, y]
    S = m[:, 1].flip(1).transpose(1, 2)         # S[x, z]
    R = m[:, 2].flip(1).transpose(1, 2)         # R[y, z]
    V = T[:, :, :, None] & S[:, :, None, :] & R[:, None, :, :]

    Tc = img[:, 0].transpose(2, 3)[:, :, :, :, None]
    Sc = img[:, 1].flip(2).transpose(2, 3)[:, :, :, None, :]
    Rc = img[:, 2].flip(2).transpose(2, 3)[:, :, None, :, :]

    w_top  = _first_hit_batch(V, dim=3, from_high=True)[:, None].float()
    w_side = _first_hit_batch(V, dim=2, from_high=True)[:, None].float()
    w_rear = _first_hit_batch(V, dim=1, from_high=False)[:, None].float()
    wsum = w_top + w_side + w_rear
    seen = (w_top * Tc + w_side * Sc + w_rear * Rc) / wsum.clamp(min=1)
    rgb = torch.where(wsum > 0, seen, (Tc + Sc + Rc) / 3.0) * V[:, None]

    return torch.cat([V[:, None].float(), rgb], dim=1), m


# ==========================================
# ROLLOUT
# ==========================================
def rollout_views(env_name, factory, steps):
    """Random-action rollout -> views (T, 4, 3, N, N) in [-1, 1], ordered (top, side, rear, fpv)."""
    env = factory()
    env.reset()
    if env_name == "MultiCar Racing":
        for _ in range(50):  # skip the zoom-in intro, as in training
            env.step(np.zeros(3))

    frames = []
    for _ in range(steps):
        top, rear, side, fpv = env.render(camera_pose="Center")
        v = [cv2.resize(x, (N, N), interpolation=cv2.INTER_AREA) for x in (top, side, rear, fpv)]
        frames.append(np.stack([x.transpose(2, 0, 1) for x in v]))
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()
    return torch.from_numpy(np.stack(frames)).float() / 127.5 - 1.0


# ==========================================
# ANIMATION
# ==========================================
def animate_env(env_name, views, vox_old, m_old, vox_new, m_new, out_dir, fps, show):
    T = views.shape[0]
    hwc = lambda chw: np.clip(chw.permute(1, 2, 0).numpy(), 0, 1)
    rgb_views = (views + 1.0) / 2.0
    renders_old = [render_voxel_image(vox_old[t]).transpose(1, 2, 0) for t in range(T)]
    renders_new = [render_voxel_image(vox_new[t]).transpose(1, 2, 0) for t in range(T)]

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    row_names = ["Input views", "Gray mask (current)", "Border mask (proposed)"]
    handles = []
    for r in range(3):
        row = []
        for c in range(4):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                im = ax.imshow(hwc(rgb_views[0, c]))
                ax.set_title(VIEW_NAMES[c] if c < 3 else "FPV (not used)")
            elif c < 3:
                m = (m_old if r == 1 else m_new)[0, c].numpy()
                im = ax.imshow(m, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{VIEW_NAMES[c]} mask")
            else:
                im = ax.imshow(renders_old[0] if r == 1 else renders_new[0])
                ax.set_title("Voxel")
            if c == 0:
                ax.set_ylabel(row_names[r], fontsize=11)
            row.append(im)
        handles.append(row)
    title = fig.suptitle("")

    def update(t):
        for c in range(4):
            handles[0][c].set_data(hwc(rgb_views[t, c]))
        for c in range(3):
            handles[1][c].set_data(m_old[t, c].numpy())
            handles[2][c].set_data(m_new[t, c].numpy())
        handles[1][3].set_data(renders_old[t])
        handles[2][3].set_data(renders_new[t])
        occ_old = vox_old[t, 0].mean().item() * 100
        occ_new = vox_new[t, 0].mean().item() * 100
        title.set_text(f"{env_name} — step {t + 1}/{T}   |   occupied voxels: gray {occ_old:.2f}%  vs  border {occ_new:.2f}%")
        return [h for row in handles for h in row] + [title]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    path = os.path.join(out_dir, env_name.replace(" ", "_").lower() + ".gif")
    anim.save(path, writer=PillowWriter(fps=fps))
    print(f"  saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    registry = _build_heldout_registry()
    if args.envs:
        registry = [(n, f) for n, f in registry if n in args.envs]

    for env_name, factory in registry:
        print(f"\n[{env_name}] {args.steps}-step rollout")
        try:
            views = rollout_views(env_name, factory, args.steps)
        except Exception as e:
            print(f"  [SKIP] rollout failed: {e}")
            continue

        vox_old, m_old = voxel_from_views(views, mask_gray)
        vox_new, m_new = voxel_from_views(views, mask_border)

        # Foreground fraction per view: ~100% means the background is being counted as object
        for k, name in enumerate(VIEW_NAMES):
            print(f"  {name:4s} foreground: gray {m_old[:, k].float().mean() * 100:5.1f}%   "
                  f"border {m_new[:, k].float().mean() * 100:5.1f}%")
        print(f"  voxel occupancy: gray {vox_old[:, 0].mean() * 100:.2f}%   border {vox_new[:, 0].mean() * 100:.2f}%")

        animate_env(env_name, views, vox_old, m_old, vox_new, m_new, args.out_dir, args.fps, not args.no_show)


def parse_args():
    parser = argparse.ArgumentParser(description="Check voxel foreground masks across all orthographic envs")
    parser.add_argument("--steps", type=int, default=16, help="Rollout steps per env")
    parser.add_argument("--envs", type=str, nargs="*", default=None, help="Env names to test (default: all registered)")
    parser.add_argument("--out_dir", type=str, default="./voxel_bg_test", help="Where to save the GIFs")
    parser.add_argument("--fps", type=int, default=4, help="Animation speed")
    parser.add_argument("--no_show", action="store_true", help="Only save GIFs (no interactive window)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
