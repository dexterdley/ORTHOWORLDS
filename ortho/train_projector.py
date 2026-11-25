import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from buffer import OrthoTransitionBuffer
from data_rollout import collect_all_envs_rollouts
from utils import _build_registry, _build_heldout_registry, set_seed

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_H, IMG_W = 96, 96
OUT_DIR = os.path.join("runs", "projector")

def silhouette_cover(mask):
    """Fraction of the Side / Rear foreground that the hull of `mask` (B, 3, H, W) still projects onto (per sample)."""
    top, side, rear = mask.unbind(1)
    occ = top[:, None] & side[:, :, None] & rear[:, :, :, None]
    cover = lambda proj, m: (proj & m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)
    return (cover(occ.any(2), side) + cover(occ.any(3), rear)) / 2

def fg_mask(views, thresh=0.05, ground_cover=0.97, return_floor=False):
    """(B, 3, 3, H, W) in [-1, 1] -> (B, 3, H, W) bool foreground. Empty everywhere = the elevations' sky (the Side's
    top-left colour), and in each view its own top-left colour. The Top's corner colour is the one ambiguity: a void
    drawn in its own colour, or the ground filling the whole Top. It is taken as ground when the hull then explains
    (nearly) all of the Side / Rear silhouettes, and no fewer than with it carved away."""
    far = lambda c: (views - c).abs().amax(2) > thresh
    not_sky = far(views[:, 1:2, :, :1, :1])
    m = not_sky & far(views[..., :1, :1])
    m_ground = m.clone()
    m_ground[:, 0] = not_sky[:, 0]
    cover_ground, cover_void = silhouette_cover(m_ground), silhouette_cover(m)
    use_ground = (cover_ground >= ground_cover) & (cover_ground >= cover_void - 1e-3)
    m[use_ground, 0] = not_sky[use_ground, 0]
    if not return_floor:
        return m
    floor = ~far(views[..., :1, :1])[:, 0] & use_ground[:, None, None]   # (B, H, W) Top pixels showing the ground
    return m, floor

def hull_depth(mask, empty=2.0, ground=None):
    """Foreground masks (B, 3, H, W) of Top[z, x], Side[u, x], Rear[u, z] (H == W) -> visual-hull depth per view: first
    occupied coordinate along each view's depth axis, `empty` (= GT's clipped background) where the hull is empty.
    ground (B, H, W): Top pixels that show the ground. Seen from above, nothing stands on them, so their column keeps
    only the occupancy joined to the bottom (removes ghosts, e.g. far walls spanning the whole Side / Rear)."""
    top, side, rear = mask.unbind(1)
    occ = top[:, None] & side[:, :, None] & rear[:, :, :, None]           # (B, u, z, x) bool
    if ground is not None:
        up = occ.flip(1)                                                  # u' = 0 at the bottom
        started = up.cumsum(1, dtype=torch.int16) > 0
        bottom_run = up & (up | ~started).to(torch.uint8).cumprod(1).bool()
        occ = torch.where(ground[:, None], bottom_run.flip(1), occ)
    c = (torch.arange(mask.shape[-1], device=mask.device) + .5) / mask.shape[-1] * 2 - 1
    return torch.stack([c[occ.byte().argmax(d)].masked_fill(~occ.any(d), empty) for d in (1, 2, 3)], 1)

def with_hull(rgb, drop=0.0):
    """Top/Side/Rear RGB (B, 3, 3, H, W) -> (B, 3, 4, H, W): RGB + hull depth; `drop` of samples get -2 (= no hull)."""
    with torch.no_grad():
        mask, floor = fg_mask(rgb, return_floor=True)
        hull = hull_depth(mask, ground=floor)
        if drop > 0:
            hull[torch.rand(hull.shape[0], device=hull.device) < drop] = -2.0
    return torch.cat([rgb, hull[:, :, None]], dim=2)

def save_gif_frame(grid, xyz, rgb, xyz_gt, rgb_gt, step):
    """GIF frame: the RGB / GT depth / predicted depth grid (Top | Side | Rear), then the 3D points lifted from the GT
    depth and from the predicted depth, in their own colour."""
    fig = plt.figure(figsize=(16, 4.5), dpi=80)

    # Left: RGB, GT depth and predicted depth (bright = near, black = far / nothing drawn)
    ax2 = fig.add_subplot(1, 3, 1)
    ax2.imshow(grid[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
    ax2.set_title(f"step {step}    rows: RGB, GT depth, predicted depth    (Top | Side | Rear)", fontsize=9)
    ax2.axis("off")

    # Middle / right: 3D points from the GT and the predicted depth (lift_ortho gives (x, up, z))
    for k, (pts, col, title) in enumerate(((xyz_gt, rgb_gt, "3D from GT depth"), (xyz, rgb, "3D from predicted depth"))):
        ax3 = fig.add_subplot(1, 3, k + 2, projection="3d")
        ax3.scatter(pts[:, 0], -pts[:, 2], pts[:, 1], c=col, s=1)
        ax3.set_xlim(-1.6, 1.6), ax3.set_ylim(-1.6, 1.6), ax3.set_zlim(-1, 1)
        ax3.set_xlabel("X (Width)"); ax3.set_ylabel("Z (Depth)"); ax3.set_zlabel("Y (Height)")
        ax3.view_init(elev=20, azim=-60)
        ax3.set_title(title, fontsize=9)

    fig.tight_layout()
    fig.canvas.draw()
    frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frame

class OrthoDepthNet(nn.Module):
    """Top/Side/Rear -> each view's nearness map (B, 3, H, W). One transformer over all three views' patches: a view
    can't see its own depth axis, but the other two views can."""
    def __init__(self, patch=8, dim=192, layers=6, heads=6, in_ch=3):
        super().__init__()
        self.p = patch
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, 3 * (IMG_H // patch) * (IMG_W // patch), dim) * 0.02)   # per view + position
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.0, activation="gelu",
                                                                       batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch)                             # nearness for every patch pixel
        self.refine = nn.Sequential(nn.Conv2d(in_ch + 1, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, views):                                                 # (B, 3, 3, H, W) in [-1, 1]
        B, V, _, H, W = views.shape
        x = self.embed(views.flatten(0, 1))                                   # (B*3, dim, h, w)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).reshape(B, V * h * w, -1) + self.pos
        x = self.head(self.norm(self.blocks(x)))                              # (B, 3*h*w, p*p)
        x = x.reshape(B * V, h, w, 1, self.p, self.p).permute(0, 3, 1, 4, 2, 5).reshape(B * V, 1, H, W)
        x = x + self.refine(torch.cat([x, views.flatten(0, 1)], 1))           # sharpen edges against the RGB
        return x.reshape(B, V, H, W)                                          # nearness

def lift_ortho(img, dep):
    """Top/Side/Rear (3, H, W, 3) in [0, 1] + depth (3, H, W), inf = nothing drawn -> points (N, 3) as (x, up, z) and
    colours (N, 3). One fixed layout for every env (Bipedal's): Top cols/rows = x/z, depth = height; Side cols = x,
    rows = height, depth = z; Rear cols = z, rows = height, depth = x."""
    H, W = dep.shape[1:]
    r, c = np.meshgrid((np.arange(H) + .5) / H * 2 - 1, (np.arange(W) + .5) / W * 2 - 1, indexing="ij")
    P = np.concatenate([np.stack(v, -1).reshape(-1, 3) for v in ((c, dep[0], r), (c, r, dep[1]), (dep[2], r, c))])
    C = img.reshape(-1, 3)
    keep = np.isfinite(P).all(1) & (np.abs(P) <= 1.6).all(1)
    return P[keep] * [1, -1, 1], C[keep]

def visualize_3D_render(xyz, rgb):
    fig = plt.figure(figsize=(6, 6))
    ax3 = fig.add_subplot(1, 1, 1, projection="3d")
    ax3.scatter(xyz[:, 0], -xyz[:, 2], xyz[:, 1], c=rgb, s=1)   # image top = far, height up
    ax3.set_xlim(-1, 1); ax3.set_ylim(-1, 1); ax3.set_zlim(-1, 1); 
    ax3.view_init(elev=35, azim=-70)
    fig.savefig(os.path.join(OUT_DIR, "depth_check.png"))
    plt.close(fig)

# ==========================================
# Main Training Loop for Ortho 3DGS Projector
# ==========================================
def main(args):
    set_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Build Envs and Buffer
    if args.hifi:
        registry, _ = get_registries(args.hifi)
    else:
        registry = _build_registry()

    # registry = [registry[0]] # For debugging
    
    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(
            capacity=args.rollout_length * min(args.num_envs, len(registry)), 
            img_h=IMG_H, 
            img_w=IMG_W, 
            max_action_len=8
    )
    active_envs, agents = collect_all_envs_rollouts(
        registry, buffer, rollout_steps=args.rollout_length, num_envs=args.num_envs,
        device=device,
    )

    # Initialize model and optimizer
    model = OrthoDepthNet(in_ch=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    warmup = 200
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, milestones=[warmup], schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps - warmup, eta_min=args.lr * 0.1),
    ])

    # Validation
    check_views, _, _, _, check_cams = buffer.sample_trajectory_batch(batch_size=1, seq_len=1)
    check_gt = (check_views[:, :3] + 1.0) / 2.0
    check_depth = torch.from_numpy(np.stack([c["depth"] for c in check_cams[0]]).astype(np.float32)).to(device)
    xyz_gt, rgb_gt = lift_ortho(check_gt[0].permute(0, 2, 3, 1).cpu().numpy(), check_depth[0].cpu().numpy())
    frames = []

    for epoch in range(args.steps):
        model.train()
        
        # Sample batch (buffer images are in range [-1, 1])
        views_t, views_next, _, _, cams = buffer.sample_trajectory_batch(batch_size=args.batch_size, seq_len=1)
        gt_depth = torch.from_numpy(np.stack([c["depth"] for c in cams[0]]).astype(np.float32)).to(views_t.device)

        # depth = np.stack([c["depth"] for c in cams[0]]).astype(np.float32)
        # imgs = ((views_t[:, :3] + 1) / 2).permute(0, 1, 3, 4, 2).cpu().numpy()
        # xyz_points, rgb_points = lift_ortho(imgs[i], depth[i])
        
        # import pdb; pdb.set_trace()
        # pred_depth = model(views_t[:, :3]) # Forward pass: predict depth from Top, Side, Rear
        pred_depth = model(with_hull(views_t[:, :3], args.hull_dropout))

        # Depth loss per orthographic view
        loss_top = F.l1_loss(pred_depth[:, 0], gt_depth[:, 0])
        loss_side = F.l1_loss(pred_depth[:, 1], gt_depth[:, 1])
        loss_rear = F.l1_loss(pred_depth[:, 2], gt_depth[:, 2])
        loss = loss_top + loss_side + loss_rear

        # Backward pass & update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Step {epoch+1} | Total Loss: {loss.item():.4f} (Top: {loss_top.item():.4f}, Side: {loss_side.item():.4f}, Rear: {loss_rear.item():.4f})")

            model.eval()
            with torch.no_grad():
                #check_pred = model(check_views[:, :3])                             # (1, 3, H, W) depth
                check_pred = model(with_hull(check_views[:, :3]))
                to_img = lambda d: ((1 - d.clamp(-1, 1)) / 2)[None].expand(3, -1, -1)   # display only: near bright, far/nothing black
                rgb_row = torch.cat(check_gt[0].unbind(0), dim=-1)                  # (3, H, 3W)
                gt_row = to_img(torch.cat(check_depth[0].unbind(0), dim=-1))
                pred_row = to_img(torch.cat(check_pred[0].unbind(0), dim=-1))
                grid = torch.cat([rgb_row, gt_row, pred_row], dim=-2)[None]
                xyz, rgb = lift_ortho(check_gt[0].permute(0, 2, 3, 1).cpu().numpy(), check_pred[0].cpu().numpy())
                frames.append(save_gif_frame(grid, xyz, rgb, xyz_gt, rgb_gt, epoch + 1))
                frames[0].save(os.path.join(OUT_DIR, "training.gif"), save_all=True,
                               append_images=frames[1:], duration=200, loop=0)
    
    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "ortho_depth_hull.pt"))

def parse_args():
    parser = argparse.ArgumentParser(description="Top/Side/Rear -> 3D Gaussians, supervised by re-rendering the ortho views")
    parser.add_argument("--rollout_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--hifi", type=str, default=None)
    parser.add_argument("--hull_dropout", type=float, default=0.2, help="fraction of training samples without the hull")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)