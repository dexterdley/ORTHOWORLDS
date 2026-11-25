"""Compare GT depth, visual-hull depth and the trained OrthoDepthNet on Top/Side/Rear.
Prints a per-env L1 table and saves one PNG per env: RGB, then GT / Hull / Model depth, and the 3D points lifted from each.
    python compare_depth.py --ckpt runs/projector/ortho_depth.pt
    python compare_depth.py --hifi              # Godot hi-fi envs
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from buffer import OrthoTransitionBuffer
from data_rollout import collect_all_envs_rollouts
from utils import _build_registry, _build_heldout_registry, set_seed
from train_projector import OrthoDepthNet, lift_ortho, IMG_H, IMG_W, device

VIEWS = ("Top", "Side", "Rear")


def silhouette_cover(mask):
    """Fraction of the Side / Rear foreground that the hull of `mask` (B, 3, H, W) still projects onto (per sample)."""
    top, side, rear = mask.unbind(1)
    occ = top[:, None] & side[:, :, None] & rear[:, :, :, None]           # (B, u, z, x)
    cover = lambda proj, m: (proj & m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)
    return (cover(occ.any(2), side) + cover(occ.any(3), rear)) / 2


def fg_mask(views, thresh=0.05, ground_cover=0.97, return_ground=False):
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
    if not return_ground:
        return m
    ground = ~far(views[..., :1, :1])[:, 0] & use_ground[:, None, None]   # (B, H, W) Top pixels showing the ground
    return m, ground


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


# Each view's depth axis is read along another view's image axis (same layout as lift_ortho / hull_depth)
PARTNER = {"top": ("side", "row"), "side": ("top", "row"), "rear": ("top", "col")}


def hifi_box_depth(env):
    """Godot depth dict (metres along each ortho camera's axis, `far` where nothing is hit) -> GymBox-style
    [top, side, rear]: the surface point's coordinate along the partner view's image axis, in [-1, 1] across that
    view, 2 where nothing is hit. Not sign-flipped to make -1 = near (DepthCanvas does that): the true coordinate
    keeps the lifted points where they are when a hi-fi camera sits on the +1 side."""
    from exporters.base import _basis
    cams = {c["view"]: c for c in env.exporter.cameras(env.env)}
    w, h = env.exporter.size                                              # Godot frame (KEEP_HEIGHT: size = height)
    out = []
    for v in ("top", "side", "rear"):
        p, axis = PARTNER[v]
        pos_v, fwd_v, _, _ = _basis(cams[v])
        pos_p, _, right_p, up_p = _basis(cams[p])
        if axis == "row":                                                 # rows run down: coordinate = -y / half height
            a, b, half = -(pos_v - pos_p) @ up_p, -(fwd_v @ up_p), cams[p]["size"] / 2
        else:                                                             # cols run right: coordinate = x / half width
            a, b, half = (pos_v - pos_p) @ right_p, fwd_v @ right_p, cams[p]["size"] / 2 * w / h
        d = env.last_depth[v]
        out.append(np.where(d >= 0.999 * cams[v]["far"], 2.0, (a + b * d) / half).astype(np.float32))
    return out


def hifi_registry(heldout):
    """Godot hi-fi envs that also render depth, with env.last_depth converted by hifi_box_depth after every render."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hifi"))
    from wrappers import build_hifi_registry, build_hifi_heldout_registry
    aov = ("rgb", "depth")
    reg = build_hifi_heldout_registry(obstacles=False, aov=aov) if heldout else build_hifi_registry(aov=aov)

    def with_box_depth(make):
        def factory():
            env = make()
            render = env.render
            def render_box_depth(*a, **k):
                out = render(*a, **k)
                env.last_depth = hifi_box_depth(env)
                return out
            env.render = render_box_depth
            return env
        return factory
    return [(name, with_box_depth(make)) for name, make in reg]


@torch.no_grad()
def estimate(views, model):
    """views (B, 3, 3, H, W) in [-1, 1] -> {name: depth (B, 3, H, W)}. No GT is used by any estimator."""
    mask, ground = fg_mask(views, return_ground=True)
    out = {"Hull": hull_depth(mask, ground=ground)}                       # silhouettes from the RGB
    if model is not None:                                                 # 4-channel model = RGB + hull input
        x = views if model.embed.in_channels == 3 else torch.cat([views, out["Hull"][:, :, None]], 2)
        out["Model"] = model(x).float()
    return out


def l1(d, gt, fg):
    """Per-view mean |d - gt| over all pixels and over GT-foreground pixels only."""
    e = (d - gt).abs()
    return e.mean((0, 2, 3)), (e * fg).sum((0, 2, 3)) / fg.sum((0, 2, 3)).clamp(min=1)


def plot_sample(rgb, gt, est, path, title):
    """rgb (3, 3, H, W) in [0, 1], gt (3, H, W), est {name: (3, H, W)} -> PNG: 2D maps on the left, 3D points right."""
    depths = {"GT": gt, **est}
    near = lambda d: ((1 - d.clamp(-1, 1)) / 2).cpu().numpy()             # near bright, far / nothing black
    rows = [("RGB", [v.permute(1, 2, 0).clamp(0, 1).cpu().numpy() for v in rgb], None)]
    rows += [(k, [near(x) for x in d], "gray") for k, d in depths.items()]

    fig = plt.figure(figsize=(6 + 4 * len(depths), 1.7 * len(rows)), dpi=80)
    gs = fig.add_gridspec(len(rows), 3 + 2 * len(depths))
    for r, (label, imgs, cmap) in enumerate(rows):
        for v, img in enumerate(imgs):
            ax = fig.add_subplot(gs[r, v])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0: ax.set_title(VIEWS[v], fontsize=9)
            if v == 0: ax.set_ylabel(label, fontsize=8)

    img = rgb.permute(0, 2, 3, 1).cpu().numpy()                           # (3, H, W, 3) for lift_ortho
    for k, (name, d) in enumerate(depths.items()):
        ax = fig.add_subplot(gs[:, 3 + 2 * k: 5 + 2 * k], projection="3d")
        P, C = lift_ortho(img, d.cpu().numpy())
        ax.scatter(P[:, 0], -P[:, 2], P[:, 1], c=C, s=1)
        ax.set_xlim(-1.6, 1.6), ax.set_ylim(-1.6, 1.6), ax.set_zlim(-1, 1)
        ax.view_init(elev=20, azim=-60)
        ax.set_title(f"3D from {name} ({len(P)} pts)", fontsize=9)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.hifi:
        registry = hifi_registry(args.heldout)
    else:
        registry = _build_heldout_registry(obstacles=False) if args.heldout else _build_registry()
    if args.envs:
        registry = [(n, f) for n, f in registry if n in args.envs]
    buffer = OrthoTransitionBuffer(capacity=args.rollout_length * min(args.num_envs, len(registry)),
                                   img_h=IMG_H, img_w=IMG_W, max_action_len=8)
    collect_all_envs_rollouts(registry, buffer, rollout_steps=args.rollout_length, num_envs=args.num_envs,
                              seed=args.seed, device=device)

    model = None
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model = OrthoDepthNet(in_ch=state["embed.weight"].shape[1]).to(device).eval()
        model.load_state_dict(state)
    else:
        print(f"No checkpoint at {args.ckpt}: comparing hull vs GT only")

    views, _, _, names, cams = buffer.sample_trajectory_batch(batch_size=args.batch_size, seq_len=1)
    views, names = views[:, :3], np.array(names[0])
    gt = torch.from_numpy(np.stack([c["depth"] for c in cams[0]]).astype(np.float32)).to(device)
    fg = (gt < 2).float()
    est = estimate(views, model)

    # L1 table: "all" is comparable to the training loss, "fg" ignores the (easy) empty background
    fmt = lambda t: "/".join(f"{v:.3f}" for v in t.tolist())
    print(f"\n{'env':<24}{'estimator':<20}{'L1 all (T/S/R)':<22}L1 fg (T/S/R)")
    for env in ["ALL"] + sorted(set(names)):
        idx = torch.from_numpy(names == env if env != "ALL" else np.ones(len(names), bool)).to(device)
        for k, d in est.items():
            a, f = l1(d[idx], gt[idx], fg[idx])
            print(f"{env[:23]:<24}{k:<20}{fmt(a):<22}{fmt(f)}")

    # One PNG per env (first sample of each)
    for env in sorted(set(names)):
        i = int(np.flatnonzero(names == env)[0])
        path = os.path.join(args.out_dir, "".join(c for c in env if c.isalnum()).lower() + ".png")
        plot_sample((views[i] + 1) / 2, gt[i], {k: d[i] for k, d in est.items()}, path, env)
        print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GT, visual-hull and model depth (2D maps + 3D points)")
    parser.add_argument("--ckpt", type=str, default=os.path.join("runs", "projector", "ortho_depth.pt"))
    parser.add_argument("--out_dir", type=str, default=os.path.join("runs", "projector", "compare"))
    parser.add_argument("--rollout_length", type=int, default=256)
    parser.add_argument("--num_envs", type=int, default=99, help="first N envs of the registry (99 = all)")
    parser.add_argument("--batch_size", type=int, default=256, help="samples used for the L1 table")
    parser.add_argument("--heldout", action="store_true", help="use the held-out (obstacle-free) envs instead")
    parser.add_argument("--hifi", action="store_true", help="render with the Godot hi-fi envs (depth from Godot)")
    parser.add_argument("--envs", nargs="*", default=None, help="only these env names, e.g. --envs \"Robot Sumo\"")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
