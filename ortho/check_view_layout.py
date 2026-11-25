"""Check that each env's Top/Side/Rear GT depth follows the one fixed layout lift_ortho and hull_depth assume
(Bipedal's), in box coords (x, u, z), u = down:
    Top  [row = z, col = x], depth = u, camera at u = -1      Side [row = u, col = x], depth = z, camera at z = -1
    Rear [row = u, col = z], depth = x, camera at x = -1
Free-space test: every GT depth pixel is lifted to a 3D point, and a point is a violation when another view, looking
through that point, sees nothing there or sees a surface behind it. The same test is repeated with the Top view
rotated/flipped and the Side/Rear views mirrored or depth-negated (128 variants); an env follows the layout when the
unchanged variant ("as is") is (close to) the best. Floors or backdrops drawn in one view only add violations to every
variant alike, so compare "as is" with "best" rather than reading the absolute rate.
    python ortho/check_view_layout.py            # GymBox training envs (add --heldout, --hifi)
"""
import os
import sys
import argparse
import itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer import OrthoTransitionBuffer
from data_rollout import collect_all_envs_rollouts
from utils import _build_registry, _build_heldout_registry, set_seed
from compare_depth import hifi_registry, IMG_H, IMG_W, device

# Variant = (Top rot90 k, Top transposed, Side mirrored, Rear mirrored, Side depth sign, Rear depth sign)
VARIANTS = list(itertools.product(range(4), (False, True), (False, True), (False, True), (1, -1), (1, -1)))


def describe(v):
    k, tr, sf, rf, ss, rs = v
    parts = ([f"Top {'transpose+' if tr else ''}rot{90 * k}"] if k or tr else []) + (["Side mirrored"] if sf else []) \
        + (["Rear mirrored"] if rf else []) + (["Side depth negated"] if ss < 0 else []) \
        + (["Rear depth negated"] if rs < 0 else [])
    return ", ".join(parts) or "as is"


def apply_variant(D, v, signs=True):
    """D (B, 3, H, W) [Top, Side, Rear], inf = nothing drawn -> the same maps under variant v."""
    k, tr, sf, rf, ss, rs = v
    T, S, R = D.unbind(1)
    T = torch.rot90(T.transpose(-1, -2) if tr else T, k, dims=(-2, -1))
    S, R = (S.flip(-1) if sf else S), (R.flip(-1) if rf else R)
    if signs:
        S = torch.where(torch.isfinite(S), ss * S, S)
        R = torch.where(torch.isfinite(R), rs * R, R)
    return torch.stack([T, S, R], 1)


def shape_area(D):
    """Pixels sharing each pixel's exact depth value in its view. The envs give every drawn shape one depth, so this
    is the shape's area: large = floor / backdrop / wall, small = object."""
    A = torch.zeros_like(D)
    for b in range(D.shape[0]):
        for v in range(3):
            _, inv, cnt = torch.unique(D[b, v], return_inverse=True, return_counts=True)
            A[b, v] = cnt[inv].float()
    return A


def violation_rate(D, A, max_area, tol=0.05, radius=2):
    """D (B, 3, H, W), inf = nothing; A shape areas -> per-sample fraction of (object point, other view) checks that
    fail free space. Only points of shapes up to max_area pixels are tested; every surface still blocks. Lookups take
    the nearest surface within `radius` px: depth is the exact near face, which lands on the other views' outlines."""
    B, _, H, W = D.shape
    Dn = -torch.nn.functional.max_pool2d(-D, 2 * radius + 1, stride=1, padding=radius)   # min-pool, inf stays inf
    rr = ((torch.arange(H, device=D.device) + .5) / H * 2 - 1).view(1, H, 1).expand(B, H, W)
    cc = ((torch.arange(W, device=D.device) + .5) / W * 2 - 1).view(1, 1, W).expand(B, H, W)
    T, S, R = D.unbind(1)
    ok = torch.isfinite(D) & (A <= max_area)
    pts = [(cc, T, rr, ok[:, 0]), (cc, rr, S, ok[:, 1]), (R, rr, cc, ok[:, 2])]  # (x, u, z)
    # view -> (row coord, col coord, coord along the view axis) of a point (x, u, z)
    look = [lambda x, u, z: (z, x, u), lambda x, u, z: (u, x, z), lambda x, u, z: (u, z, x)]
    viol = torch.zeros(B, device=D.device)
    checks = torch.zeros(B, device=D.device)
    for src, (x, u, z, valid) in enumerate(pts):
        for dst in range(3):
            if dst == src:
                continue
            row, col, along = look[dst](x, u, z)
            ri, ci = ((row + 1) / 2 * H).floor().long(), ((col + 1) / 2 * W).floor().long()
            inb = valid & (ri >= 0) & (ri < H) & (ci >= 0) & (ci < W)
            d = Dn[:, dst].flatten(1).gather(1, (ri.clamp(0, H - 1) * W + ci.clamp(0, W - 1)).flatten(1)).view(B, H, W)
            bad = inb & (torch.isinf(d) | (along < d - tol))
            viol += bad.flatten(1).sum(1)
            checks += inb.flatten(1).sum(1)
    return viol / checks.clamp(min=1)


def main(args):
    set_seed(args.seed)
    if args.hifi:
        registry = hifi_registry(args.heldout)
    else:
        registry = _build_heldout_registry(obstacles=False) if args.heldout else _build_registry()
    if args.envs:
        registry = [(n, f) for n, f in registry if n in args.envs]
    buffer = OrthoTransitionBuffer(capacity=args.rollout_length * len(registry), img_h=IMG_H, img_w=IMG_W,
                                   max_action_len=8)
    collect_all_envs_rollouts(registry, buffer, rollout_steps=args.rollout_length, num_envs=len(registry),
                              seed=args.seed, device=device)

    rows = []
    for env, _ in registry:
        _, _, _, _, cams = buffer.sample_trajectory_batch(batch_size=args.samples, seq_len=1, env_filter=env)
        D = torch.from_numpy(np.stack([c["depth"] for c in cams[0]]).astype(np.float32)).to(device)
        D = torch.where(D >= 1.999, torch.full_like(D, float("inf")), D)   # GT clips "nothing drawn" to 2
        A = shape_area(D)
        max_area = args.max_area * D.shape[-1] * D.shape[-2]
        rates = torch.stack([violation_rate(apply_variant(D, v), apply_variant(A, v, signs=False), max_area)
                             for v in VARIANTS])                                       # (variants, B)
        mean = rates.mean(1)
        best = int(mean.argmin())
        frames_ok = (rates[0] <= rates.min(0).values + 0.01).float().mean().item()   # "as is" within 1% of the best
        rows.append((env, mean[0].item(), describe(VARIANTS[best]), mean[best].item(), frames_ok))

    print(f"\n{'env':<22}{'as is':<8}{'best':<8}{'frames ok':<11}best variant")
    for env, asis, desc, best, ok in rows:
        flag = "" if asis - best < 0.01 else "   <-- layout differs"
        print(f"{env[:21]:<22}{asis:<8.3f}{best:<8.3f}{ok:<11.2f}{desc}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check each env's Top/Side/Rear GT depth against the fixed layout")
    parser.add_argument("--hifi", action="store_true", help="Godot hi-fi envs")
    parser.add_argument("--heldout", action="store_true", help="held-out (obstacle-free) envs")
    parser.add_argument("--envs", nargs="*", default=None, help="only these env names")
    parser.add_argument("--rollout_length", type=int, default=64)
    parser.add_argument("--samples", type=int, default=48, help="frames per env")
    parser.add_argument("--max_area", type=float, default=0.05, help="shapes larger than this image fraction are "
                        "floors/backdrops: they still block, but their own points are not tested")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
