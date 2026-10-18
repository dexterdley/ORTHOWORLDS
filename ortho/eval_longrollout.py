"""
eval_longrollout.py — FPV-centred long-horizon rollout evaluation.

The 3D FPV view is the main output of the world model; the orthographic top / side / rear views
are supporting views. Any number of models are rolled out from the same ground-truth start frames
with the same logged actions and camera commands, and compared on the FPV view:
  - Per-step PSNR / SSIM / LPIPS (step 1 = single-step prediction), AUC-PSNR, drift onset (PSNR < 20 dB)
  - FVD on FPV clips (R3D-18 features, see train_ortho_diffusion.compute_fvd)
Multi-view models additionally report the consistency of their supporting views:
  - Projection Error (PE) of the decoded top/side/rear silhouettes, next to the ground-truth PE floor
  - Hull IoU between the visual hulls carved from generated and ground-truth silhouettes

Models are LoRA checkpoint folders, named after the folder:
  --multi   predicts all 4 views jointly (train_voxel_wan.py / train_ortho_wan.py); FPV is view 3
  --single  predicts the FPV view only (train_single_wan.py)

Usage:
  python eval_longrollout.py \\
      --multi ./checkpoints/runs/ours ./checkpoints/runs/ours_lambda0 \\
      --single ./checkpoints/runs/fpv_only \\
      --envs all --rollout_steps 32 --n_rollouts 20

Saves to results/eval/<tag>/:
  per_step.csv        <- env, model, step, metric means (curves)
  summary.csv         <- one row per (env, model); "Train (mean)" / "Held-out (mean)" aggregate rows
  summary_main.tex    <- main-results table rows (training environments)
  summary_per_env.tex <- per-environment table rows
  <env>_curves.png    <- FPV PSNR and PE versus rollout step
  <env>_strip.png     <- GT and every model's FPV frames at selected steps
"""

import os
import re
import csv
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from tqdm import tqdm

from utils import _build_registry, _build_heldout_registry, load_base_model, load_agent, compose_player_prompts, set_seed
from buffer import OrthoTransitionBuffer
from train_voxel_wan import (
    collect_rollouts,
    encode_views_to_latents,
    decode_latents_to_views,
    predict_next_view_latents,
    silhouettes_from_views,
    projection_error,
)
from train_ortho_diffusion import compute_fvd, _get_lpips_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

FPV_IDX = 3
DRIFT_DB = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def psnr_per_sample(pred, gt):
    """pred, gt: (B, 3, H, W) in [0, 1] -> (B,) PSNR in dB."""
    mse = (pred.float() - gt.float()).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return (-10.0 * torch.log10(mse)).tolist()

def ssim_per_sample(pred, gt):
    """pred, gt: (B, 3, H, W) in [0, 1] -> (B,) SSIM."""
    p = pred.permute(0, 2, 3, 1).float().cpu().numpy()
    g = gt.permute(0, 2, 3, 1).float().cpu().numpy()
    return [float(structural_similarity(a, b, channel_axis=2, data_range=1.0)) for a, b in zip(p, g)]

@torch.no_grad()
def lpips_per_sample(pred, gt):
    """pred, gt: (B, 3, H, W) in [0, 1] -> (B,) LPIPS (VGG)."""
    fn = _get_lpips_fn()
    return fn(pred.float().to(device) * 2.0 - 1.0, gt.float().to(device) * 2.0 - 1.0).flatten().tolist()

def pe_per_sample(views):
    """views: (B, 4, 3, H, W) in [0, 1] -> (B,) projection error of the top/side/rear silhouettes."""
    return projection_error(silhouettes_from_views(views.float() * 2.0 - 1.0)).tolist()

def visual_hull(views):
    """views: (B, 4, 3, H, W) in [0, 1] -> (B, N, N, N) bool hull indexed [x, y, z] (axes as build_voxel_batch)."""
    m = silhouettes_from_views(views.float() * 2.0 - 1.0) > 0.5   # (B, 3, N, N) top, side, rear
    T = m[:, 0].transpose(1, 2)                 # T[x, y]
    S = m[:, 1].flip(1).transpose(1, 2)         # S[x, z]
    R = m[:, 2].flip(1).transpose(1, 2)         # R[y, z]
    return T[:, :, :, None] & S[:, :, None, :] & R[:, None, :, :]

def hull_iou_per_sample(pred_views, gt_views):
    """(B, 4, 3, H, W) in [0, 1] each -> (B,) occupancy IoU of the carved visual hulls."""
    a, b = visual_hull(pred_views), visual_hull(gt_views)
    inter = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).float().clamp(min=1)
    return (inter / union).tolist()

def drift_onset(psnr_curve, thresh=DRIFT_DB):
    """First 1-indexed step with PSNR below `thresh`, or None if it never drifts."""
    return next((s + 1 for s, v in enumerate(psnr_curve) if v < thresh), None)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

def model_specs(multi_paths, single_paths):
    """Checkpoint folders -> [(name, kind, path)], each model named after its folder."""
    specs = [(os.path.basename(os.path.normpath(p)), "multi", p) for p in multi_paths]
    specs += [(os.path.basename(os.path.normpath(p)), "single", p) for p in single_paths]
    names = [n for n, _, _ in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Checkpoint folders must have distinct names, got {names}")
    return specs

def load_models(base_transformer, specs):
    """Loads every LoRA as a named adapter on one shared base DiT. Returns (peft_model, [(name, kind, adapter)])."""
    from peft import PeftModel

    model, models = None, []
    for i, (name, kind, path) in enumerate(specs):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint for '{name}' not found: {path}")
        adapter = f"m{i}_" + re.sub(r"[^0-9a-zA-Z]+", "_", name)
        if model is None:
            model = PeftModel.from_pretrained(base_transformer, path, adapter_name=adapter)
        else:
            model.load_adapter(path, adapter_name=adapter)
        print(f"  [LOADED] {name} ({kind}) <- {path}")
        models.append((name, kind, adapter))
    model.eval()
    return model, models


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENTS & DATA
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_envs(which, num_envs):
    """Returns [(label, env_name, factory, split)]; held-out envs appear with and without obstacles."""
    entries = []
    if which in ("train", "all"):
        for name, factory in _build_registry()[:num_envs]:
            entries.append((name, name, factory, "train"))
    if which in ("heldout", "all"):
        for obstacles in (True, False):
            for name, factory in _build_heldout_registry(obstacles=obstacles):
                label = name if obstacles else f"{name} (no obstacles)"
                entries.append((label, name, factory, "heldout"))
    if which not in ("train", "heldout", "all"):
        wanted = [w.strip() for w in which.split(",")]
        all_entries = build_eval_envs("all", num_envs=10**6)
        entries = [e for e in all_entries if e[0] in wanted]
        missing = set(wanted) - {e[0] for e in entries}
        if missing:
            raise ValueError(f"Unknown environments: {sorted(missing)}")
    return entries

def make_agent(env_name, env):
    """Same policies as data collection in train_voxel_wan.collect_all_envs_rollouts."""
    if env_name in ("Lunar Lander", "MultiCar Racing"):
        from stable_baselines3 import PPO
        from huggingface_sb3 import load_from_hub
        repo, fname = (("Adilbai/ppo-LunarLander-v2", "ppo-LunarLander-v2.zip") if env_name == "Lunar Lander"
                       else ("igpaub/ppo-CarRacing-v2", "ppo-CarRacing-v2.zip"))
        return PPO.load(load_from_hub(repo_id=repo, filename=fname))

    obs, _ = env.reset()
    state_dim = int(np.prod(obs.shape if hasattr(obs, "shape") else env.observation_space.shape))
    clean = "".join(c for c in env_name if c.isalnum() or c in ("_", "-")).lower()
    # NOTE: ActorCritic has no .predict(), so collect_rollouts samples random actions for these envs,
    # exactly as during training-data collection.
    return load_agent(os.path.join("checkpoints", f"ppo_{clean}.pt"), state_dim, env.action_space, device)

def collect_eval_buffer(env_name, factory, n_steps, seed):
    env = factory()
    agent = make_agent(env_name, env)
    buffer = OrthoTransitionBuffer(capacity=n_steps + 16, img_h=96, img_w=96, max_action_len=8)
    collect_rollouts(env, agent, buffer, n_steps=n_steps, seed=seed, env_name=env_name)   # closes env
    return buffer


# ─────────────────────────────────────────────────────────────────────────────
# ROLLOUT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, acts, env_names, cams, dtype):
    """Same prompt format as training: action + env + camera state, 128 tokens."""
    prompts = compose_player_prompts(acts, env_name=env_names, camera_pose=cams)
    tokens = tokenizer(prompts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
    return text_encoder(**tokens).last_hidden_state.to(dtype=dtype)

@torch.no_grad()
def evaluate_env(model, models, vae, text_encoder, tokenizer, buffer, env_name,
                 rollout_steps, n_rollouts, eval_steps, batch_size, strip_steps):
    """
    Rolls out every model from the same GT start frames with the same logged actions / camera commands.
    Returns:
      curves[model][metric] -> list over steps of per-step means; curves["GT"]["pe"] is the GT floor
      clips[model] -> (N, T, 3, H, W) FPV clips in [0, 1]; clips["GT"] likewise
      strip -> {"GT": [frames], model: [frames]} FPV frames of the first rollout at `strip_steps`
    """
    acc = defaultdict(lambda: defaultdict(lambda: [[] for _ in range(rollout_steps)]))
    clips = defaultdict(list)
    strip = None

    # Sample rollout windows that do not cross episode boundaries
    seqs = []
    for _ in range(n_rollouts * 4):
        if len(seqs) == n_rollouts:
            break
        s = buffer.sample_sequence(seq_len=rollout_steps, env_filter=env_name, return_camera_states=True)
        if len(s[2]) == rollout_steps:
            seqs.append(s)
    if len(seqs) < n_rollouts:
        print(f"  [WARN] {env_name}: only {len(seqs)}/{n_rollouts} full-length rollout windows available")
    if not seqs:
        return None, None, None

    for b0 in range(0, len(seqs), batch_size):
        chunk = seqs[b0:b0 + batch_size]
        B = len(chunk)
        vt0 = torch.from_numpy(np.stack([c[0][0] for c in chunk])).float().to(device) / 127.5 - 1.0  # (B, 4, 3, H, W)

        state = {}
        for name, kind, adapter in models:
            init = vt0 if kind == "multi" else vt0[:, FPV_IDX:FPV_IDX + 1]
            state[name] = encode_views_to_latents(vae, init, out_dtype=model.dtype)

        frames = defaultdict(list)
        for step in range(rollout_steps):
            acts = [c[2][step] for c in chunk]
            envs = [c[3][step] or env_name for c in chunk]
            cams = [c[4][step] for c in chunk]
            embeds = encode_prompts(text_encoder, tokenizer, acts, envs, cams, model.dtype)

            gt = torch.from_numpy(np.stack([c[1][step] for c in chunk])).float().to(device) / 255.0  # (B, 4, 3, H, W)
            gt_fpv = gt[:, FPV_IDX]
            for v in pe_per_sample(gt):
                acc["GT"]["pe"][step].append(v)
            frames["GT"].append(gt_fpv.cpu())

            for name, kind, adapter in models:
                model.set_adapter(adapter)
                next_lat = predict_next_view_latents(model, state[name], embeds, steps=eval_steps)
                dec = decode_latents_to_views(vae, next_lat)                     # (B, V, 3, H, W) in [0, 1]
                state[name] = encode_views_to_latents(vae, dec * 2.0 - 1.0, out_dtype=model.dtype)

                fpv = dec[:, -1]
                for metric, vals in (("psnr", psnr_per_sample(fpv, gt_fpv)),
                                     ("ssim", ssim_per_sample(fpv, gt_fpv)),
                                     ("lpips", lpips_per_sample(fpv, gt_fpv))):
                    acc[name][metric][step].extend(vals)
                if kind == "multi":
                    acc[name]["pe"][step].extend(pe_per_sample(dec))
                    acc[name]["hull_iou"][step].extend(hull_iou_per_sample(dec, gt))
                frames[name].append(fpv.cpu())

        for key, fr in frames.items():
            clips[key].append(torch.stack(fr, dim=1))                           # (B, T, 3, H, W)
        if strip is None:
            strip = {key: [fr[s - 1][0] for s in strip_steps if s <= rollout_steps] for key, fr in frames.items()}

    curves = {key: {metric: [float(np.mean(v)) for v in per_step] for metric, per_step in metrics.items()}
              for key, metrics in acc.items()}
    clips = {key: torch.cat(c, dim=0) for key, c in clips.items()}
    return curves, clips, strip


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY, TABLES & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def summarize(curves, model_names, rollout_steps, fvd):
    """One summary dict per model from its per-step curves."""
    mid = min(16, rollout_steps) - 1
    gt_pe = curves["GT"]["pe"]
    rows = {}
    for name in model_names:
        c = curves[name]
        row = {
            "psnr@1": c["psnr"][0], "ssim@1": c["ssim"][0], "lpips@1": c["lpips"][0],
            f"psnr@{mid + 1}": c["psnr"][mid], f"psnr@{rollout_steps}": c["psnr"][-1],
            "auc_psnr": float(np.mean(c["psnr"])),
            "drift": drift_onset(c["psnr"]),
            "fvd": fvd.get(name, float("nan")),
            "gt_pe": float(np.mean(gt_pe)),
        }
        if "pe" in c:
            row.update({"pe@1": c["pe"][0], f"pe@{mid + 1}": c["pe"][mid], "pe_mean": float(np.mean(c["pe"])),
                        "hull_iou@1": c["hull_iou"][0], "hull_iou_mean": float(np.mean(c["hull_iou"]))})
        rows[name] = row
    return rows

def aggregate(summaries, rollout_steps):
    """Mean over environments. A rollout that never drifts counts as drifting at step T+1."""
    out = {}
    names = {n for s in summaries for n in s}
    for name in names:
        rows = [s[name] for s in summaries if name in s]
        keys = {k for r in rows for k in r}
        agg = {}
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if k == "drift":
                vals = [rollout_steps + 1 if v is None else v for v in vals]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        out[name] = agg
    return out

def fmt(v, d=2):
    if v is None:
        return "--"
    if isinstance(v, float) and np.isnan(v):
        return "--"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)

def write_summary_csv(path, table):
    keys = sorted({k for rows in table.values() for r in rows.values() for k in r})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "model"] + keys)
        for env, rows in table.items():
            for name, r in rows.items():
                w.writerow([env, name] + [fmt(r.get(k), 4) for k in keys])

def write_per_step_csv(path, all_curves):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "model", "metric", "step", "value"])
        for env, curves in all_curves.items():
            for name, metrics in curves.items():
                for metric, vals in metrics.items():
                    for s, v in enumerate(vals):
                        w.writerow([env, name, metric, s + 1, f"{v:.6f}"])

def write_main_tex(path, agg, model_names, rollout_steps):
    """Rows for the main table: FPV fidelity (single step), rollout stability, supporting-view consistency."""
    lines = ["% Model & PSNR@1 & SSIM@1 & LPIPS@1 & AUC-PSNR & Drift & FVD & PE & Hull IoU \\\\"]
    for name in model_names:
        r = agg.get(name, {})
        lines.append(" & ".join([
            name, fmt(r.get("psnr@1")), fmt(r.get("ssim@1"), 3), fmt(r.get("lpips@1"), 3),
            fmt(r.get("auc_psnr")), fmt(r.get("drift"), 1), fmt(r.get("fvd"), 1),
            fmt(r.get("pe_mean"), 3), fmt(r.get("hull_iou_mean"), 3),
        ]) + " \\\\")
    lines.append(f"% GT PE floor: {fmt(next(iter(agg.values()), {}).get('gt_pe'), 3)}; drift counts T+1={rollout_steps + 1} if none")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def write_per_env_tex(path, table, model_names, multi_names, rollout_steps):
    mid = min(16, rollout_steps)
    head = ["Environment"] + [f"PSNR@{mid} {n}" for n in model_names] + [f"PE@{mid} {n}" for n in multi_names] + ["GT PE"]
    lines = ["% " + " & ".join(head) + " \\\\"]
    for env, rows in table.items():
        if not rows:
            continue
        any_row = next(iter(rows.values()))
        cells = [env] + [fmt(rows.get(n, {}).get(f"psnr@{mid}")) for n in model_names]
        cells += [fmt(rows.get(n, {}).get(f"pe@{mid}"), 3) for n in multi_names]
        cells += [fmt(any_row.get("gt_pe"), 3)]
        lines.append(" & ".join(cells) + " \\\\")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def plot_curves(curves, model_names, out_path, title):
    """FPV PSNR (left) and supporting-view PE (right) versus rollout step."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
    for name in model_names:
        c = curves[name]
        steps = range(1, len(c["psnr"]) + 1)
        ax1.plot(steps, c["psnr"], marker="o", markersize=3, linewidth=2, label=name)
        if "pe" in c:
            ax2.plot(steps, c["pe"], marker="o", markersize=3, linewidth=2, label=name)
    ax1.axhline(DRIFT_DB, color="#ef4444", linestyle=":", linewidth=1.2, label=f"Drift threshold ({DRIFT_DB:.0f} dB)")
    gt = curves["GT"]["pe"]
    ax2.plot(range(1, len(gt) + 1), gt, color="black", linestyle="--", linewidth=1.2, label="Ground truth")
    ax1.set(xlabel="Rollout step", ylabel="FPV PSNR (dB)", title=f"{title}: FPV fidelity")
    ax2.set(xlabel="Rollout step", ylabel="Projection error", title=f"{title}: supporting-view consistency")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_strip(strip, steps, out_path, scale=2):
    """Rows: GT then each model; columns: FPV frames at `steps`."""
    rows = []
    for name, frames in strip.items():
        tiles = [(f.permute(1, 2, 0).float().numpy() * 255).clip(0, 255).astype(np.uint8) for f in frames]
        row = np.concatenate(tiles, axis=1)
        row = cv2.resize(row, (row.shape[1] * scale, row.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        label = np.full((24, row.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(label, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([label, row], axis=0))
    header = np.full((20, rows[0].shape[1], 3), 30, dtype=np.uint8)
    tile_w = rows[0].shape[1] // max(1, len(steps))
    for i, s in enumerate(steps):
        cv2.putText(header, f"t={s}", (i * tile_w + 6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.concatenate([header] + rows, axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    specs = model_specs(args.multi, args.single)
    base_transformer, vae, text_encoder, tokenizer = load_base_model(model_id=args.model_id)
    model, models = load_models(base_transformer, specs)
    model_names = [n for n, _, _ in models]
    multi_names = [n for n, k, _ in models if k == "multi"]
    strip_steps = [int(s) for s in args.strip_steps.split(",")]

    env_entries = build_eval_envs(args.envs, args.num_envs)
    print(f"\nEvaluating {len(models)} models on {len(env_entries)} environments "
          f"({args.n_rollouts} rollouts x {args.rollout_steps} steps, seed {args.seed})")

    table, all_curves, split_of = {}, {}, {}
    pooled_clips = {"train": defaultdict(list), "heldout": defaultdict(list)}

    for label, env_name, factory, split in env_entries:
        print(f"\n=== {label} [{split}] ===")
        buffer = collect_eval_buffer(env_name, factory, args.buffer_steps, args.seed)
        curves, clips, strip = evaluate_env(
            model, models, vae, text_encoder, tokenizer, buffer, env_name,
            rollout_steps=args.rollout_steps, n_rollouts=args.n_rollouts, eval_steps=args.eval_steps,
            batch_size=args.batch_size, strip_steps=strip_steps,
        )
        if curves is None:
            continue

        fvd = {}
        if not args.no_fvd and clips["GT"].shape[0] >= 2:
            for name in model_names:
                fvd[name] = compute_fvd(clips["GT"].float(), clips[name].float())
        for key, c in clips.items():
            pooled_clips[split][key].append(c)

        table[label] = summarize(curves, model_names, args.rollout_steps, fvd)
        all_curves[label] = curves
        split_of[label] = split

        clean = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
        plot_curves(curves, model_names, os.path.join(out_dir, f"{clean}_curves.png"), label)
        save_strip(strip, [s for s in strip_steps if s <= args.rollout_steps], os.path.join(out_dir, f"{clean}_strip.png"))

        for name in model_names:
            r = table[label][name]
            pe = f" | PE {fmt(r.get('pe_mean'), 3)} (GT {fmt(r['gt_pe'], 3)})" if "pe_mean" in r else ""
            print(f"  {name:<20} FPV PSNR@1 {fmt(r['psnr@1'])} | AUC {fmt(r['auc_psnr'])} | "
                  f"drift {r['drift'] or '>' + str(args.rollout_steps)} | FVD {fmt(r['fvd'], 1)}{pe}")

    # Aggregate rows over training / held-out environments (FVD pooled over all clips of the split)
    for split, title in (("train", "Train (mean)"), ("heldout", "Held-out (mean)")):
        envs = [e for e, s in split_of.items() if s == split]
        if not envs:
            continue
        agg = aggregate([table[e] for e in envs], args.rollout_steps)
        if not args.no_fvd:
            gt = torch.cat(pooled_clips[split]["GT"], dim=0)
            for name in model_names:
                agg[name]["fvd"] = compute_fvd(gt.float(), torch.cat(pooled_clips[split][name], dim=0).float())
        table[title] = agg
        all_curves[title] = {key: {m: list(np.mean([all_curves[e][key][m] for e in envs], axis=0))
                                   for m in all_curves[envs[0]][key]}
                             for key in all_curves[envs[0]]}
        clean = re.sub(r"[^0-9a-zA-Z]+", "_", title).strip("_").lower()
        plot_curves(all_curves[title], model_names, os.path.join(out_dir, f"{clean}_curves.png"), title)

    write_summary_csv(os.path.join(out_dir, "summary.csv"), table)
    write_per_step_csv(os.path.join(out_dir, "per_step.csv"), all_curves)
    if "Train (mean)" in table:
        write_main_tex(os.path.join(out_dir, "summary_main.tex"), table["Train (mean)"], model_names, args.rollout_steps)
    write_per_env_tex(os.path.join(out_dir, "summary_per_env.tex"),
                      {e: r for e, r in table.items() if e in split_of}, model_names, multi_names, args.rollout_steps)
    print(f"\n[SAVED] Results -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="FPV-centred long-horizon rollout evaluation of multi-view and single-view world models")
    p.add_argument("--multi", nargs="*", default=["./checkpoints/wan_ortho_voxel_cam_3_lora_final"],
                   help="Checkpoint folders of multi-view models (FPV + supporting views)")
    p.add_argument("--single", nargs="*", default=["./checkpoints/wan_single_cam_3_lora_final"],
                   help="Checkpoint folders of FPV-only models")
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--envs", type=str, default="all",
                   help="'train', 'heldout', 'all', or a comma-separated list of env labels")
    p.add_argument("--num_envs", type=int, default=12, help="Number of training envs (registry order) to evaluate")
    p.add_argument("--rollout_steps", type=int, default=32)
    p.add_argument("--n_rollouts", type=int, default=20, help="Rollouts per environment")
    p.add_argument("--batch_size", type=int, default=4, help="Rollouts evaluated in parallel")
    p.add_argument("--eval_steps", type=int, default=20, help="Euler steps per predicted frame")
    p.add_argument("--buffer_steps", type=int, default=1000, help="Evaluation transitions collected per environment")
    p.add_argument("--seed", type=int, default=69, help="Evaluation seed (training data uses 42)")
    p.add_argument("--strip_steps", type=str, default="1,8,16,24,32", help="Rollout steps shown in the frame strips")
    p.add_argument("--no_fvd", action="store_true", help="Skip FVD")
    p.add_argument("--out_root", type=str, default="results/eval")
    p.add_argument("--tag", type=str, default=None, help="Output sub-directory (default: timestamp)")
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
