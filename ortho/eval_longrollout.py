"""
eval_longrollout.py — Experiment 3.1
Long-Horizon Rollout Degradation Curve & Multi-View vs. Single-View Comparison

Evaluates trained world model checkpoints (Multi-View OWM vs. Single-View Baseline)
by autoregressively rolling out T steps and measuring:
  - Per-step PSNR degradation curves & AUC-PSNR
  - Drift onset step (threshold PSNR < 20 dB)
  - Fréchet Video Distance (FVD)
  - Cross-Projection Consistency Error (CPCE) across orthographic views
  - Occlusion Error in occluded / non-overlapping regions
  - Side-by-side visual rollouts (MP4, GIF, PNG strips)

Saves:
  - results/exp3_1/metrics.csv        <- per-step numbers for Table / LaTeX
  - results/exp3_1/degradation.png    <- the paper comparison figure
  - results/exp3_1/comparison_*.mp4   <- stitched video comparison

Usage:
  # Compare Multi-View OWM vs. Single-View Baseline:
  python eval_longrollout.py --ckpt_dir ./checkpoints/wan_ortho_lora_final \\
                             --single_ckpt_dir ./checkpoints/wan_single_lora_epoch_8 \\
                             --target_view 3D_FPV \\
                             --rollout_steps 32 --n_rollouts 50 --eval_steps 20

  # Evaluate Multi-View OWM standalone:
  python eval_longrollout.py --ckpt_dir ./checkpoints/wan_ortho_lora_final \\
                             --rollout_steps 32 --n_rollouts 50
"""

import os
import math
import argparse
import csv
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
from PIL import Image

from utils import _build_registry, load_base_model, compose_player_prompts, set_seed
from buffer import OrthoTransitionBuffer
from train_ortho_wan import (
    collect_rollouts,
    encode_views_to_latents,
    decode_latents_to_views,
    predict_next_view_latents,
    load_agent,
)
from train_ortho_diffusion import compute_fvd

device = "cuda" if torch.cuda.is_available() else "cpu"

VIEW_NAMES = ["Top", "Side", "Rear", "3D_FPV"]
VIEW_NAME_TO_INDEX = {
    "top": 0, "top_down": 0, "topdown": 0,
    "side": 1,
    "rear": 2, "back": 2,
    "3d_fpv": 3, "fpv": 3, "3d": 3,
}

def resolve_view_index(view_name_or_idx) -> int:
    """Resolves view name string or index to 0..3."""
    if isinstance(view_name_or_idx, int):
        return max(0, min(3, view_name_or_idx))
    clean_k = str(view_name_or_idx).lower().replace("-", "_").replace(" ", "_").strip()
    if clean_k not in VIEW_NAME_TO_INDEX:
        raise ValueError(f"Unknown view '{view_name_or_idx}'. Choose from {VIEW_NAMES}")
    return VIEW_NAME_TO_INDEX[clean_k]


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR for tensors in [0,1]."""
    mse = F.mse_loss(pred.float(), gt.float()).item()
    if mse <= 0:
        return 100.0
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(max(mse, 1e-10))


def compute_cross_projection_consistency(pred_views: torch.Tensor) -> float:
    """
    Cross-Projection Consistency Error (CPCE):
    Measures 3D geometric agreement across orthographic projections:
      - Top col (X) vs Side col (X)
      - Top row (Y) vs Rear col (Y)
      - Side row (Z) vs Rear row (Z)

    pred_views: (B, 4, 3, H, W) in [0, 1]
    """
    top  = pred_views[:, 0]  # (B, 3, H, W)
    side = pred_views[:, 1]
    rear = pred_views[:, 2]

    # Shared X axis: Top col profile vs Side col profile
    top_x  = top.mean(dim=2)   # (B, 3, W)
    side_x = side.mean(dim=2)  # (B, 3, W)
    err_top_side = F.mse_loss(top_x, side_x)

    # Shared Y axis: Top row profile vs Rear col profile
    top_y  = top.mean(dim=3)   # (B, 3, H)
    rear_y = rear.mean(dim=3)  # (B, 3, H)
    err_top_rear = F.mse_loss(top_y, rear_y)

    # Shared Z axis: Side row profile vs Rear row profile
    side_z = side.mean(dim=3)  # (B, 3, H)
    rear_z = rear.mean(dim=3)  # (B, 3, H)
    err_side_rear = F.mse_loss(side_z, rear_z)

    return float(((err_top_side + err_top_rear + err_side_rear) / 3.0).item())


def compute_occlusion_error(pred_target: torch.Tensor, gt_target: torch.Tensor,
                            other_views_gt: List[torch.Tensor], bg_thresh: float = 0.08) -> float:
    """
    Measures MSE specifically in occluded regions (pixels that are prominent in
    the target view but occluded or invisible in the other views).
    pred_target: (3, H, W) in [0, 1]
    gt_target:   (3, H, W) in [0, 1]
    other_views_gt: list of (3, H, W) tensors
    """
    # Foreground presence in target view (regions with color variance)
    std_target = gt_target.std(dim=0, keepdim=True)  # (1, H, W)
    fg_target = (std_target > bg_thresh).float()

    if other_views_gt:
        other_stds = [v.std(dim=0, keepdim=True) for v in other_views_gt]
        avg_other_fg = sum((s > bg_thresh).float() for s in other_stds) / float(len(other_stds))
        # Occluded: visible in target view, but masked/occluded in other perspectives
        occ_mask = fg_target * (1.0 - avg_other_fg)
        if occ_mask.sum() < 1.0:
            occ_mask = fg_target
    else:
        occ_mask = fg_target

    err = F.mse_loss(pred_target * occ_mask, gt_target * occ_mask)
    return float(err.item())


# ─────────────────────────────────────────────────────────────────────────────
# HEAD-TO-HEAD COMPARISON EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compare_multi_vs_single_rollouts(
    multi_transformer,
    single_transformer,
    vae,
    text_encoder,
    tokenizer,
    buffer: OrthoTransitionBuffer,
    target_view: str = "3D_FPV",
    rollout_steps: int = 32,
    n_rollouts: int = 50,
    eval_steps: int = 20,
    env_name: str = None,
):
    """
    Executes identical action sequences on both Multi-View and Single-View models
    starting from the exact same initial frames. Evaluates:
      - Target view PSNR for Multi-View vs Single-View
      - Multi-View 4-view mean PSNR
      - Cross-Projection Consistency Error (CPCE)
      - Occlusion Error on target view
      - Video clips for FVD
    """
    target_idx = resolve_view_index(target_view)
    target_name = VIEW_NAMES[target_idx]

    # Model helper to support PEFT adapter switching if sharing same base model
    is_adapter_mode = (single_transformer == "adapter:single_view")
    model_for_single = multi_transformer if is_adapter_mode else single_transformer

    if hasattr(multi_transformer, "eval"):
        multi_transformer.eval()
    if hasattr(model_for_single, "eval"):
        model_for_single.eval()

    def _pred_multi(latents, embeds):
        if is_adapter_mode and hasattr(multi_transformer, "set_adapter"):
            multi_transformer.set_adapter("default")
        return predict_next_view_latents(multi_transformer, latents, embeds, steps=eval_steps)

    def _pred_single(latents, embeds):
        if is_adapter_mode and hasattr(multi_transformer, "set_adapter"):
            multi_transformer.set_adapter("single_view")
        return predict_next_view_latents(model_for_single, latents, embeds, steps=eval_steps)

    # Accumulators
    mv_target_psnr_acc = defaultdict(list)
    sv_target_psnr_acc = defaultdict(list)
    mv_all_psnr_acc    = defaultdict(list)
    cpce_acc           = defaultdict(list)
    mv_occ_err_acc     = defaultdict(list)
    sv_occ_err_acc     = defaultdict(list)

    mv_target_clips, sv_target_clips, gt_target_clips = [], [], []
    best_vis_frames = None

    print("\n" + "=" * 80)
    print(f"  HEAD-TO-HEAD ROLLOUT EVALUATION: Multi-View vs Single-View ({target_name})")
    print(f"  Horizon: {rollout_steps} steps | Rollouts: {n_rollouts} | Euler steps: {eval_steps}")
    print("=" * 80)

    for r_idx in range(n_rollouts):
        vt_seq, vn_seq, act_seq, env_seq = buffer.sample_sequence(
            seq_len=rollout_steps, env_filter=env_name
        )
        if len(act_seq) < rollout_steps:
            continue

        # Initial frames at t=0
        init_all = torch.from_numpy(vt_seq[0:1]).float().to(device) / 127.5 - 1.0  # (1, 4, 3, H, W)
        init_single = init_all[:, target_idx : target_idx + 1]                    # (1, 1, 3, H, W)

        curr_latents_mv = encode_views_to_latents(vae, init_all, out_dtype=multi_transformer.dtype)
        curr_latents_sv = encode_views_to_latents(vae, init_single, out_dtype=model_for_single.dtype)

        rollout_gt_target = []
        rollout_mv_target = []
        rollout_sv_target = []

        for step_i in range(rollout_steps):
            sample_env = env_seq[step_i] or env_name or "Bipedal Walker"
            act = act_seq[step_i]

            prompt = compose_player_prompts([act], env_name=sample_env)
            text_inputs = tokenizer(
                prompt, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
            ).to(device)
            prompt_embeds = text_encoder(**text_inputs).last_hidden_state

            # Ground truth targets at step_i: (4, 3, H, W) in [0, 1]
            gt_4views = torch.from_numpy(vn_seq[step_i]).float() / 255.0
            gt_target_frame = gt_4views[target_idx]
            other_gt_frames = [gt_4views[i] for i in range(4) if i != target_idx]

            # ── 1. Multi-View Forward Step (predicts all 4 views jointly) ──
            next_latents_mv = _pred_multi(curr_latents_mv, prompt_embeds.to(dtype=multi_transformer.dtype))
            decoded_mv = decode_latents_to_views(vae, next_latents_mv)  # (1, 4, 3, H, W) in [0, 1]
            pred_mv_target = decoded_mv[0, target_idx].cpu()

            mv_target_psnr = psnr(pred_mv_target, gt_target_frame)
            mv_target_psnr_acc[step_i].append(mv_target_psnr)

            # Multi-View 4-view mean PSNR
            all_psnrs = [psnr(decoded_mv[0, v].cpu(), gt_4views[v]) for v in range(4)]
            mv_all_psnr_acc[step_i].append(float(np.mean(all_psnrs)))

            # CPCE across Top, Side, Rear
            cpce_val = compute_cross_projection_consistency(decoded_mv)
            cpce_acc[step_i].append(cpce_val)

            # Occlusion error for multi-view
            occ_mv = compute_occlusion_error(pred_mv_target, gt_target_frame, other_gt_frames)
            mv_occ_err_acc[step_i].append(occ_mv)

            # Autoregressive feedback for next step
            curr_latents_mv = encode_views_to_latents(vae, decoded_mv * 2.0 - 1.0, out_dtype=multi_transformer.dtype)

            # ── 2. Single-View Forward Step (predicts only target view) ──
            next_latents_sv = _pred_single(curr_latents_sv, prompt_embeds.to(dtype=model_for_single.dtype))
            decoded_sv = decode_latents_to_views(vae, next_latents_sv)  # (1, 1, 3, H, W) in [0, 1]
            pred_sv_target = decoded_sv[0, 0].cpu()

            sv_target_psnr = psnr(pred_sv_target, gt_target_frame)
            sv_target_psnr_acc[step_i].append(sv_target_psnr)

            # Occlusion error for single-view
            occ_sv = compute_occlusion_error(pred_sv_target, gt_target_frame, other_gt_frames)
            sv_occ_err_acc[step_i].append(occ_sv)

            # Autoregressive feedback for next step
            curr_latents_sv = encode_views_to_latents(vae, decoded_sv * 2.0 - 1.0, out_dtype=model_for_single.dtype)

            # Storage for FVD & side-by-side
            rollout_gt_target.append(gt_target_frame)
            rollout_mv_target.append(pred_mv_target)
            rollout_sv_target.append(pred_sv_target)

        mv_target_clips.append(torch.stack(rollout_mv_target, dim=0))
        sv_target_clips.append(torch.stack(rollout_sv_target, dim=0))
        gt_target_clips.append(torch.stack(rollout_gt_target, dim=0))

        if best_vis_frames is None and len(rollout_gt_target) == rollout_steps:
            best_vis_frames = (rollout_gt_target, rollout_mv_target, rollout_sv_target)

        if (r_idx + 1) % 10 == 0 or (r_idx + 1) == n_rollouts:
            mv_p = np.mean(mv_target_psnr_acc[step_i])
            sv_p = np.mean(sv_target_psnr_acc[step_i])
            print(f"  Rollout {r_idx+1:02d}/{n_rollouts} | Step-{rollout_steps} PSNR -> "
                  f"Multi-View: {mv_p:.2f} dB | Single-View: {sv_p:.2f} dB (Advantage: {mv_p - sv_p:+.2f} dB)")

    # Aggregated step-by-step metrics
    mv_mean_psnr = [float(np.mean(mv_target_psnr_acc[s])) for s in range(rollout_steps)]
    sv_mean_psnr = [float(np.mean(sv_target_psnr_acc[s])) for s in range(rollout_steps)]
    mv_4v_psnr   = [float(np.mean(mv_all_psnr_acc[s])) for s in range(rollout_steps)]
    mean_cpce    = [float(np.mean(cpce_acc[s])) for s in range(rollout_steps)]
    mean_mv_occ  = [float(np.mean(mv_occ_err_acc[s])) for s in range(rollout_steps)]
    mean_sv_occ  = [float(np.mean(sv_occ_err_acc[s])) for s in range(rollout_steps)]

    mv_clips_t = torch.stack(mv_target_clips, dim=0)  # (N, T, 3, H, W)
    sv_clips_t = torch.stack(sv_target_clips, dim=0)
    gt_clips_t = torch.stack(gt_target_clips, dim=0)

    # Reset adapters
    if is_adapter_mode and hasattr(multi_transformer, "set_adapter"):
        multi_transformer.set_adapter("default")

    results = {
        "Multi-View (OWM)": {
            "target_psnr": mv_mean_psnr,
            "mean_4view_psnr": mv_4v_psnr,
            "cpce": mean_cpce,
            "occlusion_err": mean_mv_occ,
        },
        "Single-View Baseline": {
            "target_psnr": sv_mean_psnr,
            "occlusion_err": mean_sv_occ,
        },
    }
    return results, mv_clips_t, sv_clips_t, gt_clips_t, best_vis_frames


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE SINGLE-MODEL EVALUATION (BACKWARD COMPATIBLE)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_rollout_degradation(
    transformer,
    vae,
    text_encoder,
    tokenizer,
    buffer: OrthoTransitionBuffer,
    rollout_steps: int = 32,
    n_rollouts: int = 50,
    eval_steps: int = 20,
    env_name: str = None,
):
    """
    Evaluates a single model standalone over `n_rollouts` x `rollout_steps`.
    Retained for backward compatibility with eval_ablations.py.
    """
    transformer.eval()
    step_psnr_acc = defaultdict(list)
    gen_clips_all = []
    gt_clips_all  = []

    for rollout_idx in range(n_rollouts):
        vt_seq, vn_seq, act_seq, env_seq = buffer.sample_sequence(
            seq_len=rollout_steps, env_filter=env_name
        )
        if len(act_seq) < rollout_steps:
            continue

        init_views = torch.from_numpy(vt_seq[0:1]).float().to(device) / 127.5 - 1.0
        curr_latents = encode_views_to_latents(vae, init_views, out_dtype=transformer.dtype)

        gen_frames = []
        gt_frames  = []

        for step_i in range(rollout_steps):
            sample_env = env_seq[step_i] or env_name or "Bipedal Walker"
            act = act_seq[step_i]

            prompt = compose_player_prompts([act], env_name=sample_env)
            text_inputs = tokenizer(
                prompt, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
            ).to(device)
            prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)

            next_latents = predict_next_view_latents(
                transformer=transformer,
                latents_t=curr_latents,
                prompt_embeds=prompt_embeds,
                steps=eval_steps,
            )
            decoded = decode_latents_to_views(vae, next_latents)  # (1, 4, 3, H, W) [0,1]
            gt_np = torch.from_numpy(vn_seq[step_i]).float() / 255.0

            # Per-step metrics (mean over all 4 views)
            step_psnr_views = [psnr(decoded[0, v].cpu(), gt_np[v]) for v in range(4)]
            step_psnr_acc[step_i].append(np.mean(step_psnr_views))

            gen_frames.append(decoded[0, 0].cpu())
            gt_frames.append(gt_np[0])

            next_norm = decoded * 2.0 - 1.0
            curr_latents = encode_views_to_latents(vae, next_norm, out_dtype=transformer.dtype)

        gen_clips_all.append(torch.stack(gen_frames, dim=0))
        gt_clips_all.append(torch.stack(gt_frames, dim=0))

    mean_psnr = [float(np.mean(step_psnr_acc[s])) for s in range(rollout_steps)]
    gen_clips = torch.stack(gen_clips_all, dim=0)
    gt_clips  = torch.stack(gt_clips_all, dim=0)

    transformer.train()
    return mean_psnr, gen_clips, gt_clips


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING & VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_degradation_curves(results: dict, out_path: str, rollout_steps: int, target_view: str = "3D_FPV"):
    """
    Plots PSNR degradation curves for Multi-View vs Single-View world models.
    Supports both comparison dict structure and simple ablation dict.
    """
    steps = list(range(1, rollout_steps + 1))
    colors = {
        "Multi-View (OWM)": "#0284c7",       # Sky blue
        "Single-View Baseline": "#ea580c",  # Vivid orange
        "OWM (Ours)": "#0284c7",
    }
    fallback_colors = ["#38bdf8", "#fb923c", "#818cf8", "#4ade80"]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), dpi=200)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

    for i, (name, data) in enumerate(results.items()):
        color = colors.get(name, fallback_colors[i % len(fallback_colors)])
        psnr_vals = data.get("target_psnr", data.get("psnr", []))
        if not psnr_vals:
            continue
        auc = float(np.mean(psnr_vals[:rollout_steps]))
        ax.plot(
            steps[:len(psnr_vals)],
            psnr_vals[:rollout_steps],
            label=f"{name} (AUC={auc:.1f} dB)",
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            zorder=3,
        )

    # 20 dB drift threshold
    ax.axhline(20.0, color="#ef4444", linestyle=":", linewidth=1.2, label="Drift Threshold (20 dB)")

    for i, (name, data) in enumerate(results.items()):
        color = colors.get(name, fallback_colors[i % len(fallback_colors)])
        psnr_vals = data.get("target_psnr", data.get("psnr", []))
        drift = next((s + 1 for s, v in enumerate(psnr_vals) if v < 20.0), None)
        if drift is not None:
            ax.axvline(drift, color=color, linestyle="--", alpha=0.6, linewidth=1.2)
            ax.text(drift + 0.3, 20.5, f"drift@{drift}", fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("Rollout Step $t$", fontsize=12)
    ax.set_ylabel(f"PSNR (dB) ↑ [{target_view} View]", fontsize=12)
    ax.set_title(f"Autoregressive Rollout Degradation: Multi-View vs. Single-View ({target_view})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9.5, framealpha=0.85, loc="upper right")
    ax.grid(alpha=0.25, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4 if rollout_steps <= 32 else 8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"  [SAVED] Degradation comparison plot -> {out_path}")
    plt.close()


def save_csv(results: dict, out_path: str, rollout_steps: int, fvd_multi: float = None, fvd_single: float = None):
    """
    Saves comprehensive step-by-step metrics & comparison summary table to CSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    is_comparison = ("Multi-View (OWM)" in results and "Single-View Baseline" in results)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        if is_comparison:
            mv_data = results["Multi-View (OWM)"]
            sv_data = results["Single-View Baseline"]

            mv_p = mv_data.get("target_psnr", [])
            sv_p = sv_data.get("target_psnr", [])
            mv_4v = mv_data.get("mean_4view_psnr", [0.0] * rollout_steps)
            cpce = mv_data.get("cpce", [0.0] * rollout_steps)
            mv_occ = mv_data.get("occlusion_err", [0.0] * rollout_steps)
            sv_occ = sv_data.get("occlusion_err", [0.0] * rollout_steps)

            writer.writerow([
                "step",
                "multiview_psnr",
                "singleview_psnr",
                "advantage_db",
                "multiview_4view_psnr",
                "cpce_consistency",
                "multiview_occ_err",
                "singleview_occ_err",
            ])

            for s in range(min(rollout_steps, len(mv_p), len(sv_p))):
                adv = mv_p[s] - sv_p[s]
                writer.writerow([
                    s + 1,
                    f"{mv_p[s]:.4f}",
                    f"{sv_p[s]:.4f}",
                    f"{adv:+.4f}",
                    f"{mv_4v[s]:.4f}" if s < len(mv_4v) else "",
                    f"{cpce[s]:.6f}" if s < len(cpce) else "",
                    f"{mv_occ[s]:.6f}" if s < len(mv_occ) else "",
                    f"{sv_occ[s]:.6f}" if s < len(sv_occ) else "",
                ])

            # Summary Rows
            writer.writerow([])
            auc_m = float(np.mean(mv_p)) if mv_p else 0.0
            auc_s = float(np.mean(sv_p)) if sv_p else 0.0
            writer.writerow(["AUC_PSNR", f"{auc_m:.4f}", f"{auc_s:.4f}", f"{auc_m - auc_s:+.4f}"])

            drift_m = next((s + 1 for s, v in enumerate(mv_p) if v < 20.0), f">{rollout_steps}")
            drift_s = next((s + 1 for s, v in enumerate(sv_p) if v < 20.0), f">{rollout_steps}")
            writer.writerow(["Drift_Onset_Step", str(drift_m), str(drift_s), ""])

            writer.writerow(["Final_Step_PSNR", f"{mv_p[-1]:.4f}", f"{sv_p[-1]:.4f}", f"{mv_p[-1] - sv_p[-1]:+.4f}"])

            if fvd_multi is not None and fvd_single is not None:
                writer.writerow(["FVD_Score", f"{fvd_multi:.2f}", f"{fvd_single:.2f}", f"{fvd_single - fvd_multi:+.2f}"])

            mean_cpce = float(np.mean(cpce)) if cpce else 0.0
            writer.writerow(["Mean_CPCE_Consistency", f"{mean_cpce:.6f}", "N/A", ""])

            auc_occ_m = float(np.mean(mv_occ)) if mv_occ else 0.0
            auc_occ_s = float(np.mean(sv_occ)) if sv_occ else 0.0
            writer.writerow(["Mean_Occlusion_Error", f"{auc_occ_m:.6f}", f"{auc_occ_s:.6f}", f"{auc_occ_m - auc_occ_s:+.6f}"])

        else:
            header = ["step"] + [f"{m}_{k}" for m in results for k in ["psnr"]]
            writer.writerow(header)
            for s in range(rollout_steps):
                row = [s + 1]
                for data in results.values():
                    psnr_vals = data.get("target_psnr", data.get("psnr", []))
                    row += [f"{psnr_vals[s]:.4f}" if s < len(psnr_vals) else ""]
                writer.writerow(row)

    print(f"  [SAVED] Metrics CSV -> {out_path}")


def create_comparison_visualizations(gt_frames, multi_frames, single_frames, out_dir, target_view="3D_FPV", fps=15):
    """
    Creates stitched side-by-side video and image strips:
    [ Ground Truth | Multi-View (OWM) | Single-View Baseline ]
    """
    os.makedirs(out_dir, exist_ok=True)
    T = len(gt_frames)
    if T == 0:
        return

    H, W = gt_frames[0].shape[1], gt_frames[0].shape[2]
    header_h = 28
    composite_h = H + header_h
    composite_w = W * 3

    frames_bgr = []
    frames_rgb = []

    for t in range(T):
        gt_np = (gt_frames[t].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        mv_np = (multi_frames[t].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        sv_np = (single_frames[t].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

        canvas = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
        canvas[:header_h, :] = (30, 30, 30)

        canvas[header_h:, 0:W] = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
        canvas[header_h:, W:2*W] = cv2.cvtColor(mv_np, cv2.COLOR_RGB2BGR)
        canvas[header_h:, 2*W:3*W] = cv2.cvtColor(sv_np, cv2.COLOR_RGB2BGR)

        cv2.line(canvas, (W, 0), (W, composite_h), (80, 80, 80), 1)
        cv2.line(canvas, (2 * W, 0), (2 * W, composite_h), (80, 80, 80), 1)
        cv2.line(canvas, (0, header_h), (composite_w, header_h), (100, 100, 100), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        cv2.putText(canvas, "Ground Truth", (8, 18), font, font_scale, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Multi-View (Ours)", (W + 8, 18), font, font_scale, (255, 200, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Single-View", (2 * W + 8, 18), font, font_scale, (100, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"t={t+1}", (W - 32, composite_h - 6), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        frames_bgr.append(canvas)
        frames_rgb.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    # Save MP4
    video_path = os.path.join(out_dir, f"comparison_rollout_{target_view.lower()}.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vwriter = cv2.VideoWriter(video_path, fourcc, fps, (composite_w, composite_h))
        for f in frames_bgr:
            vwriter.write(f)
        vwriter.release()
        print(f"  [SAVED] Comparison video -> {video_path}")
    except Exception as e:
        print(f"  [WARN] Could not save MP4 video: {e}")

    # Save GIF
    gif_path = os.path.join(out_dir, f"comparison_rollout_{target_view.lower()}.gif")
    try:
        pil_images = [Image.fromarray(f) for f in frames_rgb]
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"  [SAVED] Comparison GIF -> {gif_path}")
    except Exception as e:
        print(f"  [WARN] Could not save GIF: {e}")

    # Save static milestone strip
    strip_path = os.path.join(out_dir, f"comparison_strip_{target_view.lower()}.png")
    try:
        sample_indices = sorted(list(set([0, min(3, T - 1), min(7, T - 1), min(15, T - 1), min(23, T - 1), T - 1])))
        strip_frames = [frames_bgr[i] for i in sample_indices]
        full_strip = np.vstack(strip_frames)
        cv2.imwrite(strip_path, full_strip)
        print(f"  [SAVED] Comparison static strip -> {strip_path}")
    except Exception as e:
        print(f"  [WARN] Could not save strip: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    os.makedirs("results/exp3_1", exist_ok=True)

    from peft import PeftModel
    base_transformer, vae, text_encoder, tokenizer = load_base_model(model_id=args.model_id)

    # 1. Load Multi-View Model
    multi_transformer = None
    if os.path.exists(args.ckpt_dir):
        print(f"Loading Multi-View LoRA from: {args.ckpt_dir}")
        multi_transformer = PeftModel.from_pretrained(base_transformer, args.ckpt_dir)
        multi_transformer.eval()
    else:
        print(f"[WARN] Multi-View checkpoint not found at: {args.ckpt_dir}")

    # 2. Load Single-View Model (supports PEFT adapter sharing or separate model)
    single_view_transformer = None
    if args.single_ckpt_dir and os.path.exists(args.single_ckpt_dir):
        print(f"Loading Single-View LoRA from: {args.single_ckpt_dir}")
        if multi_transformer is not None:
            try:
                multi_transformer.load_adapter(args.single_ckpt_dir, adapter_name="single_view")
                single_view_transformer = "adapter:single_view"
                print("  [INFO] Loaded Single-View adapter onto shared base DiT (memory-efficient).")
            except Exception as e:
                print(f"  [INFO] Loading separate PeftModel instance for single-view ({e})")
                base_single, _, _, _ = load_base_model(model_id=args.model_id)
                single_view_transformer = PeftModel.from_pretrained(base_single, args.single_ckpt_dir)
                single_view_transformer.eval()
        else:
            single_view_transformer = PeftModel.from_pretrained(base_transformer, args.single_ckpt_dir)
            single_view_transformer.eval()
    else:
        print(f"[WARN] Single-View checkpoint not found at: {args.single_ckpt_dir}")

    # 3. Populate buffer with gym environment rollouts
    registry = _build_registry()
    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(capacity=8000, img_h=96, img_w=96, max_action_len=8)

    env_name = args.env_name or registry[0][0]
    env = reg_dict[env_name]()
    obs, _ = env.reset(seed=args.seed)
    state_dim = int(np.prod(obs.shape if hasattr(obs, "shape") else env.observation_space.shape))
    clean_name = "".join(c for c in env_name if c.isalnum() or c in ("_", "-")).lower()
    agent = load_agent(f"checkpoints/ppo_{clean_name}.pt", state_dim, env.action_space, device)
    collect_rollouts(env, agent, buffer, n_steps=args.buffer_steps, seed=args.seed, env_name=env_name)
    env.close()

    # 4. Evaluation Execution
    if multi_transformer is not None and single_view_transformer is not None:
        # ── HEAD-TO-HEAD COMPARISON ──
        results, mv_clips, sv_clips, gt_clips, best_vis = compare_multi_vs_single_rollouts(
            multi_transformer=multi_transformer,
            single_transformer=single_view_transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            buffer=buffer,
            target_view=args.target_view,
            rollout_steps=args.rollout_steps,
            n_rollouts=args.n_rollouts,
            eval_steps=args.eval_steps,
            env_name=env_name,
        )

        print("\nComputing FVD scores for video rollouts...")
        fvd_mv = compute_fvd(gt_clips.float(), mv_clips.float())
        fvd_sv = compute_fvd(gt_clips.float(), sv_clips.float())

        mv_p = results["Multi-View (OWM)"]["target_psnr"]
        sv_p = results["Single-View Baseline"]["target_psnr"]
        auc_m = float(np.mean(mv_p))
        auc_s = float(np.mean(sv_p))
        dm = next((s + 1 for s, v in enumerate(mv_p) if v < 20.0), f">{args.rollout_steps}")
        ds = next((s + 1 for s, v in enumerate(sv_p) if v < 20.0), f">{args.rollout_steps}")

        print("\n" + "=" * 80)
        print("  COMPARISON SUMMARY: MULTI-VIEW vs SINGLE-VIEW")
        print("=" * 80)
        print(f"  {'Metric':<30} | {'Multi-View (OWM)':<18} | {'Single-View':<18} | {'Advantage':<14}")
        print("  " + "-" * 76)
        print(f"  {'AUC-PSNR ↑':<30} | {auc_m:<15.2f} dB | {auc_s:<15.2f} dB | {auc_m - auc_s:+11.2f} dB")
        print(f"  {'Drift Onset (PSNR < 20dB)':<30} | {str(dm):<18} | {str(ds):<18} | {'-'}")
        print(f"  {'Final Step PSNR ↑':<30} | {mv_p[-1]:<15.2f} dB | {sv_p[-1]:<15.2f} dB | {mv_p[-1] - sv_p[-1]:+11.2f} dB")
        print(f"  {'FVD (lower is better) ↓':<30} | {fvd_mv:<18.2f} | {fvd_sv:<18.2f} | {fvd_sv - fvd_mv:+11.2f}")
        mean_cpce = float(np.mean(results["Multi-View (OWM)"]["cpce"]))
        print(f"  {'Cross-Proj. Consistency (CPCE)':<30} | {mean_cpce:<18.6f} | {'N/A':<18} | {'-'}")
        auc_occ_m = float(np.mean(results["Multi-View (OWM)"]["occlusion_err"]))
        auc_occ_s = float(np.mean(results["Single-View Baseline"]["occlusion_err"]))
        print(f"  {'Occlusion Error ↓':<30} | {auc_occ_m:<18.6f} | {auc_occ_s:<18.6f} | {auc_occ_m - auc_occ_s:+11.6f}")
        print("=" * 80 + "\n")

        plot_degradation_curves(results, "results/exp3_1/degradation.png", args.rollout_steps, target_view=args.target_view)
        save_csv(results, "results/exp3_1/metrics.csv", args.rollout_steps, fvd_multi=fvd_mv, fvd_single=fvd_sv)

        if best_vis is not None and args.save_video:
            create_comparison_visualizations(
                gt_frames=best_vis[0],
                multi_frames=best_vis[1],
                single_frames=best_vis[2],
                out_dir="results/exp3_1",
                target_view=args.target_view,
                fps=15,
            )

    else:
        # ── SINGLE MODEL EVALUATION (FALLBACK) ──
        active_model = multi_transformer if multi_transformer is not None else single_view_transformer
        if active_model is None:
            print("[ERROR] No valid checkpoint found to evaluate.")
            return

        label = "OWM (Ours)" if multi_transformer is not None else "Single-View Baseline"
        print(f"\nRunning standalone evaluation for: {label}")
        psnr_vals, gen_clips, gt_clips = evaluate_rollout_degradation(
            transformer=active_model, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            buffer=buffer, rollout_steps=args.rollout_steps,
            n_rollouts=args.n_rollouts, eval_steps=args.eval_steps,
            env_name=env_name,
        )

        print("Computing FVD ...")
        fvd_val = compute_fvd(gt_clips.float(), gen_clips.float())
        drift_step = next((s + 1 for s, v in enumerate(psnr_vals) if v < 20.0), None)
        print(f"\n  AUC-PSNR:        {np.mean(psnr_vals):.2f} dB")
        print(f"  Drift onset:     step {drift_step}")
        print(f"  Final PSNR:      {psnr_vals[-1]:.2f} dB")
        print(f"  FVD:             {fvd_val:.2f}")

        results = {label: {"psnr": psnr_vals}}
        plot_degradation_curves(results, "results/exp3_1/degradation.png", args.rollout_steps)
        save_csv(results, "results/exp3_1/metrics.csv", args.rollout_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3.1: Long-Horizon Rollout Degradation & Multi-View vs Single-View Comparison")
    parser.add_argument("--ckpt_dir",        type=str, default="./checkpoints/wan_ortho_lora_final", help="Path to Multi-View LoRA checkpoint")
    parser.add_argument("--single_ckpt_dir", type=str, default="./checkpoints/wan_single_lora_epoch_8", help="Path to Single-View LoRA checkpoint")
    parser.add_argument("--target_view",     type=str, default="3D_FPV", help="Target camera view to compare (3D_FPV, Top, Side, Rear)")
    parser.add_argument("--model_id",        type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--env_name",        type=str, default=None)
    parser.add_argument("--rollout_steps",   type=int, default=32)
    parser.add_argument("--n_rollouts",      type=int, default=50)
    parser.add_argument("--eval_steps",      type=int, default=20)
    parser.add_argument("--buffer_steps",    type=int, default=2000)
    parser.add_argument("--save_video",      action="store_true", default=True, help="Save comparison video, gif, and strip")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()
    main(args)
