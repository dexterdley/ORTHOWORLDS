"""
eval_longrollout.py — Experiment 3.1
Long-Horizon Rollout Degradation Curve

Evaluates a trained OWM checkpoint (or any model matching the transformer interface)
by autoregressively rolling out T steps and measuring per-step PSNR / FVD
against ground truth. Saves:
  - results/exp3_1/metrics.csv        <- per-step numbers for Table / LaTeX
  - results/exp3_1/degradation.png    <- the paper figure

Usage:
  python eval_longrollout.py --ckpt_dir ./checkpoints/wan_ortho_lora_final \
                             --rollout_steps 32 --n_rollouts 50 \
                             --eval_steps 20

Dependencies: all already present in the repo + matplotlib + pandas
"""

import os
import math
import argparse
import csv
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import _build_registry, load_base_model, compose_player_prompts, set_seed
from buffer import OrthoTransitionBuffer
from train_ortho_wan import (
    collect_rollouts,
    encode_views_to_latents,
    decode_latents_to_views,
    predict_next_view_latents,
    load_agent,
)
# Reuse existing metric implementations from the diffusion trainer
from train_ortho_diffusion import compute_fvd

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR for tensors in [0,1]."""
    mse = F.mse_loss(pred.float(), gt.float()).item()
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL ROLLOUT EVALUATION
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
    Runs `n_rollouts` autoregressive rollouts of length `rollout_steps`.
    Returns per-step mean PSNR and collected video clips for FVD.

    Returns:
        step_psnr  : list[float] of length rollout_steps, per-step mean PSNR
        gen_clips  : (n_rollouts, rollout_steps, 3, H, W) - top-down view for FVD
        gt_clips   : (n_rollouts, rollout_steps, 3, H, W)
    """
    transformer.eval()

    # Accumulators: step -> list of per-rollout scalars
    step_psnr_acc  = defaultdict(list)

    gen_clips_all = []
    gt_clips_all  = []

    for rollout_idx in range(n_rollouts):
        # Sample a contiguous sequence from the buffer
        vt_seq, vn_seq, act_seq, env_seq = buffer.sample_sequence(
            seq_len=rollout_steps, env_filter=env_name
        )
        if len(act_seq) < rollout_steps:
            continue  # skip short sequences

        # Encode initial frame
        init_views = torch.from_numpy(vt_seq[0:1]).float().to(device) / 127.5 - 1.0
        curr_latents = encode_views_to_latents(vae, init_views, out_dtype=transformer.dtype)

        gen_frames = []
        gt_frames  = []

        for step_i in range(rollout_steps):
            sample_env = env_seq[step_i] or env_name or "Bipedal Walker"
            act = act_seq[step_i]

            # Build text prompt
            prompt = compose_player_prompts([act], env_name=sample_env)
            text_inputs = tokenizer(
                prompt, padding="max_length", max_length=64,
                truncation=True, return_tensors="pt"
            ).to(device)
            prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)

            # Predict next frame
            next_latents = predict_next_view_latents(
                transformer=transformer,
                latents_t=curr_latents,
                prompt_embeds=prompt_embeds,
                steps=eval_steps,
            )
            decoded = decode_latents_to_views(vae, next_latents)  # (1, 4, 3, H, W) [0,1]

            # Ground truth next views: (4, 3, H, W) in [0,1]
            gt_np = torch.from_numpy(vn_seq[step_i]).float() / 255.0

            # Per-step metrics (mean over all 4 views)
            step_psnr_views  = []
            for v in range(4):
                pred_v = decoded[0, v].cpu()
                gt_v   = gt_np[v]
                step_psnr_views.append(psnr(pred_v, gt_v))
                
            step_psnr_acc[step_i].append(np.mean(step_psnr_views))

            # Collect top-down frame (view index 0) for FVD
            gen_frames.append(decoded[0, 0].cpu())
            gt_frames.append(gt_np[0])

            # Autoregressive: re-encode decoded RGB for next step
            next_norm = decoded * 2.0 - 1.0
            curr_latents = encode_views_to_latents(vae, next_norm, out_dtype=transformer.dtype)

        gen_clips_all.append(torch.stack(gen_frames, dim=0))  # (T, 3, H, W)
        gt_clips_all.append(torch.stack(gt_frames,  dim=0))

        if (rollout_idx + 1) % 10 == 0:
            print(f"  Rollout {rollout_idx+1}/{n_rollouts} | "
                  f"Step-1 PSNR: {step_psnr_acc[0][-1]:.2f} dB | "
                  f"Step-{rollout_steps} PSNR: {step_psnr_acc[rollout_steps-1][-1]:.2f} dB")

    # Aggregate
    mean_psnr  = [float(np.mean(step_psnr_acc[s]))  for s in range(rollout_steps)]

    gen_clips = torch.stack(gen_clips_all, dim=0)  # (N, T, 3, H, W)
    gt_clips  = torch.stack(gt_clips_all,  dim=0)

    transformer.train()
    return mean_psnr, gen_clips, gt_clips


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_degradation_curves(results: dict, out_path: str, rollout_steps: int):
    """
    results: { model_name: {"psnr": [...]} }
    Single-panel figure: PSNR vs rollout step.
    Vertical dashed lines mark drift onset (first step PSNR < 20 dB).
    """
    steps = list(range(1, rollout_steps + 1))
    COLORS = ["#38bdf8", "#818cf8", "#fb923c", "#4ade80"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                          "axes.spines.right": False})

    for i, (model_name, data) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(steps, data["psnr"], label=model_name,
                color=color, linewidth=2.2, marker="o", markersize=3.5, zorder=3)

    ax.set_xlabel("Rollout Step $t$", fontsize=12)
    ax.set_ylabel("PSNR (dB) ↑", fontsize=12)
    ax.set_title("PSNR vs. Rollout Step", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.4)
    ax.grid(alpha=0.25, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    # Drift onset annotations
    ax.axhline(20.0, color="gray", linestyle=":", linewidth=1.0, label="20 dB threshold")
    for i, (model_name, data) in enumerate(results.items()):
        drift = next((s+1 for s, v in enumerate(data["psnr"]) if v < 20.0), None)
        if drift:
            ax.axvline(drift, color=COLORS[i % len(COLORS)],
                       linestyle="--", alpha=0.6, linewidth=1.2)
            ax.text(drift + 0.3, 20.5, f"drift@{drift}", fontsize=7.5,
                    color=COLORS[i % len(COLORS)])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  [SAVED] Figure -> {out_path}")
    plt.close()


def save_csv(results: dict, out_path: str, rollout_steps: int):
    """Saves per-step metrics to CSV for LaTeX table generation."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["step"] + [f"{m}_{k}" for m in results for k in ["psnr"]]
        writer.writerow(header)
        for s in range(rollout_steps):
            row = [s + 1]
            for data in results.values():
                row += [f"{data['psnr'][s]:.4f}"]
            writer.writerow(row)
    print(f"  [SAVED] CSV -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    os.makedirs("results/exp3_1", exist_ok=True)

    # Load base transformer (no LoRA), then apply saved LoRA adapter on top.
    # Do NOT use load_model() here — it already calls get_peft_model(), which
    # causes PeftModel.from_pretrained() to double-wrap the model and mangle
    # adapter key paths into 'base_model.model.base_model.model...'.
    from peft import PeftModel
    transformer, vae, text_encoder, tokenizer = load_base_model(model_id=args.model_id)
    transformer = PeftModel.from_pretrained(transformer, args.ckpt_dir)
    transformer.eval()

    # Populate buffer
    registry = _build_registry()
    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(capacity=8000, img_h=128, img_w=128, max_action_len=8)

    env_name = args.env_name or registry[0][0]
    env = reg_dict[env_name]()
    obs, _ = env.reset(seed=args.seed)
    state_dim = int(np.prod(obs.shape if hasattr(obs, "shape") else env.observation_space.shape))
    clean_name = "".join(c for c in env_name if c.isalnum() or c in ("_", "-")).lower()
    agent = load_agent(f"checkpoints/ppo_{clean_name}.pt", state_dim, env.action_space, device)
    collect_rollouts(env, agent, buffer, n_steps=args.buffer_steps, seed=args.seed, env_name=env_name)
    env.close()

    # Evaluate
    print(f"\nRunning Exp 3.1: {args.n_rollouts} rollouts x {args.rollout_steps} steps [{env_name}]")
    psnr_vals, gen_clips, gt_clips = evaluate_rollout_degradation(
        transformer=transformer, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        buffer=buffer, rollout_steps=args.rollout_steps,
        n_rollouts=args.n_rollouts, eval_steps=args.eval_steps,
        env_name=env_name,
    )

    # FVD over full rollout clips
    print("Computing FVD ...")
    fvd_val = compute_fvd(gt_clips.float(), gen_clips.float())

    drift_step = next((s+1 for s, v in enumerate(psnr_vals) if v < 20.0), None)
    print(f"\n  AUC-PSNR:        {np.mean(psnr_vals):.2f} dB")
    print(f"  Drift onset:     step {drift_step}")
    print(f"  Final PSNR:      {psnr_vals[-1]:.2f} dB")
    print(f"  FVD:             {fvd_val:.2f}")

    results = {"OWM (Ours)": {"psnr": psnr_vals}}
    plot_degradation_curves(results, "results/exp3_1/degradation.png", args.rollout_steps)
    save_csv(results, "results/exp3_1/metrics.csv", args.rollout_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3.1: Long-Horizon Rollout Degradation")
    parser.add_argument("--ckpt_dir",      type=str, default="./checkpoints/wan_ortho_lora_final")
    parser.add_argument("--model_id",      type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--env_name",      type=str, default=None)
    parser.add_argument("--rollout_steps", type=int, default=32)
    parser.add_argument("--n_rollouts",    type=int, default=50)
    parser.add_argument("--eval_steps",    type=int, default=20)
    parser.add_argument("--buffer_steps",  type=int, default=2000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()
    main(args)
