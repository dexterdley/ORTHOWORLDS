"""
eval_ablations.py — Experiment 6.1
Architecture Ablation Study

Trains/loads multiple model variants with one design choice modified at a time,
then evaluates each on the rollout degradation benchmark and collects results
into a single LaTeX-ready table.

The ablation conditions (matching CVPR Exp 6.1):
  A  ctx_noise_max = 0.0       (no context noise)
  B  ar_prob = 0.0             (teacher-forcing only, no student forcing)
  C  train_seq_len = 1         (single-step, no trajectory unrolling)
  D  shift = 1.0               (uniform tau, no logit-normal shift)
  E  timesteps in bfloat16     (float32 -> bfloat16 cast, reproduces the known bug)
  F  OWM Full (Ours)           (all components active, the default)

Usage (trains all ablations from scratch — use pretrained checkpoints for speed):
  python eval_ablations.py --base_ckpt_dir ./checkpoints/wan_ortho_lora_final \
                            --rollout_steps 16 --n_rollouts 30 --train_ablations

Shortcut (eval only, use existing checkpoints named by ablation key):
  python eval_ablations.py --eval_only --rollout_steps 16 --n_rollouts 30
"""

import os
import copy
import math
import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import numpy as np

from utils import _build_registry, load_model, set_seed
from buffer import OrthoTransitionBuffer
from train_ortho_wan import (
    collect_rollouts,
    load_agent,
    train_ortho_trajectory_step,
    encode_views_to_latents,
    decode_latents_to_views,
    predict_next_view_latents,
    compose_player_prompts,
)
from eval_longrollout import (
    evaluate_rollout_degradation,
    plot_degradation_curves,
    save_csv,
)
from train_ortho_diffusion import compute_fvd

device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION CONFIG DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    name: str               # Short label for the table
    label: str              # Display name in plots/table
    # Training hyperparams (one modified per ablation)
    ctx_noise_max: float = 0.15
    ar_prob:       float = 0.5
    train_seq_len: int   = 16
    shift:         float = 3.0
    bfloat16_timesteps: bool = False  # if True, reproduces the dtype bug
    # Training budget (keep small for ablations)
    epochs:         int = 5
    steps_per_epoch: int = 100
    lr:             float = 1e-4
    lora_rank:      int = 32


ABLATION_CONFIGS = [
    AblationConfig("full",          "OWM Full (Ours)",          ctx_noise_max=0.15, ar_prob=0.5,  train_seq_len=16, shift=3.0),
    AblationConfig("no_ctx_noise",  "No Context Noise",         ctx_noise_max=0.0,  ar_prob=0.5,  train_seq_len=16, shift=3.0),
    AblationConfig("no_ar",         "Teacher-Forcing Only",     ctx_noise_max=0.15, ar_prob=0.0,  train_seq_len=16, shift=3.0),
    AblationConfig("seq1",          "Single-Step (seq=1)",      ctx_noise_max=0.15, ar_prob=0.5,  train_seq_len=1,  shift=3.0),
    AblationConfig("no_shift",      "Uniform Tau (shift=1)",    ctx_noise_max=0.15, ar_prob=0.5,  train_seq_len=16, shift=1.0),
    AblationConfig("bf16_timestep", "bf16 Timesteps (bug)",     ctx_noise_max=0.15, ar_prob=0.5,  train_seq_len=16, shift=3.0,
                   bfloat16_timesteps=True),
]


# ─────────────────────────────────────────────────────────────────────────────
# PATCHED train_ortho_flow_step FOR ABLATIONS THAT MODIFY INTERNAL BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

def make_flow_step_fn(cfg: AblationConfig):
    """
    Returns a drop-in replacement for train_ortho_flow_step with the ablation
    config baked in. Only ctx_noise_max, shift, and bfloat16_timesteps are
    affected here; ar_prob and train_seq_len are passed at the trajectory level.
    """
    import torch.nn.functional as F

    def _flow_step(transformer, latents_t, latents_next, prompt_embeds,
                   shift=cfg.shift, ctx_noise_max=cfg.ctx_noise_max,
                   return_pred_x0=False):

        b, c, num_views, h, w = latents_next.shape

        # 1. Shifted logit-normal tau
        u = torch.randn(b, device=latents_next.device, dtype=torch.float32)
        tau = torch.sigmoid(u)
        tau = (shift * tau) / (1.0 + (shift - 1.0) * tau)
        tau = tau.clamp(1e-4, 1.0 - 1e-4)
        tau_exp = tau.view(b, 1, 1, 1, 1)

        # 2. Context noise (ablation: ctx_noise_max=0 skips noise)
        latents_t_fp32 = latents_t.float()
        tau_ctx = torch.rand((b, 1, 1, 1, 1), device=latents_t.device, dtype=torch.float32) * ctx_noise_max
        ctx_noise = torch.randn_like(latents_t_fp32)
        cond_latents_t = ((1.0 - tau_ctx) * latents_t_fp32 + tau_ctx * ctx_noise).to(dtype=transformer.dtype)

        # 3. Noise target views
        latents_next_fp32 = latents_next.float()
        noise = torch.randn_like(latents_next_fp32)
        noisy_next_fp32 = (1.0 - tau_exp) * latents_next_fp32 + tau_exp * noise
        noisy_next = noisy_next_fp32.to(dtype=transformer.dtype)

        # 4. Combine
        combined = torch.cat([cond_latents_t, noisy_next], dim=2)

        # 5. Target velocity
        target_velocity = noise - latents_next_fp32

        # 6. Forward — ablation: optionally cast timesteps to bfloat16 (bug)
        ts_dtype = transformer.dtype if cfg.bfloat16_timesteps else torch.float32
        timesteps = (tau * 1000.0).to(dtype=ts_dtype)

        model_pred = transformer(
            hidden_states=combined,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample

        # 7. Loss on target frames only
        pred_velocity = model_pred[:, :, 4:8].float()
        per_view_losses = [
            F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
            for idx in range(num_views)
        ]
        loss = sum(per_view_losses) / float(num_views)

        if return_pred_x0:
            with torch.no_grad():
                pred_x0 = (noisy_next_fp32 - tau_exp * pred_velocity.detach()).clamp(-6.0, 6.0)
                confidence = (1.0 - tau_exp).clamp(0.2, 1.0)
                rollout_latent = (confidence * pred_x0 + (1.0 - confidence) * latents_next_fp32).to(dtype=transformer.dtype)
            return loss, [l.item() for l in per_view_losses], rollout_latent

        return loss, [l.item() for l in per_view_losses]

    return _flow_step


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE ABLATION
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation(cfg: AblationConfig, buffer, model_id, ckpt_out_dir):
    """Fine-tunes a fresh LoRA for `cfg.epochs * cfg.steps_per_epoch` steps."""
    import torch.optim as optim
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print(f"  Training ablation: [{cfg.label}]")
    print(f"  ctx_noise={cfg.ctx_noise_max}, ar_prob={cfg.ar_prob}, "
          f"seq_len={cfg.train_seq_len}, shift={cfg.shift}")
    print(f"{'='*60}")

    transformer, vae, text_encoder, tokenizer = load_model(
        model_id=model_id, lora_rank=cfg.lora_rank
    )
    transformer.train()

    flow_step_fn = make_flow_step_fn(cfg)

    # Monkey-patch train_ortho_trajectory_step to use ablated flow step
    import train_ortho_wan as _wan_module
    _orig_flow_step = _wan_module.train_ortho_flow_step
    _wan_module.train_ortho_flow_step = flow_step_fn  # swap in ablated version

    optimizer = optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4
    )

    global_step = 0
    for epoch in range(cfg.epochs):
        transformer.train()
        pbar = tqdm(range(cfg.steps_per_epoch), desc=f"  Epoch {epoch+1}/{cfg.epochs}")

        for _ in pbar:
            views_t, views_next_seq, actions_seq, env_names_seq = buffer.sample_trajectory_batch(
                batch_size=4, seq_len=cfg.train_seq_len  # small batch for ablation speed
            )

            with torch.no_grad():
                latents_t = encode_views_to_latents(vae, views_t, out_dtype=transformer.dtype)
                latents_next_seq, prompt_embeds_seq = [], []
                for s in range(cfg.train_seq_len):
                    latents_next_seq.append(encode_views_to_latents(vae, views_next_seq[s], out_dtype=transformer.dtype))
                    prompts_s = compose_player_prompts(actions_seq[s], env_name=env_names_seq[s])
                    txt = tokenizer(prompts_s, padding="max_length", max_length=64,
                                    truncation=True, return_tensors="pt").to(device)
                    prompt_embeds_seq.append(text_encoder(**txt).last_hidden_state.to(dtype=transformer.dtype))

            optimizer.zero_grad(set_to_none=True)

            loss_val, _ = _wan_module.train_ortho_trajectory_step(
                transformer=transformer,
                latents_t0=latents_t,
                latents_next_seq=latents_next_seq,
                prompt_embeds_seq=prompt_embeds_seq,
                ar_prob=cfg.ar_prob,
                ctx_noise_max=cfg.ctx_noise_max,
                shift=cfg.shift,
            )

            torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    # Restore original flow step fn
    _wan_module.train_ortho_flow_step = _orig_flow_step

    # Save checkpoint
    save_path = os.path.join(ckpt_out_dir, cfg.name)
    os.makedirs(save_path, exist_ok=True)
    transformer.save_pretrained(save_path)
    print(f"  [SAVED] Ablation checkpoint -> {save_path}")

    return transformer, vae, text_encoder, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# PRINT LATEX TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_table(table_data: list[dict], rollout_steps: int):
    """
    table_data: list of dicts with keys: label, psnr_1, psnr_4, psnr_8, psnr_16,
                lpips_1, lpips_8, lpips_16, auc_psnr, drift_step, fvd
    """
    def fmt(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None else "--"

    header = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Architecture Ablation: Per-Step Reconstruction Quality}" "\n"
        r"\label{tab:ablation}" "\n"
        r"\begin{tabular}{lcccccc}" "\n"
        r"\toprule" "\n"
        r"Model & PSNR@1$\uparrow$ & PSNR@8$\uparrow$ & PSNR@16$\uparrow$ & "
        r"LPIPS@16$\downarrow$ & AUC-PSNR$\uparrow$ & FVD$\downarrow$ \\" "\n"
        r"\midrule"
    )
    print(header)
    for row in table_data:
        line = (f"  {row['label']} & {fmt(row['psnr_1'])} & {fmt(row['psnr_8'])} & "
                f"{fmt(row['psnr_16'])} & {fmt(row['lpips_16'], 4)} & "
                f"{fmt(row['auc_psnr'])} & {fmt(row['fvd'], 1)} \\\\")
        print(line)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    os.makedirs("results/exp6_1", exist_ok=True)

    # ── Populate buffer once ────────────────────────────────────────────────
    registry = _build_registry()
    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(capacity=6000, img_h=128, img_w=128, max_action_len=8)

    env_name = args.env_name or registry[0][0]
    env = reg_dict[env_name]()
    obs, _ = env.reset(seed=args.seed)
    state_dim = int(np.prod(obs.shape if hasattr(obs, "shape") else env.observation_space.shape))
    clean_name = "".join(c for c in env_name if c.isalnum() or c in ("_", "-")).lower()
    agent = load_agent(f"checkpoints/ppo_{clean_name}.pt", state_dim, env.action_space, device)
    collect_rollouts(env, agent, buffer, n_steps=args.buffer_steps, seed=args.seed, env_name=env_name)
    env.close()

    # ── Run each ablation ───────────────────────────────────────────────────
    all_results   = {}   # name -> {psnr: [...], lpips: [...]}
    table_data    = []

    for cfg in ABLATION_CONFIGS:
        ckpt_path = os.path.join(args.ablation_ckpt_root, cfg.name)
        ckpt_exists = os.path.isdir(ckpt_path)

        if args.eval_only and not ckpt_exists:
            print(f"  [SKIP] No checkpoint found for [{cfg.name}] at {ckpt_path}")
            continue

        if not args.eval_only or not ckpt_exists:
            transformer, vae, text_encoder, tokenizer = train_ablation(
                cfg, buffer, args.model_id, args.ablation_ckpt_root
            )
        else:
            # Load existing checkpoint
            print(f"\n  [LOAD] Ablation checkpoint: {ckpt_path}")
            transformer, vae, text_encoder, tokenizer = load_model(
                model_id=args.model_id, lora_rank=cfg.lora_rank
            )
            from peft import PeftModel
            transformer = PeftModel.from_pretrained(transformer, ckpt_path)
            transformer.eval()

        # Evaluate rollout degradation for this ablation
        print(f"\n  [EVAL] {cfg.label} — {args.n_rollouts} rollouts x {args.rollout_steps} steps")
        psnr_vals, lpips_vals, gen_clips, gt_clips = evaluate_rollout_degradation(
            transformer=transformer, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            buffer=buffer, rollout_steps=args.rollout_steps,
            n_rollouts=args.n_rollouts, eval_steps=args.eval_steps,
            env_name=env_name,
        )

        fvd_val = compute_fvd(gt_clips.float(), gen_clips.float())
        drift   = next((s+1 for s, v in enumerate(psnr_vals) if v < 20.0), None)
        auc     = float(np.mean(psnr_vals))

        all_results[cfg.label] = {"psnr": psnr_vals, "lpips": lpips_vals}

        # Collect table row
        def _at(vals, step):
            idx = min(step - 1, len(vals) - 1)
            return vals[idx]

        table_data.append({
            "label":    cfg.label,
            "psnr_1":   _at(psnr_vals, 1),
            "psnr_8":   _at(psnr_vals, 8),
            "psnr_16":  _at(psnr_vals, 16),
            "lpips_16": _at(lpips_vals, 16),
            "auc_psnr": auc,
            "drift_step": drift,
            "fvd":      fvd_val,
        })

        # Incremental save
        with open("results/exp6_1/table_data.json", "w") as f:
            json.dump(table_data, f, indent=2)

        del transformer
        torch.cuda.empty_cache()

    # ── Plot & print ────────────────────────────────────────────────────────
    if all_results:
        plot_degradation_curves(
            all_results,
            "results/exp6_1/ablation_degradation.png",
            args.rollout_steps
        )
        save_csv(all_results, "results/exp6_1/ablation_metrics.csv", args.rollout_steps)

    print("\n\n" + "="*60)
    print("  LATEX TABLE (Exp 6.1)")
    print("="*60)
    print_latex_table(table_data, args.rollout_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 6.1: Architecture Ablation Study")
    parser.add_argument("--model_id",           type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--ablation_ckpt_root", type=str, default="./checkpoints/ablations")
    parser.add_argument("--env_name",           type=str, default=None)
    parser.add_argument("--rollout_steps",      type=int, default=16)
    parser.add_argument("--n_rollouts",         type=int, default=30)
    parser.add_argument("--eval_steps",         type=int, default=20)
    parser.add_argument("--buffer_steps",       type=int, default=2000)
    parser.add_argument("--eval_only",          action="store_true",
                        help="Skip training — load existing ablation checkpoints")
    parser.add_argument("--seed",               type=int, default=42)
    args = parser.parse_args()
    main(args)
