"""
Flow-matching training steps and samplers shared by the ortho training scripts.
Each works with a plain or a DDP-wrapped transformer (single-GPU runs just skip the DDP-only parts).

predict_next_view_latents_sketch is the Ortho-Project sampler (adds the projector sketch frame);
it calls transformer.sketch_proj, so it needs the unwrapped model.
"""

import contextlib

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import _unwrap


def train_ortho_flow_step(
    transformer,
    latents_t,
    latents_next,
    prompt_embeds,
    shift=3.0,
    ctx_noise_max=0.15,
):
    """
    Single transition Flow-Matching step (used inside multi-step trajectory rollout).
    Applies mild context noise to `latents_t` to prevent autoregressive drift,
    and optionally returns the model's reconstructed clean latent `pred_x0` at t+1.
    """
    b, c, num_views, h, w = latents_next.shape

    # 1. Sample shifted logit-normal flow-matching timestep tau in (0, 1)
    u = torch.randn(b, device=latents_next.device, dtype=torch.float32)
    tau = torch.sigmoid(u)
    tau = (shift * tau) / (1.0 + (shift - 1.0) * tau)
    tau = tau.clamp(1e-4, 1.0 - 1e-4)
    tau_exp = tau.view(b, 1, 1, 1, 1)

    # 2. Context noise augmentation on conditioning views `latents_t`
    latents_t_fp32 = latents_t.float()
    tau_ctx = torch.rand((b, 1, 1, 1, 1), device=latents_t.device, dtype=torch.float32) * ctx_noise_max
    ctx_noise = torch.randn_like(latents_t_fp32)
    cond_latents_t = ((1.0 - tau_ctx) * latents_t_fp32 + tau_ctx * ctx_noise).to(dtype=_unwrap(transformer).dtype)

    # 3. Noise all 4 target views at t+1 simultaneously
    latents_next_fp32 = latents_next.float()
    noise = torch.randn_like(latents_next_fp32)
    noisy_latents_next_fp32 = (1.0 - tau_exp) * latents_next_fp32 + tau_exp * noise
    noisy_latents_next = noisy_latents_next_fp32.to(dtype=_unwrap(transformer).dtype)

    # 4. Concatenate conditioning views (frames 0:4) and noisy target views (frames 4:8)
    combined_latents = torch.cat([cond_latents_t, noisy_latents_next], dim=2)

    # 5. Ground truth velocity: v_target = noise - x_0
    target_velocity = noise - latents_next_fp32

    # 6. DiT Forward pass
    timesteps = (tau * 1000.0).to(dtype=_unwrap(transformer).dtype)
    model_pred = transformer(
        hidden_states=combined_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
    ).sample

    # 7. Supervise all 4 target views at t+1 (indices 4:8)
    pred_velocity = model_pred[:, :, num_views : 2 * num_views].float()

    per_view_losses = [
        F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
        for idx in range(num_views)
    ]
    loss = sum(per_view_losses)

    # Reconstruct 1-step denoised prediction: x_0_hat = x_tau - tau * v_pred
    with torch.no_grad():
        pred_x0 = (noisy_latents_next_fp32 - tau_exp * pred_velocity.detach()).clamp(-6.0, 6.0)
        #confidence = (1.0 - tau_exp).clamp(0.2, 1.0)
        #rollout_latent = (confidence * pred_x0 + (1.0 - confidence) * latents_next_fp32).to(dtype=transformer.dtype)
        rollout_latent = pred_x0.to(dtype=_unwrap(transformer).dtype)
    return loss, [l.item() for l in per_view_losses], rollout_latent


def train_ortho_trajectory_step(
    transformer,
    latents_t0,
    latents_next_seq,
    prompt_embeds_seq,
    shift=3.0,
    teacher_forcing_prob=0.5,
    ctx_noise_max=0.15,
    sf_steps=16,
):
    """
    Unrolls a multi-step trajectory of length `seq_len` during training.
    - At step s=0, conditions on ground-truth `latents_t0`.
    - At step s>0, with probability `teacher_forcing_prob`, conditions on the model's own predicted
      latent from step s-1 (student forcing) so the model learns to recover from its
      own rollout errors over multi-step horizons.
    - Backpropagates `(loss_s / seq_len)` at each step so peak VRAM remains constant.
    - sf_steps: Euler steps used to sample the self-forced latents.
    - Under DDP, gradients accumulate locally and are all-reduced only on the final step's backward.
    """
    seq_len = len(latents_next_seq)
    num_views = latents_t0.shape[2]

    curr_latents = latents_t0
    total_loss = 0.0
    avg_per_view = [0.0] * num_views

    for s in range(seq_len):
        target_latents = latents_next_seq[s]
        prompt_embeds = prompt_embeds_seq[s]
        is_last_step = (s == seq_len - 1)

        # Skip the gradient all-reduce until the last step of the trajectory (forward and backward must both be inside no_sync)
        sync_ctx = transformer.no_sync() if (isinstance(transformer, DDP) and not is_last_step) else contextlib.nullcontext()
        with sync_ctx:
            step_loss, step_view_losses, pred_next_latents = train_ortho_flow_step(
                transformer=transformer,
                latents_t=curr_latents,
                latents_next=target_latents,
                prompt_embeds=prompt_embeds,
                shift=shift,
                ctx_noise_max=ctx_noise_max,
            )

            # Normalize loss across trajectory length and backpropagate immediately to save VRAM
            scaled_loss = step_loss / float(seq_len)
            scaled_loss.backward()

        total_loss += step_loss.item() / float(seq_len)
        for v_idx in range(num_views):
            avg_per_view[v_idx] += step_view_losses[v_idx] / float(seq_len)

        # Prepare context for step s + 1 (Student Forcing vs. Teacher Forcing per sample)
        if not is_last_step:
            if teacher_forcing_prob > 0.0:
                b = curr_latents.shape[0]
                use_ar = torch.rand(b, device=curr_latents.device) < teacher_forcing_prob   # (B,) which samples self-predict
                next_latents = target_latents.clone()
                
                if use_ar.any():
                    with torch.no_grad():
                        next_latents[use_ar] = predict_next_view_latents(
                            _unwrap(transformer), curr_latents[use_ar], prompt_embeds[use_ar], steps=sf_steps)
                curr_latents = next_latents
            else:
                curr_latents = target_latents

    return total_loss, avg_per_view


@torch.no_grad()
def predict_next_view_latents(transformer, latents_t, prompt_embeds, steps=20, shift=3.0):
    """
    Simultaneously denoises all 4 orthographic views at t+1 via reverse Euler Flow Matching.
    Input:
      latents_t: (B, C, 4, h, w) - clean past views at time t
    Output:
      latents_next: (B, C, 4, h, w) - predicted views at time t+1
    """
    b, c, num_views, h, w = latents_t.shape
    latents_t = latents_t.to(dtype=transformer.dtype)

    # Keep Euler state in float32 to prevent bfloat16 rounding stagnation
    x_tau = torch.randn((b, c, num_views, h, w), device=latents_t.device, dtype=torch.float32)

    # Shifted timestep schedule from 1.0 -> 0.0
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=latents_t.device, dtype=torch.float32)
    sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)

    for step_i in range(steps):
        sigma_curr = sigmas[step_i]
        sigma_next = sigmas[step_i + 1]
        dt = sigma_curr - sigma_next

        t_tensor = torch.full((b,), sigma_curr * 1000.0, device=latents_t.device, dtype=transformer.dtype)
        combined = torch.cat([latents_t, x_tau.to(dtype=transformer.dtype)], dim=2)  # (B, C, 8, h, w)

        pred_velocity = transformer(
            hidden_states=combined,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, num_views : 2 * num_views].float()  # (B, C, 4, h, w)
        x_tau = x_tau - dt * pred_velocity

    return x_tau.to(dtype=transformer.dtype)


@torch.no_grad()
def predict_next_view_latents_sketch(transformer, latents_t, prompt_embeds, cam, steps=20, shift=3.0):
    """
    Simultaneously denoises all 4 orthographic views at t+1 via reverse Euler Flow Matching.
    Input:
      latents_t: (B, C, 4, h, w) - clean past views at time t
    Output:
      latents_next: (B, C, 4, h, w) - predicted views at time t+1
    """
    b, c, num_views, h, w = latents_t.shape
    latents_t = latents_t.to(dtype=transformer.dtype)
    sketch_t = transformer.sketch_proj(latents_t[:, :, :3], cam).to(dtype=transformer.dtype)

    # Keep Euler state in float32 to prevent bfloat16 rounding stagnation; FPV starts from sketch + noise
    x_tau = torch.randn((b, c, num_views, h, w), device=latents_t.device, dtype=torch.float32)
    x_tau[:, :, 3:4] += sketch_t.float()

    # Shifted timestep schedule from 1.0 -> 0.0
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=latents_t.device, dtype=torch.float32)
    sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)

    for step_i in range(steps):
        sigma_curr = sigmas[step_i]
        sigma_next = sigmas[step_i + 1]
        dt = sigma_curr - sigma_next

        t_tensor = torch.full((b,), sigma_curr * 1000.0, device=latents_t.device, dtype=transformer.dtype)
        combined = torch.cat([latents_t, sketch_t, x_tau.to(dtype=transformer.dtype)], dim=2)  # (B, C, 9, h, w)

        pred_velocity = transformer(
            hidden_states=combined,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample[:, :, num_views + 1 :].float()  # (B, C, 4, h, w)
        x_tau = x_tau - dt * pred_velocity

    return x_tau.to(dtype=transformer.dtype)
