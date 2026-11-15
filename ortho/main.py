import os, sys
import argparse
import torch, numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from buffer import OrthoTransitionBuffer
from models import LinearProjector, _get_wan_latent_stats
from data_rollout import get_registries, collect_all_envs_rollouts
from models import encode_views_to_latents, decode_latents_to_views
from flow import train_ortho_flow_step, train_ortho_trajectory_step, predict_next_view_latents
from utils import _build_registry, set_seed, _psnr, load_model, compose_player_prompts

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_H, IMG_W = 96, 96

def add_diffusion_noise(latents, tau):
    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - tau) * latents + tau * noise
    return noise, noisy_latents

@torch.no_grad()
def validate_fpv(model, vae, val_clip, val_embeds, steps, shift):
    """
    Autoregressive rollout: Model only sees its own predictions after frame 0
    """
    model.eval()
    views_t, views_next_seq = val_clip[0], val_clip[1]
    curr_latents = encode_views_to_latents(vae, views_t[:, -1:], out_dtype=model.dtype)
    psnrs, gt_frames, pred_frames = [], [], []
    for s, emb in enumerate(val_embeds):
        curr_latents = predict_next_view_latents(model, curr_latents, emb, steps=steps, shift=shift)
        
        pred = decode_latents_to_views(vae, curr_latents)[:, 0].float() #images
        gt = (views_next_seq[s][:, -1].float() + 1.0) / 2.0
        
        psnrs.append(_psnr(pred, gt))
        pred_frames.append(pred)
        gt_frames.append(gt)
    return psnrs, torch.stack(gt_frames, dim=1).cpu(), torch.stack(pred_frames, dim=1).cpu()


def main(args):
    set_seed(args.seed)

    # 1. Build Envs and Buffer
    if args.hifi:
     registry, _ = get_registries(args.hifi)
    else:
        registry = _build_registry()

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
    

    # 2. Instantiate Model and LoRA
    model, vae, text_encoder, tokenizer = load_model(model_id=args.model_id, lora_rank=args.lora_rank)
    #import utils; utils.device = "cpu"   # load_model() leaves everything on CPU (UMT5 is ~11 GB)
    # model.to(device); vae.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    
    log_dir = f"./runs/main_debug"
    ckpt_dir = f"./checkpoints/main_debug"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    print(f"TensorBoard logging to: {log_dir}")

    val_envs = ["Bipedal Walker"]

    # Validation, sample from buffer
    val_clip = buffer.sample_trajectory_batch(batch_size=1, seq_len=args.rollout_video_steps)
    val_embeds = []
    for s in range(args.rollout_video_steps):
        val_prompts = compose_player_prompts(val_clip[2][s], env_name=val_clip[3][s], camera_pose=val_clip[4][s])
        val_tokens = tokenizer(val_prompts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(text_encoder.device)
        with torch.no_grad():
            val_embeds.append(text_encoder(**val_tokens).last_hidden_state.to(device=device, dtype=model.dtype))

    # 3. Main Diffusion Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress = epoch / max(1, args.epochs - 1)
        curr_self_forcing_prob = args.self_forcing_prob * min(1.0, progress * 1.5)

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} (Self-pred Prob.={curr_self_forcing_prob:.2f})")

        for _ in pbar:
            views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq = buffer.sample_trajectory_batch(
                    batch_size=args.batch_size,
                    seq_len=args.train_seq_len,
                )
            
            # Take only the main FPV
            views_t = views_t[:,-1 : :]
            views_next_seq = [v[:,-1 : :] for v in views_next_seq]
            
            total_loss = 0.0
            avg_per_view = [0.0] * views_t.shape[1]
            curr_latents = encode_views_to_latents(vae, views_t, out_dtype=model.dtype)

            # So pass the images into VAE to extract their latents, then build prompt
            for s in range(args.train_seq_len): # Unroll each sample in the training sequence

                target_latents = encode_views_to_latents(vae, views_next_seq[s], out_dtype=model.dtype)
                b, c, num_views, h, w = target_latents.shape

                # Build prompt then tokenize
                prompts_s = compose_player_prompts(
                            actions_seq[s],
                            env_name=env_names_seq[s],
                            camera_pose=camera_poses_seq[s],
                        )

                text_inputs = tokenizer( prompts_s, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(text_encoder.device)
                prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(device=device, dtype=model.dtype)
                is_last_step = (s == args.train_seq_len - 1)

                u = torch.sigmoid(torch.randn(b, 1, 1, 1, 1, device=device))
                tau = args.shift * u / (1 + (args.shift - 1) * u)
                ctx_lvl = torch.rand((b, 1, 1, 1, 1), device=device, dtype=torch.float32) * args.ctx_noise_max

                # Add diffusion noise
                _, cond_latents_t = add_diffusion_noise(curr_latents, ctx_lvl)
                noise, noisy_target_latents = add_diffusion_noise(target_latents, tau)

                combined_latents = torch.cat([cond_latents_t, noisy_target_latents], dim=2).to(model.dtype)

                # 5. Compute GT velocity: v_target = noise - x_0
                target_velocity = noise - target_latents

                # 6. Forward pass
                timesteps = (tau.flatten() * 1000.0).to(dtype=model.dtype)
                model_pred = model(
                    hidden_states=combined_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                pred_velocity = model_pred[:, :, num_views : 2 * num_views]

                per_view_losses = [
                    F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
                    for idx in range(num_views)
                ]
                loss = sum(per_view_losses)
                scaled_loss = loss / float(args.train_seq_len)
                scaled_loss.backward()
                total_loss += scaled_loss.item()
                
                for v_idx in range(num_views):
                    avg_per_view[v_idx] += per_view_losses[v_idx] / float(args.train_seq_len)

                # 7. Self-forcing step, basically run inference during training and feed its own pred autoregressively
                if not is_last_step:
                    sf_flag = torch.rand(b, device=curr_latents.device) < curr_self_forcing_prob
                    pred_latents = target_latents.clone()
                    if sf_flag.any():
                        with torch.no_grad():
                            pred_latents[sf_flag] = predict_next_view_latents(
                                model, curr_latents[sf_flag], prompt_embeds[sf_flag], steps=args.diffusion_steps, shift=args.shift)
                    curr_latents = pred_latents

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            env_summary = "+".join(list(dict.fromkeys(env_names_seq[0]))[:2])
            pbar.set_postfix({
                    "Loss": f"{total_loss:.3f}",
                    "FPV": f"{per_view_losses[-1]:.3f}",
                    "Envs": env_summary,
                })

            # Logging
            global_step += 1
            writer.add_scalar("Train/Loss", total_loss, global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

        # Validation need to loop across VAL_ENVS
        if (epoch + 1) % args.eval_every == 0:
            env_name = val_envs[0]
            clean_tag = env_name.replace(" ", "_").lower()

            psnrs, gt_video, pred_video = validate_fpv(model, vae, val_clip, val_embeds, steps=args.diffusion_steps, shift=args.shift)
            writer.add_scalar("Train/SelfForcingProb", curr_self_forcing_prob, epoch + 1)
            writer.add_scalar(f"Validation/{clean_tag}/Avg_PSNR", sum(psnrs) / len(psnrs), epoch + 1)
            writer.add_video(f"Validation/{clean_tag}/FPV", pred_video, epoch + 1, fps=args.video_fps)
            writer.add_video(f"GT/{clean_tag}/FPV", gt_video, epoch + 1, fps=args.video_fps)
            model.save_pretrained(os.path.join(ckpt_dir, f"epoch_{epoch+1}"))

    print("Ending Training")
    model.save_pretrained(os.path.join(ckpt_dir, "final"))
    writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA across multiple Single Game Views (DDP)")
    parser.add_argument("--rollout_length", type=int, default=4096, help="PPO Rollout steps per environment, collected independently on each rank")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size (effective batch = batch_size * world_size)")
    parser.add_argument("--train_seq_len", type=int, default=16, help="Contiguous trajectory steps unrolled per training update")
    parser.add_argument("--self_forcing_prob", type=float, default=0.5, help="Max prob of using self-predicted latents onto next trajectory step")
    parser.add_argument("--ctx_noise_max", type=float, default=0.15, help="Max Gaussian noise added to conditioning context latents to prevent drift")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--eval_every", type=int, default=1, help="Generate and evaluate views every N epochs")
    parser.add_argument("--diffusion_steps", type=int, default=8, help="Number of Euler flow matching sampling steps per frame")
    parser.add_argument("--rollout_video_steps", type=int, default=16, help="Autoregressive rollout steps for validation videos")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for TensorBoard video logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of training envs")
    parser.add_argument("--hifi", action="store_true", default=False, help="Render with the Godot hi-fi envs")
    parser.add_argument("--shift", type=int, default=3, help="Diffusion logit shift")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)