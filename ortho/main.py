import cv2
import os, sys
import argparse
import torch, numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from buffer import OrthoTransitionBuffer
from models import LinearProjector, _get_wan_latent_stats
from data_rollout import get_registries, collect_all_envs_rollouts, _skip_intro, _camera_state
from models import encode_views_to_latents, decode_latents_to_views
from flow import train_ortho_flow_step, train_ortho_trajectory_step, predict_next_view_latents
from utils import _build_registry, _build_heldout_registry, set_seed, _psnr, load_model, compose_player_prompts, _annotate_frame_pose

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_H, IMG_W = 128, 128

def add_diffusion_noise(latents, tau):
    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - tau) * latents + tau * noise
    return noise, noisy_latents

def lift_ortho(img, dep):
    """Top/Side/Rear (B, 3, 3, H, W) in [-1, 1] + depth (B, 3, H, W) -> 3D points (x, up, z) and their colours, both kept
    as images (B, 3, 3, H, W) in [-1, 1]. Same fixed layout as train_projector.py (Bipedal's)."""
    B, _, H, W = dep.shape
    r = ((torch.arange(H, device=dep.device) + .5) / H * 2 - 1).view(1, H, 1).expand(B, H, W)
    c = ((torch.arange(W, device=dep.device) + .5) / W * 2 - 1).view(1, 1, W).expand(B, H, W)
    xyz = [torch.stack(v, dim=1) for v in ((c, -dep[:, 0], r), (c, -r, dep[:, 1]), (dep[:, 2], -r, c))]
    return torch.stack(xyz, dim=1).clamp(-1, 1), img

@torch.no_grad()
def validate_fpv(model, vae, text_encoder, tokenizer, env, agent, env_name, rollout_steps, steps, shift, seed, camera_pose="Right", use_memory=True):
    """
    Autoregressive rollout: Model only sees its own predictions after frame 0 (live env, same protocol as validation.py)
    """
    model.eval()
    obs, _ = env.reset(seed=seed)
    obs = _skip_intro(env, env_name, obs)
    env.action_space.seed(seed)

    img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")
    views = np.stack([cv2.resize(x, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for x in [img_top, im_side, im_rear, im_fpv]])
    views_t = torch.from_numpy(views).float().unsqueeze(0).to(device) / 127.5 - 1.0 # (1, 4, 3, H, W)
    curr_latents = encode_views_to_latents(vae, views_t[:, -1:], out_dtype=model.dtype)
    gt_frames, pred_frames = [], []

    if use_memory:
        gt_depth_t = np.stack([cv2.resize(d, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) for d in env.last_depth]).clip(-2, 2) # as data_rollout.py
        gt_depth_t = torch.from_numpy(gt_depth_t).float().unsqueeze(0).to(device)
        xyz_t, rgb_t = lift_ortho(views_t[:, :3], gt_depth_t)
        guided_latents = encode_views_to_latents(vae, torch.cat([xyz_t, rgb_t], dim=1), out_dtype=model.dtype)
    
    memory = [guided_latents] if use_memory else [curr_latents[:, :, :0]]

    for s in range(rollout_steps):
        action = agent.predict(obs, deterministic=True)[0] if hasattr(agent, "predict") else env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        _, _, _, next_fpv = env.render(camera_pose=camera_pose)

        prompts_s = compose_player_prompts([np.atleast_1d(np.asarray(action, dtype=np.float32))], env_name=env_name, camera_pose=_camera_state(env, camera_pose))
        text_inputs = tokenizer(prompts_s, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(text_encoder.device)
        emb = text_encoder(**text_inputs).last_hidden_state.to(device=device, dtype=model.dtype)

        curr_latents = predict_next_view_latents(model, curr_latents, emb, memory_latents=torch.cat(memory, dim=2), steps=steps, shift=shift)

        pred = decode_latents_to_views(vae, curr_latents)[:, 0].float() #images
        gt = torch.from_numpy(cv2.resize(next_fpv, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        pred_frames.append(pred)
        gt_frames.append(gt)
        curr_latents = encode_views_to_latents(vae, pred.unsqueeze(1) * 2.0 - 1.0, out_dtype=model.dtype) # Re-encode onto the VAE manifold, as validation.py
        # memory.append(curr_latents) # Temp using 3D
        if terminated or truncated:
            break

    env.close()
    gt_video, pred_video = torch.stack(gt_frames, dim=1).cpu(), torch.stack(pred_frames, dim=1).cpu()
    return _psnr(pred_video, gt_video), gt_video, pred_video


def main(args):
    set_seed(args.seed)

    # 1. Build Envs and Buffer
    if args.hifi:
        registry, heldout_registry = get_registries(args.hifi)
    else:
        registry = _build_registry()
        heldout_registry =  _build_heldout_registry(obstacles=False)

    reg_dict = dict(registry + heldout_registry)
    buffer = OrthoTransitionBuffer(
            capacity=args.rollout_length * min(args.num_envs, len(registry)), 
            img_h=IMG_H, 
            img_w=IMG_W, 
            max_action_len=8
    )

    # Use cache if exists
    rank = 0
    cache_dir = (f"./buffer_cache/s{args.seed}_c{args.seed + rank}_r{rank}"
                f"_e{args.num_envs}_n{args.rollout_length}_{IMG_H}px{'_hifi' if args.hifi else ''}")

    cached = os.path.exists(os.path.join(cache_dir, "meta.pkl"))
    print(cache_dir)

    active_envs, agents = ([], {}) if cached else collect_all_envs_rollouts(
        registry, buffer, rollout_steps=args.rollout_length, num_envs=args.num_envs, seed=args.seed + rank,
        device=device,
    )
    if cached:
        buffer.load(cache_dir)
        print(f"[INFO] Loaded cached rollouts from {cache_dir} ({buffer.size} transitions)")
    else:
        buffer.save(cache_dir)

    # 2. Instantiate Model and LoRA
    model, vae, text_encoder, tokenizer = load_model(model_id=args.model_id, lora_rank=args.lora_rank)
    #import utils; utils.device = "cpu"   # load_model() leaves everything on CPU (UMT5 is ~11 GB)
    #model, vae, text_encoder, tokenizer = load_model(model_id=args.model_id, lora_rank=args.lora_rank)
    #model.to(device); vae.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    
    run_name = f"main_{'memory' if args.use_memory else 'no_mem'}_s{args.seed}"
    log_dir = f"./runs/{run_name}"
    ckpt_dir = f"./checkpoints/{run_name}"

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    print(f"TensorBoard logging to: {log_dir}")
    
    VAL_ENVS = ["Parkour Runner", "Lunar Lander", "MultiCar Racing"]

    # ===============================
    # 3. Main Diffusion Training loop
    # ===============================
    
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
            if args.use_memory:
                gt_depth_t = torch.from_numpy(np.stack([c["depth"] for c in camera_poses_seq[0]]).astype(np.float32)).to(views_t.device) # Temporarily use GT
                xyz_t, rgb_t = lift_ortho(views_t[:, :3], gt_depth_t) # 3D model of frame t: every Top/Side/Rear pixel = its (x, up, z) + colour
                guided_views = torch.cat([xyz_t, rgb_t], dim=1) # (B, 6, 3, H, W)
                guided_latents = encode_views_to_latents(vae, guided_views, out_dtype=model.dtype)
            
            # Take only the main FPV
            views_t = views_t[:,-1 : :]
            views_next_seq = [v[:,-1 : :] for v in views_next_seq]
            
            total_loss = 0.0
            avg_per_view = [0.0] * views_t.shape[1]
            curr_latents = encode_views_to_latents(vae, views_t, out_dtype=model.dtype)
            
            # memory = [curr_latents] # Temp Simplest memory (No 3D)
            memory = [guided_latents] if args.use_memory else [curr_latents[:, :, :0]]

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

                # Memory
                memory_latents = torch.cat(memory, dim=2)
                _, memory_latents_t = add_diffusion_noise(memory_latents, ctx_lvl)

                combined_latents = torch.cat([cond_latents_t, memory_latents_t, noisy_target_latents], dim=2).to(model.dtype)

                # 5. Compute GT velocity: v_target = noise - x_0
                target_velocity = noise - target_latents

                # 6. Forward pass
                timesteps = (tau.flatten() * 1000.0).to(dtype=model.dtype)
                model_pred = model(
                    hidden_states=combined_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                pred_velocity = model_pred[:, :, -num_views:] # Input [ cond_t | memory | noisy target ]

                per_view_losses = [
                    F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
                    for idx in range(num_views)
                ]
                loss = sum(per_view_losses)
                scaled_loss = loss / float(args.train_seq_len)
                scaled_loss.backward()
                total_loss += scaled_loss.item()
                
                for v_idx in range(num_views):
                    avg_per_view[v_idx] += per_view_losses[v_idx].item() / float(args.train_seq_len)

                # 7. Self-forcing step, basically run inference during training and feed its own pred autoregressively
                if not is_last_step:
                    sf_flag = torch.rand(b, device=curr_latents.device) < curr_self_forcing_prob
                    pred_latents = target_latents.clone()
                    if sf_flag.any():
                        with torch.no_grad():
                            pred_latents[sf_flag] = predict_next_view_latents(
                                model, curr_latents[sf_flag], prompt_embeds[sf_flag], memory_latents=memory_latents[sf_flag], steps=args.diffusion_steps, shift=args.shift)
                    curr_latents = pred_latents

                # memory.append(curr_latents) # Temp using 3D 

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
            writer.add_scalar("Train/SelfForcingProb", curr_self_forcing_prob, global_step)
            writer.add_scalar("Train/Loss_3D_FPV", float(avg_per_view[0]), global_step)

        # Validation across VAL_ENVS
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            print(f"\n[VALIDATION] Running autoregressive video rollouts for Epoch {epoch + 1}")

            for env_name in VAL_ENVS:
                env, agent = reg_dict[env_name](), agents.get(env_name, agents.get("Bipedal Walker"))

                # Next Validation Autoregressive rollouts
                clean_tag = env_name.replace(" ", "_").lower()
                psnr, gt_video, pred_video = validate_fpv(model, vae, text_encoder, tokenizer, env, agent, env_name, args.rollout_video_steps, steps=args.diffusion_steps, shift=args.shift, seed=args.seed + 1000, use_memory=args.use_memory)
                vid_fpv_vis = torch.stack([_annotate_frame_pose(f, "Right") for f in pred_video[0]]).unsqueeze(0) # pose overlay, as validation.py
                vid_gt_fpv_vis = torch.stack([_annotate_frame_pose(f, "Right") for f in gt_video[0]]).unsqueeze(0)

                writer.add_scalar(f"Validation/{clean_tag}/PSNR/FPV", psnr, epoch + 1)
                print(f"  [Epoch {epoch + 1}] [{env_name}] Autoregressive Video PSNR: FPV: {psnr:.2f}dB")
                writer.add_video(f"Test/{clean_tag}/FPV", vid_fpv_vis, epoch + 1, fps=args.video_fps)
                if args.log_gt:
                    writer.add_video(f"GT/{clean_tag}/FPV", vid_gt_fpv_vis, epoch + 1, fps=args.video_fps)


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
    parser.add_argument("--use_memory", action="store_true", default=False, help="Condition on the memory 3D latents (lifted Top/Side/Rear + depth of frame t)")
    parser.add_argument("--log_gt", action="store_true", help="Also log the ground-truth validation videos (identical every epoch)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)