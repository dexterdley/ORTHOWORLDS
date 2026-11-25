import cv2
import os, sys
import argparse
import torch, numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from buffer import OrthoTransitionBuffer
from models import LinearProjector, _get_wan_latent_stats
from data_rollout import get_registries, collect_all_envs_rollouts, _skip_intro, _camera_state
from models import encode_views_to_latents, decode_latents_to_views
from flow import train_ortho_flow_step, train_ortho_trajectory_step, predict_next_view_latents
from utils import _build_registry, _build_heldout_registry, set_seed, _psnr, load_model, compose_player_prompts, _annotate_frame_pose

import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime, timedelta
from utils import is_main

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"

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

def rotate_3D(xyz, cams):
    """xyz (B, 3 views, 3, H, W) box coords (x, up, z) -> same points in the target FPV camera's axes
    (right, up, forward). Rotation only, as bipedal_orthographic.py:190-206."""
    yaw   = torch.tensor([c["yaw"]   for c in cams], device=xyz.device).view(-1, 1, 1, 1)
    pitch = torch.tensor([c["pitch"] for c in cams], device=xyz.device).view(-1, 1, 1, 1)
    x, y, z = xyz[:, :, 0] * 1.5, xyz[:, :, 1], xyz[:, :, 2]   # 600x400 viewport -> equal units
    cy, sy, cp, sp = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin()
    right = sy * x + cy * z
    up    = -cy * sp * x + cp * y + sy * sp * z
    fwd   = cy * cp * x - sp * y - sy * cp * z
    return (torch.stack([right, up, fwd], dim=2) / 2).clamp(-1, 1)

def convert_grayscale(views):
    """ RGB -> grayscale """
    luma = torch.tensor([0.299, 0.587, 0.114], device=views.device, dtype=views.dtype).view(1, 1, 3, 1, 1)
    return (views * luma).sum(2, keepdim=True).repeat(1, 1, 3, 1, 1)

class StructureNet(nn.Module):
    """Small latent-space model: FPV latent at t + prompt (action, camera) -> grayscale structure latent at t+1."""
    def __init__(self, c=16, width=128, text_dim=4096):
        super().__init__()
        self.inp, self.out = nn.Conv2d(c, width, 3, padding=1), nn.Conv2d(width, c, 3, padding=1)
        self.film = nn.Linear(text_dim, 2 * width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=d, dilation=d)) for d in (1, 2, 4, 8))

    def forward(self, latents, emb): # (B, C, 1, h, w), (B, L, D) -> (B, C, 1, h, w)
        scale, shift = self.film(emb.float().mean(1))[..., None, None].chunk(2, dim=1)
        x = self.inp(latents[:, :, 0].float()) * (1 + scale) + shift
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x).unsqueeze(2)

@torch.no_grad()
def validate_fpv(model, snet, vae, text_encoder, tokenizer, env, agent, env_name, rollout_steps, steps, shift, seed, camera_pose="Right", use_memory=True):
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
    gt_frames, pred_frames, struct_frames = [], [], []

    if use_memory:
        gt_depth_t = np.stack([cv2.resize(d, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) for d in env.last_depth]).clip(-2, 2) # as data_rollout.py
        gt_depth_t = torch.from_numpy(gt_depth_t).float().unsqueeze(0).to(device)
        memory = [lift_ortho(views_t[:, :3], gt_depth_t)] # write frame 0: (xyz, rgb)

    memory_latents = curr_latents[:, :, :0]

    for s in range(rollout_steps):
        action = agent.predict(obs, deterministic=True)[0] if hasattr(agent, "predict") else env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        next_top, next_rear, next_side, next_fpv = env.render(camera_pose=camera_pose)

        prompts_s = compose_player_prompts([np.atleast_1d(np.asarray(action, dtype=np.float32))], env_name=env_name, camera_pose=_camera_state(env, camera_pose))
        
        if use_memory:
            xyz_s = torch.cat([rotate_3D(m[0], [_camera_state(env, camera_pose)]) for m in memory], dim=1) # read: every stored frame, turned to the camera being predicted
            guided_views = torch.cat([xyz_s] + [m[1] for m in memory], dim=1) # (B, 6 * len(memory), 3, H, W)
            memory_latents = encode_views_to_latents(vae, guided_views, out_dtype=model.dtype)

        text_inputs = tokenizer(prompts_s, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(text_encoder.device)
        emb = text_encoder(**text_inputs).last_hidden_state.to(device=device, dtype=model.dtype)

        # curr_latents = predict_next_view_latents(model, curr_latents, emb, memory_latents=memory_latents, steps=steps, shift=shift)
        structure = snet(curr_latents, emb).to(model.dtype) # 1) structure of t+1, 2) paint the target given it
        curr_latents = predict_next_view_latents(model, curr_latents, emb, memory_latents=torch.cat([memory_latents, structure], dim=2), steps=steps, shift=shift)

        pred = decode_latents_to_views(vae, curr_latents)[:, 0].float() #images
        gt = torch.from_numpy(cv2.resize(next_fpv, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        pred_struct = decode_latents_to_views(vae, structure)[:, 0].float()

        pred_frames.append(pred)
        gt_frames.append(gt)
        struct_frames.append(pred_struct)
        
        if use_memory and (s + 1) % 4 == 0: # write the frame just reached, every 4th as in training
            views = np.stack([cv2.resize(x, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for x in [next_top, next_side, next_rear]])
            depth = np.stack([cv2.resize(d, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) for d in env.last_depth]).clip(-2, 2)
            memory.append(lift_ortho(torch.from_numpy(views).float().unsqueeze(0).to(device) / 127.5 - 1.0, torch.from_numpy(depth).float().unsqueeze(0).to(device)))
            memory = memory[-2:]
        if terminated or truncated:
            break

    env.close()
    gt_video = torch.stack(gt_frames, dim=1).cpu()
    pred_video = torch.stack(pred_frames, dim=1).cpu()
    struct_video = torch.stack(struct_frames, dim=1).cpu()
    return _psnr(pred_video, gt_video), gt_video, pred_video, struct_video


def main(args):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    set_seed(args.seed + rank)

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
        if is_main():
            print(f"[INFO] Loaded cached rollouts from {cache_dir} ({buffer.size} transitions)")
    else:
        buffer.save(cache_dir)

    # 2. Instantiate Model and LoRA
    model, vae, text_encoder, tokenizer = load_model(model_id=args.model_id, lora_rank=args.lora_rank)
    ddp_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    # import utils; utils.device = "cpu"   # load_model() leaves everything on CPU (UMT5 is ~11 GB)
    # model, vae, text_encoder, tokenizer = load_model(model_id=args.model_id, lora_rank=args.lora_rank)
    # model.to(device); vae.to(device)

    snet = DDP(StructureNet(text_dim=text_encoder.config.d_model).to(device), device_ids=[local_rank]) # small structure model
    #trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params = [p for p in list(model.parameters()) + list(snet.parameters()) if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=200)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - 200, eta_min=args.lr * 0.1)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[200])

    # run_name = f"main_{'memory_rotate_ddp' if args.use_memory else 'no_mem'}_s{args.seed}"
    run_name = f"main_{'memory_rotate_ddp' if args.use_memory else 'no_mem'}_gray2_s{args.seed}"
    log_dir = f"./runs/{run_name}"
    ckpt_dir = f"./checkpoints/{run_name}"

    writer = None
    if is_main():
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

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} (Self-pred Prob.={curr_self_forcing_prob:.2f})", disable=not is_main())

        for _ in pbar:
            views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq = buffer.sample_trajectory_batch(
                    batch_size=args.batch_size,
                    seq_len=args.train_seq_len,
                )
            ortho_seq = [views_t[:, :3]] + [v[:, :3] for v in views_next_seq] # Top/Side/Rear of frames t .. t+T
            memory = [] # 3D frames written so far: (xyz, rgb)

            # Take only the main FPV
            views_t = views_t[:,-1 : :]
            views_next_seq = [v[:,-1 : :] for v in views_next_seq]
            
            total_loss = 0.0
            avg_per_view = [0.0, 0.0] * views_t.shape[1]
            curr_latents = encode_views_to_latents(vae, views_t, out_dtype=model.dtype)
            
            # memory = [curr_latents] # Temp Simplest memory (No 3D)
            # memory = [guided_latents] if args.use_memory else [curr_latents[:, :, :0]]
            memory_latents = curr_latents[:, :, :0]
            
            # So pass the images into VAE to extract their latents, then build prompt
            for s in range(args.train_seq_len): # Unroll each sample in the training sequence

                struct_views_next_seq = convert_grayscale(views_next_seq[s])
                struct_latents = encode_views_to_latents(vae, struct_views_next_seq, out_dtype=model.dtype)
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

                # Predict structure
                pred_struct_latents = snet(curr_latents, prompt_embeds)
                struct_loss = F.mse_loss(pred_struct_latents, struct_latents.float())

                u = torch.sigmoid(torch.randn(b, 1, 1, 1, 1, device=device))
                tau = args.shift * u / (1 + (args.shift - 1) * u)
                ctx_lvl = torch.rand((b, 1, 1, 1, 1), device=device, dtype=torch.float32) * args.ctx_noise_max

                # Add diffusion noise
                _, cond_latents_t = add_diffusion_noise(curr_latents, ctx_lvl)
                noise, noisy_target_latents = add_diffusion_noise(target_latents, tau)

                # Memory
                # memory_latents = torch.cat(memory, dim=2)
                if args.use_memory:
                    if s % 4 == 0: # write the frame the model is at (t+s), every 4th
                        gt_depth_s = torch.from_numpy(np.stack([c["depth"] for c in camera_poses_seq[s]]).astype(np.float32)).to(device) # Temporarily use GT
                        memory.append(lift_ortho(ortho_seq[s], gt_depth_s)) # (xyz, rgb): every Top/Side/Rear pixel = its (x, up, z) + colour
                        memory = memory[-2:]
                    xyz_s = torch.cat([rotate_3D(m[0], camera_poses_seq[s]) for m in memory], dim=1) # read: every stored frame, turned to the camera being predicted
                    guided_views = torch.cat([xyz_s] + [m[1] for m in memory], dim=1) # (B, 6 * len(memory), 3, H, W)
                    memory_latents = encode_views_to_latents(vae, guided_views, out_dtype=model.dtype)

                _, memory_latents_t = add_diffusion_noise(memory_latents, ctx_lvl)

                # combined_latents = torch.cat([cond_latents_t, memory_latents_t, noisy_target_latents], dim=2).to(model.dtype)
                combined_latents = torch.cat([cond_latents_t, memory_latents_t, pred_struct_latents.detach(), noisy_target_latents], dim=2).to(model.dtype) # Always use

                # 5. Compute GT velocity: v_target = noise - x_0
                target_velocity = noise - target_latents

                # 6. Forward pass
                timesteps = (tau.flatten() * 1000.0).to(dtype=model.dtype)

                sync_ddp = ddp_model.no_sync() if not is_last_step else contextlib.nullcontext()
                with sync_ddp:
                    model_pred =  ddp_model(
                        hidden_states=combined_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                    ).sample

                    pred_velocity = model_pred[:, :, -num_views:] # Input [ cond_t | memory | noisy target ]

                    per_view_losses = [
                        F.mse_loss(pred_velocity[:, :, idx], target_velocity[:, :, idx])
                        for idx in range(num_views)
                    ]
                    loss = sum(per_view_losses) + struct_loss
                    scaled_loss = loss / float(args.train_seq_len)
                    scaled_loss.backward()
                total_loss += scaled_loss.item()
                
                for v_idx in range(num_views):
                    avg_per_view[v_idx] += per_view_losses[v_idx].item() / float(args.train_seq_len)
                avg_per_view[1] += struct_loss.item() / float(args.train_seq_len)

                # 7. Self-forcing step, basically run inference during training and feed its own pred autoregressively
                if not is_last_step:
                    sf_flag = torch.rand(b, device=curr_latents.device) < curr_self_forcing_prob
                    pred_latents = target_latents.clone()
                    if sf_flag.any():
                        with torch.no_grad():
                            pred_latents[sf_flag] = predict_next_view_latents(
                                #model, curr_latents[sf_flag], prompt_embeds[sf_flag], memory_latents=memory_latents[sf_flag], steps=args.diffusion_steps, shift=args.shift)
                                model, curr_latents[sf_flag], prompt_embeds[sf_flag], memory_latents=torch.cat([memory_latents[sf_flag], pred_struct_latents[sf_flag].detach().to(model.dtype)], dim=2), steps=args.diffusion_steps, shift=args.shift)
                    curr_latents = pred_latents

                # memory.append(curr_latents) # Temp using 3D 

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            stats = torch.tensor([total_loss] + avg_per_view, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            total_loss, avg_per_view = stats[0].item(), stats[1:].tolist()

            env_summary = "+".join(list(dict.fromkeys(env_names_seq[0]))[:2])
            pbar.set_postfix({
                    "Loss": f"{total_loss:.3f}",
                    "FPV": f"{per_view_losses[-1]:.3f}",
                    "Envs": env_summary,
                })

            # Logging
            global_step += 1
            if is_main():
                writer.add_scalar("Train/Loss", total_loss, global_step)
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("Train/SelfForcingProb", curr_self_forcing_prob, global_step)
                writer.add_scalar("Train/Loss_3D_FPV", float(avg_per_view[0]), global_step)
                writer.add_scalar("Train/Loss_structure", float(avg_per_view[1]), global_step)

        # Validation across VAL_ENVS
        if (epoch + 1) % args.eval_every == 0 and is_main():
            model.eval()
            print(f"\n[VALIDATION] Running autoregressive video rollouts for Epoch {epoch + 1}")

            for env_name in VAL_ENVS:
                env, agent = reg_dict[env_name](), agents.get(env_name, agents.get("Bipedal Walker"))

                # Next Validation Autoregressive rollouts
                clean_tag = env_name.replace(" ", "_").lower()
                psnr, gt_video, pred_video, struct_video = validate_fpv(model, snet.module, vae, text_encoder, tokenizer, env, agent, env_name, args.rollout_video_steps, steps=args.diffusion_steps, shift=args.shift, seed=args.seed + 1000, use_memory=args.use_memory)
                vid_fpv_vis = torch.stack([_annotate_frame_pose(f, "Right") for f in pred_video[0]]).unsqueeze(0) # pose overlay, as validation.py
                vid_gt_fpv_vis = torch.stack([_annotate_frame_pose(f, "Right") for f in gt_video[0]]).unsqueeze(0)

                writer.add_scalar(f"Validation/{clean_tag}/PSNR/FPV", psnr, epoch + 1)
                gt_struct = convert_grayscale(gt_video) # GT structure (same luma as training)
                writer.add_scalar(f"Validation/{clean_tag}/PSNR/Structure", _psnr(struct_video, gt_struct), epoch + 1)
                writer.add_video(f"Test/{clean_tag}/Structure", torch.cat([gt_struct, struct_video, pred_video], dim=-1), epoch + 1, fps=args.video_fps) # [GT grey | predicted structure | painted RGB]

                print(f"  [Epoch {epoch + 1}] [{env_name}] Autoregressive Video PSNR: FPV: {psnr:.2f}dB")
                writer.add_video(f"Test/{clean_tag}/FPV", vid_fpv_vis, epoch + 1, fps=args.video_fps)
                if args.log_gt:
                    writer.add_video(f"GT/{clean_tag}/FPV", vid_gt_fpv_vis, epoch + 1, fps=args.video_fps)


            model.save_pretrained(os.path.join(ckpt_dir, f"epoch_{epoch+1}"))
            torch.save(snet.module.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch+1}", "structure_net.pt"))
        # Keep ranks in step while rank 0 validates / saves
        dist.barrier()

    if is_main():
        print("Ending Training")
        model.save_pretrained(os.path.join(ckpt_dir, "final"))
        torch.save(snet.module.state_dict(), os.path.join(ckpt_dir, "final", "structure_net.pt"))
        writer.close()

    dist.barrier()
    dist.destroy_process_group()

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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for LoRA parameters")
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