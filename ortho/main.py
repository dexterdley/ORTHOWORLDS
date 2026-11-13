import os, sys
import argparse
sys.path.insert(0, os.path.abspath(".")); os.chdir("..")
import torch, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

from models import LinearProjector, _get_wan_latent_stats
from data_rollout import get_registries, _skip_intro, _camera_state
from buffer import OrthoTransitionBuffer
from data_rollout import collect_all_envs_rollouts
from models import encode_views_to_latents, decode_latents_to_views
from flow import train_ortho_flow_step, train_ortho_trajectory_step, predict_next_view_latents
from utils import _build_registry, set_seed, _psnr, load_model, compose_player_prompts

sys.path.insert(0, os.path.abspath("hifi"))
from hifi.wrappers import make_hifi_env
device = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(42)
IMG_H, IMG_W = 96, 96
device = "cuda"
ENV = "MultiCar Racing"
agents = {}

train_reg, heldout_reg = get_registries()
env = dict(train_reg)[ENV]()
obs, _ = env.reset(seed=0)
obs = _skip_intro(env, ENV, obs)

checkpoint = load_from_hub(
                repo_id="igpaub/ppo-CarRacing-v2",
                filename="ppo-CarRacing-v2.zip"
            )
agents[ENV] = PPO.load(checkpoint)

def main(args):

    # 1. Build Envs and Buffer
    if args.hifi:
     registry, _ = get_registries(args.hifi)
    else:
        registry = _build_registry()

    reg_dict = dict(registry)
    buffer = OrthoTransitionBuffer(
            capacity=args.rollout_steps * min(args.num_envs, len(registry)), 
            img_h=IMG_H, 
            img_w=IMG_W, 
            max_action_len=8
        )
    active_envs, agents = collect_all_envs_rollouts(
        registry, buffer, rollout_steps=args.rollout_steps, num_envs=args.num_envs,
        device=device,
    )
    

    # 2. Instantiate Model and LoRA
    model, vae, text_encoder, tokenizer = load_model(model_id=args.model_id, lora_rank=args.lora_rank)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    
    # 3. Main Diffusion Training loop
    for epoch in tqdm(range(args.epochs), desc="Training"):
        views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq = buffer.sample_trajectory_batch(
                batch_size=args.batch_size,
                seq_len=args.train_seq_len,
            )
        
        # Take only the main FPV
        views_t = views_t[:,-1 : :]
        views_next_seq = [v[:,-1 : :] for v in views_next_seq]
        
        curr_latents = encode_views_to_latents(vae, views_t, out_dtype=model.dtype)

        # So pass the images into VAE to extract their latents, then build prompt
        for s in range(args.train_seq_len): # Unroll each sample in the training sequence

            target_latents = encode_views_to_latents(vae, views_next_seq[s], out_dtype=model.dtype)

            # Build prompt then tokenize
            prompts_s = compose_player_prompts(
                        actions_seq[s],
                        env_name=env_names_seq[s],
                        camera_pose=camera_poses_seq[s],
                    )

            text_inputs = tokenizer( prompts_s, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
            prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=model.dtype)
            is_last_step = (s == args.train_seq_len - 1)

            '''
            u       = torch.sigmoid(torch.randn(b, 1, 1, 1, 1, device=device))
            tau     = args.shift * u / (1 + (args.shift - 1) * u)                     # (B,1,1,1,1)
            ctx_lvl = torch.rand(b, 1, 1, 1, 1, device=device) * args.ctx_noise_max

            cond_latents_t       = (1 - ctx_lvl) * curr_latents + ctx_lvl * torch.randn_like(curr_latents)
            noise                = torch.randn_like(target_latents)
            noisy_target_latents = (1 - tau) * target_latents + tau * noise

            combined_latents = torch.cat([cond_latents_t, noisy_target_latents], dim=2).to(model.dtype)
            target_velocity  = noise - target_latents
            timesteps        = (tau.flatten() * 1000).to(model.dtype)
            '''
            b, c, num_views, h, w = target_latents.shape
            u = torch.randn(b, device=device, dtype=torch.float32)
            tau = torch.sigmoid(u)
            tau = (args.shift * tau) / (1.0 + (args.shift - 1.0) * tau)
            tau = tau.clamp(1e-4, 1.0 - 1e-4).view(b, 1, 1, 1, 1)
            tau_ctx = torch.rand((b, 1, 1, 1, 1), device=device, dtype=torch.float32) * 0.15

            # Add diffusion noise
            ctx_noise = torch.randn_like(curr_latents)
            cond_latents_t = (1.0 - tau_ctx) * curr_latents + tau_ctx * ctx_noise
            
            noise = torch.randn_like(target_latents)
            noisy_target_latents = (1.0 - tau) * target_latents + tau * noise

            combined_latents = torch.cat([cond_latents_t, noisy_target_latents], dim=2).to(model.dtype)

            # 5. Ground truth velocity: v_target = noise - x_0
            target_velocity = noise - noisy_target_latents

            # 6. DiT Forward pass
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
            print(loss)

            # continue in flow.py 
            '''
            # Reconstruct 1-step denoised prediction: x_0_hat = x_tau - tau * v_pred
    with torch.no_grad():
        pred_x0 = (noisy_latents_next_fp32 - tau_exp * pred_velocity.detach()).clamp(-6.0, 6.0)
        #confidence = (1.0 - tau_exp).clamp(0.2, 1.0)
        #rollout_latent = (confidence * pred_x0 + (1.0 - confidence) * latents_next_fp32).to(dtype=transformer.dtype)
        rollout_latent = pred_x0.to(dtype=_unwrap(transformer).dtype)
            '''

            break




def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wan2.1 with LoRA across multiple Single Game Views (DDP)")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout steps per environment, collected independently on each rank")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size (effective batch = batch_size * world_size)")
    parser.add_argument("--train_seq_len", type=int, default=16, help="Contiguous trajectory steps unrolled per training update")
    parser.add_argument("--teacher_forcing_prob", type=float, default=1.0, help="Max prob of using self-predicted latents onto next trajectory step")
    parser.add_argument("--ctx_noise_max", type=float, default=0.15, help="Max Gaussian noise added to conditioning context latents to prevent drift")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Generate and evaluate views every N epochs")
    parser.add_argument("--eval_steps", type=int, default=8, help="Number of Euler flow matching sampling steps per frame")
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