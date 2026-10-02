# Standard library imports
import math
import random
import os
import time
import argparse

# Third-party imports
import cv2
import gnwrapper
import gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame import gfxdraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn
from IPython.display import display, clear_output
from tqdm import tqdm
from PIL import Image
from utils import _build_registry, ActorCritic

# ------------------------------------------------------------------
# PPO (Proximal Policy Optimization) Implementation
# ------------------------------------------------------------------

class PPOBuffer:
    def __init__(self, size, state_dim, action_dim, is_discrete=False, gamma=0.99, gae_lambda=0.95):
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.is_discrete = is_discrete

        self.states = np.zeros((size, state_dim), dtype=np.float32)
        if is_discrete:
            self.actions = np.zeros(size, dtype=np.int64)
        else:
            self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def store(self, state, action, log_prob, reward, value, done):
        if self.ptr >= self.size:
            return
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_gae(self, last_value):
        last_gae = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
        # Normalize advantages
        adv_std = self.advantages.std()
        if adv_std > 1e-8:
            self.advantages = (self.advantages - self.advantages.mean()) / adv_std

    def get_batches(self, batch_size):
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                torch.tensor(self.states[batch_idx], dtype=torch.float32),
                torch.tensor(self.actions[batch_idx], dtype=torch.int64 if self.is_discrete else torch.float32),
                torch.tensor(self.log_probs[batch_idx], dtype=torch.float32),
                torch.tensor(self.returns[batch_idx], dtype=torch.float32),
                torch.tensor(self.advantages[batch_idx], dtype=torch.float32),
            )

    def reset(self):
        self.ptr = 0

def get_gradient_norm(gradients):
    """Helper to calculate the L2 norm of a list of gradients."""
    # Filter out None values (parameters not connected to the loss)
    valid_grads = [g.detach() for g in gradients if g is not None]
    if not valid_grads:
        return torch.tensor(0.0, device=gradients[0].device if gradients else 'cpu')
    
    # Calculate the L2 norm across all valid gradients
    return torch.norm(torch.stack([torch.norm(g) for g in valid_grads]))

class PPOAgent:
    def __init__(self, state_dim, action_space, lr=3e-4, clip_eps=0.2, c_val=0.1, c_ent=0.01, device="cpu"):
        self.device = torch.device(device)
        self.clip_eps = clip_eps
        self.c_val = c_val
        self.c_ent = c_ent
        self.action_space = action_space

        self.model = ActorCritic(state_dim, action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.model(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if not self.model.is_discrete:
                log_prob = log_prob.sum(dim=-1)

            if self.model.is_discrete:
                act_out = action.item()
            else:
                act_out = action.squeeze(0).cpu().numpy()
            
        return act_out, log_prob.item(), value.squeeze(0).item()

    def update(self, buffer, epochs=10, batch_size=64):
        policy_losses = []
        value_losses = []

        for _ in range(epochs):
            for states, actions, old_log_probs, returns, advantages in buffer.get_batches(batch_size):
                states = states.to(self.device)
                actions = actions.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)

                dist, values = self.model(states)
                values = values.squeeze(-1)

                if self.model.is_discrete:
                    log_probs = dist.log_prob(actions)
                else:
                    log_probs = dist.log_prob(actions).sum(dim=-1)

                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)
                p_grads = torch.autograd.grad(policy_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
                v_grads = torch.autograd.grad(value_loss, self.model.parameters(), retain_graph=True, allow_unused=True)

                # 3. Calculate the gradient magnitudes (norms)
                p_norm = get_gradient_norm(p_grads)
                v_norm = get_gradient_norm(v_grads)
                c_val = (p_norm / (v_norm + 1e-6)).detach()

                total_loss = policy_loss + c_val * self.c_val * value_loss - self.c_ent * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return np.mean(policy_losses), np.mean(value_losses)

def set_seed(seed):
    """Set random seed for reproducibility across random, numpy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on orthographic game environments.")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total timesteps per environment")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Rollout steps per batch")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of PPO update epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Hyperparameters: total_timesteps={args.total_timesteps}, rollout_steps={args.rollout_steps}, batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, seed={args.seed}")

    registry = _build_registry()
    if not registry:
        print("No environments could be loaded. Exiting.")
        return
    
    for idx, (env_name, make_env) in enumerate(registry):
        print(f"\n[{idx+1}/{len(registry)}] Initializing training for {env_name}...")
        print("  Loading env ...", end=" ", flush=True)
        env = make_env()
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(args.seed)
        obs, info = env.reset(seed=args.seed)
        views = env.render()
            
        action = env.action_space.sample()
        next_obs, reward, term, trunc, info = env.step(action)
            
        print("OK")
        print(f"  - State Space   : {env.observation_space} (Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)})")
        print(f"  - Action Space  : {env.action_space}")
        print(f"  - Reward Range  : {getattr(env, 'reward_range', (-np.inf, np.inf))} (Sample step reward: {reward})")
        
        print(f"\n==================================================")
        print(f" Starting PPO Training on: {env_name}")
        print(f"==================================================")

        # Flatten vector observation dimension
        if isinstance(obs, np.ndarray):
            state_dim = int(np.prod(obs.shape))
        else:
            state_dim = int(np.prod(env.observation_space.shape))

        is_discrete = hasattr(env.action_space, 'n')
        action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

        agent = PPOAgent(state_dim, env.action_space, lr=args.lr, device=device)
        buffer = PPOBuffer(args.rollout_steps, state_dim, action_dim, is_discrete=is_discrete)

        os.makedirs("checkpoints", exist_ok=True)

        timestep = 0
        update_cnt = 0
        episode_rewards = []
        curr_ep_reward = 0.0

        state, _ = env.reset(seed=args.seed)

        with tqdm(total=args.total_timesteps, desc=f"PPO [{env_name}]", unit="step") as pbar:
            while timestep < args.total_timesteps:
                buffer.reset()

                rollout_count = 0
                for step in range(args.rollout_steps):
                    if timestep >= args.total_timesteps:
                        break
                    flat_state = np.array(state, dtype=np.float32).flatten()
                    action, log_prob, value = agent.select_action(flat_state)

                    next_state, reward, term, trunc, info = env.step(action)
                    done = term or trunc

                    buffer.store(flat_state, action, log_prob, reward, value, float(done))

                    curr_ep_reward += reward
                    timestep += 1
                    rollout_count += 1
                    state = next_state

                    if done:
                        episode_rewards.append(curr_ep_reward)
                        curr_ep_reward = 0.0
                        state, _ = env.reset()

                pbar.update(rollout_count)

                # Compute GAE
                flat_state = np.array(state, dtype=np.float32).flatten()
                last_value = agent.model.get_value(torch.tensor(flat_state, dtype=torch.float32, device=agent.device).unsqueeze(0)).item()
                buffer.compute_gae(last_value)

                # Update PPO Agent
                p_loss, v_loss = agent.update(buffer, epochs=args.epochs, batch_size=args.batch_size)
                update_cnt += 1

                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                pbar.set_postfix({
                    "Upd": update_cnt,
                    "AvgR": f"{avg_reward:.2f}",
                    "PLoss": f"{p_loss:.4f}",
                    "VLoss": f"{v_loss:.4f}"
                })

        # Save Checkpoint
        clean_name = "".join(c for c in env_name if c.isalnum() or c in ('_', '-')).lower()
        ckpt_path = os.path.join("checkpoints", f"ppo_{clean_name}.pt")
        torch.save(agent.model.state_dict(), ckpt_path)
        print(f"[SUCCESS] Saved PPO model checkpoint to: {ckpt_path}\n")

        env.close()

if __name__ == "__main__":
    main()