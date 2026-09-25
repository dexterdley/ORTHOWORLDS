"""
test_ortho_ppo.py
------------------
Evaluation script for trained PPO agents on orthographic game environments.
Loads trained checkpoints from ./checkpoints/ and visualizes agent behavior live in a Matplotlib 2x2 grid.

Usage:
    cd car_racing
    python test_ortho_ppo.py                            # Interactive selection
    python test_ortho_ppo.py --env "Drone Dogfight"     # Test specific environment
    python test_ortho_ppo.py --ckpt checkpoints/ppo_dronedogfight.pt
"""

import sys
import os
import time
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("TkAgg")  # Fallback GUI backend for live playback
import matplotlib.pyplot as plt

# Ensure local package path is imported
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from train_ortho_ppo import _build_registry, ActorCritic

ANSI = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def _c(text, *keys):
    prefix = "".join(ANSI[k] for k in keys)
    return f"{prefix}{text}{ANSI['reset']}"


def parse_args():
    parser = argparse.ArgumentParser(description="Test and visualize trained PPO agents.")
    parser.add_argument("--env", type=str, default="", help="Name of the environment to test (e.g., 'Drone Dogfight')")
    parser.add_argument("--ckpt", type=str, default="", help="Path to custom model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play per env")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between rendered frames (seconds)")
    return parser.parse_args()


def load_agent(ckpt_path, state_dim, action_space, device):
    """Instantiate ActorCritic and load state dict from checkpoint."""
    ac = ActorCritic(state_dim, action_space).to(device)
    if os.path.exists(ckpt_path):
        ac.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(_c(f"  [LOADED] Model weights from {ckpt_path}", "green", "bold"))
    else:
        print(_c(f"  [WARN] Checkpoint '{ckpt_path}' not found! Playing with random initialization.", "yellow", "bold"))
    ac.eval()
    return ac


def select_greedy_action(ac, state, device):
    """Select greedy / deterministic action from ActorCritic model."""
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        features = ac.feature_net(state_t)
        if ac.is_discrete:
            logits = ac.actor_head(features)
            action = torch.argmax(logits, dim=-1).item()
        else:
            mean = torch.tanh(ac.actor_mean(features))
            action = mean.squeeze(0).cpu().numpy()
    return action


def setup_figure(env_name):
    """Initialize dark-themed 2x2 matplotlib figure for 4 orthographic views."""
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"hspace": 0.1, "wspace": 0.05})
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(f"PPO Agent Evaluation: {env_name}", fontsize=14, fontweight="bold", color="#ffffff", y=0.98)
    
    titles = ["Top-Down View", "Rear View", "Side View", "First Person View (FPV)"]
    img_plots = []
    
    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(titles[idx], fontsize=10, color="#00e5ff", pad=4)
        ax.axis("off")
        # Dummy blank image to update later
        dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
        im = ax.imshow(dummy_img)
        img_plots.append(im)
        
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, img_plots


def play_episode(env, ac, device, fig, img_plots, ep_idx, max_steps, delay):
    """Play a single episode and visualize in real-time."""
    obs, info = env.reset()
    state = np.array(obs, dtype=np.float32).flatten()
    total_reward = 0.0
    
    for step in range(1, max_steps + 1):
        action = select_greedy_action(ac, state, device)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        total_reward += reward
        state = np.array(next_obs, dtype=np.float32).flatten()

        # Render 4 orthographic views
        try:
            views = env.render()
            if isinstance(views, (tuple, list)) and len(views) == 4:
                for idx, view_img in enumerate(views):
                    img_plots[idx].set_data(view_img)
            elif isinstance(views, np.ndarray):
                img_plots[0].set_data(views)
            
            fig.suptitle(
                f"PPO Evaluation | Episode {ep_idx} | Step: {step} | Step Reward: {reward:+.2f} | Total Reward: {total_reward:+.2f}",
                fontsize=12, fontweight="bold", color="#00ffcc", y=0.98
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(max(0.001, delay))
        except Exception as e:
            print(f"[WARN] Rendering error at step {step}: {e}")

        if done:
            print(_c(f"  Episode {ep_idx} finished in {step} steps. Total Reward: {total_reward:.2f}", "cyan"))
            break

    return total_reward


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(_c(f"Using PyTorch device: {device}", "cyan", "bold"))

    registry = _build_registry()
    if not registry:
        print(_c("No environments available. Exiting.", "red"))
        return

    # Filter environment if specified
    if args.env:
        matched = [(name, fn) for name, fn in registry if args.env.lower() in name.lower()]
        if not matched:
            print(_c(f"Environment '{args.env}' not found in registry!", "red"))
            print("Available environments:")
            for name, _ in registry:
                print(f" - {name}")
            return
        selected_envs = matched
    else:
        selected_envs = registry

    for env_name, make_env in selected_envs:
        print(_c(f"\n==================================================", "yellow"))
        print(_c(f" Testing PPO Agent on: {env_name}", "yellow", "bold"))
        print(_c(f"==================================================", "yellow"))

        try:
            env = make_env()
            obs, info = env.reset()

            state_dim = int(np.prod(obs.shape if hasattr(obs, 'shape') else env.observation_space.shape))
            
            # Determine Checkpoint Path
            if args.ckpt:
                ckpt_path = args.ckpt
            else:
                clean_name = "".join(c for c in env_name if c.isalnum() or c in ('_', '-')).lower()
                ckpt_path = os.path.join("checkpoints", f"ppo_{clean_name}.pt")

            ac = load_agent(ckpt_path, state_dim, env.action_space, device)
            fig, img_plots = setup_figure(env_name)

            for ep in range(1, args.episodes + 1):
                print(f"\n--- Episode {ep}/{args.episodes} ---")
                play_episode(env, ac, device, fig, img_plots, ep, args.max_steps, args.delay)
                time.sleep(0.5)

            plt.close(fig)
            env.close()

        except Exception as e:
            print(_c(f"Error testing environment '{env_name}': {e}", "red"))


if __name__ == "__main__":
    main()
