"""
Rollout collection shared by the ortho training scripts: env registries, camera-pose rollouts into the
OrthoTransitionBuffer, and the PPO agents that drive them (also reused for validation).
"""

import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

from utils import _build_registry, _build_heldout_registry, load_agent, is_main

IMG_H, IMG_W = 224, 224


def get_registries(hifi=False):
    """(train_registry, heldout_registry): GymBox by default, Godot hi-fi with --hifi."""
    if not hifi:
        return _build_registry(), _build_heldout_registry()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hifi"))
    from wrappers import build_hifi_registry, build_hifi_heldout_registry
    return build_hifi_registry(), build_hifi_heldout_registry()


def _skip_intro(env, env_name, obs):
    """Steps CarRacing past its zoom-in intro with no-op actions so those frames never reach the buffer."""
    if env_name == "MultiCar Racing":
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.zeros(3))
    return obs


def _camera_state(env, command):
    """Camera state stored with each transition and fed to the prompt; read right after env.render(command)."""
    return {
        "command": command,
        "x": getattr(env, "cam_x", 0.0),
        "y": getattr(env, "cam_y", 0.0),
        "z": getattr(env, "cam_z", 0.0),
        "roll": getattr(env, "cam_roll", 0.0),
        "yaw": getattr(env, "cam_yaw", 0.0),
        "pitch": getattr(env, "cam_pitch", 0.0),
    }


def collect_rollouts(env, agent, buffer, n_steps, seed, env_name="Bipedal Walker"):
    """Collects camera rollouts storing stateful camera poses."""
    obs, _ = env.reset(seed=seed)
    obs = _skip_intro(env, env_name, obs)
    env.action_space.seed(seed)
    pose_hold = 0

    # Render once per env step (each render advances the camera), so views_next[k] == views_t[k+1]
    # and every transition is one camera increment, matching the validation rollout.
    # "Center" is a no-op right after reset (yaw = pitch = 0).
    img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")

    for _ in tqdm(range(n_steps), desc=f"Rollouts [{env_name}]", disable=not is_main()):
        if pose_hold <= 0:
          current_pose = random.choice(["Center", "Up", "Down", "Left", "Right"])
          pose_hold = random.randint(8, 16)
        else:
          pose_hold -= 1

        with torch.inference_mode():
          if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
          else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_top, next_rear, next_side, next_fpv = env.render(
            camera_pose=current_pose
        )

        # Extract wrapper's stateful angles if available
        cam_state = _camera_state(env, current_pose)

        buffer.push(
            top_t=img_top,
            side_t=im_side,
            rear_t=im_rear,
            fpv_t=im_fpv,
            top_next=next_top,
            side_next=next_side,
            rear_next=next_rear,
            fpv_next=next_fpv,
            action=action,
            done=done,
            env_name=env_name,
            camera_state=cam_state,
        )

        if done:
          obs, _ = env.reset()
          obs = _skip_intro(env, env_name, obs)
          pose_hold = 0
          img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")
        else:
          obs = next_obs
          img_top, im_rear, im_side, im_fpv = next_top, next_rear, next_side, next_fpv
    env.close()


def collect_all_envs_rollouts(registry, buffer, rollout_steps=150, num_envs=3, seed=42, collect=True, device="cuda"):
    """Loops through all registered environments and populates the multi-env buffer.
    Returns (active_envs, agents) where agents maps env_name -> the collection policy (reused by validation)."""
    print("=" * 60)
    print(f"Collecting rollouts across all {min(num_envs, len(registry))} environments...")
    print("=" * 60)

    active_envs = []
    agents = {}
    for idx, (env_name, factory) in enumerate(registry[:num_envs]):
        print(f"\n[{idx + 1}/{len(registry)}] Initializing environment: {env_name}")
        env = factory()

        obs, info = env.reset()
        state_dim = int(np.prod(obs.shape if hasattr(obs, 'shape') else env.observation_space.shape))
        clean_env_name = "".join(c for c in env_name if c.isalnum() or c in ('_', '-')).lower()
        ckpt_path = os.path.join("checkpoints", f"ppo_{clean_env_name}.pt")
        
        if env_name == "Lunar Lander":
            checkpoint = load_from_hub(
                repo_id="Adilbai/ppo-LunarLander-v2",
                filename="ppo-LunarLander-v2.zip"
            )
            agent = PPO.load(checkpoint)

        elif env_name == "MultiCar Racing":
            checkpoint = load_from_hub(
                repo_id="igpaub/ppo-CarRacing-v2",
                filename="ppo-CarRacing-v2.zip"
            )
            agent = PPO.load(checkpoint)

        else:
            print(f"Loading PPO agent from: {ckpt_path}")
            agent = load_agent(ckpt_path, state_dim, env.action_space, device)
        if collect:
            collect_rollouts(env, agent, buffer, n_steps=rollout_steps, seed=seed, env_name=env_name)
        active_envs.append((env_name, env))
        agents[env_name] = agent

    unique_recorded = list(dict.fromkeys([e for e in buffer.env_names[:buffer.size] if e is not None]))
    print(f"\n[INFO] Buffer populated with {buffer.size} transitions across {len(unique_recorded)} environments: {', '.join(unique_recorded)}")
    return active_envs, agents


class ReplayEnv:
    """Validation without Godot: replays a fixed cached segment of `env_name` through autoregressive_rollout.
    Acts as env *and* agent: predict() gives the recorded action, render() the recorded next frames."""
    def __init__(self, buf, env_name, seq_len):
        starts = buf._get_valid_trajectory_starts(seq_len, env_filter=env_name)
        assert starts, f"val cache has no {seq_len}-step segment of {env_name}"
        self.buf, self.i0 = buf, starts[len(starts) // 2]      # same clip every epoch
        self.action_space = SimpleNamespace(seed=lambda s: None)
    def reset(self, seed=None):
        self.i, self.started, self.camera_state = self.i0, False, None
        return None, {}
    def predict(self, obs, deterministic=True):
        return self.buf.actions[self.i, :self.buf.action_lens[self.i]], None
    def step(self, action):
        if not self.started:      # _skip_intro's no-op steps: cached frames already start after the intro
            return None, 0.0, False, False, {}
        self.camera_state = self.buf.camera_states[self.i]
        self.i += 1
        return None, 0.0, bool(self.buf.dones[self.i - 1]), False, {}
    def render(self, camera_pose=None):
        v = self.buf.views_next[self.i - 1] if self.started else self.buf.views_t[self.i]
        self.started = True
        top, side, rear, fpv = np.ascontiguousarray(v.transpose(0, 2, 3, 1))   # buffer order is top, side, rear, fpv
        return top, rear, side, fpv
    def close(self): pass
