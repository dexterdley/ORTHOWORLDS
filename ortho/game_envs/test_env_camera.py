"""
test_env_camera.py — Renders ground-truth rollouts for every registered env with a fixed
camera command and saves each view (Top, Rear, Side, FPV) at 96x96 into ./assets.

The camera command is held for the whole rollout. `--step_size` is the signed per-step
camera delta in degrees (yaw for Left/Right, pitch for Up/Down), e.g. Left with -1.0
pans the FPV camera 1 degree to the left every step.

Outputs (in ./assets):
  {Env}_{Top|Rear|Side|FPV}.gif  - 96x96 previews

Usage (from the `ortho` directory):
  python game_envs/test_env_camera.py --command Left --step_size -1.0 --steps 64
"""
import os
import sys
import math
import random
import argparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import cv2
import imageio

ORTHO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ORTHO_DIR not in sys.path:
    sys.path.insert(0, ORTHO_DIR)

from utils import _build_registry

VIEW_NAMES = ["Top", "Rear", "Side", "FPV"]  # env.render() return order


def configure_camera_step(env, command, step_size):
    """Sets the wrapper's per-step camera increment from a signed step size (degrees)."""
    cmd = command.strip().title()
    if cmd in ("Left", "Right"):
        env.step_yaw = math.radians(abs(step_size))
    elif cmd in ("Up", "Down"):
        env.step_pitch = math.radians(abs(step_size))


def rollout_env(env_name, env, command, steps, img_size, seed):
    """Steps the env with random actions while holding a fixed camera command."""
    frames = {v: [] for v in VIEW_NAMES}
    cam_states = []

    obs, _ = env.reset(seed=seed)
    if env_name == "MultiCar Racing":
        for _ in range(50):  # skip zoom-in intro frames
            obs, _, _, _, _ = env.step(np.zeros(3))

    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        views = env.render(camera_pose=command)
        for name, img in zip(VIEW_NAMES, views):
            img = np.asarray(img)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            frames[name].append(cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA))

        # Same keys as the camera dict stored by collect_rollouts
        cam_states.append({
            "command": command,
            "x": getattr(env, "cam_x", 0.0),
            "y": getattr(env, "cam_y", 0.0),
            "z": getattr(env, "cam_z", 0.0),
            "roll": getattr(env, "cam_roll", 0.0),
            "yaw": getattr(env, "cam_yaw", 0.0),
            "pitch": getattr(env, "cam_pitch", 0.0),
        })

        if terminated or truncated:
            obs, _ = env.reset()

    return frames, cam_states


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = os.path.join(ORTHO_DIR, "assets")
    os.makedirs(out_dir, exist_ok=True)

    registry = _build_registry()
    if args.env:
        registry = [(n, f) for n, f in registry if args.env.lower() in n.lower()]

    for env_name, factory in registry:
        print(f"[{env_name}] {args.steps} steps, command={args.command}, step_size={args.step_size:+.1f} deg")
        try:
            env = factory()
            env.action_space.seed(args.seed)
            configure_camera_step(env, args.command, args.step_size)
            frames, cam_states = rollout_env(env_name, env, args.command, args.steps, args.img_size, args.seed)
            env.close()
        except Exception as e:
            print(f"  [WARN] Failed on {env_name}: {e}")
            continue

        clean_name = env_name.replace(" ", "_")
        for view in VIEW_NAMES:
            gif_path = os.path.join(out_dir, f"{clean_name}_{view}.gif")
            imageio.mimsave(gif_path, frames[view], duration=1.0 / args.fps, loop=0)

        yaw_deg = [math.degrees(c["yaw"]) for c in cam_states]
        print(f"  saved {len(VIEW_NAMES)} GIFs (yaw {yaw_deg[0]:+.1f} -> {yaw_deg[-1]:+.1f} deg)")


def parse_args():
    parser = argparse.ArgumentParser(description="Render fixed-command GT camera rollouts for every env/view as 96x96 GIFs")
    parser.add_argument("--env", type=str, default=None, help="Substring filter on env name (default: all envs)")
    parser.add_argument("--command", type=str, default="Left", choices=["Center", "Up", "Down", "Left", "Right"], help="Camera command held for the whole rollout")
    parser.add_argument("--step_size", type=float, default=-1.0, help="Signed per-step camera delta in degrees (yaw for Left/Right, pitch for Up/Down)")
    parser.add_argument("--steps", type=int, default=64, help="Rollout length per env")
    parser.add_argument("--img_size", type=int, default=96, help="Output frame size (square)")
    parser.add_argument("--fps", type=int, default=15, help="GIF playback FPS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
