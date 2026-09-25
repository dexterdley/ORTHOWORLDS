"""
review_game_envs.py
-------------------
Interactive reviewer for all game environments in game_envs/.

For each environment it:
  1. Runs N steps with random actions
  2. Captures up to CAPTURE_EVERY-th frame and renders a 2x2 grid
     (Top-Down | FPV | Rear | Side) in a matplotlib window
  3. Asks you in the terminal:  [k]eep  [r]eject  [s]kip  [q]uit
  4. Saves a Markdown report (review_report.md) in the same directory

Usage:
    cd car_racing
    python review_game_envs.py

Optional CLI flags:
    --steps  N      Number of env steps per env  (default: 60)
    --every  N      Show every N-th frame        (default: 10)
    --output FILE   Report output path           (default: review_report.md)
"""

import sys
import os
import time
import argparse
import traceback
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # switch to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make sure the car_racing package root is on sys.path
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# Registry -- add / remove entries here as you create new environments
# ---------------------------------------------------------------------------
def _build_registry():
    """Return a list of (display_name, factory_fn) tuples.
    Each factory_fn must return a fully-wrapped env that exposes:
        env.reset()  -> (obs, info)
        env.step(a)  -> (obs, reward, term, trunc, info)
        env.render() -> (im_top_down, im_rear, im_side, im_fpv)  [H x W x 3 uint8]
        env.action_space.sample()
        env.close()
    """
    registry = []

    def _try(name, factory):
        registry.append((name, factory))

    # ------------------------------------------------------------------
    # Tank Duel
    try:
        from game_envs.tank_duel_orthographic import (
            TankDuelEnv, TankDuelOrthographicWrapper)
        _try("Tank Duel",
             lambda: TankDuelOrthographicWrapper(TankDuelEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Catapult War: {e}")

    # Catapult War
    try:
        from game_envs.catapult_war_orthographic import (
            CatapultWarEnv, CatapultWarOrthographicWrapper)
        _try("Catapult War",
             lambda: CatapultWarOrthographicWrapper(CatapultWarEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Catapult War: {e}")
    
    # Robot Sumo
    try:
        from game_envs.robot_sumo_orthographic import (
            RobotSumoEnv, RobotSumoOrthographicWrapper)
        _try("Robot Sumo",
             lambda: RobotSumoOrthographicWrapper(RobotSumoEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Robot Sumo: {e}")

    # Mountain Goat
    try:
        from game_envs.mountain_goat_orthographic import (
            MountainGoatEnv, MountainGoatOrthographicWrapper)
        _try("Mountain Goat",
             lambda: MountainGoatOrthographicWrapper(MountainGoatEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Mountain Goat: {e}")

    # Wrecking Ball
    try:
        from game_envs.wrecking_ball_orthographic import (
            WreckingBallEnv, WreckingBallOrthographicWrapper)
        _try("Wrecking Ball",
             lambda: WreckingBallOrthographicWrapper(WreckingBallEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Wrecking Ball: {e}")

    # ------------------------------------------------------------------
    # Toxic Gas Escape
    try:
        from game_envs.toxic_gas_escape_orthographic import (
            ToxicGasEscapeEnv, ToxicGasEscapeOrthographicWrapper)
        _try("Toxic Gas Escape",
             lambda: ToxicGasEscapeOrthographicWrapper(ToxicGasEscapeEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Toxic Gas Escape: {e}")

    # ------------------------------------------------------------------
    # Drone Dogfight
    try:
        from game_envs.drone_dogfight_orthographic import (
            DroneDogfightEnv, DroneDogfightOrthographicWrapper)
        _try("Drone Dogfight",
             lambda: DroneDogfightOrthographicWrapper(DroneDogfightEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Drone Dogfight: {e}")

    # ------------------------------------------------------------------
    # Excavator
    try:
        from game_envs.excavator_orthographic import (
            ExcavatorEnv, ExcavatorOrthographicWrapper)
        _try("Excavator",
             lambda: ExcavatorOrthographicWrapper(ExcavatorEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Excavator: {e}")

    # ------------------------------------------------------------------
    # Mario Escape
    try:
        from game_envs.mario_orthographic import (
            MarioEscapeEnv, MarioOrthographicWrapper)
        _try("Mario Escape",
             lambda: MarioOrthographicWrapper(MarioEscapeEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Mario Escape: {e}")

    # ------------------------------------------------------------------
    # Lunar Lander (wraps gymnasium's LunarLander-v3)
    try:
        import gymnasium
        from game_envs.lunar_orthographic import LunarOrthographicWrapper
        _try("Lunar Lander",
             lambda: LunarOrthographicWrapper(
                 gymnasium.make("LunarLander-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register Lunar Lander: {e}")

    # ------------------------------------------------------------------
    # Bipedal Walker (wraps gymnasium's BipedalWalker-v3)
    try:
        import gymnasium
        from game_envs.bipedal_orthographic import BipedalOrthographicWrapper
        _try("Bipedal Walker",
             lambda: BipedalOrthographicWrapper(
                 gymnasium.make("BipedalWalker-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register Bipedal Walker: {e}")

    # ------------------------------------------------------------------
    # MultiCar Racing (wraps gymnasium's CarRacing-v3)
    try:
        import gymnasium
        from game_envs.multicar_racing_orthographic import MultiCarOrthographicWrapper
        _try("MultiCar Racing",
             lambda: MultiCarOrthographicWrapper(
                 gymnasium.make("CarRacing-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register MultiCar Racing: {e}")

    return registry


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
PANEL_LABELS = [
    "Top-Down (Plan View)",
    "3D Perspective (FPV)",
    "Orthographic Rear",
    "Orthographic Side",
]


def _render_frame(fig, axes, images, env_name, step, total_steps):
    """Update a 2x2 matplotlib figure with the four views."""
    # images tuple: (top_down, rear, side, fpv)
    view_order = [images[0], images[3], images[1], images[2]]  # top, fpv, rear, side
    for ax, img, label in zip(axes.flat, view_order, PANEL_LABELS):
        ax.clear()
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#1a1a2e")
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=14)
        ax.set_title(f"{label}  [step {step}/{total_steps}]",
                     fontsize=9, color="#e0e0e0", pad=4)
        ax.axis("off")
    fig.suptitle(env_name, fontsize=14, fontweight="bold", color="#ffffff", y=0.98)
    fig.canvas.draw()
    fig.canvas.flush_events()


def _setup_figure():
    """Create a dark-themed 2x2 matplotlib figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                              gridspec_kw={"hspace": 0.08, "wspace": 0.05})
    fig.patch.set_facecolor("#0d0d1a")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
    plt.ion()
    plt.show(block=False)
    return fig, axes


# ---------------------------------------------------------------------------
# Terminal prompt helpers
# ---------------------------------------------------------------------------
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


def _ask_decision(env_name):
    """Block until the user enters k / r / s / q."""
    options = (
        _c("[k]", "green", "bold") + " keep   " +
        _c("[r]", "red",   "bold") + " reject   " +
        _c("[s]", "yellow","bold") + " skip   " +
        _c("[q]", "cyan",  "bold") + " quit"
    )
    while True:
        raw = input(f"\n  Decision for {_c(env_name, 'bold')}: {options}\n  > ").strip().lower()
        if raw in ("k", "r", "s", "q"):
            return raw
        print("  Please type one of: k, r, s, q")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _write_report(results, output_path):
    lines = [
        "# Game Environment Review Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| # | Environment | Decision | Notes |",
        "|---|-------------|----------|-------|",
    ]
    for i, (name, decision, note) in enumerate(results, 1):
        emoji = {"keep": "OK", "reject": "REJECT", "skip": "SKIP", "error": "ERROR"}.get(decision, "?")
        lines.append(f"| {i} | {name} | {emoji} {decision.capitalize()} | {note} |")

    kept    = sum(1 for _, d, _ in results if d == "keep")
    rejected= sum(1 for _, d, _ in results if d == "reject")
    skipped = sum(1 for _, d, _ in results if d == "skip")
    errored = sum(1 for _, d, _ in results if d == "error")

    lines += [
        "",
        "## Summary",
        f"- Total reviewed : {len(results)}",
        f"- Kept           : {kept}",
        f"- Rejected        : {rejected}",
        f"- Skipped         : {skipped}",
        f"- Errors          : {errored}",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{_c('Report saved to:', 'cyan')} {output_path}")


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------
def run_review(steps: int, every: int, output: str):
    registry = _build_registry()
    if not registry:
        print("No environments could be loaded. Exiting.")
        return

    print(_c(f"\n{'='*60}", "cyan"))
    print(_c(f"  Game Env Reviewer  --  {len(registry)} environments", "bold"))
    print(_c(f"{'='*60}\n", "cyan"))
    print(f"  steps per env : {steps}")
    print(f"  show every    : {every} steps")
    print(f"  report output : {output}\n")
    print(f"  Controls during playback: just wait -- frames advance automatically.")
    print(f"  Controls after playback:  k=keep  r=reject  s=skip  q=quit\n")

    fig, axes = _setup_figure()
    results = []

    for idx, (env_name, make_env) in enumerate(registry):
        print(_c(f"\n[{idx+1}/{len(registry)}]  {env_name}", "cyan", "bold"))
        print("  Loading env ...", end=" ", flush=True)

        try:
            env = make_env()
            obs, info = env.reset()
            print("OK")
        except Exception as exc:
            print(f"FAILED\n  {traceback.format_exc()}")
            results.append((env_name, "error", str(exc)[:200]))
            continue

        # --- Step loop ---
        last_images = None
        frame_count = 0

        for step_idx in range(1, steps + 1):
            try:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                images = env.render()
            except Exception as exc:
                print(f"\n  [WARN] step/render error at step {step_idx}: {exc}")
                break

            if images is None or all(im is None for im in images):
                continue

            last_images = images
            frame_count += 1

            if frame_count == 1 or step_idx % every == 0:
                _render_frame(fig, axes, images, env_name, step_idx, steps)
                plt.pause(0.001)

            if terminated or truncated:
                # Reset to show more variety
                try:
                    obs, info = env.reset()
                except Exception:
                    break

        # Final frame pause so the user can look before deciding
        if last_images is not None:
            _render_frame(fig, axes, last_images, env_name, steps, steps)
            plt.pause(0.1)

        # --- User decision ---
        decision_char = _ask_decision(env_name)
        decision = {"k": "keep", "r": "reject", "s": "skip", "q": "quit"}[decision_char]

        note = ""
        if decision in ("reject"):
            note = input("  State the issues (press Enter to skip): ").strip()

        results.append((env_name, decision, note))

        try:
            env.close()
        except Exception:
            pass

        if decision == "quit":
            print(_c("\n  Quitting early ...", "yellow"))
            break

    # --- Finish ---
    plt.ioff()
    plt.close(fig)
    _write_report(results, output)

    print(_c("\nReview complete! Summary:", "bold"))
    icons = {"keep": "OK ", "reject": "NO ", "skip": "-- ", "error": "ERR", "quit": "..."}
    for name, decision, note in results:
        icon = icons.get(decision, " ? ")
        print(f"  [{icon}]  {name}" + (f"  -- {note}" if note else ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively review game environments.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of env steps per environment (default: 60)"
    )
    parser.add_argument(
        "--every", type=int, default=1,
        help="Render every N-th step in the matplotlib window (default: 10)"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(HERE, "review_report.md"),
        help="Path for the Markdown review report (default: review_report.md)"
    )
    args = parser.parse_args()
    run_review(steps=args.steps, every=args.every, output=args.output)
