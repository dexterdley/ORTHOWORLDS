"""
play_hifi.py — Play MultiCar Racing with GymBox physics and the Godot high-fidelity renderer.

Python runs the Box2D simulation at 50 Hz and streams state to Godot; Godot renders the
scene and sends the player's keyboard/gamepad input back. The blue opponent is driven by
the autopilot in multicar_env.py.

Usage (from any directory):
  python hifi/play_hifi.py                      # then launch the Godot project (hifi/multicar_racing)
  python hifi/play_hifi.py --seed 3 --lowfi top # also show the GymBox top-down render side by side
  python hifi/play_hifi.py --lowfi grid         # all four GymBox views (slower, rendered every 3rd step)
  python hifi/play_hifi.py --ego autopilot      # watch mode: both cars on autopilot

Godot controls: W/S or Up/Down gas/brake, A/D or Left/Right steer, gamepad supported,
1-5 or C camera, R reset, F1 toggle GI, Esc quit.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from godot_bridge import GodotBridge, export_state, export_track
from multicar_env import MultiCarRacing

# The GymBox envs (for --lowfi grid) live in ../ortho
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ortho"))

FPS = 50
VIEW_NAMES = ["Top", "Rear", "Side", "3D FPV"]


def make_lowfi_viewer(env, mode):
    """Returns a callable that shows the GymBox render(s) in an OpenCV window, or None."""
    if mode == "top":
        def show(step):
            cv2.imshow("GymBox (low-fi)", cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return show
    if mode == "grid":
        from game_envs.multicar_racing_orthographic import MultiCarOrthographicWrapper
        wrapper = MultiCarOrthographicWrapper(env)

        def show(step):
            if step % 3:
                return
            views = [cv2.resize(v, (400, 266)) for v in wrapper.render()]
            for v, name in zip(views, VIEW_NAMES):
                cv2.putText(v, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            grid = np.vstack([np.hstack(views[:2]), np.hstack(views[2:])])
            cv2.imshow("GymBox (low-fi)", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return show
    return None


def main(args):
    env = MultiCarRacing(opponent_speed=args.opponent_speed)
    bridge = GodotBridge(host=args.host, port=args.port)
    lowfi = make_lowfi_viewer(env, args.lowfi)
    seed = args.seed

    print(f"Waiting for Godot on {args.host}:{args.port} ... (open hifi/multicar_racing and press Play)")
    try:
        while True:
            bridge.wait_for_client()
            print("Godot connected.")
            env.reset(seed=seed)
            bridge.send(export_track(env, seed))
            step, ep_return, episode = 0, 0.0, 1
            next_tick = time.perf_counter()

            while bridge.connected:
                events = bridge.poll()
                if any(e.get("type") == "reset" for e in events):
                    print(f"Episode {episode} reset by player: return {ep_return:+.1f} in {step} steps")
                    seed = None if seed is None else seed + 1
                    env.reset(seed=seed)
                    bridge.send(export_track(env, seed))
                    step, ep_return, episode = 0, 0.0, episode + 1

                action = env.autopilot_action(env.car) if args.ego == "autopilot" else bridge.action()
                _, reward, terminated, truncated, info = env.step(action)
                step += 1
                ep_return += reward
                bridge.send(export_state(env, step, ep_return))
                if lowfi is not None:
                    lowfi(step)

                if terminated or truncated:
                    why = "lap finished" if info.get("lap_finished") else "left the playfield"
                    print(f"Episode {episode} over ({why}): return {ep_return:+.1f} in {step} steps")
                    seed = None if seed is None else seed + 1
                    env.reset(seed=seed)
                    bridge.send(export_track(env, seed))
                    step, ep_return, episode = 0, 0.0, episode + 1

                # Real-time pacing at the env's native 50 Hz
                next_tick += 1.0 / FPS
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
            print("Godot disconnected, waiting for it to reconnect ...")
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        env.close()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="MultiCar Racing: GymBox physics + Godot hi-fi renderer")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Interface to listen on; 0.0.0.0 lets Godot connect from another machine/VM")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--seed", type=int, default=None, help="Track seed (incremented on each new episode)")
    p.add_argument("--opponent_speed", type=float, default=1.0, help="Autopilot speed multiplier")
    p.add_argument("--ego", choices=["human", "autopilot"], default="human",
                   help="Who drives the red car: Godot input, or the autopilot (watch mode)")
    p.add_argument("--lowfi", choices=["none", "top", "grid"], default="none",
                   help="Also show the GymBox render for side-by-side comparison")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
