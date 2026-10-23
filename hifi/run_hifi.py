"""
run_hifi.py — Play (or watch) any OrthoWorlds env with the Godot hi-fi renderer.

Python runs the GymBox env at its native rate and streams state to Godot ("live" mode); Godot
interpolates, shows the four views and sends the pressed keys back.

Usage (from the repo root, GAME conda env):
  python hifi/run_hifi.py --env tank                        # play Tank Duel
  python hifi/run_hifi.py --env multicar --opponent         # two cars (second on autopilot)
  python hifi/run_hifi.py --env lunar --policy random       # watch random actions
  python hifi/run_hifi.py --env snowball --no-obstacles     # held-out env, obstacle-free twin
  python hifi/run_hifi.py --env goat --lowfi                # also show the GymBox views (OpenCV window)
  python hifi/run_hifi.py --env drone --godot hifi/tools/bin/Godot_v4.3-stable_win64_console.exe

Godot window: game keys (printed below), arrows orbit the 3D FPV camera, 1-4 Top / Rear / Side /
FPV, 5 hi-fi chase camera, 0/G/Tab 2x2 grid, M object-ID masks (with --masks), F5/Backspace new
episode, F1 SDFGI, Esc quit.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from bridge import BridgeError  # noqa: E402
from controls import HELP, read_action  # noqa: E402
from exporters import ENVS, get_exporter, resolve  # noqa: E402
from renderer import GodotRenderer  # noqa: E402

VIEW_NAMES = ["Top", "Rear", "Side", "3D FPV"]


def lowfi_grid(views, cell=(400, 266)):
    cells = [cv2.resize(np.asarray(v), cell) for v in views]
    for c, name in zip(cells, VIEW_NAMES):
        cv2.putText(c, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])


def main(args):
    name = resolve(args.env)
    kw = {}
    ex_probe = get_exporter(name)
    if ex_probe.heldout:
        kw["obstacles"] = not args.no_obstacles
    if name == "multicar_racing" and args.opponent:
        kw["opponent"] = True
    exporter = get_exporter(name, **kw)
    env = exporter.make_env()
    print(f"[{ENVS[name][0]}] {HELP.get(name, 'no key bindings')}" +
          ("   (policy: random)" if args.policy == "random" else ""))
    print("Godot: arrows orbit the FPV camera, 1-5 views, 0 grid, F5 reset, Esc quit")
    renderer = GodotRenderer(godot=args.godot, display=args.display, window=tuple(args.window))
    aov = ("rgb", "mask") if args.masks else ("rgb",)
    renderer.hello(name, exporter.size, aov=aov, mode="live", fps=args.fps or exporter.fps)
    fps = args.fps or exporter.fps
    seed = args.seed
    episode, ret, step, total = 1, 0.0, 0, 0

    def new_episode():
        nonlocal step, ret
        env.reset(seed=seed)
        exporter.on_reset(env)
        renderer.scene(exporter.scene(env))
        step, ret = 0, 0.0

    try:
        new_episode()
        state = {"aim": 0.0}
        next_tick = time.perf_counter()
        reward = 0.0
        while renderer.alive():
            for ev in renderer.poll():
                if ev.get("type") == "reset":
                    print(f"Episode {episode} reset by player: return {ret:+.2f} in {step} steps")
                    seed = None if seed is None else seed + 1
                    episode += 1
                    state = {"aim": 0.0}
                    new_episode()
                elif ev.get("type") == "bye":
                    raise KeyboardInterrupt
            inp = renderer.latest_input
            if args.policy == "random":
                action = env.action_space.sample()
            else:
                action = read_action(name, env, inp.get("keys", []), state)
            _, reward, term, trunc, _ = env.step(action)
            exporter.on_step(env, action)
            step += 1
            total += 1
            ret += float(reward)
            cam_cmd = inp.get("cam", "Center")
            if args.lowfi:
                cv2.imshow("GymBox (low-fi)", cv2.cvtColor(lowfi_grid(env.render(camera_pose=cam_cmd)), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            else:
                exporter.advance_camera(env, cam_cmd)
            msg = exporter.state(env)
            msg.update(step=step, t=round(step / float(exporter.fps), 4), cams=exporter.cameras(env))
            msg["hud"] = dict(exporter.hud(env), reward=float(reward), ret=ret, cam=cam_cmd)
            renderer.state(msg)
            if term or trunc:
                print(f"Episode {episode} {'terminated' if term else 'truncated'}: return {ret:+.2f} in {step} steps")
                seed = None if seed is None else seed + 1
                episode += 1
                state = {"aim": 0.0}
                new_episode()
            if args.max_steps and total >= args.max_steps:
                print(f"[hifi] reached --max-steps {args.max_steps}")
                break
            next_tick += 1.0 / fps
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    except (KeyboardInterrupt, BridgeError) as e:
        if isinstance(e, BridgeError):
            print(f"[hifi] {e}")
    finally:
        renderer.close()
        env.close()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Play an OrthoWorlds env with the Godot hi-fi renderer")
    p.add_argument("--env", required=True, help="env name or substring, e.g. 'tank', 'lunar', 'rescue'")
    p.add_argument("--godot", default=None, help="Godot binary (default: see renderer.py)")
    p.add_argument("--policy", choices=["keyboard", "random"], default="keyboard")
    p.add_argument("--fps", type=int, default=None, help="sim steps per second (default: env FPS)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--opponent", action="store_true", help="MultiCar Racing: add the autopilot opponent car")
    p.add_argument("--no-obstacles", action="store_true", help="held-out envs: obstacle-free twin")
    p.add_argument("--lowfi", action="store_true", help="also show the GymBox views in an OpenCV window")
    p.add_argument("--masks", action="store_true", help="also render object-ID masks (M in Godot)")
    p.add_argument("--window", type=int, nargs=2, default=[1280, 860])
    p.add_argument("--display", choices=["window", "xvfb"], default="window", help="xvfb: headless smoke test")
    p.add_argument("--max-steps", type=int, default=0, help="quit after this many steps (0 = run until closed)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
