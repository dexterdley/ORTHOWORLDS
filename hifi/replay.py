"""
replay.py — Record GymBox rollouts as renderer messages, then render them offline with Godot.

Recording needs no Godot: it runs the GymBox env and logs, per step, exactly what the hi-fi
wrapper would send (scene per episode, state + four cameras per frame) plus the action, reward
and camera command. Rendering replays the log through one or more Godot instances in lockstep,
so a dataset can be re-rendered (new looks, resolutions, masks) without re-running physics.

  python hifi/replay.py record --env tank --episodes 4 --steps 200 --out hifi/test_out/tank.jsonl.gz
  python hifi/replay.py render --log hifi/test_out/tank.jsonl.gz --out hifi/test_out/tank_frames \\
      --workers 2 --aov rgb mask [--godot hifi/tools/bin/Godot_v4.3-stable_win64_console.exe]

Output of `render`: <out>/ep<k>/<step>.npz with arrays top, rear, side, fpv (HxWx3 uint8, the
GymBox view size) and, with --aov mask, top_mask ... fpv_mask; plus <out>/meta.jsonl (action,
reward, camera command and camera state per frame).
"""
import argparse
import gzip
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from exporters import ENVS, get_exporter, resolve  # noqa: E402
from exporters.base import VIEWS  # noqa: E402

POSES = ["Center", "Up", "Down", "Left", "Right"]


def _jsonable(a):
    if isinstance(a, (np.integer,)):
        return int(a)
    if isinstance(a, (np.floating,)):
        return float(a)
    if isinstance(a, np.ndarray):
        return a.tolist()
    return a


# ---------------------------------------------------------------------- record
def record(args):
    name = resolve(args.env)
    kw = {"obstacles": not args.no_obstacles} if get_exporter(name).heldout else {}
    exporter = get_exporter(name, **kw)
    env = exporter.make_env()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with gzip.open(args.out, "wt") as f:
        f.write(json.dumps({"type": "header", "env": name, "size": list(exporter.size),
                            "view_size": list(exporter.view_size), "fps": exporter.fps, "kwargs": kw}) + "\n")
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            env.action_space.seed(args.seed + ep)
            exporter.on_reset(env)
            f.write(json.dumps({"type": "scene", "episode": ep, "msg": exporter.scene(env)}) + "\n")
            pose, hold = "Center", 0
            for t in range(args.steps):
                if hold <= 0:   # same camera-command schedule as collect_rollouts
                    pose, hold = rng.choice(POSES), rng.randint(8, 16)
                hold -= 1
                exporter.advance_camera(env, pose)
                msg = exporter.state(env)
                msg.update(step=t, t=round(t / float(exporter.fps), 4), cams=exporter.cameras(env))
                action = env.action_space.sample()
                _, reward, term, trunc, _ = env.step(action)
                exporter.on_step(env, action)
                cam = {"command": pose, "x": 0.0, "y": 0.0, "z": 0.0, "roll": float(getattr(env, "cam_roll", 0.0)),
                       "yaw": float(getattr(env, "cam_yaw", 0.0)), "pitch": float(getattr(env, "cam_pitch", 0.0))}
                cam["x"], cam["y"], cam["z"] = (float(c) for c in exporter.cam_log(env))
                f.write(json.dumps({"type": "frame", "episode": ep, "step": t, "msg": msg,
                                    "action": _jsonable(action), "reward": float(reward),
                                    "done": bool(term or trunc), "camera_state": cam}) + "\n")
                n += 1
                if term or trunc:
                    break
    env.close()
    print(f"recorded {n} frames of {ENVS[name][0]} -> {args.out}")
    return 0


# ---------------------------------------------------------------------- render
def _load(log):
    header, episodes = None, {}
    with gzip.open(log, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "header":
                header = rec
            elif rec["type"] == "scene":
                episodes.setdefault(rec["episode"], {"scene": rec["msg"], "frames": []})
            elif rec["type"] == "frame":
                episodes[rec["episode"]]["frames"].append(rec)
    return header, episodes


def _render_shard(header, items, args, worker):
    from renderer import GodotRenderer
    size = tuple(args.size) if args.size else tuple(header["size"])
    view_w, view_h = (args.size if args.size else header["view_size"])
    r = GodotRenderer(godot=args.godot, display=args.display, window=(640, 420))
    try:
        r.hello(header["env"], size, aov=args.aov, mode="lockstep", fps=header["fps"])
        done = 0
        for ep, data in items:
            r.scene(data["scene"])
            ep_dir = os.path.join(args.out, f"ep{ep:04d}")
            os.makedirs(ep_dir, exist_ok=True)
            for rec in data["frames"]:
                r.state(rec["msg"])
                frames = r.render()
                out = {}
                for (view, aov), img in frames.items():
                    if img.shape[1] != view_w or img.shape[0] != view_h:
                        img = cv2.resize(img, (view_w, view_h),
                                         interpolation=cv2.INTER_NEAREST if aov == "mask" else cv2.INTER_AREA)
                    out[view if aov == "rgb" else f"{view}_mask"] = img
                np.savez_compressed(os.path.join(ep_dir, f"{rec['step']:05d}.npz"), **out)
                done += 1
        return done, r.mask_table
    finally:
        r.close()


def render(args):
    header, episodes = _load(args.log)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "meta.jsonl"), "w") as f:
        for ep, data in sorted(episodes.items()):
            for rec in data["frames"]:
                f.write(json.dumps({k: rec[k] for k in ("episode", "step", "action", "reward", "done", "camera_state")}) + "\n")
    items = sorted(episodes.items())
    shards = [items[i::args.workers] for i in range(args.workers)]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda w: _render_shard(header, shards[w], args, w), range(args.workers)))
    n = sum(d for d, _ in results)
    with open(os.path.join(args.out, "mask_table.json"), "w") as f:
        json.dump(results[0][1] if results else {}, f, indent=1)
    dt = time.time() - t0
    print(f"rendered {n} frames in {dt:.1f}s ({n / max(dt, 1e-6):.1f} fps) -> {args.out}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Record GymBox rollouts / render them with the Godot hi-fi renderer")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("record")
    pr.add_argument("--env", required=True)
    pr.add_argument("--episodes", type=int, default=2)
    pr.add_argument("--steps", type=int, default=150)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--no-obstacles", action="store_true")
    pr.add_argument("--out", required=True)
    pd = sub.add_parser("render")
    pd.add_argument("--log", required=True)
    pd.add_argument("--out", required=True)
    pd.add_argument("--workers", type=int, default=1, help="parallel Godot instances (episodes are sharded)")
    pd.add_argument("--aov", nargs="+", default=["rgb"], choices=["rgb", "mask"])
    pd.add_argument("--size", type=int, nargs=2, default=None, help="render/output size override (w h)")
    pd.add_argument("--godot", default=None)
    pd.add_argument("--display", default="auto", choices=["auto", "xvfb", "window"])
    args = p.parse_args()
    return record(args) if args.cmd == "record" else render(args)


if __name__ == "__main__":
    sys.exit(main())
