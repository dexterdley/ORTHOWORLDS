"""
test_hifi.py — Automated checks of the hi-fi Godot envs against their GymBox twins.

For each env it runs a seeded random-action rollout with a camera-command schedule
(Center, Left, Up, Right, Down), renders GymBox and Godot side by side, and checks:
  shapes         hi-fi frames have exactly the GymBox frame shapes (all four views)
  not_blank      every hi-fi view has image content (std > 6)
  masks          the registered objects show up in the object-ID masks (views with any object)
  deterministic  re-rendering the same state gives the same pixels (max abs diff <= 2)
  camera_state   exporter.advance_camera() reproduces the wrapper's own render() camera state
                 (cam_x/y/z/yaw/pitch) over a rollout, so logged camera dicts stay comparable
                 (exporters that fix a GymBox camera bug declare it in camera_fix; reported, not failed)
  probe          Godot's cameras project test points where the camera dicts say (<= 1 px)
  godot_errors   no SCRIPT ERROR / ERROR lines in the Godot log
  views          cross-view consistency (tools/check_views.py): no object inside a view's frame is
                 hidden, clipped or culled there while another view shows it, except cutaways in
                 front of the subject
Outputs (hifi/test_out/<env>/): sheet_<step>.png (rows: GymBox, Godot, masks), hifi_<step>.png
(2x2 Godot grid at native resolution), and hifi/test_out/report.{json,md}.

Usage (from the repo root, GAME conda env):
  python hifi/test_hifi.py --env lunar                       # software Vulkan under xvfb, no window
  python hifi/test_hifi.py --all                             # every env with an exporter
  python hifi/test_hifi.py --all --godot hifi/tools/bin/Godot_v4.3-stable_win64_console.exe   # GPU
"""
import argparse
import copy
import json
import math
import os
import re
import sys
import time
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
HIFI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIFI)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from exporters import ENVS, available, get_exporter, resolve  # noqa: E402
from exporters.base import VIEWS, project  # noqa: E402
from renderer import GodotRenderer, mask_ids  # noqa: E402
from wrappers import HiFiOrthographicWrapper  # noqa: E402

sys.path.insert(0, os.path.join(HIFI, "tools"))
from check_views import analyse  # noqa: E402

CAM_SCHEDULE = ["Center"] * 6 + ["Left"] * 8 + ["Up"] * 6 + ["Right"] * 12 + ["Down"] * 8 + ["Center"] * 10
LABELS = ["Top", "Rear", "Side", "3D FPV"]


def schedule(t):
    return CAM_SCHEDULE[t % len(CAM_SCHEDULE)]


def env_kwargs(name, obstacles=True):
    return {"obstacles": obstacles} if get_exporter(name).heldout else {}


# ---------------------------------------------------------------------- checks
def check_camera_state(name, seed, steps=60):
    """advance_camera() must leave the wrapper in the same camera state as its render()."""
    ex_a, ex_b = get_exporter(name, **env_kwargs(name)), get_exporter(name, **env_kwargs(name))
    a, b = ex_a.make_env(), ex_b.make_env()
    a.reset(seed=seed)
    b.reset(seed=seed)
    a.action_space.seed(seed)
    worst = 0.0
    for t in range(steps):
        cmd = schedule(t)
        a.render(camera_pose=cmd)
        ex_b.advance_camera(b, cmd)
        la = [a.cam_x, a.cam_y, a.cam_z, getattr(a, "cam_yaw", 0.0), getattr(a, "cam_pitch", 0.0)]
        lb = list(ex_b.cam_log(b)) + [getattr(b, "cam_yaw", 0.0), getattr(b, "cam_pitch", 0.0)]
        worst = max(worst, float(np.max(np.abs(np.asarray(la, float) - np.asarray(lb, float)))))
        act = a.action_space.sample()
        ra, rb = a.step(act), b.step(act)
        if ra[2] or ra[3]:
            a.reset(seed=seed + t + 1)
            b.reset(seed=seed + t + 1)
    a.close()
    b.close()
    return worst


def probe_error(renderer, cams, w, h, rng):
    pts = []
    for c in cams:
        look = np.asarray(c["look"], float)
        for _ in range(3):
            pts.append(look + rng.uniform(-2.0, 2.0, size=3))
    got = renderer.probe(pts)
    worst = 0.0
    for c in cams:
        for p, g in zip(pts, got[c["view"]]):
            u, v, behind = project(c, p, w, h)
            if behind or g[2] or not np.isfinite(u):
                continue
            if c["proj"] == "p" and not (0 <= u <= w and 0 <= v <= h):
                continue
            worst = max(worst, abs(u - g[0]), abs(v - g[1]))
    return worst


def godot_log_errors(path):
    if not path or not os.path.exists(path):
        return []
    errs = []
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    benign = ("leaked at exit", "Pages in use exist at exit", "were never freed", "RIDs of type")
    for i, line in enumerate(lines):
        if any(b in line for b in benign):
            continue
        if "SCRIPT ERROR" in line or re.match(r"^\s*ERROR:", line):
            ctx = lines[i + 1].strip() if i + 1 < len(lines) else ""
            errs.append(f"{line.strip()} | {ctx}")
    return errs


# ---------------------------------------------------------------------- images
def colorize_mask(m):
    idx, cls = mask_ids(m)
    idx = idx.astype(np.int64)
    key = (idx * 2654435761 + cls.astype(np.int64) * 97) & 0xFFFFFF
    out = np.stack([(key >> 16) & 255, (key >> 8) & 255, key & 255], -1).astype(np.uint8)
    out[(idx == 0)] = 0
    return out


def sheet(lowfi, hifi, masks, cell_w=320):
    rows = []
    for label, views in (("GymBox", lowfi), ("Godot", hifi), ("mask", masks)):
        if views is None:
            continue
        cells = []
        for i, v in enumerate(views):
            h = int(round(cell_w * v.shape[0] / v.shape[1]))
            c = cv2.resize(np.ascontiguousarray(v[..., :3]), (cell_w, h), interpolation=cv2.INTER_AREA)
            cv2.putText(c, f"{label} {LABELS[i]}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(c, f"{label} {LABELS[i]}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cells.append(c)
        rows.append(np.hstack(cells))
    return np.vstack(rows)


def grid2x2(views):
    return np.vstack([np.hstack(views[:2]), np.hstack(views[2:])])


def save_rgb(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------- per env
def run_env(name, args):
    out_dir = os.path.join(HIFI, "test_out", name)
    os.makedirs(out_dir, exist_ok=True)
    res = {"env": name, "title": ENVS[name][0], "checks": {}, "notes": []}
    t0 = time.time()
    renderer = None
    try:
        res["checks"]["camera_state"] = {"max_abs_diff": check_camera_state(name, args.seed)}
        fix = get_exporter(name, **env_kwargs(name)).camera_fix
        res["checks"]["camera_state"]["pass"] = res["checks"]["camera_state"]["max_abs_diff"] < 1e-6 or bool(fix)
        if fix:
            res["checks"]["camera_state"]["deliberate_fix"] = fix
        renderer = GodotRenderer(godot=args.godot, display=args.display, window=(640, 420), timeout=args.timeout,
                                 log_tag=name)
        exporter = get_exporter(name, **env_kwargs(name))
        env = HiFiOrthographicWrapper(exporter, renderer=renderer, aov=("rgb", "mask"), pair=True)
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        res["renderer"] = {k: renderer.info.get(k) for k in ("renderer", "device")}
        res["mask_table"] = renderer.mask_table
        checkpoints = sorted({0, args.steps // 3, (2 * args.steps) // 3, args.steps - 1})
        shapes_ok, blank, mask_views, render_s, view_missing = True, [], [], [], {}
        w, h = exporter.size
        rng = np.random.default_rng(args.seed)
        for t in range(args.steps):
            cmd = schedule(t)
            t1 = time.time()
            hifi = env.render(camera_pose=cmd)
            render_s.append(time.time() - t1)
            lowfi = env.last_lowfi
            for hv, lv in zip(hifi, lowfi):
                if hv.shape != np.asarray(lv).shape:
                    shapes_ok = False
            if t in checkpoints:
                masks = [colorize_mask(env.last_masks[v]) for v in VIEWS]
                save_rgb(os.path.join(out_dir, f"sheet_{t:03d}.png"), sheet(lowfi, hifi, masks))
                save_rgb(os.path.join(out_dir, f"hifi_{t:03d}.png"), grid2x2(list(hifi)))
                for i, v in enumerate(hifi):
                    if float(np.std(v)) <= 6.0:
                        blank.append(f"{VIEWS[i]}@{t}")
                mask_views.append(sum(int(np.any(mask_ids(env.last_masks[v])[0] > 0)) for v in VIEWS))
                for path, rr in analyse(renderer.inspect(), exporter.cameras(env.env), w, h).items():
                    for v, why, n in rr["missing"]:
                        key = f"{path} [{v}:{why}]"
                        view_missing[key] = max(view_missing.get(key, 0), n)
            if t == checkpoints[-1]:
                # determinism: same state twice
                msg = env.exporter.state(env.env)
                msg["step"], msg["t"] = env._step, round(env._step / float(exporter.fps), 4)
                msg["cams"] = exporter.cameras(env.env)
                renderer.state(msg)
                f1 = renderer.render()
                renderer.state(copy.deepcopy(msg))
                f2 = renderer.render()
                diff = max(int(np.max(np.abs(f1[k].astype(np.int16) - f2[k].astype(np.int16)))) for k in f1)
                res["checks"]["deterministic"] = {"max_abs_diff": diff, "pass": diff <= 2}
                res["checks"]["probe"] = {"max_px_error": round(probe_error(renderer, msg["cams"], w, h, rng), 3)}
                res["checks"]["probe"]["pass"] = res["checks"]["probe"]["max_px_error"] <= 1.0
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                env.reset(seed=args.seed + t + 1)
        res["checks"]["shapes"] = {"pass": shapes_ok, "view_shape": list(np.asarray(lowfi[0]).shape)}
        res["checks"]["not_blank"] = {"pass": not blank, "blank": blank}
        res["checks"]["masks"] = {"pass": min(mask_views) >= 2, "views_with_objects": mask_views}
        res["checks"]["views"] = {"pass": not view_missing, "missing_objects": len(view_missing),
                                  "first": sorted(view_missing, key=lambda k: -view_missing[k])[:8]}
        res["timing"] = {"render_mean_s": round(float(np.mean(render_s[1:])), 4),
                         "first_render_s": round(render_s[0], 3)}
        env.env.close()
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    finally:
        log = renderer.log_path if renderer is not None else None
        if renderer is not None:
            renderer.close()
        errs = godot_log_errors(log)
        res["checks"]["godot_errors"] = {"pass": not errs, "count": len(errs), "first": errs[:8], "log": log}
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = "error" not in res and all(c.get("pass", False) for c in res["checks"].values())
    return res


def write_report(results):
    out = os.path.join(HIFI, "test_out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "report.json")
    old = {}
    if os.path.exists(path):
        try:
            old = {r["env"]: r for r in json.load(open(path))}
        except (ValueError, KeyError):
            old = {}
    for r in results:
        old[r["env"]] = r
    merged = [old[n] for n in ENVS if n in old]
    with open(path, "w") as f:
        json.dump(merged, f, indent=1, default=str)
    names = ["shapes", "not_blank", "masks", "deterministic", "camera_state", "probe", "views", "godot_errors"]
    lines = ["| env | result | " + " | ".join(names) + " | render s | renderer |", "|---" * (len(names) + 4) + "|"]
    for r in merged:
        cells = []
        for n in names:
            c = r["checks"].get(n)
            cells.append("-" if c is None else ("ok" if c.get("pass") else "FAIL"))
        rend = r.get("renderer", {}) or {}
        lines.append(f"| {r['title']} | {'PASS' if r['pass'] else 'FAIL'} | " + " | ".join(cells) +
                     f" | {r.get('timing', {}).get('render_mean_s', '-')} | {rend.get('device', '-')} |")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return merged


def main(args):
    names = available() if args.all else [resolve(e) for e in args.env]
    results = []
    for name in names:
        print(f"=== {ENVS[name][0]} ({name})", flush=True)
        r = run_env(name, args)
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        summary = ", ".join(f"{k}={'ok' if v.get('pass') else 'FAIL'}" for k, v in r["checks"].items())
        print(f"    {status} in {r['seconds']}s: {summary}", flush=True)
        if "error" in r:
            print("    error:", r["error"])
            print(r["traceback"])
        if r["checks"]["godot_errors"]["count"]:
            for e in r["checks"]["godot_errors"]["first"]:
                print("    godot:", e)
        for k in ("deterministic", "probe", "camera_state", "masks", "views"):
            if k in r["checks"]:
                print(f"    {k}: {r['checks'][k]}")
        print(f"    timing: {r.get('timing')}  renderer: {r.get('renderer')}")
    write_report(results)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test hi-fi Godot envs against GymBox")
    p.add_argument("--env", nargs="*", default=[], help="env names / substrings")
    p.add_argument("--all", action="store_true", help="every env that has an exporter")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--godot", type=str, default=None, help="Godot binary (default: see renderer.py)")
    p.add_argument("--display", default="auto", choices=["auto", "xvfb", "window"])
    p.add_argument("--timeout", type=float, default=180.0)
    sys.exit(main(p.parse_args()))
