"""
check_views.py — Cross-view consistency of the hi-fi scenes: is every object that lies inside a
view's frame actually drawn there?

An object can go missing from one view (while the others show it) through these mechanisms, all
of which are visible in the scene graph, so this test is exact rather than image-based:
  layer   hidden in that view by set_views() or the per-step state["vis"] cut
  near    inside the frame laterally, but entirely in front of the near plane
  far     inside the frame laterally, but entirely behind the far plane
  card    behind the ortho sky card (placed at 0.97 * far), so painted over by the backdrop
  range   culled by its visibility_range_end at that camera's distance
For each geometry (each MultiMesh instance separately) and each of the four views this reports
the status, then lists every object that is drawn in some view but missing from another view
whose frame contains it. Objects hidden ("layer") or near-clipped ("near") while they lie wholly
between the camera and its look point (the subject) are "cutaway": section cuts that keep the
subject visible, which are allowed. Section-cap geometry (meta "section_cap") is exempt.

Policy: one consistent world. Scenery appears in every view whose frame contains it, except where
it is cut away in front of the subject.

Usage (from the repo root, GAME conda env):
  python hifi/tools/check_views.py --all                    # writes hifi/test_out/views_report.{json,md}
  python hifi/tools/check_views.py --env tank --steps 60
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
HIFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HIFI)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from exporters import ENVS, available, get_exporter, resolve  # noqa: E402
from exporters.base import VIEWS, _basis  # noqa: E402
from renderer import GodotRenderer  # noqa: E402
from wrappers import HiFiOrthographicWrapper  # noqa: E402

CAM_SCHEDULE = ["Center"] * 6 + ["Left"] * 8 + ["Up"] * 6 + ["Right"] * 12 + ["Down"] * 8 + ["Center"] * 10
MISSING = ("layer", "near", "far", "card", "range")


def corners(b):
    x0, y0, z0, x1, y1, z1 = b
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def classify(cam, box, w, h, layers, vr_end, idx, node_box):
    """Status of one AABB in one view: 'shown', 'out' (outside the frame) or a MISSING reason."""
    pos, fwd, right, up = _basis(cam)
    c = corners(box) - pos
    x, y, z = c @ right, c @ up, c @ fwd
    near, far = float(cam["near"]), float(cam["far"])
    if cam["proj"] == "o":
        hh = 0.5 * float(cam["size"])
        hw = hh * w / h
        if x.max() < -hw or x.min() > hw or y.max() < -hh or y.min() > hh:
            return "out"
    else:
        t = np.tan(np.radians(float(cam["fov"])) / 2.0)
        tw = t * w / h
        zz = np.maximum(z, 1e-6)
        # all corners outside one side plane (or all behind the eye) -> outside the frustum
        if (z <= 0).all() or (x > tw * zz).all() or (x < -tw * zz).all() or (y > t * zz).all() or (y < -t * zz).all():
            return "out"
    if z.max() < near:
        return "near"
    if z.min() > far:
        return "far"
    if cam["proj"] == "o" and z.min() > 0.97 * far:
        return "card"
    if not (layers & 1 or layers & (1 << (idx + 1))):
        return "layer"
    if vr_end > 0:
        nb = np.asarray(node_box, float)
        if np.linalg.norm(0.5 * (nb[:3] + nb[3:]) - pos) > vr_end:
            return "range"
    return "shown"


def in_front_of_subject(cam, box):
    """True when the box lies wholly between the camera and its look point (a cutaway occluder)."""
    pos, fwd, _, _ = _basis(cam)
    z = (corners(box) - pos) @ fwd
    return z.max() < float(np.dot(np.asarray(cam["look"], float) - pos, fwd))


def union(boxes):
    a = np.asarray(boxes, float)
    return list(a[:, :3].min(0)) + list(a[:, 3:].max(0))


def analyse(nodes, cams, w, h):
    """-> {path: {"cls", "owner", "n", "status": {view: {status: count}}, "missing": [...], "cutaway": [...]}}"""
    cams = {c["view"]: c for c in cams}
    out = {}
    for nd in nodes:
        if not nd["aabbs"] or nd.get("cap"):
            continue
        nb = union(nd["aabbs"])
        status = {v: defaultdict(int) for v in VIEWS}
        missing, cutaway = defaultdict(int), defaultdict(int)
        for box in nd["aabbs"]:
            per = {v: classify(cams[v], box, w, h, nd["layers"], nd["vr"][1], i, nb) for i, v in enumerate(VIEWS)}
            for v, s in per.items():
                status[v][s] += 1
            if not any(s == "shown" for s in per.values()):
                continue
            for v, s in per.items():
                if s in MISSING:
                    key = (v, s)
                    if s in ("layer", "near") and in_front_of_subject(cams[v], box):
                        cutaway[key] += 1
                    else:
                        missing[key] += 1
        out[nd["path"]] = {"cls": nd["cls"], "owner": nd["owner"], "n": nd["n"], "box": [round(float(c), 2) for c in nb],
                           "status": {v: dict(s) for v, s in status.items()},
                           "missing": [[v, s, n] for (v, s), n in missing.items()],
                           "cutaway": [[v, s, n] for (v, s), n in cutaway.items()]}
    return out


def reveal_sheet(name, t, env, renderer, hifi, revealed):
    """Renders the same state with every geometry shown in every view; saves GymBox / Godot /
    revealed / changed-pixel rows and records the changed-pixel fraction per view."""
    renderer.reveal(True)
    frames = renderer.render()
    renderer.reveal(False)
    rev = [env._fit(frames[(v, "rgb")]) for v in VIEWS]
    diffs = [np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 24 for a, b in zip(hifi, rev)]
    revealed[str(t)] = {v: round(float(d.mean()), 4) for v, d in zip(VIEWS, diffs)}
    rows = []
    for imgs in (env.last_lowfi, hifi, rev, [np.where(d[..., None], np.uint8([255, 0, 255]), (b * 0.35).astype(np.uint8)) for d, b in zip(diffs, rev)]):
        cells = []
        for v in imgs:
            v = np.ascontiguousarray(np.asarray(v)[..., :3])
            cells.append(cv2.resize(v, (320, int(round(320 * v.shape[0] / v.shape[1]))), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(cells))
    out = os.path.join(HIFI, "test_out", name)
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, f"views_{t:03d}.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def run_env(name, args, renderer):
    kw = {"obstacles": True} if get_exporter(name).heldout else {}
    exporter = get_exporter(name, **kw)
    env = HiFiOrthographicWrapper(exporter, renderer=renderer, aov=("rgb",), pair=True)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    w, h = exporter.size
    checkpoints = sorted({0, args.steps // 2, args.steps - 1})
    merged, revealed = {}, {}
    for t in range(args.steps):
        hifi = env.render(camera_pose=CAM_SCHEDULE[t % len(CAM_SCHEDULE)])
        if t in checkpoints:
            reveal_sheet(name, t, env, renderer, hifi, revealed)
            res = analyse(renderer.inspect(args.max_instances), exporter.cameras(env.env), w, h)
            for path, r in res.items():
                m = merged.setdefault(path, dict(r, missing={}, cutaway={}, steps=[]))
                m["status"] = r["status"]
                for kind in ("missing", "cutaway"):
                    for v, s, n in r[kind]:
                        k = f"{v}:{s}"
                        m[kind][k] = max(m[kind].get(k, 0), n)
                if r["missing"]:
                    m["steps"].append(t)
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset(seed=args.seed + t + 1)
    env.env.close()
    merged["__revealed__"] = revealed
    return merged


def write_report(all_res, out=None):
    out = out or os.path.join(HIFI, "test_out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "views_report.json"), "w") as f:
        json.dump(all_res, f, indent=1)
    lines = ["# Cross-view consistency", "",
             "Objects drawn in at least one view but missing from another view whose frame contains them.",
             "`layer` = set_views / state vis, `near`/`far` = clip planes, `card` = behind ortho sky card, "
             "`range` = visibility_range_end.", ""]
    for name, res in all_res.items():
        rev = res.get("__revealed__", {})
        res = {p: r for p, r in res.items() if p != "__revealed__"}
        bad = {p: r for p, r in res.items() if r["missing"]}
        cut = {p: r for p, r in res.items() if r["cutaway"] and not r["missing"]}
        lines.append(f"## {ENVS[name][0]}: {len(bad)} with missing views, {len(cut)} cutaway-only, {len(res)} geometries")
        lines.append(f"pixels changed when everything is revealed (per step): {rev}")
        for p, r in sorted(bad.items(), key=lambda kv: -sum(kv[1]["missing"].values())):
            lines.append(f"- `{p}` ({r['cls']}, n={r['n']}{', owner ' + r['owner'] if r['owner'] else ''}) "
                         f"missing {r['missing']} box={r['box']}")
        for p, r in cut.items():
            lines.append(f"- cutaway `{p}` {r['cutaway']}")
        lines.append("")
    with open(os.path.join(out, "views_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    names = available() if args.all else [resolve(e) for e in args.env]
    renderer = GodotRenderer(godot=args.godot, display=args.display, window=(640, 420), timeout=180.0, log_tag="views")
    all_res = {}
    try:
        for name in names:
            t0 = time.time()
            res = run_env(name, args, renderer)
            all_res[name] = res
            write_report({name: res}, os.path.join(HIFI, "test_out", name))
            bad = sum(1 for p, r in res.items() if p != "__revealed__" and r["missing"])
            print(f"{ENVS[name][0]:20s} {len(res):4d} geometries, {bad:3d} missing in some view  ({time.time() - t0:.0f}s)",
                  flush=True)
    finally:
        renderer.close()
    if args.all:
        write_report(all_res)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--env", nargs="*", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--max-instances", type=int, default=4000)
    p.add_argument("--godot", default=None)
    p.add_argument("--display", default="auto", choices=["auto", "xvfb", "window"])
    main(p.parse_args())
