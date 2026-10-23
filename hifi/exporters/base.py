"""
base.py — Shared pieces of the per-env exporters.

An exporter turns a GymBox orthographic wrapper (the low-fi env from ortho/game_envs) into the
messages the Godot renderer understands (see hifi/bridge.py):
  scene(env)    once per reset: static, procedurally generated content (terrain, obstacles, ...)
  state(env)    every step: "xf" transforms of persistent objects, "dyn" transient objects
                (projectiles, effects), "p" extra signals (flags, flashes, gas front, ...)
  cameras(env)  every render: the four views (top, rear, side, fpv) as camera dicts

Coordinate conventions (Godot: metres, +Y up, right-handed):
  side-view envs  (Box2D x-y is the vertical plane):  Box2D (x, y), extrusion z -> Godot (x, y, z),
                  body angle a -> rotation about +Z.             helpers: xf_side, P_side
  top-down envs   (Box2D x-y is the ground plane):    Box2D (x, y), height h -> Godot (x, h, -y),
                  body angle a -> rotation about +Y (yaw).       helpers: xf_top, P_top
GymBox wrappers sometimes work in pixel units (SCALE px per metre); exporters divide those out.

Camera policy (decided in the plan): each view keeps the GymBox wrapper's framing (what it
follows, its px-per-metre scale, its viewing axis) but is a real camera, so mirrored painter
orders, fake horizons and agent-always-on-top artefacts disappear. The FPV uses the wrapper's own
orbit state (cam_yaw / cam_pitch and its cam_pos), so camera commands and logged camera dicts
are unchanged.
"""
import math

import numpy as np

VIEWS = ("top", "rear", "side", "fpv")


def r(x, nd=4):
    return round(float(x), nd)


def rv(v, nd=4):
    return [round(float(c), nd) for c in v]


# ---------------------------------------------------------------------- rotations
def quat_axis(axis, a):
    s = math.sin(a / 2.0)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(a / 2.0)]


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


def xf(pos, quat=(0.0, 0.0, 0.0, 1.0)):
    return rv(pos) + rv(quat, 5)


def xf_side(x, y, a=0.0, z=0.0):
    """Side-view body pose: Box2D (x, y, angle) -> Godot transform (rotation about +Z)."""
    return xf((x, y, z), quat_axis((0.0, 0.0, 1.0), a))


def xf_top(x, y, a=0.0, h=0.0):
    """Top-down body pose: Box2D (x, y, angle) at height h -> Godot transform (yaw about +Y)."""
    return xf((x, h, -y), quat_axis((0.0, 1.0, 0.0), a))


def P_side(x, y, z=0.0):
    return [float(x), float(y), float(z)]


def P_top(x, y, h=0.0):
    return [float(x), float(h), -float(y)]


def body_xf_side(body, z=0.0):
    return xf_side(body.position[0], body.position[1], body.angle, z)


def body_xf_top(body, h=0.0):
    return xf_top(body.position[0], body.position[1], body.angle, h)


# ---------------------------------------------------------------------- Box2D geometry
def fixture_shapes(body, world=False):
    """Fixture shapes of a Box2D body: [{"poly": [[x, y], ...]}] or [{"circle": [cx, cy, r]}].
    Local coordinates unless world=True."""
    out = []
    for f in body.fixtures:
        sh = f.shape
        if hasattr(sh, "vertices"):
            vs = [body.transform * v for v in sh.vertices] if world else list(sh.vertices)
            out.append({"poly": [rv(v) for v in vs]})
        elif hasattr(sh, "radius"):
            c = body.transform * sh.pos if world else sh.pos
            out.append({"circle": [r(c[0]), r(c[1]), r(sh.radius)]})
    return out


# ---------------------------------------------------------------------- cameras
def vfov_from_focal(focal_px, height_px):
    """Vertical field of view (degrees) of a pinhole with the given focal length in pixels."""
    return math.degrees(2.0 * math.atan(0.5 * height_px / focal_px))


def cam_persp(view, pos, look, up=(0.0, 1.0, 0.0), vfov=60.0, near=0.05, far=4000.0):
    return {"view": view, "proj": "p", "pos": rv(pos), "look": rv(look), "up": rv(up),
            "fov": r(vfov), "near": r(near), "far": r(far)}


def cam_ortho(view, pos, look, up=(0.0, 1.0, 0.0), size=10.0, near=0.05, far=4000.0):
    """Orthographic camera; size = visible height in metres (= frame height px / GymBox px per m)."""
    return {"view": view, "proj": "o", "pos": rv(pos), "look": rv(look), "up": rv(up),
            "size": r(size), "near": r(near), "far": r(far)}


def _basis(cam):
    pos, look, up = (np.asarray(cam[k], dtype=np.float64) for k in ("pos", "look", "up"))
    fwd = look - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    return pos, fwd, right, true_up


def project(cam, point, w, h):
    """Pixel (u, v) of a Godot-space point in a camera dict (Godot KEEP_HEIGHT conventions).
    Used to cross-check the renderer (hifi/test_hifi.py probe test)."""
    pos, fwd, right, up = _basis(cam)
    d = np.asarray(point, dtype=np.float64) - pos
    x, y, z = d @ right, d @ up, d @ fwd
    if cam["proj"] == "o":
        s = h / cam["size"]
        return w / 2 + x * s, h / 2 - y * s, z < 0
    f = (h / 2) / math.tan(math.radians(cam["fov"]) / 2)
    if z <= 1e-9:
        return float("nan"), float("nan"), True
    return w / 2 + f * x / z, h / 2 - f * y / z, False


# ---------------------------------------------------------------------- exporter base
class Exporter:
    """Base class: subclass per env (see hifi/exporters/*.py)."""
    name = "base"            # Godot world folder: hifi/godot/envs/<name>/world.gd
    title = "Base"           # display name, as in ortho/utils.py registries
    fps = 50
    view_size = (600, 400)   # (w, h) of each GymBox view, i.e. what render() returns
    render_size = None       # (w, h) Godot renders at, if the GymBox aspect differs (resized after)
    heldout = False          # held-out obstacle-transfer env (ortho/utils.py _build_heldout_registry)
    camera_fix = None        # str: why cam_log() deliberately differs from the wrapper (a GymBox camera bug fix)
    mask_groups = {}         # render_masks() aliases: {"obstacle": ["hurdle", "crate", ...]} (classes or ids)

    def __init__(self, **env_kwargs):
        self.env_kwargs = env_kwargs

    # -- env
    def make_env(self):
        raise NotImplementedError

    # -- messages
    def scene(self, env):
        return {}

    def state(self, env):
        return {"xf": {}}

    def cameras(self, env):
        raise NotImplementedError

    def on_reset(self, env):
        """Called after every env.reset() (before scene())."""

    def on_step(self, env, action):
        """Called after every env.step(action): keep what state() needs but the env does not store
        (e.g. which engines the last action fired)."""

    def hud(self, env):
        """Live-mode HUD extras: {"text": str}."""
        return {}

    # -- camera state (must mirror what the wrapper's render() does before it draws)
    def advance_camera(self, env, command):
        env.update_camera_pose(command)

    def cam_log(self, env):
        """cam_x/y/z in the wrapper's own units, as its render() stores them."""
        return tuple(float(c) for c in env.cam_pos)

    # -- helpers
    @property
    def size(self):
        return self.render_size or self.view_size

    def base(self, env):
        return env.unwrapped
