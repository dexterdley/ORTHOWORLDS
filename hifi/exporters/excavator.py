"""
Excavator (ExcavatorEnv + ExcavatorOrthographicWrapper), side-view.

Box2D (x, y) = Godot (x, y), z = extrusion depth. The wrapper works in pixels (SCALE = 30 px/m);
every camera value is divided back to metres here.

GymBox views (excavator_orthographic.py, 800x600, 30 px/m -> ortho size 20 m):
  top   looks down, follows the chassis x (centre column), +z points down the screen, z = 0 on the
        centre row
  rear  (z + cx, 600 - y): no x follow, z = 0 on the centre column, y = 0 on the bottom row; the
        painter draws far x first, so the viewer stands at -X looking +X (+z to the right)
  side  (x - track_x + cx, 600 - y): follows the chassis x, y = 0 on the bottom row. The painter sorts
        larger z first (a viewer at -Z), but that projection only works for a camera on +Z looking
        -Z: real camera on +Z (the mirrored painter order is the GymBox artefact)
  fpv   orbit camera 10 m behind / 4 m above the chassis, pivot 4 m above the chassis centre,
        focal 400 px on 600 (vfov 73.7 deg), near clip 30 px = 1 m. Its lateral axis is -z (a
        mirrored image); the hi-fi view is a real camera at the wrapper's cam_pos looking at the pivot.
"""
import numpy as np
from game_envs.excavator_orthographic import (BIN_X, GROUND_Y, WORLD_HALF_W, ExcavatorEnv,
                                              ExcavatorOrthographicWrapper)

from .base import Exporter, body_xf_side, cam_ortho, cam_persp, fixture_shapes, r, vfov_from_focal

SCALE = 30.0
W_PX, H_PX = 800, 600
VIEW_M = H_PX / SCALE              # ortho size: 20 m
REAR_X = -62.0                      # rear camera plane (the terrain strip starts at x = -60)
SIDE_Z = 65.0                       # side camera plane (the terrain strip ends at z = 60)
# Ortho far planes: one consistent world, so they reach past the farthest scenery inside the frame
# (the sky card sits at 0.97 * far): rear -> the quarry's far end wall (x 300..700), side -> the
# back quarry wall (z -240..-520). Directional shadows stop at the Look's shadow_distance anyway.
REAR_FAR = 790.0
SIDE_FAR = 610.0


class ExcavatorExporter(Exporter):
    name = "excavator"
    title = "Excavator"
    fps = 50
    view_size = (W_PX, H_PX)

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self.load = 0.0

    def make_env(self):
        return ExcavatorOrthographicWrapper(ExcavatorEnv(**self.env_kwargs))

    def on_reset(self, env):
        self.load = 0.0

    def on_step(self, env, action):
        # engine load (exhaust smoke) from the commanded track / hydraulic speeds
        self.load = float(np.clip(np.mean(np.abs(np.nan_to_num(np.asarray(action, dtype=np.float64).ravel()[:4]))), 0.0, 1.0))

    def scene(self, env):
        b = env.unwrapped
        return {
            "ground_y": r(GROUND_Y), "half_w": r(WORLD_HALF_W), "bin_x": r(BIN_X),
            "bin": {"pos": [r(b.goal_bin.position[0]), r(b.goal_bin.position[1])], "shapes": fixture_shapes(b.goal_bin)},
            "face": {"pos": [r(b.rock_face.position[0]), r(b.rock_face.position[1])], "shapes": fixture_shapes(b.rock_face)},
            "payload": fixture_shapes(b.payload),
            "bucket": fixture_shapes(b.bucket),
        }

    def state(self, env):
        b = env.unwrapped
        if getattr(b, "chassis", None) is None:
            return {"xf": {}}
        xf = {k: body_xf_side(getattr(b, k)) for k in ("chassis", "boom", "dipper", "bucket", "payload")}
        # tread phase: the tracks roll without slip, so the shoe offset is the distance driven
        return {"xf": xf, "p": {"tread": r(b.chassis.position[0]), "vx": r(b.chassis.linearVelocity[0], 3),
                                   "load": r(self.load, 3)}}

    def cameras(self, env):
        b = env.unwrapped
        if getattr(b, "chassis", None) is None:
            cx, cy = 8.0, 3.0
        else:
            cx, cy = float(b.chassis.position[0]), float(b.chassis.position[1])
        mid_y = VIEW_M / 2.0        # y = 0 on the bottom row of the rear / side views
        px, py, pz = (c / SCALE for c in env.cam_pos)
        pivot = (cx, cy + env.camera_height / SCALE, 0.0)
        return [
            cam_ortho("top", (cx, 30.0, 0.0), (cx, 0.0, 0.0), (0.0, 0.0, -1.0), VIEW_M, 2.0, 34.0),
            cam_ortho("rear", (REAR_X, mid_y, 0.0), (REAR_X + 1.0, mid_y, 0.0), (0.0, 1.0, 0.0), VIEW_M, 1.0, REAR_FAR),
            cam_ortho("side", (cx, mid_y, SIDE_Z), (cx, mid_y, SIDE_Z - 1.0), (0.0, 1.0, 0.0), VIEW_M, 1.0, SIDE_FAR),
            cam_persp("fpv", (px, py, pz), pivot, (0.0, 1.0, 0.0), vfov_from_focal(env.fov, H_PX), 30.0 / SCALE, 2500.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        return {"text": "payload x %.1f  bin %.1f" % (b.payload.position[0], BIN_X)}
