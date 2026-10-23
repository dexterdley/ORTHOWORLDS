"""
Wrecking Ball (WreckingBallEnv + WreckingBallOrthographicWrapper), side-view.

Box2D (x, y) = Godot (x, y), z = extrusion depth. The wrapper works in pixels (SCALE = 30 px/m);
camera values are divided back to metres here. Same view code as the Excavator wrapper:
  top   looks down, follows the chassis x, +z down the screen, z = 0 on the centre row (20 m tall)
  rear  (z + cx, 600 - y): no x follow, y = 0 on the bottom row; far x painted first -> viewer at -X
  side  (x - track_x + cx, 600 - y): follows the chassis x, y = 0 on the bottom row; its painter
        sorts larger z first (mirrored), the projection is a camera on +Z looking -Z (used here)
  fpv   orbit camera 12 m behind / 4.5 m above the chassis, pivot 4.5 m above the chassis centre,
        focal 400 px on 600 (vfov 73.7 deg), near clip 15 px = 0.5 m; real camera at cam_pos.
Cable: env.cable_points() (boom-tip sheave -> top of the ball, with the visual sag when slack).
Knocked blocks raise a dust puff whose phase comes from the steps since the knock (latched flags).
"""
import numpy as np
from game_envs.wrecking_ball_orthographic import (BARRIER_HALF, BARRIER_X, BLOCK_HALF, GROUND_TOP, WreckingBallEnv,
                                                  WreckingBallOrthographicWrapper)

from .base import Exporter, body_xf_side, cam_ortho, cam_persp, r, rv, vfov_from_focal, xf_side

SCALE = 30.0
W_PX, H_PX = 800, 600
VIEW_M = H_PX / SCALE              # ortho size: 20 m
REAR_X = -72.0                      # rear camera plane (the terrain strip starts at x = -70)
SIDE_Z = 65.0                       # side camera plane (the terrain strip ends at z = 60)
DUST_LIFE = 45                      # steps a knock dust puff lives
# Ortho far planes: one consistent world, so they reach past the farthest scenery inside the frame
# (the sky card sits at 0.97 * far). Rear (fixed frame): the city block at x 274..293 behind the
# site. Side (follows the crane): any building of the backdrop ring behind the lot (z >= -343).
REAR_FAR = 385.0
SIDE_FAR = 425.0


class WreckingBallExporter(Exporter):
    name = "wrecking_ball"
    title = "Wrecking Ball"
    fps = 50
    view_size = (W_PX, H_PX)
    mask_groups = {"block": ["target_block", "base_block"]}

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self.knock = {}
        self.load = 0.0

    def make_env(self):
        return WreckingBallOrthographicWrapper(WreckingBallEnv(**self.env_kwargs))

    def on_reset(self, env):
        self.knock = {}
        self.load = 0.0

    def on_step(self, env, action):
        b = env.unwrapped
        self.load = float(np.clip(np.mean(np.abs(np.nan_to_num(np.asarray(action, dtype=np.float64).ravel()[:3]))), 0.0, 1.0))
        for i, k in enumerate(getattr(b, "knocked", [])):
            if k and i not in self.knock:
                p = b.blocks[i].position
                self.knock[i] = (int(b.steps), float(p[0]), float(p[1]))

    def scene(self, env):
        b = env.unwrapped
        return {
            "ground_y": r(GROUND_TOP), "barrier": [r(BARRIER_X), r(GROUND_TOP + BARRIER_HALF[1]), r(BARRIER_HALF[0]), r(BARRIER_HALF[1])],
            "block_half": rv(BLOCK_HALF), "site_x": r(b.site_x), "pillar_x": rv(b.pillar_x),
            "target": [int(t) for t in b.is_target],
        }

    def state(self, env):
        b = env.unwrapped
        if getattr(b, "chassis", None) is None:
            return {"xf": {}}
        xf = {"chassis": body_xf_side(b.chassis), "boom": body_xf_side(b.boom), "ball": body_xf_side(b.ball)}
        for i, blk in enumerate(b.blocks):
            xf[f"block{i}"] = body_xf_side(blk)
        dyn = []
        for i, (t0, x, y) in sorted(self.knock.items()):
            age = int(b.steps) - t0
            if 0 <= age < DUST_LIFE:
                dyn.append({"id": f"dust{i}", "k": "dust", "xf": xf_side(x, y, 0.0), "t": r(age / float(DUST_LIFE), 3)})
        return {"xf": xf, "dyn": dyn, "p": {
            "cable": [rv(p) for p in b.cable_points()], "knocked": [int(k) for k in b.knocked],
            "tread": r(b.chassis.position[0]), "cable_len": r(b.cable_length, 3),
            "load": r(self.load, 3)}}

    def cameras(self, env):
        b = env.unwrapped
        if getattr(b, "chassis", None) is None:
            cx, cy = 0.0, 3.0
        else:
            cx, cy = float(b.chassis.position[0]), float(b.chassis.position[1])
        mid_y = VIEW_M / 2.0
        px, py, pz = (c / SCALE for c in env.cam_pos)
        pivot = (cx, cy + env.camera_height / SCALE, 0.0)
        return [
            cam_ortho("top", (cx, 30.0, 0.0), (cx, 0.0, 0.0), (0.0, 0.0, -1.0), VIEW_M, 2.0, 34.0),
            cam_ortho("rear", (REAR_X, mid_y, 0.0), (REAR_X + 1.0, mid_y, 0.0), (0.0, 1.0, 0.0), VIEW_M, 1.0, REAR_FAR),
            cam_ortho("side", (cx, mid_y, SIDE_Z), (cx, mid_y, SIDE_Z - 1.0), (0.0, 1.0, 0.0), VIEW_M, 1.0, SIDE_FAR),
            cam_persp("fpv", (px, py, pz), pivot, (0.0, 1.0, 0.0), vfov_from_focal(env.fov, H_PX), 15.0 / SCALE, 2500.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        return {"text": "blocks knocked %d/%d  cable %.1f m" % (b._targets_knocked(), b.num_targets, b.cable_length)}
