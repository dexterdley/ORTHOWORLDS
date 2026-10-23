"""
Toxic Gas Escape (ToxicGasEscapeEnv + ToxicGasEscapeOrthographicWrapper), top-down.

GymBox views (toxic_gas_escape_orthographic.py, 800x600, SCALE 20 px/m; the wrapper's camera state is in px):
  top   plan view, north up, centred on the car, 20 px/m
  rear  section elevation looking North (+y), East to the right, 140 px/m (zoom 7), ground line on row
        468 (VH * 0.78) -> frame centre at height 1.2 m; geometry wholly South of the car is dropped
  side  section elevation looking West (-x), North to the right, same scale / ground line; geometry
        wholly East of the car is dropped
  fpv   spring-arm chase camera: pivot = the car at target_height (0.45 m), arm camera_dist 6.5 m
        shortened by wall hits (min 1.8 m), camera_height 2.5 m absolute, orbit cam_yaw / cam_pitch,
        focal 420 px (vfov 71.1 deg), near clip 0.5 m

Hi-fi cameras are real cameras with the same framing, looking at one consistent world: every
object is drawn in every view whose frame contains it. The only exception is the cutaway in front
of the car: walls (and the exit gantry / door) lying wholly on the viewer's side of the car are
hidden in that view through state["vis"], the exterior props / sheds South of the facility (Rear)
or East of it (Side) are hidden by the world, and the ground is a slab whose South / East faces sit
just behind the Rear / Side near planes, so its cross-section fills the band below the ground line
(GymBox's checkered ground band). The Rear / Side far planes reach past the farthest skyline shed
in their frames (the ortho sky card sits at 0.97 * far). GymBox quirks
dropped: car always drawn on top, depth fade of the scenery, fake FPV horizon, per-view gas hacks
(one volumetric gas body with a hard front at gas_x is used in every view).
"""
import math

from game_envs.toxic_gas_escape_orthographic import (ARENA_H, ARENA_W, MAX_STEER, SCALE, WALL_H,
                                                      ToxicGasEscapeEnv, ToxicGasEscapeOrthographicWrapper)

from .base import Exporter, P_top, cam_ortho, cam_persp, r, vfov_from_focal, xf_top

W_PX, H_PX = 800, 600
WHEEL_R = 0.22                  # rover wheel radius (Box2D wheel boxes are 0.44 m long)
GATE_BACK = 1.0                 # exit gantry this far West of exit_pos (GymBox arch centre)
REAR_PLANE_Y = -3.1             # Rear near plane (Box2D y); the ground slab's South face is at -3.0
SIDE_PLANE_X = ARENA_W + 3.1    # Side near plane (Box2D x); the slab's East face is at ARENA_W + 3.0
REAR_CAM_Y = -40.0
SIDE_CAM_X = ARENA_W + 40.0
TOP_CAM_H = 30.0
REAR_SCENERY_Y = 102.0          # beyond the North skyline shed (world.gd: y up to ah + 79 = 101)
SIDE_SCENERY_X = -74.0          # beyond the West skyline shed (world.gd: x down to -73)
CARD = 0.97                     # the ortho sky card sits at CARD * far


class ToxicGasEscapeExporter(Exporter):
    name = "toxic_gas_escape"
    title = "Toxic Gas Escape"
    fps = 50
    view_size = (W_PX, H_PX)

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self.phase = 0.0
        self.steer = 0.0
        self.v_fwd = 0.0

    def make_env(self):
        return ToxicGasEscapeOrthographicWrapper(ToxicGasEscapeEnv(**self.env_kwargs))

    # ------------------------------------------------------------------ camera state
    def advance_camera(self, env, command):
        # Mirror everything the wrapper's render() updates before drawing
        env.update_camera_pose(command)
        if getattr(env.unwrapped, "player", None) is not None:
            env.pivot_depth = env._arm()[2]
            env.cam_x, env.cam_y, env.cam_z = env.cam_pos

    def on_reset(self, env):
        self.phase = 0.0
        self.steer = 0.0
        self.v_fwd = 0.0

    def on_step(self, env, action):
        b = env.unwrapped
        if b.player is None:
            return
        a = float(b.player.angle)
        v = b.player.linearVelocity
        self.v_fwd = float(v[0] * math.cos(a) + v[1] * math.sin(a))
        self.phase = (self.phase + self.v_fwd / self.fps / WHEEL_R) % (2.0 * math.pi * 64.0)
        try:
            st = float(action[1])
        except (TypeError, IndexError):
            st = 0.0
        self.steer = max(-1.0, min(1.0, st)) * MAX_STEER

    # ------------------------------------------------------------------ messages
    @staticmethod
    def _gate(b):
        """Exit gantry x and the free corridor span (y0, y1) it bridges (nearest wall faces)."""
        ex, ey = float(b.exit_pos[0]), float(b.exit_pos[1])
        gx = ex - GATE_BACK
        y0, y1 = 0.0, ARENA_H
        for cx, cy, hw, hh in b.wall_defs:
            if abs(gx - cx) <= hw:
                if cy + hh <= ey:
                    y0 = max(y0, cy + hh)
                elif cy - hh >= ey:
                    y1 = min(y1, cy - hh)
        return gx, y0, y1

    def scene(self, env):
        b = env.unwrapped
        gx, y0, y1 = self._gate(b)
        return {"walls": [[r(cx), r(cy), r(hw), r(hh)] for cx, cy, hw, hh in b.wall_defs],
                "exit": [r(b.exit_pos[0]), r(b.exit_pos[1])], "gate": [r(gx), r(y0), r(y1)],
                "W": ARENA_W, "H": ARENA_H, "wall_h": WALL_H,
                "slab": [-25.0, REAR_PLANE_Y + 0.1, SIDE_PLANE_X - 0.1, 40.0]}

    def state(self, env):
        b = env.unwrapped
        if b.player is None:
            return {"xf": {}}
        px, py = float(b.player.position[0]), float(b.player.position[1])
        rear, side = [], []
        for i, (cx, cy, hw, hh) in enumerate(b.wall_defs):
            if cy + hh < py:
                rear.append(f"w{i}")
            if cx - hw > px:
                side.append(f"w{i}")
        gx, y0, y1 = self._gate(b)
        ex, ey = float(b.exit_pos[0]), float(b.exit_pos[1])
        if ey + 2.1 < py:        # sign arm at ey, beacons at ey +- 1.9
            rear.append("gate")
        if gx - 0.3 > px:
            side.append("gate")
        if ey + 1.8 < py:
            rear.append("door")
        side.append("door")   # on the East wall's inner face: always East of the car
        return {"xf": {"car": xf_top(px, py, float(b.player.angle), 0.0)},
                "vis": {"rear": rear, "side": side},
                "p": {"gas_x": r(b.gas_x), "phase": r(self.phase), "steer": r(self.steer),
                      "speed": r(self.v_fwd), "gap": r(px - b.gas_x),
                      "released": int(b.step_count >= b.gas_delay)}}

    def cameras(self, env):
        b = env.unwrapped
        if b.player is None:
            px, py = ARENA_W / 2, ARENA_H / 2
        else:
            px, py = float(b.player.position[0]), float(b.player.position[1])
        top_size = H_PX / SCALE
        ppm = SCALE * env.ortho_zoom                          # 140 px/m
        sec_size = H_PX / ppm
        mid_h = (H_PX * env.ortho_ground - H_PX / 2) / ppm    # height on the centre row (1.2 m)
        rear_near = REAR_PLANE_Y - REAR_CAM_Y
        side_near = SIDE_CAM_X - SIDE_PLANE_X
        rear_far = (REAR_SCENERY_Y - REAR_CAM_Y) / CARD
        side_far = (SIDE_CAM_X - SIDE_SCENERY_X) / CARD
        cx, cy, cz = (c / SCALE for c in env.cam_pos)
        th = env.target_height / SCALE
        return [
            cam_ortho("top", P_top(px, py, TOP_CAM_H), P_top(px, py, 0.0), (0.0, 0.0, -1.0), top_size,
                      TOP_CAM_H - 14.0, TOP_CAM_H + 2.0),
            cam_ortho("rear", P_top(px, REAR_CAM_Y, mid_h), P_top(px, py, mid_h), (0.0, 1.0, 0.0), sec_size,
                      rear_near, rear_far),
            cam_ortho("side", P_top(SIDE_CAM_X, py, mid_h), P_top(px, py, mid_h), (0.0, 1.0, 0.0), sec_size,
                      side_near, side_far),
            cam_persp("fpv", P_top(cx, cy, cz), P_top(px, py, th), (0.0, 1.0, 0.0),
                      vfov_from_focal(env.fov, H_PX), 0.5, 600.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        if b.player is None:
            return {}
        gap = float(b.player.position[0]) - float(b.gas_x)
        wait = max(0, int(b.gas_delay) - int(b.step_count))
        return {"text": "gas gap %.1f m%s" % (gap, ("  release in %d" % wait) if wait else "")}
