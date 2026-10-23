"""
Robot Sumo (RobotSumoEnv + RobotSumoOrthographicWrapper), top-down, origin at the ring centre.

A robot's forward (plow side) is its local +y: world heading (-sin a, cos a); its right is local +x
(cos a, sin a). Body box 0.8 x 1.0 m, 1.5 m tall; two free-spinning wheels (r 0.25) on revolute
joints at local (+-0.48, 0); plow box 1.2 x 0.3 m (0.8 m tall) on a motorised hinge at (0, 0.6).

GymBox views (robot_sumo_orthographic.py, 800x600):
  top   50 px/m, north-up, centred on the player body (plus a crosshair overlay)
  rear  player frame from behind: screen x = 400 + right * 50, screen y = 400 - 100 z (vertical
        stretched 2x), painter-sorted by forward depth (nothing is cut away)
  side  screen x = 400 + forward * 50 from the robot's right, same 2x vertical
  fpv   orbit camera 2 m behind the body along its heading at z = 2.0, looking at the pivot
        (body x, y, 2.0); focal 200 px on 600 (vfov 112.6 deg), near clip 0.1 m; HUD crosshair
Hi-fi fixes: rear / side are true-scale ortho elevations (50 px/m on both axes, ortho size 12 m)
with z = 0 kept on row 400 (the centre row is z = 2 m); the cameras sit 7 m out (inside the
hall), so near objects occlude like GymBox's depth sort. Crosshair overlays are dropped.
Masks: ids "player", "player_plow", "player_wheel0/1" (class "player") and the same with
"opponent"; ring / hall are unregistered occluders.
"""
import math

from game_envs.robot_sumo_orthographic import (
    RING_RADIUS, WHEEL_RADIUS, RobotSumoEnv, RobotSumoOrthographicWrapper, heading_vector,
)

from .base import Exporter, P_top, cam_ortho, cam_persp, r, vfov_from_focal, xf_top

W_PX, H_PX = 800, 600
TOP_PPM = 50.0
ELEV_PPM = 50.0
GROUND_ROW = 400.0
ELEV_ZC = (GROUND_ROW - H_PX / 2.0) / ELEV_PPM     # 2.0 m on the centre row
CAM_BACK = 7.0                                     # elevation cameras: distance from the player


class RobotSumoExporter(Exporter):
    name = "robot_sumo"
    title = "Robot Sumo"
    fps = 50
    view_size = (W_PX, H_PX)
    mask_groups = {"robot": ["player", "opponent"]}

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self.roll = {"player": [0.0, 0.0], "opponent": [0.0, 0.0]}

    def make_env(self):
        return RobotSumoOrthographicWrapper(RobotSumoEnv(**self.env_kwargs))

    def on_reset(self, env):
        self.roll = {"player": [0.0, 0.0], "opponent": [0.0, 0.0]}

    def _robots(self, b):
        return (("player", b.player_body, b.player_plow, b.player_wheels),
                ("opponent", b.opponent_body, b.opponent_plow, b.opponent_wheels))

    def on_step(self, env, action):
        b = env.unwrapped
        for name, body, _, wheels in self._robots(b):
            fx, fy = heading_vector(body.angle)
            for k, (wheel, _) in enumerate(wheels):
                v = wheel.linearVelocity
                self.roll[name][k] += (v[0] * fx + v[1] * fy) / self.fps / WHEEL_RADIUS

    # ------------------------------------------------------------------ messages
    def scene(self, env):
        return {"ring": RING_RADIUS, "wheel_r": WHEEL_RADIUS}

    def state(self, env):
        b = env.unwrapped
        if b.player_body is None:
            return {"xf": {}}
        out, robots = {}, []
        for name, body, plow, wheels in self._robots(b):
            a = body.angle
            out[name] = xf_top(body.position[0], body.position[1], a, 0.0)
            if plow is not None:
                out[name + "_plow"] = xf_top(plow.position[0], plow.position[1], plow.angle, 0.0)
            for k, (wheel, _) in enumerate(wheels):
                out[f"{name}_wheel{k}"] = xf_top(wheel.position[0], wheel.position[1], a, WHEEL_RADIUS)
            d = math.hypot(body.position[0], body.position[1])
            robots.append([r(self.roll[name][0], 4), r(self.roll[name][1], 4), r(d / RING_RADIUS, 3)])
        return {"xf": out, "p": {"robots": robots}}

    # ------------------------------------------------------------------ cameras
    def cameras(self, env):
        b = env.unwrapped
        body = b.player_body
        px, py, a = float(body.position[0]), float(body.position[1]), float(body.angle)
        fx, fy = heading_vector(a)
        rx, ry = math.cos(a), math.sin(a)
        size = H_PX / ELEV_PPM
        look = P_top(px, py, ELEV_ZC)
        cx, cy, cz = env.cam_pos
        return [
            cam_ortho("top", P_top(px, py, 20.0), P_top(px, py, 0.0), (0.0, 0.0, -1.0), H_PX / TOP_PPM, 1.0, 25.0),
            # rear: behind the robot looking along its heading (its right on the screen right)
            cam_ortho("rear", P_top(px - fx * CAM_BACK, py - fy * CAM_BACK, ELEV_ZC), look, (0.0, 1.0, 0.0), size, 0.3,
                      CAM_BACK + 30.0),
            # side: from the robot's right looking left (forward on the screen right)
            cam_ortho("side", P_top(px + rx * CAM_BACK, py + ry * CAM_BACK, ELEV_ZC), look, (0.0, 1.0, 0.0), size, 0.3,
                      CAM_BACK + 30.0),
            cam_persp("fpv", P_top(cx, cy, cz), P_top(px, py, env.camera_height), (0.0, 1.0, 0.0),
                      vfov_from_focal(200.0, H_PX), 0.1, 400.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        p, o = b._center_dists()
        return {"text": "edge: you %.1f m  them %.1f m" % (RING_RADIUS - p, RING_RADIUS - o)}
