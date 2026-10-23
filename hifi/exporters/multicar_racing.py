"""
MultiCar Racing (gymnasium CarRacing-v3 + MultiCarOrthographicWrapper), top-down.

Like ortho/utils.py the default env is plain CarRacing-v3 (one car). opponent=True uses
hifi/multicar_env.py's MultiCarRacing (a second car on a pure-pursuit autopilot); the wrapper
draws it too.

GymBox views (multicar_racing_orthographic.py; every view drawn at 1000x800 and resized to
600x400, so Godot renders 5:4 and the wrapper resizes the same way):
  top   CarRacing's own render: car-up, 16.2 px/unit after a 1 s zoom-in from 0.6 px/unit,
        car a quarter of the height above the bottom edge
  rear  ortho 25 px/m from behind (car frame), z = 8 m on the centre row, culls > 9.5 m behind
  side  ortho 25 px/m from the car's right, forward to the right
  fpv   orbit 12 m behind / 8 m up around the pivot 8 m above the car, focal 500 (vfov 77.3)
GymBox bug fixed: the wrapper's FPV uses (sin a, cos a) as the car's forward while CarRacing's is
(-sin a, cos a), so at headings other than 0/180 deg its camera ends up beside or in front of the
car. The hi-fi FPV (and the cam_x/y/z it logs) uses the true forward.
"""
import math

import gymnasium
import numpy as np
from gymnasium.envs.box2d.car_racing import PLAYFIELD, TRACK_WIDTH

from game_envs.multicar_racing_orthographic import MultiCarOrthographicWrapper

from .base import Exporter, P_top, cam_ortho, cam_persp, r, rv, vfov_from_focal, xf_top

INTERNAL_H = 800.0
# Elevation cutaway: the hi-fi car reaches 2.7 m ahead / behind and 1.5 m beside its centre. The
# rear / side near planes sit that far plus ~1.5 m in front of the car (a section plane that
# follows it), so trees, grandstand or gantry between the camera and the car are cut away while
# everything at and behind the car stays in view (one consistent world).
CAR_HALF_L, CAR_HALF_W, CUT_MARGIN = 2.7, 1.5, 1.5
SIDE_DIST = 60.0
CAR_COLORS = {"ego": (0.8, 0.08, 0.06), "opponent": (0.1, 0.25, 0.9)}


class MultiCarRacingExporter(Exporter):
    name = "multicar_racing"
    title = "MultiCar Racing"
    fps = 50
    view_size = (600, 400)
    render_size = (750, 600)       # GymBox draws 1000x800 (5:4) and squashes to 600x400
    camera_fix = "FPV forward vector: the wrapper uses (sin a, cos a); CarRacing's forward is (-sin a, cos a)"

    def __init__(self, opponent=False, **env_kwargs):
        super().__init__(**env_kwargs)
        self.opponent = opponent
        self._vel = {}

    def make_env(self):
        if self.opponent:
            from multicar_env import MultiCarRacing
            # same 1000-step limit gymnasium.make("CarRacing-v3") adds; without it episodes never truncate
            spec = gymnasium.spec("CarRacing-v3")
            base = MultiCarRacing(render_mode="rgb_array", **self.env_kwargs)
            return MultiCarOrthographicWrapper(gymnasium.wrappers.TimeLimit(base, spec.max_episode_steps))
        return MultiCarOrthographicWrapper(gymnasium.make("CarRacing-v3", render_mode="rgb_array", **self.env_kwargs))

    def _cars(self, env):
        b = env.unwrapped
        out = []
        if getattr(b, "car", None) is not None:
            out.append(("ego", b.car))
        if getattr(b, "opponent_car", None) is not None:
            out.append(("opponent", b.opponent_car))
        return out

    def on_reset(self, env):
        self._vel = {}

    # ------------------------------------------------------------------ messages
    def scene(self, env):
        b = env.unwrapped
        curbs = []
        for poly, color in b.road_poly:
            if isinstance(color, tuple):  # kerbs are (255,255,255)/(255,0,0) tuples; road tiles are arrays
                curbs.append([r(c) for xy in poly for c in xy] + [1.0 if color[1] == 0 else 0.0])
        return {"center": [[r(t[2]), r(t[3]), r(t[1], 5)] for t in b.track], "curbs": curbs,
                "width": float(TRACK_WIDTH), "playfield": float(PLAYFIELD),
                "cars": [{"name": n, "color": list(CAR_COLORS[n])} for n, _ in self._cars(env)]}

    def state(self, env):
        xf, cars, skids = {}, [], []
        for name, car in self._cars(env):
            h = car.hull
            xf[name] = xf_top(h.position[0], h.position[1], h.angle)
            vel = np.array([h.linearVelocity[0], h.linearVelocity[1]], dtype=np.float64)
            prev = self._vel.get(name, vel)
            acc = (vel - prev) * self.fps
            self._vel[name] = vel
            fwd = np.array([-math.sin(h.angle), math.cos(h.angle)])
            right = np.array([math.cos(h.angle), math.sin(h.angle)])
            wheels = []
            for w in car.wheels:
                skid = w.skid_start is not None or w.skid_particle is not None
                wheels.append([r(w.joint.angle), r(w.phase), int(skid), int(len(w.tiles) == 0)])
            cars.append({"name": name, "w": wheels, "brake": r(car.wheels[0].brake), "gas": r(car.wheels[2].gas),
                         "speed": r(np.linalg.norm(vel)),
                         # cosmetic body roll / pitch from the (deterministic) acceleration
                         "roll": r(np.clip(float(acc @ right) * 0.0025, -0.06, 0.06)),
                         "pitch": r(np.clip(float(acc @ fwd) * -0.0035, -0.045, 0.045))})
            for p in car.particles:
                pts = [c for xy in p.poly[-24:] for c in (r(xy[0], 3), r(xy[1], 3))]
                if len(pts) >= 4:
                    skids.append([int(p.grass)] + pts)
        return {"xf": xf, "p": {"cars": cars, "skids": skids}}

    def _fpv(self, env):
        car = env.unwrapped.car
        a = car.hull.angle + env.cam_yaw
        p = env.cam_pitch
        cx, cy = float(car.hull.position[0]), float(car.hull.position[1])
        fx, fy = -math.sin(a), math.cos(a)
        d, hgt = env.camera_dist, env.camera_height
        return (cx - d * fx * math.cos(p), cy - d * fy * math.cos(p), hgt + d * math.sin(p)), (cx, cy, hgt)

    def cam_log(self, env):
        if env.unwrapped.car is None:
            return (0.0, 0.0, 0.0)
        return self._fpv(env)[0]

    def cameras(self, env):
        b = env.unwrapped
        car = b.car
        cx, cy, a = float(car.hull.position[0]), float(car.hull.position[1]), float(car.hull.angle)
        fx, fy = -math.sin(a), math.cos(a)
        rx, ry = math.cos(a), math.sin(a)
        fwd_g = (fx, 0.0, -fy)
        # Top: CarRacing zoom-in over the first second, car a quarter of the height above the bottom
        zoom = 0.1 * 6.0 * max(1.0 - b.t, 0.0) + 2.7 * 6.0 * min(b.t, 1.0)
        size_top = INTERNAL_H / zoom
        ahead = (INTERNAL_H / 4.0) / zoom
        tx, ty = cx + fx * ahead, cy + fy * ahead
        hgt = env.camera_height
        size_side = INTERNAL_H / env.scale_factor
        (px, py, pz), (lx, ly, lz) = self._fpv(env)
        return [
            cam_ortho("top", P_top(tx, ty, 60.0), P_top(tx, ty, 0.0), fwd_g, size_top, 1.0, 120.0),
            cam_ortho("rear", P_top(cx - env.camera_dist * fx, cy - env.camera_dist * fy, hgt), P_top(cx, cy, hgt),
                      (0.0, 1.0, 0.0), size_side, env.camera_dist - CAR_HALF_L - CUT_MARGIN, 1500.0),
            cam_ortho("side", P_top(cx + SIDE_DIST * rx, cy + SIDE_DIST * ry, hgt), P_top(cx, cy, hgt), (0.0, 1.0, 0.0),
                      size_side, SIDE_DIST - CAR_HALF_W - CUT_MARGIN, 1500.0),
            cam_persp("fpv", P_top(px, py, pz), P_top(lx, ly, lz), (0.0, 1.0, 0.0),
                      vfov_from_focal(env.fov, INTERNAL_H), 1.0, 4000.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        v = np.linalg.norm(b.car.hull.linearVelocity) if b.car is not None else 0.0
        lap = b.tile_visited_count / max(len(b.track), 1)
        return {"text": "%3d km/h   lap %3d%%" % (v * 3.6, lap * 100)}
