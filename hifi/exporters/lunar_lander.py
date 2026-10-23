"""
Lunar Lander (gymnasium LunarLander-v3 + LunarOrthographicWrapper), side-view.

GymBox views (lunar_orthographic.py, 600x400, 30 px/m, all real cameras):
  top   looks down, follows the lander's x, +z points down the screen, z = 0 on the centre row
  rear  slot = the wrapper's "front" elevation: x-y plane seen from +z, y = 0 on the bottom row
  side  looks along +x from -x (z to the right), y = 0 on the bottom row, z = 0 centred
  fpv   orbit 12 m behind / 3 m above the lander, looking at a pivot 3 m above it,
        focal 400 px (vfov 53.1 deg), near clip 10 px = 0.33 m, pitch limits +60 / -30 deg
"""
import gymnasium
import numpy as np

from game_envs.lunar_orthographic import LunarOrthographicWrapper

from .base import Exporter, cam_ortho, cam_persp, r, rv, vfov_from_focal, xf_side

SCALE = 30.0
W_PX, H_PX = 600, 400


class LunarLanderExporter(Exporter):
    name = "lunar_lander"
    title = "Lunar Lander"
    fps = 50
    view_size = (600, 400)

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self.last_action = 0

    def make_env(self):
        return LunarOrthographicWrapper(gymnasium.make("LunarLander-v3", render_mode="rgb_array", **self.env_kwargs))

    def on_reset(self, env):
        self.last_action = 0

    def on_step(self, env, action):
        self.last_action = action

    def scene(self, env):
        b = env.unwrapped
        pts = set()
        for f in b.moon.fixtures:
            for v in f.shape.vertices:
                pts.add((round(float(v[0]), 5), round(float(v[1]), 5)))
        terrain = sorted(pts)
        return {"terrain": [list(p) for p in terrain], "pad": [r(b.helipad_x1), r(b.helipad_x2), r(b.helipad_y)],
                "W": W_PX / SCALE, "H": H_PX / SCALE}

    def _engines(self, env):
        a = self.last_action
        b = env.unwrapped
        if getattr(b, "continuous", False):
            a = np.asarray(a, dtype=np.float64).ravel()
            main = float(np.clip(a[0], 0.0, 1.0) * 0.5 + 0.5) if a[0] > 0 else 0.0
            side = float(a[1]) if abs(a[1]) > 0.5 else 0.0
            return main, max(-side, 0.0), max(side, 0.0)
        a = int(a) if a is not None else 0
        # action 1 fires the nozzle on the lander's +x side (exhaust +x), action 3 the -x side
        return (1.0 if a == 2 else 0.0), (1.0 if a == 1 else 0.0), (1.0 if a == 3 else 0.0)

    def state(self, env):
        b = env.unwrapped
        if b.lander is None:
            return {"xf": {}}
        xf = {"lander": xf_side(b.lander.position[0], b.lander.position[1], b.lander.angle)}
        for i, leg in enumerate(b.legs):
            xf[f"leg{i}"] = xf_side(leg.position[0], leg.position[1], leg.angle)
        main, plus_x, minus_x = self._engines(env)
        if b.game_over or not b.lander.awake:
            main = plus_x = minus_x = 0.0
        return {"xf": xf, "p": {"main": main, "rcs_px": plus_x, "rcs_nx": minus_x,
                                "contact": [int(l.ground_contact) for l in b.legs],
                                "landed": int(not b.lander.awake)}}

    def cameras(self, env):
        b = env.unwrapped
        if b.lander is None:
            lx, ly = 10.0, 10.0
        else:
            lx, ly = float(b.lander.position[0]), float(b.lander.position[1])
        size = H_PX / SCALE
        mid_y = size / 2.0
        cx, cy, cz = (c / SCALE for c in env.cam_pos)
        return [
            # One consistent world: the far planes reach the far field's edge (crater rims at
            # z = -460 behind the rear view, x = 340 behind the side view; the ortho sky card sits
            # at 0.97 * far). The near planes are section cuts just outside the terrain strip's
            # front face (z = 12) / west end (x = -60), so the far field between those cameras
            # and the strip is cut away and the strip's soil section stays in view.
            cam_ortho("top", (lx, 45.0, 0.0), (lx, 0.0, 0.0), (0.0, 0.0, -1.0), size, 1.0, 60.0),
            cam_ortho("rear", (lx, mid_y, 30.0), (lx, mid_y, 0.0), (0.0, 1.0, 0.0), size, 17.9, 510.0),
            cam_ortho("side", (-70.0, mid_y, 0.0), (lx, mid_y, 0.0), (0.0, 1.0, 0.0), size, 9.0, 425.0),
            cam_persp("fpv", (cx, cy, cz), (lx, ly + env.camera_height / SCALE, 0.0), (0.0, 1.0, 0.0),
                      vfov_from_focal(env.fov, H_PX), 10.0 / SCALE, 3000.0),
        ]

    def hud(self, env):
        main, px, nx = self._engines(env)
        return {"text": "engines: main %d  rcs %d/%d" % (main > 0, px > 0, nx > 0)}
