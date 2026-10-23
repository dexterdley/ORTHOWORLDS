"""
Mountain Goat (MountainGoatEnv + MountainGoatOrthographicWrapper), side-view.

The goat climbs WEST (-X) from the meadow (x 60..65, y 2) up a ledged slope to the summit plateau
(x 25..20, y 12). Box2D gravity -15, 28 px/m, every GymBox view is 800x600.

GymBox views (mountain_goat_orthographic.py) and their hi-fi cameras:
  top   (relx + 400, z + 300): looks down, centred on the torso x, west on the left, +z down the
        screen.                                            -> ortho camera above the torso, up = -Z
  rear  (z + 400, 600 - (rely + 300)) with depth along +X: the GymBox viewer stands WEST of the goat
        looking +X, i.e. it shows the goat's FACE (the docstring says "rear"). Fixed to a true rear
        view: viewer EAST of the goat looking -X (the travel direction), the torso height on the
        centre row, z = 0 on the centre column; its look point is the torso. Its near plane is a
        section that cuts away everything between the camera and the goat (the flat valley floor
        and the boulders east of x = 68), so below the valley level the view shows the receding
        ground backdrop instead of a terrain end cap. Its far plane reaches the western edge of the
        backdrop ranges (x = -500), so the pine forests and ranges ahead of the climb are drawn.
  side  (relx + 400, 600 - (rely + 300)), mirrored painter order -> real camera on +Z looking -Z,
        centred on the torso in both axes; below the surface it sees the terrain's front section.
        Its far plane reaches the back edge of the backdrop ranges (z = -480).
  fpv   orbit camera: anchor 3 m behind (east) / 1 m above the torso looking -X, orbit radius 3 m
        around the pivot 1 m above the torso; focal 320 px on 600 (vfov 86.3 deg), near 14 px.
        GymBox projects it as a real camera already (screen right = -Z); position = cam_pos,
        look-at = the orbit pivot.
  Ortho size = 600 px / 28 px per m = 21.43 m. Clouds are hidden in the top view (GymBox draws them
  under the terrain there; a real camera would look through them at the goat).
"""
import numpy as np

from game_envs.mountain_goat_orthographic import (
    MOUNTAIN_BASE_H, MOUNTAIN_MAX_H, MOUNTAIN_PEAK_X, MOUNTAIN_START_X, SCALE, SNOW_C, TERRAIN_STEP, VALLEY_H,
    MountainGoatEnv, MountainGoatOrthographicWrapper)

from .base import Exporter, cam_ortho, cam_persp, r, vfov_from_focal, xf_side

W_PX, H_PX = 800, 600
REAR_CAM_X = 86.0       # true-rear camera, east of the terrain's east end (x = 84)
REAR_CUT_X = 68.0       # its near plane: east of the goat (terminated beyond x = 65.5), cuts the flat valley floor
SIDE_CAM_Z = 18.0       # side camera, in front of the terrain's front face (z = +15)
# Ortho far planes: the sky card sits at 0.97 * far, behind the backdrop's far edges (x = -500 for
# the rear view, z = -480 for the side view). Shadow cascades are capped by the look's
# shadow_distance (150 m) anyway, so the longer range costs no shadow resolution beyond that.
REAR_FAR = 610.0
SIDE_FAR = 520.0


class MountainGoatExporter(Exporter):
    name = "mountain_goat"
    title = "Mountain Goat"
    fps = 50
    view_size = (W_PX, H_PX)

    def make_env(self):
        return MountainGoatOrthographicWrapper(MountainGoatEnv(**self.env_kwargs))

    # ------------------------------------------------------------------ scene
    def scene(self, env):
        b = env.unwrapped
        segs = b.terrain_polys[1:]          # mountain segments, x descending (65 -> 20)
        pts = {}
        snow = []
        for t in segs:
            (x0, y0), (x1, y1) = t["top_edge"]
            pts[round(float(x0), 4)] = float(y0)
            pts[round(float(x1), 4)] = float(y1)
            lo, hi = sorted((float(x0), float(x1)))
            snow.append([r(lo), r(hi), int(tuple(t["top_color"]) == tuple(SNOW_C))])
        snow.sort()
        profile = [[r(x), r(pts[x])] for x in sorted(pts)]
        valley = b.terrain_polys[0]["top_edge"]
        clouds = [[r(c["x"]), r(c["y"]), r(c["z"]), r(c["w"]), r(c["h"]), r(c["d"])] for c in b.clouds]
        return {"profile": profile, "snow": snow, "seg": TERRAIN_STEP,
                "valley": [r(valley[0][0]), r(valley[1][0]), r(VALLEY_H)],
                "meadow": [r(MOUNTAIN_START_X), r(MOUNTAIN_BASE_H)], "summit": [r(MOUNTAIN_PEAK_X), r(MOUNTAIN_MAX_H)],
                "clouds": clouds, "lane": 1.2}

    # ------------------------------------------------------------------ state
    def state(self, env):
        b = env.unwrapped
        if b.torso is None:
            return {"xf": {}}
        xf = {"torso": xf_side(b.torso.position[0], b.torso.position[1], b.torso.angle),
              "head": xf_side(b.head.position[0], b.head.position[1], b.head.angle),
              "horn0": xf_side(b.horn_left.position[0], b.horn_left.position[1], b.horn_left.angle),
              "horn1": xf_side(b.horn_right.position[0], b.horn_right.position[1], b.horn_right.angle)}
        # legs: [front_near thigh, shin, rear_near thigh, shin, front_far thigh, shin, rear_far thigh, shin]
        for i, leg in enumerate(b.legs):
            xf[f"leg{i}"] = xf_side(leg.position[0], leg.position[1], leg.angle)
        return {"xf": xf}

    # ------------------------------------------------------------------ cameras
    def cameras(self, env):
        b = env.unwrapped
        if b.torso is None:
            tx, ty = MOUNTAIN_START_X + 2.0, MOUNTAIN_BASE_H + 1.25
        else:
            tx, ty = float(b.torso.position[0]), float(b.torso.position[1])
        size = H_PX / SCALE
        cx, cy, cz = (c / SCALE for c in env.cam_pos)
        pivot = (tx, ty + env.camera_height / SCALE, 0.0)
        return [
            cam_ortho("top", (tx, 40.0, 0.0), (tx, 0.0, 0.0), (0.0, 0.0, -1.0), size, 14.0, 58.0),
            cam_ortho("rear", (REAR_CAM_X, ty, 0.0), (tx, ty, 0.0), (0.0, 1.0, 0.0), size,
                      REAR_CAM_X - REAR_CUT_X, REAR_FAR),
            cam_ortho("side", (tx, ty, SIDE_CAM_Z), (tx, ty, 0.0), (0.0, 1.0, 0.0), size, 0.5, SIDE_FAR),
            cam_persp("fpv", (cx, cy, cz), pivot, (0.0, 1.0, 0.0), vfov_from_focal(env.fov, H_PX), 0.5, 3000.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        if b.torso is None:
            return {}
        prog = (MOUNTAIN_START_X - float(b.torso.position[0])) / (MOUNTAIN_START_X - MOUNTAIN_PEAK_X)
        return {"text": "climb %3d%%  alt %.1f m" % (int(np.clip(prog, 0, 1) * 100), float(b.torso.position[1]))}
