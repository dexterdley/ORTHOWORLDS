"""
Mario Escape (MarioEscapeEnv + MarioOrthographicWrapper), side-view platformer.

The level is fixed (module constants: ground segments with 4 pits, 7 one-way brick platforms, a
10 m flagpole at x = 70, an 18 m west wall); only the spawn x is random. Box2D gravity -50,
28 px/m, every GymBox view is 800x600, elevation views shifted up by 80 px (y = 0 on row 520).

GymBox views (mario_orthographic.py) and their hi-fi cameras:
  top   (x - px + 400, z + 300): looks down, centred on the player x, +z down the screen
                                                    -> ortho camera above the player, up = -Z
  rear  (z + 400, 520 - y), depth = x - px: viewer at -X looking +X (a true rear view, the player
        runs +X), z = 0 on the centre column, ABSOLUTE vertical framing (centre row = y 7.857 m).
        Real camera behind the player (its look point is the player's x on the same axis) whose
        near plane sits just behind him: a section plane that cuts away everything between the
        camera and him (level geometry, trees, shrubs west of him; GymBox draws those under the
        player). Everything beyond him, the tree rows included, is drawn; far reaches the
        easternmost trees (x ~ 124). The west wall, always between this camera and the player
        (GymBox draws it translucent over him), is hidden from this view.
  side  (x - px + 400, 520 - y), mirrored painter order -> real camera on +Z looking -Z, centred
        on the player x, same absolute vertical framing.
  fpv   follow camera: anchor 2.5 m behind (-X) / 0.8 m above the eased follow height, orbit radius
        2.5 m, then tilted by look_pitch (keeps the player framed while follow_y eases after him).
        GymBox projects it mirrored (+z on screen-left); here it is a real camera. Focal 280 px on
        600 (vfov 93.9 deg), near 14 px. GymBox leaves the ground boxes out of the FPV (pits are
        invisible over its floor grid): the hi-fi shows the real ground and pits. Near the spawn
        the follow camera ends up behind (west of) the west wall; the wall is built from stone
        blocks and the FPV cuts away the blocks that stand between the camera and the player
        (wall_cut_fpv), so the rest of the wall stays in view.
  Ortho size = 600 px / 28 px per m = 21.43 m.
Cross-view consistency: every scenery object is drawn in every view whose frame contains it; the
only exceptions are cutaways of things standing between a camera and the player.
"""
import math

from game_envs.mario_orthographic import (
    FLAGPOLE_X, FLAGPOLE_Y_TOP, GROUND_SEGS, LEVEL_H, LEVEL_W, MAX_VX, P_HH, P_HW, PITS, PLATFORMS, SCALE,
    MarioEscapeEnv, MarioOrthographicWrapper)

import numpy as np

from .base import Exporter, _basis, cam_ortho, cam_persp, r, vfov_from_focal, xf_side

W_PX, H_PX = 800, 600
Y_SHIFT = 80.0
MID_Y = (H_PX / 2.0 - Y_SHIFT) / SCALE      # world y on the centre row of the elevation views (7.857 m)
REAR_BACK = 30.0       # rear camera distance behind the player
REAR_CUT = 0.75        # its near plane, metres behind the player's centre (clears his trailing limbs)
REAR_REACH = 128.0     # its far plane, metres beyond the camera's look point: the sky card (0.97 far)
                       # stays behind the easternmost trees (x ~ 124) wherever the player stands
SIDE_CAM_Z = 24.0      # side camera, in front of the meadow's front face (z = +16)
TOP_CAM_Y = 32.0       # above the west wall (18 m)
WALL_X = (-0.4, 0.0)   # west wall body (GymBox: x -0.4..0, z -2.5..2.5, y 0..LEVEL_H)
WALL_Z = 2.5
WALL_ROW_H = 17.76 / 18.0  # stone block courses below the coping


def _box(cx, cy, hw, hh):
    return [r(cx - hw), r(cx + hw), r(cy - hh), r(cy + hh)]


def wall_blocks():
    """The west wall as stone blocks in running bond, a coping course and merlons:
    [[x0, x1, y0, y1, z0, z1, kind], ...] (kind "s" stone, "c" coping). One id per block ("wall<i>")
    so the FPV can cut away exactly the blocks between its camera and the player."""
    x0, x1 = WALL_X
    out = []
    for row in range(18):
        y0, y1 = row * WALL_ROW_H, (row + 1) * WALL_ROW_H
        zs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] if row % 2 == 0 else [-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5]
        for za, zb in zip(zs[:-1], zs[1:]):
            out.append([x0, x1, r(y0), r(y1), za, zb, "s"])
    for k in range(5):
        za = -2.56 + k * 1.024
        out.append([x0 - 0.06, x1 + 0.06, 17.76, 18.0, r(za), r(za + 1.024), "c"])
    for k in range(5):
        zc = -2.2 + k * 1.1
        out.append([x0 - 0.05, x1 + 0.05, 18.0, 18.4, r(zc - 0.3), r(zc + 0.3), "c"])
    return out


WALL_BLOCKS = wall_blocks()
WALL_IDS = [f"wall{i}" for i in range(len(WALL_BLOCKS))]
_WALL_CORNERS = np.array([[[b[0 + (m & 1)], b[2 + ((m >> 1) & 1)], b[4 + ((m >> 2) & 1)]] for m in range(8)]
                          for b in WALL_BLOCKS], float)          # (n, 8, 3)
_WALL_LO = np.array([[b[0], b[2], b[4]] for b in WALL_BLOCKS], float)
_WALL_HI = np.array([[b[1], b[3], b[5]] for b in WALL_BLOCKS], float)


def wall_cut_fpv(cam, player_xy):
    """Ids of the wall blocks the FPV cuts away. Only while its camera is west of the wall's east
    face (behind or inside the wall): the blocks wholly between the camera and its look point (the
    player), plus any block that a sight line from the camera to the player's body passes through
    (he can stand right against the wall)."""
    pos, fwd, _, _ = _basis(cam)
    if pos[0] > WALL_X[1] + 0.08:
        return []
    look_d = float(np.dot(np.asarray(cam["look"], float) - pos, fwd))
    cut = ((_WALL_CORNERS - pos) @ fwd).max(1) < look_d
    # sight lines to a grid of points on the player's (slightly padded) box: slab test per block
    px, py = player_xy
    pts = np.array([[px + sx, py + sy, sz] for sx in (-0.45, 0.0, 0.45) for sy in np.linspace(-0.75, 0.9, 8)
                    for sz in (-0.45, 0.0, 0.45)]) - pos                      # (m, 3) segments from the camera
    inv = 1.0 / np.where(np.abs(pts) < 1e-9, 1e-9, pts)
    ta = (_WALL_LO - pos)[:, None, :] * inv[None]                          # (n, m, 3)
    tb = (_WALL_HI - pos)[:, None, :] * inv[None]
    t0 = np.maximum(np.minimum(ta, tb).max(-1), 0.0)
    t1 = np.minimum(np.maximum(ta, tb).min(-1), 1.0)
    cut |= (t0 <= t1).any(1)
    return [WALL_IDS[i] for i in np.nonzero(cut)[0]]


class MarioEscapeExporter(Exporter):
    name = "mario_escape"
    title = "Mario Escape"
    fps = 50
    view_size = (W_PX, H_PX)

    def make_env(self):
        return MarioOrthographicWrapper(MarioEscapeEnv(**self.env_kwargs))

    # ------------------------------------------------------------------ scene
    def scene(self, env):
        return {"ground": [_box(*g) for g in GROUND_SEGS], "platforms": [_box(*p) for p in PLATFORMS],
                "pits": [[r(a), r(b)] for a, b in PITS], "flag": [r(FLAGPOLE_X), r(FLAGPOLE_Y_TOP), 0.1],
                "wall": [WALL_X[0], WALL_X[1], 0.0, r(LEVEL_H)], "wall_blocks": WALL_BLOCKS, "level_w": r(LEVEL_W), "player": [r(P_HW), r(P_HH)]}

    # ------------------------------------------------------------------ state
    def state(self, env):
        b = env.unwrapped
        if b.player is None:
            return {"xf": {}}
        p, v = b.player.position, b.player.linearVelocity
        # the west wall: cut away from the rear view (always between that camera and the player;
        # GymBox draws it translucent over him), and in the FPV the blocks between its camera and
        # the player (GymBox's FPV looks through the wall)
        vis = {"rear": list(WALL_IDS)}
        fpv_cut = wall_cut_fpv(self.cameras(env)[3], (float(p[0]), float(p[1])))
        if fpv_cut:
            vis["fpv"] = fpv_cut
        return {"xf": {"player": xf_side(p[0], p[1], 0.0)}, "vis": vis,
                "p": {"vx": r(v[0], 3), "vy": r(v[1], 3), "g": int(bool(b.is_grounded)),
                      "fr": int(bool(getattr(b, "facing_right", True))), "step": int(b.step_count),
                      "vmax": MAX_VX}}

    # ------------------------------------------------------------------ cameras
    def advance_camera(self, env, command):
        # render() orbits the camera, then eases follow_y toward the player and re-aims (look_pitch)
        env.update_camera_pose(command)
        env._update_follow_y()

    def cameras(self, env):
        b = env.unwrapped
        px = float(b.player.position[0]) if b.player is not None else 2.0
        size = H_PX / SCALE
        # FPV: the wrapper's cam_pos; optical axis = orbit axis tilted up by look_pitch
        # (orbit_camera_frame then rotate_camera_frame(pitch=look_pitch) in render()):
        # direction (X, Y, Z) = (cos(lp - p) cos y, sin(lp - p), cos(lp - p) sin y)
        cx, cy, cz = (c / SCALE for c in env.cam_pos)
        yaw, pitch, lp = env.cam_yaw, env.cam_pitch, env.look_pitch
        c = math.cos(lp - pitch)
        d = (c * math.cos(yaw), math.sin(lp - pitch), c * math.sin(yaw))
        dist = env.pivot_depth / SCALE
        look = (cx + d[0] * dist, cy + d[1] * dist, cz + d[2] * dist)
        rear_x = px - REAR_BACK
        # rear: look point = the player's x on the camera axis (the subject of its section cut)
        return [
            cam_ortho("top", (px, TOP_CAM_Y, 0.0), (px, 0.0, 0.0), (0.0, 0.0, -1.0), size, 1.0, TOP_CAM_Y + 12.0),
            cam_ortho("rear", (rear_x, MID_Y, 0.0), (px, MID_Y, 0.0), (0.0, 1.0, 0.0), size,
                      REAR_BACK - REAR_CUT, REAR_BACK + REAR_REACH),
            cam_ortho("side", (px, MID_Y, SIDE_CAM_Z), (px, MID_Y, 0.0), (0.0, 1.0, 0.0), size, 0.5, SIDE_CAM_Z + 150.0),
            cam_persp("fpv", (cx, cy, cz), look, (0.0, 1.0, 0.0), vfov_from_focal(env.fov, H_PX), 0.5, 3000.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        if b.player is None:
            return {}
        return {"text": "flag in %.1f m" % abs(FLAGPOLE_X - float(b.player.position[0]))}
