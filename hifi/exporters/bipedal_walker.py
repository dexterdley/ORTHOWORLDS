"""
Bipedal Walker (gymnasium BipedalWalker-v3 + BipedalOrthographicWrapper), side-view.

GymBox (ortho/game_envs/bipedal_orthographic.py): 600x400 views, SCALE = 30 px/m, the wrapper
works in pixels. Box2D shapes are extruded along z: terrain z +-10, hull z +-1.0, legs[0..1]
(the i = -1 pair) z 1.1..1.5, legs[2..3] z -1.5..-1.1, start flag z +-0.5.

Views (decided camera policy: same follow target / scale / axis, but real cameras):
  top   looks down (-Y), screen x = x - scroll (scroll = hull.x - 4 m, hull at 1/5 of the width),
        +z points down the screen, z = 0 on the centre row; size 400/30 = 13.33 m
  rear  GymBox "front" slot: screen (z, y) seen from -X looking +X (behind the walker), z = 0 centred,
        y = 0 on the bottom row. GymBox draws every terrain slab with no culling and the walker on
        top; the real camera sits REAR_BACK behind the hull and its near plane cuts the world
        REAR_CUT behind the hull (the world caps the cut with a terrain cross-section), so terrain
        and the start flag behind the walker no longer hide it
  side  screen (x - scroll, y): the classic 2D view, y = 0 on the bottom row. GymBox's painter order
        is mirrored (it sorts as if seen from -Z); the real camera is on +Z looking -Z, +X right
  fpv   the wrapper's orbit camera: position = env.cam_pos (px -> m), look-at = the orbit pivot
        50 px above the hull, focal 400 px on a 400 px frame (vfov 53.13 deg), near clip 10 px
"""
import gymnasium

from game_envs.bipedal_orthographic import BipedalOrthographicWrapper

from .base import Exporter, cam_ortho, cam_persp, r, vfov_from_focal, xf_side

SCALE = 30.0
W_PX, H_PX = 600, 400
TERRAIN_STEP = 14.0 / SCALE
FLAG_X = 3 * TERRAIN_STEP                     # start flag pole (GymBox draws it in every view)
FLAG_Y1 = H_PX / SCALE / 4.0                  # = TERRAIN_HEIGHT
FLAG_Y2 = FLAG_Y1 + 50.0 / SCALE
LEG_Z = (1.3, 1.3, -1.3, -1.3)                # centre of each leg body's GymBox extrusion

VIEW_H = H_PX / SCALE                         # ortho size (m): frame height / px per metre
VIEW_W = W_PX / SCALE
TOP_Y = 10.5                                  # top camera height (walker <= ~8 m, nothing taller in frame)
REAR_BACK = 6.0                               # rear camera distance behind the hull
REAR_CUT = 1.8                                # section cut (near plane) behind the hull; world.gd caps it
REAR_FAR = 166.0                              # rear far plane (core sky card at 0.97 far, past the hills)
SIDE_Z = 10.6                                 # side camera just in front of the terrain face at z = +10
SIDE_FAR_Z = -172.0                           # side far plane: behind the back band (z >= -160)


class BipedalWalkerExporter(Exporter):
    name = "bipedal_walker"
    title = "Bipedal Walker"
    fps = 50
    view_size = (600, 400)

    def make_env(self):
        return BipedalOrthographicWrapper(gymnasium.make("BipedalWalker-v3", render_mode="rgb_array", **self.env_kwargs))

    # ------------------------------------------------------------------ messages
    def scene(self, env):
        b = env.unwrapped
        return {"tx": [r(x) for x in b.terrain_x], "ty": [r(y) for y in b.terrain_y],
                "step": r(TERRAIN_STEP), "flag": [r(FLAG_X), r(FLAG_Y1), r(FLAG_Y2)],
                "W": r(VIEW_W), "H": r(VIEW_H), "rear_cut": REAR_CUT}

    def state(self, env):
        b = env.unwrapped
        if b.hull is None:
            return {"xf": {}}
        hx, hy = float(b.hull.position[0]), float(b.hull.position[1])
        xf = {"hull": xf_side(hx, hy, b.hull.angle)}
        for i, leg in enumerate(b.legs):
            xf[f"leg{i}"] = xf_side(leg.position[0], leg.position[1], leg.angle, LEG_Z[i])
        # GymBox cloth colour: red once game over, green once the hull passed the pole, else orange
        flag = 2 if b.game_over else (1 if hx > FLAG_X else 0)
        return {"xf": xf, "p": {"flag": flag, "hx": r(hx), "hy": r(hy), "scroll": r(b.scroll),
                                "contact": [int(b.legs[1].ground_contact), int(b.legs[3].ground_contact)]}}

    def cameras(self, env):
        b = env.unwrapped
        if b.hull is None:
            hx, hy, scroll = 4.67, 5.6, 0.67
        else:
            hx, hy, scroll = float(b.hull.position[0]), float(b.hull.position[1]), float(b.scroll)
        cx = scroll + VIEW_W / 2.0                # frame centre column (GymBox screen x = x - scroll)
        mid_y = VIEW_H / 2.0                      # y = 0 on the bottom row
        px, py, pz = (c / SCALE for c in env.cam_pos)
        pivot = (hx, hy + env.camera_height / SCALE, 0.0)
        return [
            cam_ortho("top", (cx, TOP_Y, 0.0), (cx, 0.0, 0.0), (0.0, 0.0, -1.0), VIEW_H, 0.1, TOP_Y + 6.0),
            cam_ortho("rear", (hx - REAR_BACK, mid_y, 0.0), (hx, mid_y, 0.0), (0.0, 1.0, 0.0), VIEW_H,
                      REAR_BACK - REAR_CUT - 0.05, REAR_FAR),
            cam_ortho("side", (cx, mid_y, SIDE_Z), (cx, mid_y, 0.0), (0.0, 1.0, 0.0), VIEW_H,
                      0.2, SIDE_Z - SIDE_FAR_Z),
            cam_persp("fpv", (px, py, pz), pivot, (0.0, 1.0, 0.0), vfov_from_focal(env.fov, H_PX),
                      10.0 / SCALE, 2500.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        if b.hull is None:
            return {}
        return {"text": "x %.1f m  contact %d/%d" % (b.hull.position[0], b.legs[1].ground_contact, b.legs[3].ground_contact)}
