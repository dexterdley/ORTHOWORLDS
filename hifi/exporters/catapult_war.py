"""
Catapult War (CatapultWarEnv + CatapultWarOrthographicWrapper), side-view, units are metres.

Box2D (x, y) = Godot (x, y), z = extrusion depth. Every GymBox view is anchored to the static
catapult base (10, 2):
  top   12 px/m (ortho size 50 m), centred on x = 10, z = 0 on the centre row. GymBox draws +z up
        the screen (a mirrored plan) and only the AABB of each body; the hi-fi view is a real camera
        looking down (+z down the screen, which is the same picture for this z-symmetric scene)
        and shows the real, rotated geometry
  rear  15 px/m (size 40 m): screen x = z * 15 + 400, y = 0 on the bottom row; far x painted first ->
        viewer at -X looking +X. GymBox draws AABBs; the hi-fi view shows the real geometry
  side  12 px/m (size 50 m): screen ((x - 10) * 12 + 400, 600 - (y + 1) * 12) -> centre (x 10, y 24);
        real camera on +Z looking -Z
  fpv   orbit camera around the pivot (10, 4, 0), 6.5 m out, base pitch 50 deg (env.orbit_pitch),
        focal 200 px on 600 (vfov 112.6 deg), near 0.1 m; the GymBox fake ground half-plane and
        crosshair are replaced by the real scene
The boulder is destroyed and recreated every shot: it travels as a transient object with a
logical id (shots fired so far, minus the one in flight). Toppled blocks raise a dust puff.
"""
from game_envs.catapult_war_orthographic import CatapultWarEnv, CatapultWarOrthographicWrapper

from .base import Exporter, body_xf_side, cam_ortho, cam_persp, fixture_shapes, r, rv, vfov_from_focal, xf_side

W_PX, H_PX = 800, 600
REAR_X = -62.0                      # rear camera plane (the terrain strip starts at x = -60)
SIDE_Z = 50.0                       # side camera plane (the terrain strip ends at z = 45)
DUST_LIFE = 50
# Ortho far planes: one consistent world, so they reach past the farthest scenery inside the frame
# (the sky card sits at 0.97 * far): rear -> the rolling hills across the far end (x 320..800),
# side -> the hill range behind the field (z -300..-650). Directional shadows stop at the Look's
# shadow_distance anyway.
REAR_FAR = 890.0
SIDE_FAR = 725.0


class CatapultWarExporter(Exporter):
    name = "catapult_war"
    title = "Catapult War"
    fps = 50
    view_size = (W_PX, H_PX)

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self.topple = {}

    def make_env(self):
        return CatapultWarOrthographicWrapper(CatapultWarEnv(**self.env_kwargs))

    def on_reset(self, env):
        self.topple = {}

    def on_step(self, env, action):
        b = env.unwrapped
        for i, k in enumerate(b.block_toppled):
            if k and i not in self.topple:
                p = b.enemy_blocks[i].position
                self.topple[i] = (int(b.step_count), float(p[0]), float(p[1]))

    def _base(self, env):
        b = env.unwrapped
        if b.catapult_base is not None:
            return float(b.catapult_base.position[0]), float(b.catapult_base.position[1])
        return 10.0, 2.0

    def scene(self, env):
        b = env.unwrapped
        bx, by = self._base(env)
        return {
            "base": [r(bx), r(by)], "base_shape": fixture_shapes(b.catapult_base) if b.catapult_base else [],
            "pivot": rv(b.PIVOT), "tower_x": r(b.tower_x), "wind": r(b.wind, 3),
            "blocks": [fixture_shapes(blk) for blk in b.enemy_blocks],
        }

    def state(self, env):
        b = env.unwrapped
        xf = {}
        if b.catapult_arm is not None:
            xf["arm"] = body_xf_side(b.catapult_arm)
        for i, blk in enumerate(b.enemy_blocks):
            xf[f"block{i}"] = body_xf_side(blk)
        dyn = []
        if b.boulder is not None:
            serial = b.SHOT_BUDGET - b.shots_left - int(bool(b.boulder_launched))
            dyn.append({"id": f"boulder{serial}", "k": "boulder", "mask": "boulder",
                        "xf": xf_side(b.boulder.position[0], b.boulder.position[1], b.boulder.angle)})
        for i, (t0, x, y) in sorted(self.topple.items()):
            age = int(b.step_count) - t0
            if 0 <= age < DUST_LIFE:
                dyn.append({"id": f"dust{i}", "k": "dust", "xf": xf_side(x, y, 0.0), "t": r(age / float(DUST_LIFE), 3)})
        return {"xf": xf, "dyn": dyn, "p": {
            "tension": r(b.tension, 3), "wind": r(b.wind, 3), "aim": r(b.aim_angle, 4),
            "loaded": int(b.boulder is not None and not b.boulder_launched), "shots_left": int(b.shots_left),
            "toppled": [int(t) for t in b.block_toppled]}}

    def cameras(self, env):
        bx, by = self._base(env)
        cx, cy, cz = env.cam_pos
        pivot = (bx, by + env.camera_height, 0.0)
        return [
            cam_ortho("top", (bx, 60.0, 0.0), (bx, 0.0, 0.0), (0.0, 0.0, -1.0), H_PX / 12.0, 1.0, 70.0),
            cam_ortho("rear", (REAR_X, 20.0, 0.0), (REAR_X + 1.0, 20.0, 0.0), (0.0, 1.0, 0.0), H_PX / 15.0, 1.0, REAR_FAR),
            cam_ortho("side", (bx, by + 22.0, SIDE_Z), (bx, by + 22.0, SIDE_Z - 1.0), (0.0, 1.0, 0.0), H_PX / 12.0, 1.0, SIDE_FAR),
            cam_persp("fpv", (cx, cy, cz), pivot, (0.0, 1.0, 0.0), vfov_from_focal(200.0, H_PX), 0.1, 1500.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        return {"text": "tension %.2f  aim %.0f deg  shots %d  wind %+.1f" % (
            b.tension, b.aim_angle * 57.2958, b.shots_left, b.wind)}
