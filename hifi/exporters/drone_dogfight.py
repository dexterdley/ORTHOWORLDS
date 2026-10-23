"""
Drone Dogfight (DroneDogfightEnv + DroneDogfightOrthographicWrapper), top-down (gravity 0).

Box2D (x, y) is the arena floor plan (25 x 20 m, origin at the south-west corner); every drone
flies at the fixed altitude DRONE_ALT = 5 m. Godot: (x, h, -y), heading = yaw about +Y.

GymBox views (drone_dogfight_orthographic.py, 800x600, SCALE 30 px/m; the wrapper's camera state
is in pixels and is divided out here):
  top   plan view centred on the player, north up, 30 px/m (size 20 m). GymBox also draws a
        fixed-screen "arena outline" rectangle that does not follow the arena (bug): not copied.
  rear  axis-aligned Y-Z elevation: screen x = (y - py) * 30 + 400, screen y = 600 - z * 30, so
        z = 0 is the bottom row and the frame centre is z = 10 m. GymBox's painter order is
        inverted; the real camera stands on the East side looking West, screen-right = North.
        Its near plane sits just outside the East wall: scenery between the camera and the
        arena is cut away; the far plane reaches past the farthest tree behind the arena.
  side  X-Z elevation: screen x = (x - px) * 30 + 400, same vertical; camera on the South side
        looking North (screen-right = East), near plane just outside the South wall (cutaway),
        far plane past the tree line behind the arena.
  fpv   orbit camera: anchor 3.5 m behind the player along its heading, 1 m above it (z = 6 m),
        pivot (px, py, 6 m); position = wrapper cam_pos / 30; focal 400 px on 600 -> vfov 73.7 deg;
        near clip 1 px = 1/30 m. GymBox omits the floor/walls and fakes the horizon in the FPV;
        the hi-fi FPV shows the real arena.
Moving ids: player, opponent (class "drone", hidden while downed). dyn: bolt (class "bolt"),
explosion (effect, no mask). p: alive flags, velocities (cosmetic bank), hp.
"""
import math

from game_envs.drone_dogfight_orthographic import (ARENA_H, ARENA_W, BULLET_LIFE, DRONE_ALT, WALL_HEIGHT,
                                                   WALL_T, DroneDogfightEnv, DroneDogfightOrthographicWrapper,
                                                   SCALE)

from .base import Exporter, P_top, cam_ortho, cam_persp, r, vfov_from_focal, xf_top

W_PX, H_PX = 800, 600
EX_FRAMES = 14
CAM_H = 1.0          # FPV anchor height above the drone (wrapper camera_height / SCALE)
# Elevation far plane: the world's tree ring (radius <= 95 m around the arena centre, crowns
# <= ~5 m) ends <= 144 m from either elevation camera; the ortho sky card sits at 0.97 * far.
ELEV_FAR = 150.0


class DroneDogfightExporter(Exporter):
    name = "drone_dogfight"
    title = "Drone Dogfight"
    fps = 50
    view_size = (W_PX, H_PX)

    def make_env(self):
        return DroneDogfightOrthographicWrapper(DroneDogfightEnv(**self.env_kwargs))

    # ------------------------------------------------------------------ camera state
    def advance_camera(self, env, command):
        # render() only updates the orbit angles and then logs cam_pos (pixels)
        env.update_camera_pose(command)
        env.cam_x, env.cam_y, env.cam_z = env.cam_pos

    # ------------------------------------------------------------------ messages
    def scene(self, env):
        return {"W": ARENA_W, "H": ARENA_H, "wall_t": WALL_T, "wall_h": WALL_HEIGHT, "alt": DRONE_ALT}

    @staticmethod
    def _bolts(b, lst, owner):
        out = []
        for bl in lst:
            body = bl["body"]
            vx, vy = float(body.linearVelocity[0]), float(body.linearVelocity[1])
            # birth step = step_count + life - (BULLET_LIFE - 1): unique per owner (fire cooldown >= 8)
            birth = int(b.step_count) + int(bl["life"]) - (BULLET_LIFE - 1)
            out.append({"id": f"{owner}b{birth}", "k": "bolt", "mask": "bolt", "c": 0 if owner == "p" else 1,
                        "xf": xf_top(body.position[0], body.position[1], math.atan2(vy, vx), DRONE_ALT)})
        return out

    def state(self, env):
        b = env.unwrapped
        if getattr(b, "player", None) is None:
            return {"xf": {}}
        xf = {}
        p = {"alive": [int(b.player_alive), int(b.opponent_alive)],
             "hp": [int(b.player_hp), int(b.opponent_hp)]}
        for key, body, head in (("player", b.player, b.player_heading), ("opponent", b.opponent, b.opp_heading)):
            xf[key] = xf_top(body.position[0], body.position[1], body.angle, DRONE_ALT)
            p[key[0] + "v"] = [r(body.linearVelocity[0], 3), r(body.linearVelocity[1], 3)]
        dyn = self._bolts(b, b.bullets, "p") + self._bolts(b, b.opp_bullets, "o")
        for owner, ex in (("p", b.player_explosion), ("o", b.opp_explosion)):
            if ex is None:
                continue
            dyn.append({"id": f"{owner}x", "k": "explosion", "t": r((EX_FRAMES - ex["frames"]) / EX_FRAMES),
                        "r": r(ex["radius"]), "xf": [r(ex["pos"][0]), r(DRONE_ALT), r(-ex["pos"][1])]})
        return {"xf": xf, "dyn": dyn, "p": p}

    def cameras(self, env):
        b = env.unwrapped
        if getattr(b, "player", None) is None:
            px, py = ARENA_W / 2, ARENA_H / 2
        else:
            px, py = float(b.player.position[0]), float(b.player.position[1])
        size = H_PX / SCALE                       # 20 m
        mid = size / 2.0                          # z = 0 on the bottom row -> centre at z = 10 m
        cx, cy, cz = (c / SCALE for c in env.cam_pos)
        piv = DRONE_ALT + CAM_H
        return [
            cam_ortho("top", (px, 40.0, -py), (px, 0.0, -py), (0.0, 0.0, -1.0), size, 1.0, 44.0),
            # East side looking West (-X): screen-right = -Z = Box2D +y. Near plane at x = 25.6 m.
            cam_ortho("rear", (ARENA_W + 30.6, mid, -py), (px, mid, -py), (0.0, 1.0, 0.0), size, 30.0, ELEV_FAR),
            # South side looking North (-Z): screen-right = +X. Near plane at y = -0.6 m.
            cam_ortho("side", (px, mid, 30.6), (px, mid, -py), (0.0, 1.0, 0.0), size, 30.0, ELEV_FAR),
            cam_persp("fpv", P_top(cx, cy, cz), P_top(px, py, piv), (0.0, 1.0, 0.0),
                      vfov_from_focal(env.fov, H_PX), 1.0 / SCALE, 900.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        return {"text": "hp %d / opp %d   kills %d" % (b.player_hp, b.opponent_hp, b.score)}
