"""
Tank Duel (TankDuelEnv + TankDuelOrthographicWrapper), top-down.

GymBox views (tank_duel_orthographic.py, 800x600):
  top   15 px/m, north-up, centred on the player hull
  rear  30 px/m lateral, 60 px/m vertical, z = 0 on row 420; draws the player-frame *forward*
        axis horizontally (screen x = -forward), i.e. a mirrored side profile, not a rear view
  side  30 px/m lateral, 60 px/m vertical, z = 0 on row 420, from the hull's right (forward to
        the screen right)
  fpv   orbit camera 4.5 m behind the hull centre along its heading at z = 2.5, looking at the
        pivot (hull x, y, 2.5); focal 400 px on 600 (vfov 73.7 deg), near clip 0.1 m; fake
        horizon (sky / sand split at the centre row)
GymBox draws every body as a slab z_base..z_base + 0.5 (hull 0.5-1.0, turret 1.0-1.5, shell
0.8-1.3, obstacle 1.5-2.0, wall 2.0-2.5), so nothing touches the ground.

Hi-fi decisions:
  * Everything stands on the ground and keeps its GymBox top height: tracks/hull 0-1.0 m,
    turret 1.0-1.5 m, barrel axis 1.25 m, obstacles 0-2.0 m, walls 0-2.5 m. Shells fly along the
    barrel axis (1.25 m) so tracers leave the muzzle; explosions keep the env's z.
  * rear: a true rear view (camera behind the hull looking along its heading, the player's right
    on the screen right), 30 px/m on both axes (ortho size 20 m), z = 0 kept on row 420.
  * side: same scale fix (30 px/m both axes), z = 0 on row 420, forward to the screen right.
  * One consistent world: every object (cover, walls, desert props, the hill ring) is drawn in
    every view whose frame contains it. The only exception is the cutaway the real elevation
    cameras need (GymBox's floating slabs never cover the player): tanks / obstacles lying wholly
    between the camera and the player (in front of the hull centre's depth) whose screen extent
    overlaps the player's are hidden (state["vis"]), as are perimeter walls across the view
    wholly in front of the player; everything else stays visible and the ortho near plane sits
    just behind the deepest kept object (>= CUT), pulled in further where a kept wall would
    cross in front of the player, so it only slices walls. The world removes desert props in
    front of the player the same way (tank_duel/world.gd _cut_props). The far plane reaches past
    the hill ring (sky card at 0.97 * far). Below z = 0 the core's ortho sky card paints a
    receding sand ground (look "card_ground"), like GymBox's ground band.
  * fpv: the wrapper's own cam_pos / orbit pivot, real 3D scene instead of the fake horizon.
Masks: "player" / "enemy" (hull + turret), "obs<i>" class "obstacle", "wall<i>" class "wall",
shells are transient objects with class "shell". Effects never occlude masks.
"""
import math

from game_envs.tank_duel_orthographic import (
    BARREL_END, HULL_HALF_L, HULL_HALF_W, TankDuelEnv, TankDuelOrthographicWrapper, WORLD_H, WORLD_W,
)

from .base import Exporter, P_top, cam_ortho, cam_persp, r, rv, vfov_from_focal, xf, xf_top

W_PX, H_PX = 800, 600
TOP_PPM = 15.0
ELEV_PPM = 30.0
GROUND_ROW = 420.0                     # GymBox rear/side: z = 0 on row 0.7 * 600
ELEV_ZC = (GROUND_ROW - H_PX / 2.0) / ELEV_PPM   # height on the centre row of the true-scale views (4 m)
CUT = 2.6                              # minimum near-plane distance behind / beside the hull centre
SHELL_H = 1.25                         # barrel axis height
TANK_REACH = BARREL_END + 0.15         # hull corner / barrel tip radius around a tank centre
TRACK_HALF = 0.75                      # track centre lines at +-0.75 m (hull half width 1.0)
HULL_REACH = math.hypot(HULL_HALF_L, HULL_HALF_W) + 0.1   # near plane never cuts into the hull
HILLS_C = (20.0, 15.0)                 # tank_duel/world.gd _build_hills: ring centre (Box2D)
HILLS_R = 300.5                        # outer hill ring radius (+ margin)


class _Ids:
    """Stable sequential ids for env-owned Python objects (shell userData / explosion dicts)."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.map = {}
        self.n = 0

    def get(self, obj):
        e = self.map.get(id(obj))
        if e is None or e[0] is not obj:
            self.n += 1
            e = (obj, f"{self.prefix}{self.n}")
            self.map[id(obj)] = e
        return e[1]

    def has(self, obj):
        e = self.map.get(id(obj))
        return e is not None and e[0] is obj

    def keep(self, objs):
        live = {id(o) for o in objs}
        self.map = {k: v for k, v in self.map.items() if k in live}

    def reset(self):
        self.map = {}
        self.n = 0


class TankDuelExporter(Exporter):
    name = "tank_duel"
    title = "Tank Duel"
    fps = 50
    view_size = (W_PX, H_PX)
    mask_groups = {"tank": ["player", "enemy"]}

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self._shell_ids = _Ids("s")
        self._boom_ids = _Ids("e")
        self._reset_memory()

    def _reset_memory(self):
        self._shell_ids.reset()
        self._boom_ids.reset()
        self.steps = 0
        self.tracks = {"player": [0.0, 0.0], "enemy": [0.0, 0.0]}
        self.fire_step = {"player": -1000, "enemy": -1000}

    def make_env(self):
        return TankDuelOrthographicWrapper(TankDuelEnv(**self.env_kwargs))

    def on_reset(self, env):
        self._reset_memory()

    def on_step(self, env, action):
        b = env.unwrapped
        self.steps += 1
        dt = 1.0 / self.fps
        for name, body in (("player", b.player_body), ("enemy", b.enemy_body)):
            a = body.angle
            v = body.linearVelocity
            speed = v[0] * math.cos(a) + v[1] * math.sin(a)
            w = body.angularVelocity
            # left track (Box2D +y side) / right track speeds
            self.tracks[name][0] += (speed - w * TRACK_HALF) * dt
            self.tracks[name][1] += (speed + w * TRACK_HALF) * dt
        for p in b.projectiles:
            ud = p.userData
            if not self._shell_ids.has(ud):
                self._shell_ids.get(ud)
                self.fire_step[ud["owner"]] = self.steps
        self._shell_ids.keep([p.userData for p in b.projectiles])
        self._boom_ids.keep(b.explosions)

    # ------------------------------------------------------------------ messages
    def scene(self, env):
        b = env.unwrapped
        walls = []
        for wb in b.walls:
            for f in wb.fixtures:
                walls.append([rv(wb.transform * v) for v in f.shape.vertices])
        obstacles = []
        for i, ob in enumerate(b.obstacles):
            f = ob.fixtures[0]
            vs = [ob.transform * v for v in f.shape.vertices]
            xs, ys = [v[0] for v in vs], [v[1] for v in vs]
            obstacles.append({"c": [r(ob.position[0]), r(ob.position[1])],
                              "half": [r((max(xs) - min(xs)) / 2.0), r((max(ys) - min(ys)) / 2.0)],
                              "kind": ("concrete", "hesco", "rock")[i % 3]})
        return {"W": WORLD_W, "H": WORLD_H, "walls": walls, "obstacles": obstacles,
                "hull": [HULL_HALF_L, HULL_HALF_W]}

    def _tanks(self, b):
        return (("player", b.player_body, b.player_turret_angle), ("enemy", b.enemy_body, b.enemy_turret_angle))

    def state(self, env):
        b = env.unwrapped
        if b.player_body is None:
            return {"xf": {}}
        out_xf, tanks = {}, []
        for name, body, turret in self._tanks(b):
            out_xf[name] = xf_top(body.position[0], body.position[1], body.angle, 0.0)
            v = body.linearVelocity
            speed = v[0] * math.cos(body.angle) + v[1] * math.sin(body.angle)
            tanks.append([r(turret, 5), r(b.hit_flash.get(name, 0) / 6.0, 3), r(self.tracks[name][0], 4),
                          r(self.tracks[name][1], 4), int(min(self.steps - self.fire_step[name], 99)), r(speed, 3)])
        dyn = []
        for p in b.projectiles:
            ud = p.userData
            vx, vy = p.linearVelocity
            dyn.append({"id": self._shell_ids.get(ud), "k": "shell", "mask": "shell",
                        "xf": xf_top(p.position[0], p.position[1], math.atan2(vy, vx), SHELL_H),
                        "o": 0 if ud["owner"] == "player" else 1})
        for e in b.explosions:
            kind = "boom" if e["target"] in ("player", "enemy") else "dust"
            dyn.append({"id": self._boom_ids.get(e), "k": kind, "xf": xf(P_top(e["x"], e["y"], e["z"])),
                        "t": r(e["age"] / float(e["life"]), 4), "r": r(e["radius"][1], 3)})
        msg = {"xf": out_xf, "dyn": dyn, "p": {"tanks": tanks}}
        self._section(env, msg)
        return msg

    # ------------------------------------------------------------------ section cut (rear / side)
    def _objects_2d(self, b):
        """(id, centre (x, y), corner points) of the compact objects the elevation section can hide."""
        objs = [("enemy", (b.enemy_body.position[0], b.enemy_body.position[1]), None)]
        for i, ob in enumerate(b.obstacles):
            vs = [ob.transform * v for v in ob.fixtures[0].shape.vertices]
            objs.append((f"obs{i}", (ob.position[0], ob.position[1]), vs))
        return objs

    @staticmethod
    def _wall_dir(wb):
        """Unit direction of a perimeter wall's long axis."""
        vs = [wb.transform * v for v in wb.fixtures[0].shape.vertices]
        dx = max(v[0] for v in vs) - min(v[0] for v in vs)
        dy = max(v[1] for v in vs) - min(v[1] for v in vs)
        return (1.0, 0.0) if dx > dy else (0.0, 1.0)

    def _cut(self, b, axis):
        """Hidden ids and near distance for an elevation camera looking along `axis` (unit 2D).
        GymBox's floating slabs never cover the player, so whatever stands wholly between the
        camera and the player (in front of the hull centre's depth, overlapping its screen
        extent) is hidden (a cutaway); everything else stays, and the near plane sits just behind
        the deepest kept object, or in front of the player where a kept wall would cover it."""
        px, py = b.player_body.position[0], b.player_body.position[1]
        lat = (-axis[1], axis[0])

        def dl(x, y):
            return (x - px) * axis[0] + (y - py) * axis[1], (x - px) * lat[0] + (y - py) * lat[1]

        hidden, deepest = [], -CUT
        for oid, (cx, cy), corners in self._objects_2d(b):
            d, l = dl(cx, cy)
            if corners is None:
                lo, hi, dmin, dmax = l - TANK_REACH, l + TANK_REACH, d - TANK_REACH, d + TANK_REACH
            else:
                pts = [dl(v[0], v[1]) for v in corners]
                lo, hi = min(q[1] for q in pts), max(q[1] for q in pts)
                dmin, dmax = min(q[0] for q in pts), max(q[0] for q in pts)
            if dmax < 0.0 and hi > -TANK_REACH and lo < TANK_REACH:
                hidden.append(oid)
            else:
                deepest = min(deepest, dmin)
        # perimeter walls across the view wholly in front of the player are hidden; a kept wall
        # that would still cross in front of the player pulls the near plane in (section cut)
        near_cap = 52.0
        for i, wb in enumerate(b.walls):
            wd = self._wall_dir(wb)
            vs = [wb.transform * v for v in wb.fixtures[0].shape.vertices]
            pts = [dl(v[0], v[1]) for v in vs]
            if max(q[0] for q in pts) < 0.0 and abs(wd[0] * axis[0] + wd[1] * axis[1]) < 0.7:
                hidden.append(f"wall{i}")
                continue
            # deepest point of the wall footprint (sampled) inside the player's screen band
            band = -math.inf
            for u in range(41):
                for v in range(3):
                    a0, a1, a3 = pts[0], pts[1], pts[3]
                    fu, fv = u / 40.0, v / 2.0
                    qd = a0[0] + (a1[0] - a0[0]) * fu + (a3[0] - a0[0]) * fv
                    ql = a0[1] + (a1[1] - a0[1]) * fu + (a3[1] - a0[1]) * fv
                    if qd < 0.0 and abs(ql) < TANK_REACH:
                        band = max(band, qd)
            if band > -math.inf:
                near_cap = min(near_cap, max(-band - 0.1, HULL_REACH))
        near = min(-deepest + 0.05, near_cap, 52.05)
        for p in b.projectiles:
            d, l = dl(p.position[0], p.position[1])
            if d + 0.4 < 0.0 and (d < -near or abs(l) < TANK_REACH):
                hidden.append(self._shell_ids.get(p.userData))
        for e in b.explosions:
            d, l = dl(e["x"], e["y"])
            ext = 1.5 * float(e["radius"][1]) + 0.3
            if d + ext < 0.0 and (d < -near or abs(l) < TANK_REACH + ext):
                hidden.append(self._boom_ids.get(e))
        return hidden, near

    def _views(self, b):
        a = b.player_body.angle
        fwd = (math.cos(a), math.sin(a))
        left = (-math.sin(a), math.cos(a))   # side camera looks from the right towards the left
        return {"rear": fwd, "side": left}

    def _section(self, env, msg):
        b = env.unwrapped
        msg["vis"] = {view: self._cut(b, axis)[0] for view, axis in self._views(b).items()}

    # ------------------------------------------------------------------ cameras
    def cameras(self, env):
        b = env.unwrapped
        px, py = float(b.player_body.position[0]), float(b.player_body.position[1])
        cams = [cam_ortho("top", P_top(px, py, 30.0), P_top(px, py, 0.0), (0.0, 0.0, -1.0), H_PX / TOP_PPM, 1.0, 40.0)]
        size = H_PX / ELEV_PPM
        for view, axis in self._views(b).items():
            hidden, near = self._cut(b, axis)
            back = near + 1.0
            pos = P_top(px - axis[0] * back, py - axis[1] * back, ELEV_ZC)
            look = P_top(px, py, ELEV_ZC)
            # far plane past the outer hill ring, with the sky card (0.97 * far) behind it
            reach = (HILLS_C[0] - px) * axis[0] + (HILLS_C[1] - py) * axis[1] + HILLS_R
            cams.append(cam_ortho(view, pos, look, (0.0, 1.0, 0.0), size, 1.0, round((back + reach) / 0.97 + 1.0, 1)))
        ax, ay, az = env._cam_anchor
        a = b.player_body.angle
        pd = env.pivot_depth
        pivot = (ax + pd * math.cos(a), ay + pd * math.sin(a), az)
        cx, cy, cz = env.cam_pos
        cams.append(cam_persp("fpv", P_top(cx, cy, cz), P_top(*pivot), (0.0, 1.0, 0.0),
                              vfov_from_focal(400.0, H_PX), 0.1, 2000.0))
        return cams

    def hud(self, env):
        b = env.unwrapped
        return {"text": "hull %3d  enemy %3d" % (b.player_health, b.enemy_health)}
