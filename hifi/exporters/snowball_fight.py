"""
Snowball Fight (SnowballFightEnv + SnowballFightOrthographicWrapper), top-down, held-out env.

GymBox views (snowball_fight_orthographic.py, 800x600, real projections with painter sorting):
  top   20 px/m, north-up, follows the player but the centre is clamped to x in [10, 20],
        y in [5, 15] so the whole 30 x 20 m arena stays on screen
  rear  orthographic elevation along the player's facing, 36 px/m on both axes, player centred,
        z = 0 on row 432 (0.72 * 600); cutaway: whole objects whose anchor is more than
        REAR_CUT = 1 m behind the player are dropped (fence panels by their own centroid)
  side  same from the player's right (screen right = facing), cutaway 1 m to the right
  fpv   orbit camera 4 m behind the player along its facing at z = 2.0 m, looking at the pivot
        (player x, y, 2.0); focal 400 px on 600 (vfov 73.7 deg), near clip 0.1 m
Occlusion semantics are kept exact: forts 1.0 m, pillars 2.5 m, fence 1.2 m, character coat
0-1.12 m / head 1.12-1.45 m / beanie 1.45-1.62 m, snowballs fly flat at 1.25 m, FPV eye 2.0 m.

Hi-fi: the elevation cutaway is reproduced with state["vis"] (cover "obs<i>", the opponent,
snowballs and splats whose anchor is > 1 m behind / beside the player are hidden) plus the ortho
near plane, which sits at the cut or just behind the deepest kept object and slices the fence
(fence sides across the view that lie behind the cut, ids "fence<i>", are hidden as well).
Below z = 0 the core's ortho sky card paints a receding snow field (GymBox's flat snow band).
One consistent world: the pine forest, snowman, sled, fence and snowfall are drawn in every view
whose frame contains them; only the cutaway in front of the player (the hidden ids above and the
near plane) removes anything. The elevation far planes reach the far edge of the forest inside
each frame (the ortho sky card sits at 0.97 * far).
Masks: ids "player" / "opponent" (classes "player" / "opponent"), cover i -> id "obs<i>" with
class "fort" or "pillar" (group "cover"), snowballs are transient objects with class "snowball",
fence unregistered (black occluder). Splat effects never occlude masks (GymBox draws them).
"""
import math

from game_envs.snowball_fight_orthographic import (
    ARENA_H, ARENA_W, COVER_HEIGHT, FENCE_HEIGHT, FENCE_T, HIT_FLASH_STEPS,
    SnowballFightEnv, SnowballFightOrthographicWrapper, Z_SNOWBALL,
)

from .base import Exporter, P_top, cam_ortho, cam_persp, r, rv, vfov_from_focal, xf, xf_top

W_PX, H_PX = 800, 600
TOP_PPM = 20.0
ELEV_PPM = 36.0
GROUND_ROW = 0.72 * H_PX                       # 432
ELEV_ZC = (GROUND_ROW - H_PX / 2.0) / ELEV_PPM   # 3.667 m on the centre row
CUT = SnowballFightOrthographicWrapper.REAR_CUT
CHAR_REACH = 0.9                               # coat radius / arm stub reach around a character centre
FOREST = (-47.0, -42.5, ARENA_W + 48.5, ARENA_H + 44.0)   # pine forest bounds incl. crowns (world.gd), Box2D
CARD = 0.97                                    # the ortho sky card sits at CARD * far


def _strip_depth(px, py, axis, half_w, box):
    """Max depth along `axis` from (px, py) over the part of `box` inside the view's lateral strip
    (|lateral offset| <= half_w), or None when the strip misses the box."""
    lat = (-axis[1], axis[0])
    poly = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    for sgn in (1.0, -1.0):          # clip with sgn * lateral <= half_w (Sutherland-Hodgman)
        def g(q):
            return sgn * ((q[0] - px) * lat[0] + (q[1] - py) * lat[1]) - half_w
        out = []
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            ga, gb = g(a), g(b)
            if ga <= 0.0:
                out.append(a)
            if (ga <= 0.0) != (gb <= 0.0):
                t = ga / (ga - gb)
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        poly = out
        if not poly:
            return None
    return max((q[0] - px) * axis[0] + (q[1] - py) * axis[1] for q in poly)


class _Ids:
    """Stable sequential ids for env-owned Python objects (snowball / effect dicts)."""

    def __init__(self, prefix):
        self.prefix, self.map, self.n = prefix, {}, 0

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
        self.map, self.n = {}, 0


class SnowballFightExporter(Exporter):
    name = "snowball_fight"
    title = "Snowball Fight"
    heldout = True
    fps = 50
    view_size = (W_PX, H_PX)
    mask_groups = {"cover": ["fort", "pillar"]}

    def __init__(self, obstacles=True, **env_kwargs):
        super().__init__(**env_kwargs)
        self.obstacles = bool(obstacles)
        self._balls = _Ids("b")
        self._fx = _Ids("f")
        self._reset_memory()

    def _reset_memory(self):
        self._balls.reset()
        self._fx.reset()
        self.steps = 0
        self.throw_step = {"player": -1000, "opponent": -1000}
        self.gait = {"player": 0.0, "opponent": 0.0}

    def make_env(self):
        return SnowballFightOrthographicWrapper(SnowballFightEnv(cover=self.obstacles, **self.env_kwargs))

    def on_reset(self, env):
        self._reset_memory()

    def on_step(self, env, action):
        b = env.unwrapped
        self.steps += 1
        for name, body in (("player", b.player), ("opponent", b.opponent)):
            v = body.linearVelocity
            self.gait[name] += math.hypot(float(v.x), float(v.y)) / self.fps
        for s in b.snowballs:
            if not self._balls.has(s):
                self._balls.get(s)
                self.throw_step[s["owner"]] = self.steps
        self._balls.keep(b.snowballs)
        self._fx.keep(b.effects)

    # ------------------------------------------------------------------ messages
    def scene(self, env):
        b = env.unwrapped
        fences = []
        for fb in b.fences:
            for f in fb.fixtures:
                fences.append([rv(fb.transform * v) for v in f.shape.vertices])
        cover = [{"kind": c["kind"], "x": r(c["x"]), "y": r(c["y"]), "w": r(c["w"]), "h": r(c["h"]),
                  "a": r(c["angle"], 5), "height": COVER_HEIGHT[c["kind"]]} for c in b.cover]
        return {"W": ARENA_W, "H": ARENA_H, "fence_t": FENCE_T, "fence_h": FENCE_HEIGHT,
                "fences": fences, "cover": cover}

    def state(self, env):
        b = env.unwrapped
        if b.player is None:
            return {"xf": {}}
        chars = []
        out_xf = {}
        for name, body, facing in (("player", b.player, b.player_facing), ("opponent", b.opponent, b.opp_facing)):
            out_xf[name] = xf_top(body.position.x, body.position.y, facing, 0.0)
            v = body.linearVelocity
            fwd = float(v.x) * math.cos(facing) + float(v.y) * math.sin(facing)
            side = float(v.x) * math.sin(facing) - float(v.y) * math.cos(facing)
            chars.append([r(b.hit_flash.get(name, 0) / float(HIT_FLASH_STEPS), 3), r(self.gait[name], 4),
                          r(fwd, 3), r(side, 3), int(min(self.steps - self.throw_step[name], 99))])
        dyn = []
        for s in b.snowballs:
            p = s["body"].position
            vel = s["body"].linearVelocity
            dyn.append({"id": self._balls.get(s), "k": "ball", "mask": "snowball",
                        "xf": xf_top(p.x, p.y, math.atan2(vel.y, vel.x), Z_SNOWBALL),
                        "o": 0 if s["owner"] == "player" else 1})
        for e in b.effects:
            dyn.append({"id": self._fx.get(e), "k": "splat", "xf": xf(P_top(e["x"], e["y"], e["z"])),
                        "t": r(e["age"] / float(e["life"]), 4), "r": r(e["radius"][1], 3),
                        "hit": 1 if e["target"] in ("player", "opponent") else 0})
        msg = {"xf": out_xf, "dyn": dyn, "p": {"chars": chars}}
        msg["vis"] = {view: self._cut(b, axis)[0] for view, axis in self._views(b).items()}
        return msg

    # ------------------------------------------------------------------ elevation cutaway
    def _views(self, b):
        a = b.player_facing
        return {"rear": (math.cos(a), math.sin(a)),          # looks along the facing
                "side": (-math.sin(a), math.cos(a))}         # from the player's right towards its left

    def _cut(self, b, axis):
        """GymBox's REAR_CUT for a camera looking along `axis`: hidden ids and the near distance."""
        px, py = float(b.player.position.x), float(b.player.position.y)

        def depth(x, y):
            return (x - px) * axis[0] + (y - py) * axis[1]

        hidden, near = [], CUT
        o = b.opponent
        d = depth(float(o.position.x), float(o.position.y))
        if d < -CUT:
            hidden.append("opponent")
        else:
            near = max(near, CHAR_REACH - d)
        for i, c in enumerate(b.cover):
            d = depth(c["x"], c["y"])
            if d < -CUT:
                hidden.append(f"obs{i}")
                continue
            ca, sa = math.cos(c["angle"]), math.sin(c["angle"])
            hw, hh = c["w"] / 2.0, c["h"] / 2.0
            deepest = min(depth(c["x"] + lx * ca - ly * sa, c["y"] + lx * sa + ly * ca)
                          for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)))
            near = max(near, -deepest)
        # fence sides across the view whose centre line is behind the cut are dropped (GymBox drops
        # those panels); the others are sliced by the near plane
        for i, fb in enumerate(b.fences):
            vs = [fb.transform * v for v in fb.fixtures[0].shape.vertices]
            dx = max(v[0] for v in vs) - min(v[0] for v in vs)
            dy = max(v[1] for v in vs) - min(v[1] for v in vs)
            along = abs(axis[0]) if dx > dy else abs(axis[1])
            if depth(float(fb.position.x), float(fb.position.y)) < -CUT and along < 0.7:
                hidden.append(f"fence{i}")
        for s in b.snowballs:
            p = s["body"].position
            if depth(float(p.x), float(p.y)) < -CUT:
                hidden.append(self._balls.get(s))
        for e in b.effects:
            if depth(e["x"], e["y"]) < -CUT:
                hidden.append(self._fx.get(e))
        return hidden, near + 0.05

    # ------------------------------------------------------------------ cameras
    def cameras(self, env):
        b = env.unwrapped
        px, py = float(b.player.position.x), float(b.player.position.y)
        hx, hy = W_PX / 2.0 / TOP_PPM, H_PX / 2.0 / TOP_PPM
        tx = min(max(px, ARENA_W - hx), hx) if 2 * hx >= ARENA_W else px
        ty = min(max(py, ARENA_H - hy), hy) if 2 * hy >= ARENA_H else py
        cams = [cam_ortho("top", P_top(tx, ty, 25.0), P_top(tx, ty, 0.0), (0.0, 0.0, -1.0), H_PX / TOP_PPM, 1.0, 35.0)]
        size = H_PX / ELEV_PPM
        for view, axis in self._views(b).items():
            near = self._cut(b, axis)[1]
            back = near + 1.0
            deep = _strip_depth(px, py, axis, W_PX / 2.0 / ELEV_PPM, FOREST)
            far = max(back + 48.0, (back + (deep or 0.0) + 0.5) / CARD)
            cams.append(cam_ortho(view, P_top(px - axis[0] * back, py - axis[1] * back, ELEV_ZC), P_top(px, py, ELEV_ZC),
                                  (0.0, 1.0, 0.0), size, 1.0, far))
        ax, ay, az = env._cam_anchor
        a = b.player_facing
        pd = env.pivot_depth
        cx, cy, cz = env.cam_pos
        cams.append(cam_persp("fpv", P_top(cx, cy, cz), P_top(ax + pd * math.cos(a), ay + pd * math.sin(a), az),
                              (0.0, 1.0, 0.0), vfov_from_focal(env.fov, H_PX), env.Z_NEAR, 1500.0))
        return cams

    def hud(self, env):
        b = env.unwrapped
        return {"text": "you %3d  them %3d" % (b.player_health, b.opponent_health)}
