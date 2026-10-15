import math
import numpy as np
import pygame
from .camera_utils import orbit_camera_frame, orbit_camera_offset

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from Box2D.b2 import (
    world as b2World,
    rayCastCallback as b2RayCastCallback,
)

FPS = 50
DT = 1.0 / FPS
SCALE = 30.0  # pixels per metre of the 2D debug render

# Arena dimensions in physics units (metres); the fence is inside [0, ARENA_W] x [0, ARENA_H]
ARENA_W = 30.0
ARENA_H = 20.0
FENCE_T = 0.3        # fence thickness; interior is [FENCE_T, ARENA_W - FENCE_T] x [FENCE_T, ARENA_H - FENCE_T]
FENCE_HEIGHT = 1.2   # wooden perimeter fence (renderer only)

# Characters: circle body, rendered as coat cylinder + head + hat (~1.6 m tall) + scarf/arm stub
CHAR_RADIUS = 0.45
Z_BODY_TOP = 1.12    # coat cylinder 0 -> 1.12 m
Z_HEAD_TOP = 1.45    # head 1.12 -> 1.45 m (above a 1.0 m snow fort)
Z_HAT_TOP = 1.62     # beanie 1.45 -> 1.62 m (total height)
HEAD_RADIUS = 0.27
HAT_RADIUS = 0.29
ARM_SPAN = (0.25, 0.85)   # scarf/arm stub along the facing (local +X), metres from the centre
ARM_HALF_W = 0.09
Z_ARM = (0.82, 1.0)

# Cover: two kinds that differ only in height (this split matters for occlusion experiments)
FORT_HEIGHT = 1.0    # low snow walls: a character's head stays visible behind them
PILLAR_HEIGHT = 2.5  # tall ice blocks: fully hide a character from the FPV / rear cameras
COVER_HEIGHT = {"fort": FORT_HEIGHT, "pillar": PILLAR_HEIGHT}
COVER_DEFAULT_SIZE = {"fort": (3.0, 0.7), "pillar": (1.2, 1.2)}
COVER_TYPES = ("fort", "pillar")

# Movement / combat tuning
PLAYER_MAX_V = 4.0        # m/s (holonomic, commanded in the facing frame)
OPP_MAX_V = 3.5
MOVE_ACCEL = 20.0         # m/s^2 velocity-tracking acceleration
PLAYER_TURN_RATE = 3.0    # rad/s at |turn| = 1
OPP_TURN_RATE = 5.0       # rad/s
SNOWBALL_SPEED = 12.0     # m/s (0.24 m per step)
SNOWBALL_RADIUS = 0.15
SNOWBALL_LIFE = 63        # steps (~15 m range)
Z_SNOWBALL = 1.25         # flight height (renderer only; physics is planar)
PLAYER_THROW_CD = 20      # steps between player throws
OPP_THROW_CD = 40
OPP_AIM_NOISE = 0.05      # rad std of the opponent's aim error per throw
HIT_RADIUS = CHAR_RADIUS + SNOWBALL_RADIUS
MAX_HEALTH = 100.0
HIT_DAMAGE = 10.0

# Reward weights (same scheme as Tank Duel)
REWARD_HIT = 2.0
REWARD_HURT = -1.0
REWARD_WIN = 20.0
REWARD_LOSE = -20.0
AIM_SHAPING = 0.5         # potential-based: phi = AIM_SHAPING * cos(aim_error)

# Observation
N_COVER_OBS = 4           # nearest-K cover slots
N_LIDAR = 8               # rays around the player's facing
LIDAR_RANGE = 15.0
OBS_DIM = 17 + 6 * N_COVER_OBS + N_LIDAR

# Opponent hide-and-peek controller
HIDE_GAP = 0.3            # clearance between the hiding character and its cover (m)
PEEK_GAP = 0.45           # how far past the cover edge the opponent steps out (m)
ARRIVE_R = 0.25
NAV_MARGIN = CHAR_RADIUS + 0.45   # detour corners are this far outside a blocking cover box

# Snow splat / hit puffs (purely visual: never read by physics, observations, rewards or the RNG)
PUFF_LIFE = 8
PUFF_RADIUS = (0.2, 0.6)
HIT_FLASH_STEPS = 6

# Collision categories
CAT_WORLD = 0x0001
CAT_CHAR = 0x0002
CAT_SNOWBALL = 0x0004

# Colors (winter theme)
COLOR_SKY = (196, 216, 236)
COLOR_SNOW_A = (247, 249, 253)
COLOR_SNOW_B = (234, 239, 247)
COLOR_SNOW_OUT = (226, 233, 243)
COLOR_FENCE = (128, 86, 50)
COLOR_FORT = (226, 233, 244)
COLOR_PILLAR = (160, 210, 240)
COLOR_PLAYER = (40, 90, 210)
COLOR_OPP = (205, 45, 45)
COLOR_PLAYER_HAT = (25, 55, 140)
COLOR_OPP_HAT = (130, 25, 25)
COLOR_PLAYER_SCARF = (250, 200, 40)
COLOR_OPP_SCARF = (40, 160, 80)
COLOR_SKIN = (245, 215, 185)
COLOR_SNOWBALL_PLAYER = (255, 200, 0)   # Tank Duel's player shell colour: stands out on snow and ice
COLOR_SNOWBALL_OPP = (255, 80, 80)      # Tank Duel's enemy shell colour
COLOR_PUFF = (240, 245, 252)
COLOR_HIT_FLASH = (255, 240, 140)


def _wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _mix_color(c0, c1, t):
    """Linear blend of two RGB colors (t = 0 -> c0, t = 1 -> c1)."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))


def _cover_support(c, dx, dy):
    """Half extent of a cover box (dict with x, y, w, h, angle) along the unit direction (dx, dy)."""
    ca, sa = math.cos(c["angle"]), math.sin(c["angle"])
    return 0.5 * c["w"] * abs(dx * ca + dy * sa) + 0.5 * c["h"] * abs(-dx * sa + dy * ca)


def _cover_corners(c, margin=0.0):
    """World-space corners (CCW) of a cover box grown by margin on every side."""
    ca, sa = math.cos(c["angle"]), math.sin(c["angle"])
    hw, hh = 0.5 * c["w"] + margin, 0.5 * c["h"] + margin
    return [(c["x"] + lx * ca - ly * sa, c["y"] + lx * sa + ly * ca)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


class _AllHitsRayCast(b2RayCastCallback):
    """Records every non-sensor fixture on the ray as (fraction, userData); the env filters them."""

    def __init__(self):
        super().__init__()
        self.hits = []

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.sensor:
            return -1.0  # snowballs: ignore and continue
        self.hits.append((float(fraction), fixture.body.userData or {}))
        return 1.0       # keep going: report all fixtures on the segment


# ==============================================================================
# 1. SNOWBALL FIGHT BASE ENVIRONMENT (Top-Down, No Gravity)
# ==============================================================================
class SnowballFightEnv(gym.Env):
    """
    Snowball Fight Task (winter blend of Drone Dogfight + Tank Duel):
    Two kids in a fenced 30 x 20 m snow field. Both move holonomically in the X-Y plane (like the
    drones), have a facing that turns at a limited rate, and throw straight-flying snowballs along it.
    Static cover blocks both characters and snowballs (like the tank obstacles):
      * snow forts  — low white walls, 1.0 m tall (a character's head stays visible behind them)
      * ice pillars — tall pale-blue blocks, 2.5 m tall (fully hide a character in FPV / rear views)
    The scripted opponent plays hide-and-peek: pick a cover piece, run to its far side (relative to
    the player), wait, step out to one side, throw at the player (with aim noise), go back, repeat.
    The episode terminates when either side's health reaches 0 and truncates after max_steps.

    Constructor:
      cover=True   random forts + pillars each reset. cover=False is the obstacle-free twin: the same
                   layout is still sampled (identical RNG stream) but no bodies are created; the
                   opponent then uses the sampled spots as invisible waypoints.

    Actions (4D continuous, all in [-1, 1]):
      0: Strafe           — +1 = full speed to the player's right  (holonomic, facing frame)
      1: Forward          — +1 = full speed along the facing; the (0, 1) vector is clipped to unit length
      2: Turn             — facing rate, +1 = counter-clockwise at PLAYER_TURN_RATE rad/s
      3: Throw            — > 0 throws a snowball along the facing when ready (PLAYER_THROW_CD steps)

    Observation (49D float32, clipped to [-1, 1]):
      0-1:   Player x, y (arena-normalised to [-1, 1])
      2-3:   Player vx, vy / PLAYER_MAX_V
      4-5:   Player facing cos, sin
      6:     Player health fraction
      7:     Player throw cooldown fraction (0 = ready)
      8-9:   Opponent relative position dX / ARENA_W, dY / ARENA_H
      10-11: Opponent vx, vy / PLAYER_MAX_V
      12:    Opponent visible flag (1 = not hidden behind an ice pillar, see info["opponent_hidden"])
      13:    Clear-shot flag (1 = no cover / fence on the centre line to the opponent)
      14:    Opponent health fraction
      15-16: cos, sin of the aim error (bearing to opponent - player facing)
      17-40: Nearest N_COVER_OBS=4 cover pieces (by centre distance), 6 floats each:
             dX / ARENA_W, dY / ARENA_H, world-AABB width / 4, world-AABB depth / 4, is_pillar, present
             (all zeros for empty slots, e.g. always with cover=False)
      41-48: N_LIDAR=8 ray distances / LIDAR_RANGE to cover or fence (characters / snowballs ignored),
             ray k at angle facing + k * 45 deg (k = 0 straight ahead, counter-clockwise)

    Reward (Tank Duel scheme):
      +2 per hit on the opponent, -1 per hit taken, +20 win, -20 lose (simultaneous KO = draw, no bonus),
      plus potential-based aim shaping Phi(s') - Phi(s), Phi = 0.5 * cos(aim error).

    reset(options=...) keys (all optional; random if absent; random draws are consumed either way):
      "cover":          list of {"kind": "fort"|"pillar", "x", "y", "w", "h", "angle"(opt, rad)}; w / h are
                        full box sizes along the local X / Y axes. Placed exactly (overrides the cover flag).
      "player_start":   (x, y, facing_rad)
      "opponent_start": (x, y, facing_rad)
      "hide_schedule":  list of {"cover_index": i, "hide_steps": k, "peek_side": "left"|"right"}; cycled.
                        "left"/"right" are as seen by the player looking at the cover.
      "opponent_fires": bool (default True)

    info keys (every reset / step):
      player_pos, opponent_pos           (x, y) tuples
      opponent_hidden                    line of sight from the player blocked by an ice pillar (centre ray
                                         and both silhouette-edge rays all hit a pillar; Box2D raycast)
      player_collision                   player touching cover / fence this step
      snowball_hits_cover                # snowballs that splatted on a fort / pillar this step
      cover                              list of placed cover dicts (kind, x, y, w, h, angle)
      health, opponent_health
      hits_dealt, hits_taken             this step
      opponent_state                     hide-and-peek controller state
      is_success
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, cover=True):
        self.render_mode = render_mode
        self.cover_enabled = bool(cover)
        self.screen = None
        self.clock = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.world = None
        self.player = None
        self.opponent = None
        self.fences = []
        self.cover_bodies = []
        self.cover = []          # placed cover dicts (index = cover_index)
        self.virtual_cover = []  # sampled layout (used as opponent waypoints when nothing is placed)
        self.snowballs = []
        self.max_steps = 1000
        # Visual-only effects: splat / hit puffs and per-character hit-flash countdowns
        self.effects = []
        self.hit_flash = {"player": 0, "opponent": 0}

    # ------------------------------------------------------------------
    def _sample_cover(self):
        """Random layout in the central band (spawn columns at x < 5 and x > 25 stay clear):
        2-4 snow forts (randomly turned 0 / 90 deg) and 2-3 ice pillars, with a >= 2 m walkway between."""
        rng = self.np_random
        n_forts = int(rng.integers(2, 5))
        n_pillars = int(rng.integers(2, 4))
        kinds = ["fort"] * n_forts + ["pillar"] * n_pillars
        placed, boxes = [], []
        for kind in kinds:
            for _ in range(60):
                if kind == "fort":
                    w, h = float(rng.uniform(2.5, 4.0)), float(rng.uniform(0.6, 0.8))
                    angle = 0.0 if rng.random() < 0.5 else math.pi / 2
                else:
                    w, h = float(rng.uniform(0.9, 1.4)), float(rng.uniform(0.9, 1.4))
                    angle = 0.0
                x = float(rng.uniform(8.0, ARENA_W - 8.0))
                y = float(rng.uniform(3.0, ARENA_H - 3.0))
                hx, hy = (w / 2, h / 2) if angle == 0.0 else (h / 2, w / 2)
                if any(abs(x - bx) < hx + bw + 2.0 and abs(y - by) < hy + bh + 2.0
                       for bx, by, bw, bh in boxes):
                    continue
                boxes.append((x, y, hx, hy))
                placed.append({"kind": kind, "x": x, "y": y, "w": w, "h": h, "angle": angle})
                break
        return placed

    @staticmethod
    def _parse_cover(items):
        out = []
        for c in items:
            kind = str(c.get("kind", "fort")).lower()
            if kind not in COVER_TYPES:
                raise ValueError(f"cover kind must be 'fort' or 'pillar', got {kind!r}")
            dw, dh = COVER_DEFAULT_SIZE[kind]
            out.append({"kind": kind, "x": float(c["x"]), "y": float(c["y"]),
                        "w": float(c.get("w", dw)), "h": float(c.get("h", dh)),
                        "angle": float(c.get("angle", 0.0))})
        return out

    def _make_character(self, x, y, facing, ctype):
        body = self.world.CreateDynamicBody(position=(float(x), float(y)), angle=float(facing),
                                            linearDamping=0.0, fixedRotation=True)
        body.CreateCircleFixture(radius=CHAR_RADIUS, density=1.0, friction=0.0,
                                 categoryBits=CAT_CHAR, maskBits=CAT_WORLD | CAT_CHAR)
        body.userData = {"type": ctype}
        return body

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        rng = self.np_random

        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.fences, self.cover_bodies, self.snowballs = [], [], []
        self.effects = []
        self.hit_flash = {"player": 0, "opponent": 0}
        self.steps = 0
        self.player_hits = 0
        self.opp_hits = 0

        # Perimeter fence (4 static boxes inside the arena rectangle)
        h = FENCE_T / 2
        for cx, cy, hw, hh in ((ARENA_W / 2, h, ARENA_W / 2, h), (ARENA_W / 2, ARENA_H - h, ARENA_W / 2, h),
                               (h, ARENA_H / 2, h, ARENA_H / 2), (ARENA_W - h, ARENA_H / 2, h, ARENA_H / 2)):
            b = self.world.CreateStaticBody(position=(cx, cy))
            b.CreatePolygonFixture(box=(hw, hh), density=0, friction=0.0, categoryBits=CAT_WORLD)
            b.userData = {"type": "fence"}
            self.fences.append(b)

        # Every random draw happens up front in a fixed order, whatever the options / cover flag say,
        # so the cover / no-cover twins and scripted variants share one RNG stream.
        sampled = self._sample_cover()
        p_start = (float(rng.uniform(2.5, 4.5)), float(rng.uniform(4.0, ARENA_H - 4.0)),
                   float(rng.uniform(-0.2, 0.2)))
        o_start = (float(rng.uniform(ARENA_W - 4.5, ARENA_W - 2.5)), float(rng.uniform(4.0, ARENA_H - 4.0)),
                   math.pi + float(rng.uniform(-0.2, 0.2)))
        self.opp_cd = int(rng.integers(10, OPP_THROW_CD))

        self.virtual_cover = sampled
        if "cover" in options and options["cover"] is not None:
            self.cover = self._parse_cover(options["cover"])
        else:
            self.cover = [dict(c) for c in sampled] if self.cover_enabled else []
        for i, c in enumerate(self.cover):
            b = self.world.CreateStaticBody(position=(c["x"], c["y"]), angle=c["angle"])
            b.CreatePolygonFixture(box=(c["w"] / 2, c["h"] / 2), density=0, friction=0.0,
                                   categoryBits=CAT_WORLD)
            b.userData = {"type": c["kind"], "cover_index": i}
            self.cover_bodies.append(b)

        if options.get("player_start") is not None:
            p_start = tuple(float(v) for v in options["player_start"])
        if options.get("opponent_start") is not None:
            o_start = tuple(float(v) for v in options["opponent_start"])
        self.player_facing = _wrap(p_start[2])
        self.opp_facing = _wrap(o_start[2])
        self.player = self._make_character(p_start[0], p_start[1], self.player_facing, "player")
        self.opponent = self._make_character(o_start[0], o_start[1], self.opp_facing, "opponent")

        self.player_health = MAX_HEALTH
        self.opponent_health = MAX_HEALTH
        self.player_cd = 0

        # Opponent hide-and-peek controller state
        self.opponent_fires = bool(options.get("opponent_fires", True))
        self.hide_schedule = list(options.get("hide_schedule") or [])
        self.sched_i = 0
        self.opp_state = "pick"
        self.opp_state_steps = 0
        self.opp_cover_idx = None
        self.opp_hide_steps = 0
        self.opp_peek_side = "left"
        self.opp_hide_spot = (float(self.opponent.position.x), float(self.opponent.position.y))
        self.opp_peek_spot = self.opp_hide_spot
        self.opp_timer = 0
        self.opp_thrown = False

        self._aim_potential = AIM_SHAPING * math.cos(self._aim_error())
        return self._get_obs(), self._get_info(0, 0, 0, False)

    # ------------------------------------------------------------------
    # Geometry queries (Box2D raycasts)
    # ------------------------------------------------------------------
    def _ray_hits(self, p1, p2):
        """All cover / fence hits on segment p1 -> p2 sorted by fraction (characters and snowballs skipped)."""
        if (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 < 1e-10:
            return []
        cb = _AllHitsRayCast()
        self.world.RayCast(cb, (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        hits = [(f, d) for f, d in cb.hits if d.get("type") not in ("player", "opponent")]
        hits.sort(key=lambda t: t[0])
        return hits

    def _side_rays(self, p, q, half_width):
        """Centre ray plus two rays offset sideways by +-half_width (a character-wide corridor)."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return [(p, q)]
        nx, ny = -dy / d * half_width, dx / d * half_width
        return [(p, q), ((p[0] + nx, p[1] + ny), (q[0] + nx, q[1] + ny)),
                ((p[0] - nx, p[1] - ny), (q[0] - nx, q[1] - ny))]

    def _opponent_hidden(self):
        """True when the centre ray and both silhouette-edge rays from the player to the opponent all hit an ice pillar."""
        p = (float(self.player.position.x), float(self.player.position.y))
        o = (float(self.opponent.position.x), float(self.opponent.position.y))
        for a, b in self._side_rays(p, o, CHAR_RADIUS):
            if not any(d.get("type") == "pillar" for _, d in self._ray_hits(a, b)):
                return False
        return True

    def _clear_shot(self):
        p = (float(self.player.position.x), float(self.player.position.y))
        o = (float(self.opponent.position.x), float(self.opponent.position.y))
        return not self._ray_hits(p, o)

    def _first_cover_on_path(self, p, q):
        """Index of the nearest cover piece crossing a character-wide corridor p -> q, or None."""
        best, best_f = None, 2.0
        for a, b in self._side_rays(p, q, CHAR_RADIUS):
            for f, d in self._ray_hits(a, b):
                if "cover_index" in d and f < best_f:
                    best, best_f = d["cover_index"], f
                    break
        return best

    def _player_touching(self):
        for ce in self.player.contacts:
            if ce.contact.touching and (ce.other.userData or {}).get("type") in ("fence",) + COVER_TYPES:
                return True
        return False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    @staticmethod
    def _drive(body, tvx, tvy):
        """Holonomic velocity tracking with bounded acceleration (world frame)."""
        v = body.linearVelocity
        dvx, dvy = tvx - float(v.x), tvy - float(v.y)
        dv = math.hypot(dvx, dvy)
        lim = MOVE_ACCEL * DT
        if dv > lim:
            dvx, dvy = dvx * lim / dv, dvy * lim / dv
        body.linearVelocity = (float(v.x) + dvx, float(v.y) + dvy)

    def _throw(self, shooter, heading, owner):
        """Spawn a snowball just in front of the thrower. Snowballs are sensors that collide with nothing
        physically; hits / cover impacts are resolved geometrically in step() from 'prev' (the thrower's
        centre on the first step, so a ball thrown point-blank into cover still splats)."""
        dx, dy = math.cos(heading), math.sin(heading)
        sx, sy = float(shooter.position.x), float(shooter.position.y)
        off = CHAR_RADIUS + SNOWBALL_RADIUS + 0.05
        body = self.world.CreateDynamicBody(position=(sx + off * dx, sy + off * dy), angle=heading,
                                            linearDamping=0.0, fixedRotation=True)
        body.CreateCircleFixture(radius=SNOWBALL_RADIUS, density=0.05, isSensor=True,
                                 categoryBits=CAT_SNOWBALL, maskBits=0)
        body.linearVelocity = (SNOWBALL_SPEED * dx, SNOWBALL_SPEED * dy)
        body.userData = {"type": "snowball", "owner": owner}
        self.snowballs.append({"body": body, "life": SNOWBALL_LIFE, "owner": owner, "prev": (sx, sy)})

    def _resolve_snowballs(self):
        """Advance-and-collide: a ball dies on the first thing its swept segment touches (target character,
        cover or fence) or on expiry. Returns (hits on opponent, hits on player, cover splats)."""
        dealt = taken = splats = 0
        alive = []
        for b in self.snowballs:
            b["life"] -= 1
            body = b["body"]
            p0 = b["prev"]
            vx, vy = float(body.linearVelocity.x), float(body.linearVelocity.y)
            sp = math.hypot(vx, vy) + 1e-9
            ux, uy = vx / sp, vy / sp
            p1 = (float(body.position.x) + ux * SNOWBALL_RADIUS, float(body.position.y) + uy * SNOWBALL_RADIUS)
            seg = (p1[0] - p0[0], p1[1] - p0[1])
            hits = self._ray_hits(p0, p1)
            f_obs = hits[0][0] if hits else 2.0

            target = self.opponent if b["owner"] == "player" else self.player
            c = (float(target.position.x), float(target.position.y))
            dd = seg[0] ** 2 + seg[1] ** 2
            t = 0.0 if dd < 1e-12 else min(1.0, max(0.0, ((c[0] - p0[0]) * seg[0] + (c[1] - p0[1]) * seg[1]) / dd))
            qx, qy = p0[0] + t * seg[0], p0[1] + t * seg[1]
            if math.hypot(qx - c[0], qy - c[1]) < HIT_RADIUS and t <= f_obs:
                who = "opponent" if b["owner"] == "player" else "player"
                if who == "opponent":
                    dealt += 1
                else:
                    taken += 1
                self._spawn_puff(qx, qy, who)
                self.world.DestroyBody(body)
                continue
            if hits:
                kind = hits[0][1].get("type")
                if kind in COVER_TYPES:
                    splats += 1
                f = hits[0][0]
                self._spawn_puff(p0[0] + f * seg[0] - ux * SNOWBALL_RADIUS,
                                 p0[1] + f * seg[1] - uy * SNOWBALL_RADIUS, kind)
                self.world.DestroyBody(body)
                continue
            if b["life"] <= 0:
                self.world.DestroyBody(body)
                continue
            b["prev"] = (float(body.position.x), float(body.position.y))
            alive.append(b)
        self.snowballs = alive
        return dealt, taken, splats

    # ------------------------------------------------------------------
    # Visual-only effects
    # ------------------------------------------------------------------
    def _age_effects(self):
        for e in self.effects:
            e["age"] += 1
        self.effects = [e for e in self.effects if e["age"] < e["life"]]
        for k in self.hit_flash:
            self.hit_flash[k] = max(0, self.hit_flash[k] - 1)

    def _spawn_puff(self, x, y, hit_type):
        """Record a white snow splat at the impact point (visual state only)."""
        self.effects.append({"x": float(x), "y": float(y), "z": Z_SNOWBALL, "age": 0,
                             "life": PUFF_LIFE, "radius": PUFF_RADIUS, "target": hit_type or "fence"})
        if hit_type in self.hit_flash:
            self.hit_flash[hit_type] = HIT_FLASH_STEPS

    # ------------------------------------------------------------------
    # Opponent: scripted hide-and-peek controller
    # ------------------------------------------------------------------
    def _spots(self):
        """Cover pieces the opponent hides behind: the placed cover, else the sampled (virtual) layout."""
        return self.cover if self.cover else self.virtual_cover

    @staticmethod
    def _clamp_interior(x, y):
        m = FENCE_T + CHAR_RADIUS + 0.2
        return (min(max(x, m), ARENA_W - m), min(max(y, m), ARENA_H - m))

    def _hide_spot(self, idx):
        """Point on the far side of cover idx as seen from the player's current position."""
        c = self._spots()[idx]
        px, py = float(self.player.position.x), float(self.player.position.y)
        ux, uy = c["x"] - px, c["y"] - py
        d = math.hypot(ux, uy)
        ux, uy = (ux / d, uy / d) if d > 1e-6 else (1.0, 0.0)
        r = _cover_support(c, ux, uy) + CHAR_RADIUS + HIDE_GAP
        return self._clamp_interior(c["x"] + ux * r, c["y"] + uy * r)

    def _peek_spot(self, idx, side):
        """Step out sideways from the hide spot, just past the cover edge ('left' = player's left)."""
        c = self._spots()[idx]
        hx, hy = self.opp_hide_spot
        px, py = float(self.player.position.x), float(self.player.position.y)
        ux, uy = c["x"] - px, c["y"] - py
        d = math.hypot(ux, uy)
        ux, uy = (ux / d, uy / d) if d > 1e-6 else (1.0, 0.0)
        nx, ny = (-uy, ux) if side == "left" else (uy, -ux)
        r = _cover_support(c, nx, ny) + CHAR_RADIUS + PEEK_GAP
        return self._clamp_interior(hx + nx * r, hy + ny * r)

    def _opp_set_state(self, st):
        self.opp_state = st
        self.opp_state_steps = 0

    def _opp_pick(self):
        rng = self.np_random
        spots = self._spots()
        ox, oy = float(self.opponent.position.x), float(self.opponent.position.y)
        px, py = float(self.player.position.x), float(self.player.position.y)
        if self.hide_schedule:
            e = self.hide_schedule[self.sched_i % len(self.hide_schedule)]
            self.sched_i += 1
            self.opp_cover_idx = int(e.get("cover_index", 0)) % len(spots) if spots else None
            self.opp_hide_steps = int(e.get("hide_steps", 40))
            self.opp_peek_side = "right" if str(e.get("peek_side", "left")).lower() == "right" else "left"
        else:
            if spots:
                # Prefer nearby cover that is not right next to the player
                w = np.array([math.exp(-math.hypot(c["x"] - ox, c["y"] - oy) / 6.0)
                              * (math.hypot(c["x"] - px, c["y"] - py) > 4.0) for c in spots]) + 1e-9
                self.opp_cover_idx = int(rng.choice(len(spots), p=w / w.sum()))
            else:
                self.opp_cover_idx = None
            self.opp_hide_steps = int(rng.integers(20, 61))
            self.opp_peek_side = "left" if rng.random() < 0.5 else "right"
        if self.opp_cover_idx is None:
            self.opp_hide_spot = (ox, oy)
        else:
            self.opp_hide_spot = self._hide_spot(self.opp_cover_idx)
        self._opp_set_state("run")

    def _nav_waypoint(self, pos, target):
        """Straight to target if the corridor is clear, else to the best visible corner of the blocking cover."""
        blk = self._first_cover_on_path(pos, target)
        if blk is None:
            return target
        best, best_cost = target, np.inf
        for corner in _cover_corners(self.cover[blk], NAV_MARGIN):
            corner = self._clamp_interior(*corner)
            d0 = math.hypot(corner[0] - pos[0], corner[1] - pos[1])
            if d0 < 0.3 or self._first_cover_on_path(pos, corner) is not None:
                continue
            cost = d0 + math.hypot(target[0] - corner[0], target[1] - corner[1])
            if cost < best_cost:
                best, best_cost = corner, cost
        return best

    def _opponent_ai(self):
        rng = self.np_random
        o = self.opponent
        ox, oy = float(o.position.x), float(o.position.y)
        px, py = float(self.player.position.x), float(self.player.position.y)
        want_player = math.atan2(py - oy, px - ox)
        self.opp_state_steps += 1

        if self.opp_state == "pick":
            self._opp_pick()
        st = self.opp_state
        # Keep the hide spot on the far side of the cover as the player moves
        if st in ("run", "hide", "return") and self.opp_cover_idx is not None and self.steps % 10 == 0:
            self.opp_hide_spot = self._hide_spot(self.opp_cover_idx)

        def arrived(spot):
            return math.hypot(spot[0] - ox, spot[1] - oy) < ARRIVE_R

        target, face_player = None, True
        if st == "run":
            target, face_player = self.opp_hide_spot, False
            if arrived(target) or self.opp_state_steps > 250:
                self._opp_set_state("hide")
                self.opp_timer = self.opp_hide_steps
        elif st == "hide":
            target = self.opp_hide_spot
            self.opp_timer -= 1
            if self.opp_timer <= 0:
                if self.opp_cover_idx is None:
                    self.opp_peek_spot = self.opp_hide_spot
                else:
                    self.opp_peek_spot = self._peek_spot(self.opp_cover_idx, self.opp_peek_side)
                self._opp_set_state("peek")
        elif st == "peek":
            target = self.opp_peek_spot
            if arrived(target) or self.opp_state_steps > 120:
                self._opp_set_state("throw")
                self.opp_thrown = False
        elif st == "throw":
            target = self.opp_peek_spot
            if not self.opp_thrown:
                aligned = abs(_wrap(want_player - self.opp_facing)) < 0.15
                if self.opponent_fires and self.opp_cd == 0 and aligned:
                    self._throw(o, self.opp_facing + float(rng.normal(0.0, OPP_AIM_NOISE)), "opponent")
                    self.opp_cd = OPP_THROW_CD
                    self.opp_thrown = True
                    self.opp_timer = 10
                elif self.opp_state_steps > (50 if self.opponent_fires else 15):
                    self._opp_set_state("return")
            else:
                self.opp_timer -= 1
                if self.opp_timer <= 0:
                    self._opp_set_state("return")
        elif st == "return":
            target, face_player = self.opp_hide_spot, False
            if arrived(target) or self.opp_state_steps > 150:
                self._opp_set_state("pick")

        tvx = tvy = 0.0
        if target is not None:
            wx, wy = self._nav_waypoint((ox, oy), target)
            dx, dy = wx - ox, wy - oy
            d = math.hypot(dx, dy)
            if d > 1e-6:
                sp = OPP_MAX_V * min(1.0, d / 0.8)
                tvx, tvy = dx / d * sp, dy / d * sp
        self._drive(o, tvx, tvy)

        want = want_player
        if not face_player and math.hypot(tvx, tvy) > 0.3:
            want = math.atan2(tvy, tvx)
        rate = OPP_TURN_RATE * DT
        self.opp_facing = _wrap(self.opp_facing + float(np.clip(_wrap(want - self.opp_facing), -rate, rate)))
        o.angle = self.opp_facing

    # ------------------------------------------------------------------
    def _aim_error(self):
        dx = float(self.opponent.position.x - self.player.position.x)
        dy = float(self.opponent.position.y - self.player.position.y)
        return _wrap(math.atan2(dy, dx) - self.player_facing)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.steps += 1
        self._age_effects()
        self.player_cd = max(0, self.player_cd - 1)
        self.opp_cd = max(0, self.opp_cd - 1)

        # ---- Player: holonomic move in the facing frame (strafe, forward), turn rate, throw ----
        strafe, fwd = float(action[0]), float(action[1])
        n = math.hypot(strafe, fwd)
        if n > 1.0:
            strafe, fwd = strafe / n, fwd / n
        c, s = math.cos(self.player_facing), math.sin(self.player_facing)
        # forward axis = (c, s), right axis = (s, -c)
        self._drive(self.player, (fwd * c + strafe * s) * PLAYER_MAX_V, (fwd * s - strafe * c) * PLAYER_MAX_V)
        self.player_facing = _wrap(self.player_facing + float(action[2]) * PLAYER_TURN_RATE * DT)
        self.player.angle = self.player_facing
        if float(action[3]) > 0.0 and self.player_cd == 0:
            self._throw(self.player, self.player_facing, "player")
            self.player_cd = PLAYER_THROW_CD

        # ---- Opponent hide-and-peek ----
        self._opponent_ai()

        # ---- Physics ----
        self.world.Step(DT, 6, 2)
        self.world.ClearForces()
        collision = self._player_touching()

        # ---- Snowballs ----
        dealt, taken, splats = self._resolve_snowballs()
        self.player_hits += dealt
        self.opp_hits += taken
        self.opponent_health = max(0.0, self.opponent_health - HIT_DAMAGE * dealt)
        self.player_health = max(0.0, self.player_health - HIT_DAMAGE * taken)

        reward = REWARD_HIT * dealt + REWARD_HURT * taken
        potential = AIM_SHAPING * math.cos(self._aim_error())
        reward += potential - self._aim_potential
        self._aim_potential = potential

        terminated = False
        if self.opponent_health <= 0 and self.player_health <= 0:
            terminated = True
        elif self.opponent_health <= 0:
            reward += REWARD_WIN
            terminated = True
        elif self.player_health <= 0:
            reward += REWARD_LOSE
            terminated = True
        truncated = (not terminated) and self.steps >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, \
            self._get_info(dealt, taken, splats, collision)

    # ------------------------------------------------------------------
    def _get_info(self, dealt, taken, splats, collision):
        return {
            "player_pos": (float(self.player.position.x), float(self.player.position.y)),
            "opponent_pos": (float(self.opponent.position.x), float(self.opponent.position.y)),
            "opponent_hidden": bool(self._opponent_hidden()),
            "player_collision": bool(collision),
            "snowball_hits_cover": int(splats),
            "cover": [dict(c) for c in self.cover],
            "health": float(self.player_health),
            "opponent_health": float(self.opponent_health),
            "hits_dealt": int(dealt),
            "hits_taken": int(taken),
            "opponent_state": self.opp_state,
            "is_success": bool(self.opponent_health <= 0 < self.player_health),
        }

    def _get_obs(self):
        p, o = self.player, self.opponent
        px, py = float(p.position.x), float(p.position.y)
        dx, dy = float(o.position.x) - px, float(o.position.y) - py
        aim = self._aim_error()
        obs = [
            px / ARENA_W * 2.0 - 1.0, py / ARENA_H * 2.0 - 1.0,
            float(p.linearVelocity.x) / PLAYER_MAX_V, float(p.linearVelocity.y) / PLAYER_MAX_V,
            math.cos(self.player_facing), math.sin(self.player_facing),
            self.player_health / MAX_HEALTH, self.player_cd / PLAYER_THROW_CD,
            dx / ARENA_W, dy / ARENA_H,
            float(o.linearVelocity.x) / PLAYER_MAX_V, float(o.linearVelocity.y) / PLAYER_MAX_V,
            0.0 if self._opponent_hidden() else 1.0,
            1.0 if self._clear_shot() else 0.0,
            self.opponent_health / MAX_HEALTH,
            math.cos(aim), math.sin(aim),
        ]
        # Nearest cover pieces (world-axis bounding box sizes)
        near = sorted(self.cover, key=lambda c: math.hypot(c["x"] - px, c["y"] - py))[:N_COVER_OBS]
        for c in near:
            ca, sa = abs(math.cos(c["angle"])), abs(math.sin(c["angle"]))
            obs += [(c["x"] - px) / ARENA_W, (c["y"] - py) / ARENA_H,
                    (c["w"] * ca + c["h"] * sa) / 4.0, (c["w"] * sa + c["h"] * ca) / 4.0,
                    1.0 if c["kind"] == "pillar" else 0.0, 1.0]
        obs += [0.0] * (6 * (N_COVER_OBS - len(near)))
        # Lidar to cover / fence
        for k in range(N_LIDAR):
            a = self.player_facing + k * 2.0 * math.pi / N_LIDAR
            hits = self._ray_hits((px, py), (px + LIDAR_RANGE * math.cos(a), py + LIDAR_RANGE * math.sin(a)))
            obs.append(hits[0][0] if hits else 1.0)
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    # ------------------------------------------------------------------
    def render(self):
        """2D pygame debug render (top-down bird's-eye)."""
        if self.render_mode is None:
            return
        sw, sh = int(ARENA_W * SCALE), int(ARENA_H * SCALE)
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((sw, sh))
            else:
                self.screen = pygame.Surface((sw, sh))
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_SNOW_A)
        colors = {"fence": COLOR_FENCE, "fort": COLOR_FORT, "pillar": COLOR_PILLAR}
        for b in self.fences + self.cover_bodies:
            for f in b.fixtures:
                verts = [(int((b.transform * v)[0] * SCALE), int(sh - (b.transform * v)[1] * SCALE))
                         for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, colors[b.userData["type"]], verts)
                pygame.draw.polygon(self.screen, (90, 100, 120), verts, 1)
        for b in self.snowballs:
            pos = b["body"].position
            color = COLOR_SNOWBALL_PLAYER if b["owner"] == "player" else COLOR_SNOWBALL_OPP
            pygame.draw.circle(self.screen, color, (int(pos.x * SCALE), int(sh - pos.y * SCALE)), 5)
        for body, facing, color in ((self.player, self.player_facing, COLOR_PLAYER),
                                    (self.opponent, self.opp_facing, COLOR_OPP)):
            c = (int(body.position.x * SCALE), int(sh - body.position.y * SCALE))
            pygame.draw.circle(self.screen, color, c, int(CHAR_RADIUS * SCALE))
            tip = (int(c[0] + ARM_SPAN[1] * SCALE * math.cos(facing)),
                   int(c[1] - ARM_SPAN[1] * SCALE * math.sin(facing)))
            pygame.draw.line(self.screen, (30, 30, 30), c, tip, 3)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ==============================================================================
# 2. 3D ORTHOGRAPHIC MULTI-VIEW WRAPPER FOR SNOWBALL FIGHT
# ==============================================================================
class SnowballFightOrthographicWrapper(gym.Wrapper):
    """
    Extrudes the 2D top-down physics into a 3D volumetric snow scene.

    Coordinate mapping:
      physics X  -> 3D X (East-West), physics Y -> 3D Y (North-South), extrusion -> 3D Z (up)

    Views produced (all 800 x 600, painter's algorithm with back-face culling):
      Top-Down  : (X, Y) plan view, world-aligned (north up), scrolling with the player but clamped so
                  the whole arena is always on screen
      Rear      : orthographic elevation looking along the player's facing from just behind the
                  player (objects whose centre is more than REAR_CUT behind the player are dropped)
      Side      : orthographic elevation from the player's right looking left (screen-right = facing;
                  objects whose centre is more than REAR_CUT to the player's right are dropped)
      FPV       : true 3D third-person view; orbit camera trailing 4 m behind the player and 2 m up,
                  looking along the facing (Left / Right / Up / Down / Center orbit like Tank Duel)

    render_masks(target) returns visible (occlusion-aware) binary masks of the same 4 views.
    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """

    REAR_CUT = 1.0   # metres behind / beside the player where the elevation views start
    Z_NEAR = 0.1

    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.VIEWPORT_W = 800
        self.VIEWPORT_H = 600
        self.fov = 400.0          # FPV focal length (pixels)
        self.top_scale = 20.0     # pixels per metre, top-down
        self.elev_scale = 36.0    # pixels per metre, rear / side
        self.ground_y = int(self.VIEWPORT_H * 0.72)   # screen row of z = 0 in rear / side

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0

        # Follow/Orbit camera baseline (metres): trails the player along its facing. With the camera at
        # z = 2.0 a 2.5 m pillar hides a character standing right behind it, a 1.0 m fort does not hide its head.
        self.camera_dist = 4.0
        self.camera_height = 2.0

        # Orbit pivot distance (m) along the FPV optical axis: the player itself
        self.pivot_depth = self.camera_dist

        # Last computed camera world position
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        return self.env.reset(**kwargs)

    @property
    def _cam_anchor(self):
        """Un-orbited FPV camera anchor (world X, Y, Z) used for camera-relative projection."""
        base = self.env.unwrapped
        if getattr(base, "player", None) is None:
            return (0.0, 0.0, 0.0)
        a = base.player_facing
        px, py = float(base.player.position.x), float(base.player.position.y)
        return (px - math.cos(a) * self.camera_dist, py - math.sin(a) * self.camera_dist, self.camera_height)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the orbited camera in world space."""
        base = self.env.unwrapped
        if getattr(base, "player", None) is None:
            return (0.0, 0.0, 0.0)
        ax, ay, az = self._cam_anchor
        a = base.player_facing
        d_depth, d_lat, d_vert = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)
        # Inverse of the FPV mapping: depth axis = (cos a, sin a), lateral (right) axis = (sin a, -cos a)
        cos_a, sin_a = math.cos(a), math.sin(a)
        return (ax + d_depth * cos_a + d_lat * sin_a, ay + d_depth * sin_a - d_lat * cos_a, az + d_vert)

    def update_camera_pose(self, command):
        """Accumulates continuous orbital displacement for the 3D FPV camera."""
        if isinstance(command, (list, tuple)):
            command = command[0] if len(command) > 0 else "Center"
        cmd = str(command).strip().title()

        if cmd == "Left":
            self.cam_yaw -= self.step_yaw
        elif cmd == "Right":
            self.cam_yaw += self.step_yaw
        elif cmd == "Up":
            self.cam_pitch = min(self.cam_pitch + self.step_pitch, math.radians(45.0))
        elif cmd == "Down":
            self.cam_pitch = max(self.cam_pitch - self.step_pitch, math.radians(-30.0))
        elif cmd == "Center":
            self.cam_yaw *= 0.7
            self.cam_pitch *= 0.7

    # ------------------------------------------------------------------
    # Scene construction: list of primitives
    #   poly: {"kind": "poly", "verts": [(x, y, z)...], "n": outward normal, "color", "tag", "layer", "outline"}
    #   disk: {"kind": "disk", "c": (x, y, z), "r": metres, ...}  (camera-facing billboard)
    # tag = (type, cover_index) selects primitives for render_masks; layer 0 = ground (drawn first, in order)
    # ------------------------------------------------------------------
    @staticmethod
    def _shade(color, n):
        """Simple directional shading for vertical faces (light from the south-west); tops keep their color."""
        if abs(n[2]) > 0.5:
            return color
        k = 0.80 + 0.15 * (-0.6 * n[0] - 0.8 * n[1])
        return tuple(max(0, min(255, int(c * k))) for c in color)

    def _prism(self, prims, poly2d, z0, z1, color, tag, panel=None, anchor=None):
        """Extrude a CCW 2D polygon into a prism (top + sides). With panel set, long side faces and (for
        quads) the top face are split into <= panel metre pieces so painter's sorting stays local.
        anchor (x, y, z): object reference point; the rear / side cutaway keeps or drops the whole object
        by its depth (None = each face is judged by its own centroid)."""
        first = len(prims)
        n = len(poly2d)
        for i in range(n):
            (x0, y0), (x1, y1) = poly2d[i], poly2d[(i + 1) % n]
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln < 1e-9:
                continue
            nrm = ((y1 - y0) / ln, -(x1 - x0) / ln, 0.0)
            col = self._shade(color, nrm)
            k = max(1, int(math.ceil(ln / panel))) if panel else 1
            for j in range(k):
                ta, tb = j / k, (j + 1) / k
                ax, ay = x0 + (x1 - x0) * ta, y0 + (y1 - y0) * ta
                bx, by = x0 + (x1 - x0) * tb, y0 + (y1 - y0) * tb
                prims.append({"kind": "poly", "verts": [(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)],
                              "n": nrm, "color": col, "tag": tag, "layer": 1, "outline": True})
        if panel and n == 4:
            (ax, ay), (bx, by), (cx, cy), (dx, dy) = poly2d
            nu = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / panel)))
            nv = max(1, int(math.ceil(math.hypot(dx - ax, dy - ay) / panel)))

            def bil(u, v):
                return (ax + (bx - ax) * u + (dx - ax) * v + (ax - bx + cx - dx) * u * v,
                        ay + (by - ay) * u + (dy - ay) * v + (ay - by + cy - dy) * u * v)
            for iu in range(nu):
                for iv in range(nv):
                    q = [bil(iu / nu, iv / nv), bil((iu + 1) / nu, iv / nv),
                         bil((iu + 1) / nu, (iv + 1) / nv), bil(iu / nu, (iv + 1) / nv)]
                    prims.append({"kind": "poly", "verts": [(x, y, z1) for x, y in q], "n": (0.0, 0.0, 1.0),
                                  "color": color, "tag": tag, "layer": 1, "outline": True})
        else:
            prims.append({"kind": "poly", "verts": [(x, y, z1) for x, y in poly2d], "n": (0.0, 0.0, 1.0),
                          "color": color, "tag": tag, "layer": 1, "outline": True})
        for pr in prims[first:]:
            vs = pr["verts"]
            pr["anchor"] = anchor if anchor is not None else (
                sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs), sum(v[2] for v in vs) / len(vs))

    def _scene_prims(self):
        base = self.env.unwrapped
        prims = []

        # A. Snow ground: a huge outer field, then the checkered arena tiles (2 m) on top
        g = 500.0
        prims.append({"kind": "poly", "verts": [(-g, -g, 0.0), (ARENA_W + g, -g, 0.0), (ARENA_W + g, ARENA_H + g, 0.0),
                                                (-g, ARENA_H + g, 0.0)],
                      "n": (0.0, 0.0, 1.0), "color": COLOR_SNOW_OUT, "tag": ("ground", None), "layer": 0,
                      "outline": False})
        tile = 2.0
        for xi, x0 in enumerate(np.arange(0.0, ARENA_W, tile)):
            x1 = min(x0 + tile, ARENA_W)
            for yi, y0 in enumerate(np.arange(0.0, ARENA_H, tile)):
                y1 = min(y0 + tile, ARENA_H)
                prims.append({"kind": "poly", "verts": [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)],
                              "n": (0.0, 0.0, 1.0), "color": COLOR_SNOW_A if (xi + yi) % 2 == 0 else COLOR_SNOW_B,
                              "tag": ("ground", None), "layer": 0, "outline": False})

        # B. Wooden perimeter fence (Box2D fixtures, extruded to FENCE_HEIGHT, 1.5 m panels)
        for b in base.fences:
            for f in b.fixtures:
                verts = [tuple(b.transform * v) for v in f.shape.vertices]
                self._prism(prims, verts, 0.0, FENCE_HEIGHT, COLOR_FENCE, ("fence", None), panel=1.5)

        # C. Cover: snow forts (1.0 m) and ice pillars (2.5 m), 1 m panels
        for b in base.cover_bodies:
            kind, idx = b.userData["type"], b.userData["cover_index"]
            color = COLOR_FORT if kind == "fort" else COLOR_PILLAR
            anchor = (float(b.position.x), float(b.position.y), 0.5 * COVER_HEIGHT[kind])
            for f in b.fixtures:
                verts = [tuple(b.transform * v) for v in f.shape.vertices]
                self._prism(prims, verts, 0.0, COVER_HEIGHT[kind], color, (kind, idx), panel=1.0, anchor=anchor)

        # D. Characters: coat cylinder, head, beanie and a scarf/arm stub pointing along the facing
        for body, facing, coat, hat, scarf, ctype in (
                (base.player, base.player_facing, COLOR_PLAYER, COLOR_PLAYER_HAT, COLOR_PLAYER_SCARF, "player"),
                (base.opponent, base.opp_facing, COLOR_OPP, COLOR_OPP_HAT, COLOR_OPP_SCARF, "opponent")):
            flash = base.hit_flash.get(ctype, 0)
            if flash > 0:
                coat = _mix_color(coat, COLOR_HIT_FLASH, 0.7 * flash / HIT_FLASH_STEPS)
            cx, cy = float(body.position.x), float(body.position.y)
            tag = (ctype, None)
            anchor = (cx, cy, 0.5 * Z_HAT_TOP)

            def ring(r):
                return [(cx + r * math.cos(facing + (i + 0.5) * math.pi / 4),
                         cy + r * math.sin(facing + (i + 0.5) * math.pi / 4)) for i in range(8)]
            self._prism(prims, ring(CHAR_RADIUS), 0.0, Z_BODY_TOP, coat, tag, anchor=anchor)
            self._prism(prims, ring(HEAD_RADIUS), Z_BODY_TOP, Z_HEAD_TOP, COLOR_SKIN, tag, anchor=anchor)
            self._prism(prims, ring(HAT_RADIUS), Z_HEAD_TOP, Z_HAT_TOP, hat, tag, anchor=anchor)
            c, s = math.cos(facing), math.sin(facing)
            arm = [(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in
                   ((ARM_SPAN[0], -ARM_HALF_W), (ARM_SPAN[1], -ARM_HALF_W),
                    (ARM_SPAN[1], ARM_HALF_W), (ARM_SPAN[0], ARM_HALF_W))]
            self._prism(prims, arm, Z_ARM[0], Z_ARM[1], scarf, tag, anchor=anchor)

        # E. Snowballs (billboards)
        for b in base.snowballs:
            pos = b["body"].position
            prims.append({"kind": "disk", "c": (float(pos.x), float(pos.y), Z_SNOWBALL), "r": SNOWBALL_RADIUS,
                          "color": COLOR_SNOWBALL_PLAYER if b["owner"] == "player" else COLOR_SNOWBALL_OPP,
                          "tag": ("snowball", None), "layer": 1, "outline": True})

        # F. Snow splats (visual only; occlude like everything else but are never a mask target)
        for e in base.effects:
            frac = e["age"] / float(e["life"])
            r0, r1 = e["radius"]
            r = r0 + (r1 - r0) * (1.0 - (1.0 - frac) ** 2)
            prims.append({"kind": "disk", "c": (e["x"], e["y"], e["z"]), "r": r,
                          "color": _mix_color((255, 255, 255), COLOR_PUFF, frac), "tag": ("effect", None),
                          "layer": 1, "outline": True})
        return prims

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def _view_defs(self):
        """Per-view transform world -> (a, b, depth), screen mapping, near cut and culling rule."""
        base = self.env.unwrapped
        VW, VH = self.VIEWPORT_W, self.VIEWPORT_H
        cx, cy = VW / 2.0, VH / 2.0
        px, py = float(base.player.position.x), float(base.player.position.y)
        h = base.player_facing
        fx, fy = math.cos(h), math.sin(h)     # facing
        rx, ry = math.sin(h), -math.cos(h)    # player's right
        ST, SE, GY = self.top_scale, self.elev_scale, self.ground_y
        f = self.fov
        # Top-down scrolls with the player, clamped so the whole arena always stays on screen
        hx, hy = VW / 2.0 / ST, VH / 2.0 / ST
        tx = min(max(px, ARENA_W - hx), hx) if 2 * hx >= ARENA_W else px
        ty = min(max(py, ARENA_H - hy), hy) if 2 * hy >= ARENA_H else py
        anc_x, anc_y, anc_z = self._cam_anchor
        yaw, pitch, pivot = self.cam_yaw, self.cam_pitch, self.pivot_depth

        def fpv_tf(v):
            dx, dy, dz = v[0] - anc_x, v[1] - anc_y, v[2] - anc_z
            d_c, x_c, z_c = orbit_camera_frame(dx * fx + dy * fy, dx * rx + dy * ry, dz, yaw, pitch, pivot)
            return (x_c, z_c, d_c)

        return {
            "top": {"tf": lambda v: (v[0] - tx, v[1] - ty, -v[2]), "near": None, "cull": (0.0, 0.0, -1.0),
                    "scr": lambda a, b, d: (cx + a * ST, cy - b * ST), "rad": lambda r, d: r * ST},
            "rear": {"tf": lambda v: ((v[0] - px) * rx + (v[1] - py) * ry, v[2], (v[0] - px) * fx + (v[1] - py) * fy),
                     "near": -self.REAR_CUT, "cull": (fx, fy, 0.0),
                     "scr": lambda a, b, d: (cx + a * SE, GY - b * SE), "rad": lambda r, d: r * SE},
            "side": {"tf": lambda v: ((v[0] - px) * fx + (v[1] - py) * fy, v[2], -((v[0] - px) * rx + (v[1] - py) * ry)),
                     "near": -self.REAR_CUT, "cull": (-rx, -ry, 0.0),
                     "scr": lambda a, b, d: (cx + a * SE, GY - b * SE), "rad": lambda r, d: r * SE},
            "fpv": {"tf": fpv_tf, "near": self.Z_NEAR, "cull": None, "cam": self.cam_pos,
                    "scr": lambda a, b, d: (cx + a / d * f, cy - b / d * f), "rad": lambda r, d: r / d * f},
        }

    @staticmethod
    def _clip_depth(tv, near):
        """Sutherland-Hodgman clip of view-space vertices (a, b, depth) against depth >= near."""
        out = []
        for i in range(len(tv)):
            cur, prev = tv[i], tv[i - 1]
            ci, pi_ = cur[2] >= near, prev[2] >= near
            if ci != pi_:
                t = (near - prev[2]) / (cur[2] - prev[2])
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1]), near))
            if ci:
                out.append(cur)
        return out

    def _clip_screen(self, pts):
        """Sutherland-Hodgman clip of screen points to the viewport (keeps pygame's int16 coords in range)."""
        W, H = self.VIEWPORT_W, self.VIEWPORT_H
        edges = ((0, -1.0, True), (0, W + 1.0, False), (1, -1.0, True), (1, H + 1.0, False))
        poly = list(pts)
        for axis, lim, lower in edges:
            out = []
            for i in range(len(poly)):
                cur, prev = poly[i], poly[i - 1]
                ci = cur[axis] >= lim if lower else cur[axis] <= lim
                pi_ = prev[axis] >= lim if lower else prev[axis] <= lim
                if ci != pi_:
                    t = (lim - prev[axis]) / (cur[axis] - prev[axis])
                    q = [prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])]
                    q[axis] = lim
                    out.append((q[0], q[1]))
                if ci:
                    out.append(cur)
            poly = out
            if len(poly) < 3:
                return []
        return poly

    def _project(self, prims, view):
        """Project, cull and clip all primitives for one view; returns draw list sorted back to front."""
        tf, near, scr, cull = view["tf"], view["near"], view["scr"], view["cull"]
        cam = view.get("cam")
        out = []
        for idx, pr in enumerate(prims):
            if pr["kind"] == "poly":
                n = pr["n"]
                if cull is not None:
                    if n[0] * cull[0] + n[1] * cull[1] + n[2] * cull[2] > -1e-6:
                        continue
                else:
                    vs = pr["verts"]
                    k = 1.0 / len(vs)
                    mx = sum(v[0] for v in vs) * k - cam[0]
                    my = sum(v[1] for v in vs) * k - cam[1]
                    mz = sum(v[2] for v in vs) * k - cam[2]
                    if n[0] * mx + n[1] * my + n[2] * mz >= 0.0:
                        continue
                if near is not None and cull is not None and tf(pr["anchor"])[2] < near:
                    continue  # elevation cutaway: drop whole objects behind the cut, never slice them open
                tv = [tf(v) for v in pr["verts"]]
                if near is not None and cull is None:
                    tv = self._clip_depth(tv, near)   # FPV near plane
                    if len(tv) < 3:
                        continue
                pts = self._clip_screen([scr(a, b, d) for a, b, d in tv])
                if len(pts) < 3:
                    continue
                depth = sum(t[2] for t in tv) / len(tv)
            else:
                a, b, d = tf(pr["c"])
                if near is not None and d < near + (pr["r"] if cull is None else 0.0):
                    continue
                sx, sy = scr(a, b, d)
                rp = max(1.0, view["rad"](pr["r"], d))
                pts = self._clip_screen([(sx + rp * math.cos(t), sy + rp * math.sin(t))
                                         for t in np.linspace(0.0, 2.0 * math.pi, 14, endpoint=False)])
                if len(pts) < 3:
                    continue
                depth = d
            key = (pr["layer"], idx if pr["layer"] == 0 else -depth)
            out.append((key, pts, pr))
        out.sort(key=lambda t: t[0])
        return out

    @staticmethod
    def _mask_match(tag, target):
        ttype, tidx = tag
        if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
            return tidx is not None and tidx == int(target)
        target = str(target).lower()
        if target == "cover":
            return ttype in COVER_TYPES
        return ttype == target

    def _render_views(self, mask_target=None):
        if not pygame.get_init():
            pygame.init()
        base = self.env.unwrapped
        VW, VH = self.VIEWPORT_W, self.VIEWPORT_H
        if getattr(base, "player", None) is None:
            shape = (VH, VW) if mask_target is not None else (VH, VW, 3)
            return tuple(np.zeros(shape, np.uint8) for _ in range(4))

        prims = self._scene_prims()
        views = self._view_defs()
        out = []
        for name in ("top", "rear", "side", "fpv"):
            surf = pygame.Surface((VW, VH))
            if mask_target is not None:
                surf.fill((0, 0, 0))
            elif name == "top":
                surf.fill(COLOR_SNOW_OUT)
            else:
                surf.fill(COLOR_SKY)
                if name != "fpv":
                    # Elevation views: the ground plane is edge-on; fill below the z = 0 line with snow
                    pygame.draw.rect(surf, COLOR_SNOW_OUT, (0, self.ground_y, VW, VH - self.ground_y))
                    pygame.draw.line(surf, (200, 210, 225), (0, self.ground_y), (VW, self.ground_y), 2)
            for _, pts, pr in self._project(prims, views[name]):
                ipts = [(int(round(x)), int(round(y))) for x, y in pts]
                if mask_target is not None:
                    col = (255, 255, 255) if self._mask_match(pr["tag"], mask_target) else (0, 0, 0)
                    pygame.draw.polygon(surf, col, ipts)
                else:
                    pygame.draw.polygon(surf, pr["color"], ipts)
                    if pr["outline"]:
                        pygame.draw.polygon(surf, tuple(int(c * 0.72) for c in pr["color"]), ipts, 1)
            img = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
            out.append((img[..., 0] > 127).astype(np.uint8) if mask_target is not None else img)
        return tuple(out)

    # ------------------------------------------------------------------
    def render(self, camera_pose="Center"):
        """Returns tuple of 4 views: (top_down, rear, side, fpv), each H x W x 3 uint8."""
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)
        # Get dynamically calculated position from our property
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        return self._render_views()

    def render_masks(self, target):
        """Visible binary masks (values 0/1, uint8, H x W) of `target` in (top, rear, side, fpv).
        target: 'opponent' | 'player' | 'cover' | 'pillar' | 'fort' | 'snowball' | int cover index.
        The full scene is drawn with the same projection / painter ordering as render(), target
        primitives in white and everything else in black, so occluded parts are excluded.
        Does not advance the camera pose or change any state."""
        return self._render_views(mask_target=target)


# Main test
if __name__ == "__main__":
    pygame.init()

    env = SnowballFightEnv()
    wrapper = SnowballFightOrthographicWrapper(env)
    obs, info = wrapper.reset(seed=0)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print("Controls: W/S forward/back along the facing, A/D strafe, Q/E turn, SPACE throw, arrows orbit FPV camera, "
          "C toggle cover (next reset), R reset, ESC quit")

    half_w, half_h = wrapper.VIEWPORT_W // 2, wrapper.VIEWPORT_H // 2
    display = pygame.display.set_mode((half_w * 2, half_h * 2))
    pygame.display.set_caption("Snowball Fight - Orthographic Views")
    clock = pygame.time.Clock()

    running, total_reward = True, 0.0
    while running:
        reset = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                env.cover_enabled = not env.cover_enabled
                print(f"cover = {env.cover_enabled} (applies on reset)")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset = True
        keys = pygame.key.get_pressed()
        action = np.array([
            float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
            float(keys[pygame.K_w]) - float(keys[pygame.K_s]),
            float(keys[pygame.K_q]) - float(keys[pygame.K_e]),
            1.0 if keys[pygame.K_SPACE] else -1.0,
        ], dtype=np.float32)
        pose = "Center"
        for k, p in ((pygame.K_LEFT, "Left"), (pygame.K_RIGHT, "Right"), (pygame.K_UP, "Up"), (pygame.K_DOWN, "Down")):
            if keys[k]:
                pose = p

        obs, reward, terminated, truncated, info = wrapper.step(action)
        total_reward += reward
        views = wrapper.render(camera_pose=pose)
        for i, im in enumerate(views):
            s = pygame.transform.smoothscale(pygame.surfarray.make_surface(im.transpose(1, 0, 2)), (half_w, half_h))
            display.blit(s, ((i % 2) * half_w, (i // 2) * half_h))
        pygame.display.flip()
        clock.tick(FPS)

        if terminated or truncated or reset:
            print(f"Episode ended at step {env.steps}, total reward: {total_reward:.2f}, "
                  f"health {info['health']:.0f} vs {info['opponent_health']:.0f}")
            obs, info = wrapper.reset()
            total_reward = 0.0

    env.close()
    pygame.quit()
