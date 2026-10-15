import math
import numpy as np
import pygame
from pygame import gfxdraw
from .camera_utils import orbit_camera_frame, orbit_camera_offset

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from Box2D.b2 import (world, polygonShape)

FPS = 50
SCALE = 30.0

# Arena dimensions in physics units (metres)
ARENA_W = 25.0   # X: 0 → 25
ARENA_H = 20.0   # Y: 0 → 20  (top-down, no gravity)
WALL_T = 0.5     # Wall thickness (metres); interior is [WALL_T, ARENA_W - WALL_T] x [WALL_T, ARENA_H - WALL_T]
DRONE_ALT = 5.0  # Fixed 3D altitude for all drones (metres, used only in renderer)
WALL_HEIGHT = 3.0  # Visual height of 3D arena walls in renderer

# Combat tuning
PLAYER_HP = 5            # hits the player can take
OPP_HP = 3               # hits needed to down the opponent
HIT_RADIUS = 0.75        # bullet-vs-drone hit radius (metres)
BULLET_SPEED = 35.0      # m/s (0.7 m per step)
BULLET_LIFE = 30         # steps (~21 m range)
PLAYER_FIRE_CD = 8       # steps between player shots
OPP_FIRE_CD = 24         # steps between opponent shots
PLAYER_TURN = 0.12       # max heading slew (rad/step) of the player's gun/nose
OPP_TURN = 0.08          # max heading slew (rad/step) of the opponent
OPP_AIM_ERR = 0.08       # std of opponent aim error (rad)
PLAYER_MAX_V = 12.0      # speed caps (m/s)
OPP_MAX_V = 8.0
OPP_RESPAWN = 40         # steps the opponent stays down after a kill
ENGAGE_R = 7.0           # preferred engagement range for potential shaping (metres)
SHAPING_K = 0.05         # potential = -SHAPING_K * |dist - ENGAGE_R|


def _wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ==============================================================================
# 1. DRONE DOGFIGHT BASE ENVIRONMENT (Top-Down, No Gravity)
# ==============================================================================
class DroneDogfightEnv(gym.Env):
    """
    Drone Dogfight Task:
    Outmaneuver and shoot down an opposing quadrotor in a closed top-down arena.
    Drones move freely in the X-Y horizontal plane (no Z / altitude DOF). Each drone has a
    nose/gun heading that slews toward the commanded aim at a limited turn rate; bolts fly
    along the heading. The opponent wanders the arena, turns toward the player and fires
    back with some aim error. Downed opponents respawn after OPP_RESPAWN steps elsewhere.
    The episode terminates when the player is shot down and truncates after max_steps.

    Actions (4D continuous, all in [-1, 1]):
      0: East-West Thrust   — +1 = full thrust east (+X), -1 = full thrust west
      1: North-South Thrust — +1 = full thrust north (+Y), -1 = full thrust south
      2: Aim Heading        — desired gun heading = a * 180° (0 = east, +0.5 = north,
                              -0.5 = south, ±1 = west); the nose turns toward it at
                              PLAYER_TURN rad/step
      3: Fire Weapon        — > 0.5 fires a bolt along the current heading when the gun is
                              ready (cooldown PLAYER_FIRE_CD steps)

    Observation (25D, metres / m/s / radians unless noted):
      0-3:   Player X, Y, Vx, Vy
      4-5:   Player heading cos, sin
      6:     Player HP fraction
      7:     Player gun cooldown fraction (0 = ready to fire)
      8:     Player alive flag
      9-10:  Opponent relative position dX, dY (opponent - player)
      11-12: Opponent Vx, Vy
      13-14: Opponent heading cos, sin
      15:    Opponent HP fraction
      16:    Opponent alive flag
      17:    Opponent respawn timer fraction (0 when alive)
      18:    Distance to opponent
      19:    Signed bearing of opponent relative to player heading (rad, + = to the left)
      20-23: Nearest incoming opponent bolt: relative dX, dY, Vx, Vy (zeros if none)
      24:    Incoming bolt present flag

    Reward:
      +1 per hit on the opponent, +5 extra when the hit downs it;
      -1 per hit taken, -5 extra when the player is downed (episode terminates);
      plus potential-based range shaping  Phi(s') - Phi(s),  Phi = -0.05 * |dist - 7 m|
      (only while both drones are alive, so it telescopes and cannot be farmed).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        # Zero gravity — top-down horizontal arena
        self.world = world(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self.bullets = []
        self.opp_bullets = []
        self.boundaries = []
        self.max_steps = 600

    # ------------------------------------------------------------------
    def _make_drone(self, pos, angle=0.0):
        """Create a quadrotor body with canopy + 4 diagonal arms. Physics rotation locked;
        the heading is set kinematically each step (nose = local +X)."""
        body = self.world.CreateDynamicBody(
            position=pos, angle=angle,
            linearDamping=1.2, angularDamping=5.0,
            fixedRotation=True   # heading is driven by the aim controller, not by physics
        )
        # Main canopy (long axis along the heading)
        body.CreatePolygonFixture(box=(0.6, 0.25), density=1.5, friction=0.3)
        # 4 diagonal rotor arms (at 45° to the heading)
        body.CreatePolygonFixture(box=(0.35, 0.06, ( 0.45,  0.30),  0.785), density=0.2)
        body.CreatePolygonFixture(box=(0.35, 0.06, (-0.45,  0.30), -0.785), density=0.2)
        body.CreatePolygonFixture(box=(0.35, 0.06, ( 0.45, -0.30), -0.785), density=0.2)
        body.CreatePolygonFixture(box=(0.35, 0.06, (-0.45, -0.30),  0.785), density=0.2)
        return body

    def _fire(self, shooter, heading, bullet_list):
        """Spawn a bolt just ahead of the shooter's nose. Bolts are sensors that collide with
        nothing physically; hits / wall impacts are resolved geometrically in step()."""
        dx, dy = math.cos(heading), math.sin(heading)
        pos = (float(shooter.position.x) + 1.0 * dx, float(shooter.position.y) + 1.0 * dy)
        body = self.world.CreateDynamicBody(position=pos, angle=heading, linearDamping=0.0)
        body.CreateCircleFixture(radius=0.12, density=0.05, isSensor=True, maskBits=0)
        body.linearVelocity = (BULLET_SPEED * dx, BULLET_SPEED * dy)
        bullet_list.append({'body': body, 'life': BULLET_LIFE})

    @staticmethod
    def _slew(cur, target, rate):
        return _wrap(cur + float(np.clip(_wrap(target - cur), -rate, rate)))

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for body in self.bodies:
            self.world.DestroyBody(body)
        for b in self.bullets + self.opp_bullets:
            self.world.DestroyBody(b['body'])
        self.bodies = []
        self.bullets = []
        self.opp_bullets = []
        self.boundaries = []

        self.step_count = 0
        rng = self.np_random

        # Static arena walls (thin boxes on all 4 sides)
        h = WALL_T / 2
        wall_defs = [
            # (center_x, center_y, half_w, half_h)
            (ARENA_W / 2,       h,           ARENA_W / 2, h),            # South wall
            (ARENA_W / 2,       ARENA_H - h, ARENA_W / 2, h),            # North wall
            (h,                 ARENA_H / 2, h,           ARENA_H / 2),  # West wall
            (ARENA_W - h,       ARENA_H / 2, h,           ARENA_H / 2),  # East wall
        ]
        for cx, cy, hw, hh in wall_defs:
            b = self.world.CreateStaticBody(
                position=(cx, cy),
                shapes=polygonShape(box=(hw, hh))
            )
            self.bodies.append(b)
            self.boundaries.append(b)

        # Player drone — west side facing east; opponent — east side facing west
        p_y = ARENA_H / 2 + float(rng.uniform(-4.0, 4.0))
        o_y = ARENA_H / 2 + float(rng.uniform(-5.0, 5.0))
        self.player_heading = 0.0
        self.player = self._make_drone((4.0, p_y), self.player_heading)
        self.bodies.append(self.player)
        self.player_alive = True
        self.player_hp    = PLAYER_HP
        self.player_cd    = 0

        self.opp_heading = math.pi
        self.opponent = self._make_drone((ARENA_W - 4.0, o_y), self.opp_heading)
        self.bodies.append(self.opponent)
        self.opponent_alive = True
        self.opponent_hp    = OPP_HP
        self.opp_cd         = int(rng.integers(10, OPP_FIRE_CD + 10))
        self.opp_respawn_timer = 0
        # Random phases of the opponent's Lissajous patrol path
        self.opp_phase = rng.uniform(0.0, 2.0 * math.pi, size=2)

        # Explosion state per drone {'pos','frames','radius'}
        self.player_explosion = None
        self.opp_explosion    = None

        self.score  = 0   # player kills
        self.deaths = 0   # player deaths
        self.hits = 0
        self.hits_taken = 0
        self._prev_phi = self._potential()

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _potential(self):
        if not (self.player_alive and self.opponent_alive):
            return None
        d = (self.opponent.position - self.player.position).length
        return -SHAPING_K * abs(d - ENGAGE_R)

    def _trigger_explosion(self, pos, owner, radius=1.0):
        """Trigger an orange explosion at pos for the given owner ('player'/'opp')."""
        ex = {'pos': pos, 'frames': 14, 'radius': radius}
        if owner == 'player':
            self.player_explosion = ex
        else:
            self.opp_explosion = ex

    def _respawn_opponent(self):
        """Re-activate the (kept) opponent body at a random spot away from the player."""
        rng = self.np_random
        px, py = float(self.player.position.x), float(self.player.position.y)
        for _ in range(20):
            sx = float(rng.uniform(3.0, ARENA_W - 3.0))
            sy = float(rng.uniform(3.0, ARENA_H - 3.0))
            if math.hypot(sx - px, sy - py) >= 9.0:
                break
        self.opp_heading = math.atan2(py - sy, px - sx)
        self.opponent.position = (sx, sy)
        self.opponent.angle = self.opp_heading
        self.opponent.linearVelocity = (0.0, 0.0)
        self.opponent.active = True
        self.opponent.awake = True
        self.opponent_alive = True
        self.opponent_hp    = OPP_HP
        self.opp_cd         = OPP_FIRE_CD
        self.opp_phase = rng.uniform(0.0, 2.0 * math.pi, size=2)

    @staticmethod
    def _disable(body):
        """Take a downed drone out of the simulation without destroying it (keeps every
        reference, e.g. the wrapper's camera, pointing at a valid body)."""
        body.linearVelocity = (0.0, 0.0)
        body.active = False

    @staticmethod
    def _seg_hits(p0, p1, c, r):
        """True if segment p0→p1 passes within r of point c."""
        d = p1 - p0
        dd = float(d[0] * d[0] + d[1] * d[1])
        t = 0.0 if dd < 1e-9 else float(np.clip(np.dot(c - p0, d) / dd, 0.0, 1.0))
        q = p0 + t * d
        return float(np.hypot(*(q - c))) < r

    def _resolve_bullets(self, bullet_list, prev, target, target_alive):
        """Advance-and-collide bolts: destroy on hit, wall impact or expiry. Returns (#hits, last hit pos)."""
        hits, hit_pos, alive = 0, None, []
        c = np.array([target.position.x, target.position.y]) if target_alive else None
        for b, p0 in zip(bullet_list, prev):
            b['life'] -= 1
            p1 = np.array([b['body'].position.x, b['body'].position.y])
            if c is not None and self._seg_hits(p0, p1, c, HIT_RADIUS):
                hits += 1
                hit_pos = (float(p1[0]), float(p1[1]))
                self.world.DestroyBody(b['body'])
                continue
            if (b['life'] <= 0 or p1[0] <= WALL_T or p1[0] >= ARENA_W - WALL_T
                    or p1[1] <= WALL_T or p1[1] >= ARENA_H - WALL_T):
                self.world.DestroyBody(b['body'])
                continue
            alive.append(b)
        return alive, hits, hit_pos

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.step_count += 1
        reward = 0.0
        terminated = False
        rng = self.np_random

        # ---- Decay explosions & countdown timers ----
        for attr in ('player_explosion', 'opp_explosion'):
            ex = getattr(self, attr)
            if ex is not None:
                ex['frames'] -= 1
                if ex['frames'] <= 0:
                    setattr(self, attr, None)
        self.player_cd = max(0, self.player_cd - 1)
        self.opp_cd    = max(0, self.opp_cd - 1)

        if not self.opponent_alive and self.player_alive:
            self.opp_respawn_timer -= 1
            if self.opp_respawn_timer <= 0:
                self._respawn_opponent()
                self._prev_phi = self._potential()

        # ---- Player control: world-axis thrust, slewed aim heading, gun ----
        if self.player_alive:
            self.player.ApplyForceToCenter((float(action[0]) * 25.0, float(action[1]) * 25.0), wake=True)
            self.player_heading = self._slew(self.player_heading, float(action[2]) * math.pi, PLAYER_TURN)
            self.player.angle = self.player_heading
            if float(action[3]) > 0.5 and self.player_cd == 0:
                self._fire(self.player, self.player_heading, self.bullets)
                self.player_cd = PLAYER_FIRE_CD

        # ---- Opponent AI: damped patrol along a Lissajous path, turn to player, fire ----
        if self.opponent_alive:
            opp_t = self.step_count * 0.04
            target_x = ARENA_W / 2 + 7.0 * math.cos(0.9 * opp_t + self.opp_phase[0])
            target_y = ARENA_H / 2 + 6.0 * math.sin(opp_t + self.opp_phase[1])
            o_pos, o_vel = self.opponent.position, self.opponent.linearVelocity
            fx = 10.0 * (target_x - o_pos.x) - 4.0 * o_vel.x
            fy = 10.0 * (target_y - o_pos.y) - 4.0 * o_vel.y
            self.opponent.ApplyForceToCenter((float(fx), float(fy)), wake=True)

            if self.player_alive:
                rel = self.player.position - o_pos
                want = math.atan2(rel.y, rel.x)
                self.opp_heading = self._slew(self.opp_heading, want, OPP_TURN)
                self.opponent.angle = self.opp_heading
                if (self.opp_cd == 0 and abs(_wrap(want - self.opp_heading)) < 0.25
                        and rel.length < 16.0):
                    aim = self.opp_heading + float(rng.normal(0.0, OPP_AIM_ERR))
                    self._fire(self.opponent, aim, self.opp_bullets)
                    self.opp_cd = OPP_FIRE_CD

        # ---- Physics ----
        prev_p = [np.array([b['body'].position.x, b['body'].position.y]) for b in self.bullets]
        prev_o = [np.array([b['body'].position.x, b['body'].position.y]) for b in self.opp_bullets]
        self.world.Step(1.0 / FPS, 6, 2)
        for body, vmax in ((self.player, PLAYER_MAX_V), (self.opponent, OPP_MAX_V)):
            v = body.linearVelocity
            if v.length > vmax:
                body.linearVelocity = v * (vmax / v.length)

        # ---- Player bolts vs opponent ----
        self.bullets, n_hit, hit_pos = self._resolve_bullets(
            self.bullets, prev_p, self.opponent, self.opponent_alive)
        if n_hit:
            self.hits += n_hit
            reward += 1.0 * n_hit
            self.opponent_hp -= n_hit
            self._trigger_explosion(hit_pos, 'opp', 0.8)
            if self.opponent_hp <= 0:
                # Opponent downed — big explosion, deactivate body, schedule respawn
                self._trigger_explosion((float(self.opponent.position.x), float(self.opponent.position.y)), 'opp', 2.0)
                self._disable(self.opponent)
                self.opponent_alive    = False
                self.opponent_hp       = 0
                self.opp_respawn_timer = OPP_RESPAWN
                reward += 5.0
                self.score += 1

        # ---- Opponent bolts vs player ----
        self.opp_bullets, n_hit, hit_pos = self._resolve_bullets(
            self.opp_bullets, prev_o, self.player, self.player_alive)
        if n_hit:
            self.hits_taken += n_hit
            reward -= 1.0 * n_hit
            self.player_hp -= n_hit
            self._trigger_explosion(hit_pos, 'player', 0.8)
            if self.player_hp <= 0:
                # Player downed — episode ends (body kept, just deactivated)
                self._trigger_explosion((float(self.player.position.x), float(self.player.position.y)), 'player', 2.0)
                self._disable(self.player)
                self.player_alive = False
                self.player_hp    = 0
                reward -= 5.0
                self.deaths += 1
                terminated = True

        # ---- Potential-based engagement-range shaping (both alive only) ----
        phi = self._potential()
        if phi is not None and self._prev_phi is not None:
            reward += phi - self._prev_phi
        self._prev_phi = phi

        truncated = (not terminated) and self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, \
               {'score': self.score, 'deaths': self.deaths, 'hits': self.hits,
                'hits_taken': self.hits_taken,
                'player_hp': self.player_hp, 'opp_hp': self.opponent_hp}

    # ------------------------------------------------------------------
    def _get_obs(self):
        p, o = self.player, self.opponent
        px, py = float(p.position.x), float(p.position.y)
        pvx, pvy = float(p.linearVelocity.x), float(p.linearVelocity.y)
        dx, dy = float(o.position.x) - px, float(o.position.y) - py
        ovx, ovy = float(o.linearVelocity.x), float(o.linearVelocity.y)
        dist = math.hypot(dx, dy)
        bearing = _wrap(math.atan2(dy, dx) - self.player_heading)

        # Nearest incoming opponent bolt (closing on the player)
        bul = [0.0, 0.0, 0.0, 0.0, 0.0]
        best = np.inf
        for b in self.opp_bullets:
            bx, by = float(b['body'].position.x) - px, float(b['body'].position.y) - py
            bvx, bvy = float(b['body'].linearVelocity.x), float(b['body'].linearVelocity.y)
            d = math.hypot(bx, by)
            if bx * bvx + by * bvy < 0.0 and d < best:
                best = d
                bul = [bx, by, bvx, bvy, 1.0]

        return np.array([
            px, py, pvx, pvy,
            math.cos(self.player_heading), math.sin(self.player_heading),
            self.player_hp / PLAYER_HP, self.player_cd / PLAYER_FIRE_CD, float(self.player_alive),
            dx, dy, ovx, ovy,
            math.cos(self.opp_heading), math.sin(self.opp_heading),
            self.opponent_hp / OPP_HP, float(self.opponent_alive),
            self.opp_respawn_timer / OPP_RESPAWN if not self.opponent_alive else 0.0,
            dist, bearing,
            *bul,
        ], dtype=np.float32)

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

        self.screen.fill((60, 120, 60))

        for b in self.boundaries:
            for f in b.fixtures:
                verts = [(int((b.transform * v)[0] * SCALE), int(sh - (b.transform * v)[1] * SCALE))
                         for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, (100, 110, 130), verts)

        # Player bullets
        for b in self.bullets:
            pos = b['body'].position
            pygame.draw.circle(self.screen, (80, 200, 255),
                               (int(pos.x * SCALE), int(sh - pos.y * SCALE)), 4)
        # Opponent bullets
        for b in self.opp_bullets:
            pos = b['body'].position
            pygame.draw.circle(self.screen, (255, 80, 80),
                               (int(pos.x * SCALE), int(sh - pos.y * SCALE)), 4)

        # Draw alive drones
        for body, alive, color in ((self.player, self.player_alive, (0, 220, 255)),
                                   (self.opponent, self.opponent_alive, (255, 60, 60))):
            if not alive:
                continue
            for f in body.fixtures:
                verts = [(int((body.transform * v)[0] * SCALE),
                          int(sh - (body.transform * v)[1] * SCALE))
                         for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, color, verts)

        # Draw explosions (2D)
        for ex in [self.player_explosion, self.opp_explosion]:
            if ex is not None:
                t = max(0.0, ex['frames'] / 14.0)
                ec = (255, int(140 * t), 0)
                ex_sc = (int(ex['pos'][0] * SCALE), int(sh - ex['pos'][1] * SCALE))
                pygame.draw.circle(self.screen, ec, ex_sc, int(ex['radius'] * SCALE * (0.4 + 0.6*t)))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


# ==============================================================================
# 2. 3D ORTHOGRAPHIC MULTI-VIEW WRAPPER FOR DRONE DOGFIGHT
# ==============================================================================
class DroneDogfightOrthographicWrapper(gym.Wrapper):
    """
    Extrudes the 2D top-down physics into a 3D volumetric scene.

    Coordinate mapping:
      physics X  → 3D X (East-West)
      physics Y  → 3D Y (North-South)   [top-down horizontal plane]
      extrusion  → 3D Z (Up/Down altitude, fixed for all drones)

    Views produced:
      Top-Down  : (X, Y) plan view  — bird's eye over the arena
      Rear      : (Y, Z) elevation  — looking from the West wall
      Side      : (X, Z) elevation  — looking from the South wall
      FPV       : true 3D third-person view; orbit camera trailing behind and above the
                  player drone, looking along heading in X-Y plane (drone fully visible)

    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """

    DRONE_Z_LO = (DRONE_ALT - 0.6)   # bottom extrusion (metres)
    DRONE_Z_HI = (DRONE_ALT + 0.6)   # top extrusion   (metres)

    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.SCALE = SCALE
        self.VIEWPORT_W = 800
        self.VIEWPORT_H = 600
        self.fov = 400.0

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state

        # Follow/Orbit camera baseline offsets (relative to player drone, in pixel units)
        # The camera trails the drone's gun heading (body angle, driven by action dim 2). The env
        # never destroys the player body mid-episode (downed drones are only deactivated), so the
        # player pose read here is always valid memory.
        self.camera_dist = 3.5 * SCALE    # Distance behind the drone along its heading
        self.camera_height = 1.0 * SCALE  # Vertical offset above the drone

        # Orbit focus: the player drone (camera_dist ahead along the optical axis)
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
        """Un-orbited follow-camera anchor (behind and above the drone) in world space; the projection is relative to it."""
        base = self.env.unwrapped
        if not hasattr(base, 'player') or base.player is None:
            return (0.0, 0.0, 0.0)

        # Camera — trailing camera_dist behind the drone, camera_height above it,
        # looking along heading (horizontal, in X-Y plane)
        S = self.SCALE
        p_angle = float(base.player.angle)
        p_x = base.player.position.x * S
        p_y = base.player.position.y * S
        p_z = DRONE_ALT * S    # fixed altitude

        cam_x = p_x - self.camera_dist * math.cos(p_angle)
        cam_y = p_y - self.camera_dist * math.sin(p_angle)
        cam_z = p_z + self.camera_height

        return (cam_x, cam_y, cam_z)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the orbited camera in world space."""
        base = self.env.unwrapped
        if not hasattr(base, 'player') or base.player is None:
            return (0.0, 0.0, 0.0)

        ax, ay, az = self._cam_anchor
        p_angle = float(base.player.angle)

        # Camera-frame (depth, lateral, vertical) offset -> world (inverse of render's mapping:
        # depth = dx*cos + dy*sin, lateral = dx*sin - dy*cos, vertical = dz)
        d_off, lat_off, vert_off = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)
        cam_x = ax + d_off * math.cos(p_angle) + lat_off * math.sin(p_angle)
        cam_y = ay + d_off * math.sin(p_angle) - lat_off * math.cos(p_angle)
        cam_z = az + vert_off

        return (cam_x, cam_y, cam_z)

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
    def _surf_to_img(self, surf):
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    # ------------------------------------------------------------------
    def render(self, camera_pose="Center"):
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)

        base = self.env.unwrapped

        # Initialise surfaces
        top_surf  = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        rear_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        side_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        fpv_surf  = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))

        SKY   = (135, 206, 235)   # light blue sky
        GRASS = (75,  145,  75)   # green ground

        # Sky fills all views
        for s in [top_surf, rear_surf, side_surf]:
            s.fill(SKY)

        # FPV: draw sky/ground horizon split instead of flat fill
        # Upper half = sky, lower half = grass, sharp horizon at mid-height
        fpv_surf.fill(SKY)
        pygame.draw.rect(fpv_surf, GRASS, (0, self.VIEWPORT_H // 2, self.VIEWPORT_W, self.VIEWPORT_H // 2))
        # Thin bright horizon line
        pygame.draw.line(fpv_surf, (200, 230, 200), (0, self.VIEWPORT_H // 2), (self.VIEWPORT_W, self.VIEWPORT_H // 2), 2)

        if not hasattr(base, 'player'):
            return [np.zeros((self.VIEWPORT_H, self.VIEWPORT_W, 3), np.uint8)] * 4

        S = self.SCALE
        VW, VH = self.VIEWPORT_W, self.VIEWPORT_H
        cx, cy = VW / 2.0, VH / 2.0

        polys_3d = []  # list of (vertices_3d, color, sort_layer [, is_player])

        # ----------------------------------------------------------------
        # A. Green Grass Floor — flat quad at Z = 0 across full arena
        #    Tagged no_fpv=True so they are skipped in the FPV projection
        # ----------------------------------------------------------------
        tile = 2.0  # metres per tile
        x0s = np.arange(0.0, ARENA_W, tile)
        y0s = np.arange(0.0, ARENA_H, tile)
        for xi, x0 in enumerate(x0s):
            x1 = min(x0 + tile, ARENA_W)
            for yi, y0 in enumerate(y0s):
                y1 = min(y0 + tile, ARENA_H)
                c  = (80, 150, 70) if (xi + yi) % 2 == 0 else (65, 130, 58)
                polys_3d.append((
                    [(x0*S, y0*S, 0), (x1*S, y0*S, 0), (x1*S, y1*S, 0), (x0*S, y1*S, 0)],
                    c, 0, False, True   # (verts, color, layer, is_player, no_fpv)
                ))

        # ----------------------------------------------------------------
        # B. Arena Walls — 4 sides, extruded from Z=0 to Z=WALL_HEIGHT*S
        #    Tagged no_fpv=True so they don't pollute the FPV background
        # ----------------------------------------------------------------
        WH = WALL_HEIGHT * S
        AW = ARENA_W * S
        AH = ARENA_H * S
        WALL_C1 = (130, 140, 155)
        WALL_C2 = (100, 110, 120)

        # (verts, color, layer, is_player=False, no_fpv=True)
        polys_3d.append(([( 0,  0,  0), (AW,  0,  0), (AW,  0, WH), ( 0,  0, WH)], WALL_C1, 1, False, True))
        polys_3d.append(([(0,AH, 0), (AW,AH, 0), (AW,AH, WH), (0,AH, WH)], WALL_C2, 1, False, True))
        polys_3d.append(([( 0,  0,  0), ( 0, AH,  0), ( 0, AH, WH), ( 0,  0, WH)], WALL_C1, 1, False, True))
        polys_3d.append(([(AW,  0, 0), (AW, AH,  0), (AW, AH, WH), (AW,  0, WH)], WALL_C2, 1, False, True))

        # ----------------------------------------------------------------
        # C. Laser Bullets — player (magenta) and opponent (red-orange)
        # ----------------------------------------------------------------
        z_lo = (DRONE_ALT - 0.1) * S
        z_hi = (DRONE_ALT + 0.1) * S
        for b_list, b_color in [(base.bullets, (255, 0, 255)), (base.opp_bullets, (255, 90, 40))]:
            for b in b_list:
                bx = b['body'].position.x * S
                by = b['body'].position.y * S
                vx, vy = b['body'].linearVelocity.x, b['body'].linearVelocity.y
                v_len = math.sqrt(vx**2 + vy**2) + 1e-5
                ux = (vx / v_len) * 0.8 * S
                uy = (vy / v_len) * 0.8 * S
                # Perpendicular (horizontal) half-width so bolts stay visible end-on in FPV
                wx = -(vy / v_len) * 0.1 * S
                wy = (vx / v_len) * 0.1 * S
                z_mid_b = DRONE_ALT * S
                b_verts = [
                    (bx - ux, by - uy, z_lo),
                    (bx + ux, by + uy, z_lo),
                    (bx + ux, by + uy, z_hi),
                    (bx - ux, by - uy, z_hi),
                ]
                polys_3d.append((b_verts, b_color, 3))
                polys_3d.append(([(bx - ux - wx, by - uy - wy, z_mid_b), (bx + ux - wx, by + uy - wy, z_mid_b),
                                  (bx + ux + wx, by + uy + wy, z_mid_b), (bx - ux + wx, by - uy + wy, z_mid_b)],
                                 b_color, 3))
                # Cross-section cap at the bolt head (faces the direction of travel)
                polys_3d.append(([(bx + ux - wx, by + uy - wy, z_lo), (bx + ux + wx, by + uy + wy, z_lo),
                                  (bx + ux + wx, by + uy + wy, z_hi), (bx + ux - wx, by + uy - wy, z_hi)],
                                 b_color, 3))

        # ----------------------------------------------------------------
        # C2. Orange Explosion Bursts (player and/or opponent)
        # ----------------------------------------------------------------
        for ex in [base.player_explosion, base.opp_explosion]:
            if ex is None:
                continue
            ex_x, ex_y = ex['pos'][0] * S, ex['pos'][1] * S
            t = max(0.0, ex['frames'] / 14.0)
            ex_c1 = (255, int(120 * t), int(20 * t))
            ex_c2 = (255, min(255, int(200 * t)), 0)
            ex_rad = ex['radius'] * S * (0.5 + 0.5 * t)
            for scale, ec in [(1.0, ex_c1), (0.6, ex_c2), (0.3, (255, 255, int(180*t)))]:
                r_s = ex_rad * scale
                ring_top = [(ex_x + r_s*math.cos(a), ex_y + r_s*math.sin(a), z_hi + 0.3*S)
                            for a in np.linspace(0, 2*math.pi, 10, endpoint=False)]
                ring_bot = [(ex_x + r_s*math.cos(a), ex_y + r_s*math.sin(a), z_lo - 0.2*S)
                            for a in np.linspace(0, 2*math.pi, 10, endpoint=False)]
                polys_3d.append((ring_top, ec, 4))
                polys_3d.append((ring_bot, ec, 4))

        # ----------------------------------------------------------------
        # D. Rich 3D Quadrotor Meshes (only for alive drones)
        # ----------------------------------------------------------------
        drone_specs = []
        if base.player_alive:
            drone_specs.append((base.player,   (0, 220, 255), (0, 160, 200), (255, 215, 0),   True))
        if base.opponent_alive:
            drone_specs.append((base.opponent, (255, 60, 60), (190, 30, 30), (220, 220, 220), False))

        z_lo = self.DRONE_Z_LO * S
        z_hi = self.DRONE_Z_HI * S
        z_mid = (self.DRONE_Z_LO + self.DRONE_Z_HI) / 2.0 * S

        for d_body, main_c, sub_c, accent_c, is_player in drone_specs:
            tf = d_body.transform

            # Canopy fixture
            canopy_fix = d_body.fixtures[0]
            c_verts_2d = [(tf * v) * S for v in canopy_fix.shape.vertices]
            V_lo = [(vx, vy, z_lo) for vx, vy in c_verts_2d]
            V_hi = [(vx, vy, z_hi) for vx, vy in c_verts_2d]
            n = len(V_lo)
            polys_3d.append((V_lo,  main_c, 2, is_player))
            polys_3d.append((V_hi,  main_c, 2, is_player))
            for i in range(n):
                side_c = sub_c if i % 2 == 0 else main_c
                polys_3d.append(([V_lo[i], V_lo[(i+1)%n], V_hi[(i+1)%n], V_hi[i]], side_c, 2, is_player))

            # Glass cockpit bubble (front hemisphere, thin disc)
            bc = (tf * (0.1, 0.0)) * S
            bubble_verts = [(bc[0] + 0.28*S*math.cos(a), bc[1] + 0.28*S*math.sin(a), z_hi)
                            for a in np.linspace(0, 2*math.pi, 8, endpoint=False)]
            polys_3d.append((bubble_verts, (60, 100, 130), 3, is_player))

            # 4 rotor arms + rotor discs
            for af in d_body.fixtures[1:5]:
                a_verts_2d = [(tf * v) * S for v in af.shape.vertices]
                Va_lo = [(vx, vy, z_lo) for vx, vy in a_verts_2d]
                Va_hi = [(vx, vy, z_mid) for vx, vy in a_verts_2d]
                polys_3d.append((Va_hi, (55, 60, 70), 2, is_player))
                m = len(Va_lo)
                for i in range(m):
                    polys_3d.append(([Va_lo[i], Va_lo[(i+1)%m], Va_hi[(i+1)%m], Va_hi[i]], (38, 42, 50), 2, is_player))

                # Rotor disc at arm tip
                tip = a_verts_2d[0]
                rotor = [(tip[0] + 0.4*S*math.cos(a), tip[1] + 0.4*S*math.sin(a), z_hi)
                         for a in np.linspace(0, 2*math.pi, 7, endpoint=False)]
                polys_3d.append((rotor, accent_c, 3, is_player))

        # ----------------------------------------------------------------
        # E. Multi-View Projections
        # ----------------------------------------------------------------
        # Camera — follow cam behind/above the drone, looking along heading (in X-Y plane)
        p_angle = float(base.player.angle)
        p_x = base.player.position.x * S
        p_y = base.player.position.y * S

        # Get dynamically calculated position from our property (orbited, for logging)
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        # Un-orbited anchor for camera-relative vectors (orbit_camera_frame applies the orbit)
        anc_x, anc_y, anc_z = self._cam_anchor

        cos_a = math.cos(-p_angle)
        sin_a = math.sin(-p_angle)

        # Scrolling offset centred on player for top-down and side views
        scroll_x = p_x
        scroll_y = p_y

        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []

        for poly_item in polys_3d:
            if len(poly_item) == 5:
                p_verts, color, layer, is_player, no_fpv = poly_item
            elif len(poly_item) == 4:
                p_verts, color, layer, is_player = poly_item
                no_fpv = False
            else:
                p_verts, color, layer = poly_item
                is_player = False
                no_fpv = False

            pts_t, pts_r, pts_s = [], [], []
            dt, dr, ds = [], [], []

            for vx, vy, vz in p_verts:
                sx = vx - scroll_x   # scrolled X
                sy = vy - scroll_y   # scrolled Y

                # Top-Down: X-Y horizontal plane (bird's eye)
                #   screen_x = sx + cx,  screen_y = VH/2 - sy   (Y up → screen down)
                pts_t.append((sx + cx, cy - sy))
                dt.append(vz)   # sort by altitude (higher = drawn later = on top)

                # Rear (West-facing): Y (depth) on horizontal axis, Z (altitude) vertical
                pts_r.append((sy + cx, VH - vz))
                dr.append(sx)

                # Side (South-facing): X (horizontal) axis, Z (altitude) vertical
                pts_s.append((sx + cx, VH - vz))
                ds.append(-sy)

            if len(pts_t) >= 3: proj_top.append( (sum(dt)/len(dt),  pts_t, color, layer))
            if len(pts_r) >= 3: proj_rear.append((sum(dr)/len(dr),  pts_r, color, layer))
            if len(pts_s) >= 3: proj_side.append((sum(ds)/len(ds),  pts_s, color, layer))

            # FPV — skip env geometry (floor/walls); the player drone is drawn (third-person view)
            if no_fpv:
                continue

            # Camera-relative vertices, orbited around the player (depth, lateral, vertical)
            rel_verts = []
            for vx, vy, vz in p_verts:
                dx, dy = vx - anc_x, vy - anc_y
                dz = vz - anc_z
                # Rotate into camera space (rotate around Z-axis)
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                rel_verts.append(orbit_camera_frame(rx, -ry, dz, self.cam_yaw, self.cam_pitch, self.pivot_depth))

            # Sutherland-Hodgman near-plane clip on the orbited depth (d_c >= 1.0)
            clipped = []
            for i in range(len(rel_verts)):
                pc, pp = rel_verts[i], rel_verts[i - 1]
                if pc[0] >= 1.0 and pp[0] >= 1.0:
                    clipped.append(pc)
                elif (pc[0] >= 1.0) != (pp[0] >= 1.0):
                    t = (1.0 - pp[0]) / (pc[0] - pp[0])
                    clipped.append((1.0, pp[1] + t*(pc[1]-pp[1]), pp[2] + t*(pc[2]-pp[2])))
                    if pc[0] >= 1.0:
                        clipped.append(pc)

            pts_fpv, df = [], []
            for d_c, lat_c, rz_c in clipped:
                ry_c = -lat_c
                if d_c > 0.1:
                    px_f = -(ry_c / d_c) * self.fov + cx
                    py_f = -(rz_c / d_c) * self.fov + cy
                    pts_fpv.append((px_f, py_f))
                    df.append(d_c)

            if len(pts_fpv) >= 3:
                proj_fpv.append((sum(df)/len(df), pts_fpv, color, layer))

        # ----------------------------------------------------------------
        # F. Painter's Algorithm Z-Sort & Draw
        # ----------------------------------------------------------------
        def draw_proj(proj_list, surf):
            proj_list.sort(key=lambda x: (x[3], -x[0]))
            for _, pts, c, _ in proj_list:
                ipts = [(int(round(px)), int(round(py))) for px, py in pts]
                if len(ipts) >= 3:
                    try:
                        gfxdraw.aapolygon(surf, ipts, c)
                        gfxdraw.filled_polygon(surf, ipts, c)
                    except Exception:
                        pass

        draw_proj(proj_top,  top_surf)
        draw_proj(proj_rear, rear_surf)
        draw_proj(proj_side, side_surf)
        draw_proj(proj_fpv,  fpv_surf)

        # ----------------------------------------------------------------
        # G. FPV HUD Reticle Overlay
        # ----------------------------------------------------------------
        HUD = (0, 255, 200)
        hx, hy = int(cx), int(cy)
        pygame.draw.circle(fpv_surf, HUD, (hx, hy), 28, 2)
        pygame.draw.line(fpv_surf, HUD, (hx - 42, hy), (hx - 16, hy), 2)
        pygame.draw.line(fpv_surf, HUD, (hx + 16, hy), (hx + 42, hy), 2)
        pygame.draw.line(fpv_surf, HUD, (hx, hy - 42), (hx, hy - 16), 2)
        pygame.draw.line(fpv_surf, HUD, (hx, hy + 16), (hx, hy + 42), 2)

        # Draw arena compass/outline on top-down view
        pygame.draw.rect(top_surf, (200, 200, 200), (int(cx - ARENA_W*S/2), int(cy - ARENA_H*S/2),
                                                      int(ARENA_W*S), int(ARENA_H*S)), 1)

        return [self._surf_to_img(s) for s in [top_surf, rear_surf, side_surf, fpv_surf]]
