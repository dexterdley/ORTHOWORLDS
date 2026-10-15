import math
import numpy as np
from .camera_utils import orbit_camera_frame, orbit_camera_offset

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import Box2D
from Box2D import b2ContactListener
from Box2D.b2 import (
    world as b2World,
    polygonShape as b2PolygonShape,
    circleShape as b2CircleShape,
    rayCastCallback as b2RayCastCallback,
    vec2 as b2Vec2,
)

import pygame
import pygame.gfxdraw

# Module-level constants
Z_NEAR = 0.1
Z_FAR = 100.0
SCREEN_W = 800
SCREEN_H = 600
WORLD_W = 40.0
WORLD_H = 30.0
PPM = SCREEN_W / WORLD_W  # pixels per meter
FPS = 50
DT = 1.0 / FPS

# Z-extrusion depths
Z_GROUND = 0.0
Z_TANK_BODY = 0.5
Z_TANK_TURRET = 1.0
Z_PROJECTILE = 0.8
Z_WALL = 2.0
Z_OBSTACLE = 1.5

# Colors (desert/military theme)
COLOR_GROUND = (194, 178, 128)
COLOR_WALL = (100, 80, 60)
COLOR_PLAYER_BODY = (60, 120, 60)
COLOR_PLAYER_TURRET = (40, 90, 40)
COLOR_ENEMY_BODY = (150, 50, 50)
COLOR_ENEMY_TURRET = (120, 30, 30)
COLOR_PROJECTILE_PLAYER = (255, 200, 0)
COLOR_PROJECTILE_ENEMY = (255, 80, 80)
COLOR_OBSTACLE = (139, 119, 101)
COLOR_SKY = (135, 206, 235)

MAX_PROJECTILES = 10
PROJECTILE_SPEED = 20.0
PROJECTILE_RADIUS = 0.2
PROJECTILE_LIFE = 100  # steps
FIRE_COOLDOWN = 15  # steps
ENEMY_FIRE_COOLDOWN = 30  # steps (+ random 0..20)
TANK_MAX_SPEED = 5.0  # m/s along the hull heading
TANK_ACCEL = 10.0  # m/s^2 tread acceleration
HULL_TURN_RATE = 1.5  # rad/s
TURRET_MAX_RATE = 2.0  # rad/s (player turret, relative to hull)
ENEMY_TURRET_RATE = 1.5  # rad/s
ENEMY_SPEED = 0.6  # fraction of TANK_MAX_SPEED
ENEMY_AIM_NOISE = 0.06  # rad std of enemy aim error per shot
MAX_HEALTH = 100.0
HIT_DAMAGE = 10.0

# Shell impact effects (purely visual: never read by physics, observations, rewards or the RNG)
EXPLOSION_LIFE = 14  # steps (0.28 s) a tank-hit fireball stays on screen
EXPLOSION_RADIUS = (0.5, 1.8)  # fireball radius (m) at impact -> fully grown
PUFF_LIFE = 8  # steps (0.16 s) of the dust puff when a shell hits a wall / obstacle
PUFF_RADIUS = (0.15, 0.7)
Z_EXPLOSION = Z_TANK_TURRET  # fireball centre height (top of the hull / turret base)
HIT_FLASH_STEPS = 6  # steps the hit tank is tinted toward COLOR_HIT_FLASH
COLOR_HIT_FLASH = (255, 235, 190)

# Tank geometry (half extents, metres)
HULL_HALF_L = 1.5
HULL_HALF_W = 1.0
TURRET_HALF = (0.6, 0.5)
BARREL_START = 0.4
BARREL_END = 2.0
BARREL_HALF_W = 0.12

# Collision categories
CAT_WORLD = 0x0001
CAT_TANK = 0x0002
CAT_PROJECTILE = 0x0004

# Reward weights
REWARD_HIT = 2.0  # per hit on the enemy
REWARD_HURT = -1.0  # per hit taken
REWARD_WIN = 20.0
REWARD_LOSE = -20.0
AIM_SHAPING = 0.5  # potential-based: phi = AIM_SHAPING * cos(aim_error)


def wrap_angle(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _mix_color(c0, c1, t):
    """Linear blend of two RGB colors (t = 0 -> c0, t = 1 -> c1)."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))


def draw_fireball(surface, cx, cy, rx, ry, rgba):
    """Alpha-blended filled ellipse centred at (cx, cy) with pixel radii (rx, ry); skipped when off-screen."""
    w, h = surface.get_size()
    limit = 2 * max(w, h)  # keeps gfxdraw's int16 coordinates in range
    rx = int(min(max(rx, 1), limit))
    ry = int(min(max(ry, 1), limit))
    if not (-rx <= cx < w + rx and -ry <= cy < h + ry):
        return
    cx, cy = int(cx), int(cy)
    if rx == ry:
        pygame.gfxdraw.filled_circle(surface, cx, cy, rx, rgba)
    else:
        pygame.gfxdraw.filled_ellipse(surface, cx, cy, rx, ry, rgba)


class MyRayCastCallback(b2RayCastCallback):
    """Closest-hit ray cast that skips sensors (projectiles) and bodies of an optional ignored type."""

    def __init__(self, ignore_type=None):
        super().__init__()
        self.ignore_type = ignore_type
        self.hit = False
        self.fixture = None
        self.point = None
        self.normal = None
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        data = fixture.body.userData
        if fixture.sensor or (data is not None and data.get("type") == self.ignore_type):
            return -1.0  # ignore and continue
        self.hit = True
        self.fixture = fixture
        self.point = point
        self.normal = normal
        self.fraction = fraction
        return fraction


class ProjectileContactListener(b2ContactListener):
    """Records projectile contacts during world.Step; the env resolves them after the step."""

    def __init__(self):
        super().__init__()
        self.events = []

    def BeginContact(self, contact):
        data_a = contact.fixtureA.body.userData
        data_b = contact.fixtureB.body.userData
        for proj_data, other_data in ((data_a, data_b), (data_b, data_a)):
            if proj_data is not None and proj_data.get("type") == "projectile":
                other_type = other_data.get("type") if other_data is not None else None
                self.events.append((proj_data, other_type))


class TankDuelEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": FPS}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Action (all in [-1, 1]):
        #   a0: drive throttle along the hull heading (+ forward, - reverse)
        #   a1: hull turn rate (+ turn left / counter-clockwise, - turn right / clockwise)
        #   a2: turret rotation rate relative to the hull (+ left / counter-clockwise, - right / clockwise)
        #   a3: fire trigger (shoots when > 0.5 and the cannon has reloaded)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation (26 floats, all in [-1, 1]):
        #   0-9   player: x, y (arena-normalised), sin/cos hull heading, vx, vy (/ max speed),
        #         sin/cos turret angle relative to hull, health fraction, reload fraction (0 = ready)
        #   10-19 enemy: same layout as the player
        #   20-25 relative: enemy dx / WORLD_W, dy / WORLD_H, distance / arena diagonal,
        #         sin/cos of player aim error (bearing to enemy - world turret angle), line-of-sight flag (1/0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(26,), dtype=np.float32
        )

        self.world = None
        self.contact_listener = None
        self.player_body = None
        self.player_turret = None  # turrets are rendered from angles only (no physics body)
        self.enemy_body = None
        self.enemy_turret = None
        self.projectiles = []
        self.player_health = MAX_HEALTH
        self.enemy_health = MAX_HEALTH
        self.player_turret_angle = 0.0  # relative to the player hull, wrapped to [-pi, pi)
        self.enemy_turret_angle = 0.0  # relative to the enemy hull, wrapped to [-pi, pi)
        self.player_fire_cooldown = 0
        self.enemy_fire_cooldown = 0
        self.enemy_mode = "approach"
        self.enemy_mode_timer = 0
        self.enemy_aim_offset = 0.0
        self.enemy_stuck_steps = 0
        self.enemy_flank_heading = None
        self.player_hits = 0
        self.enemy_hits = 0
        self._aim_potential = 0.0
        self.steps = 0
        self.max_steps = 1000
        self.obstacles = []
        self.walls = []
        # Visual-only shell impact effects: explosion records and per-tank hit-flash countdowns
        self.explosions = []
        self.hit_flash = {"player": 0, "enemy": 0}

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self.contact_listener = ProjectileContactListener()
        self.world = b2World(gravity=(0, 0), contactListener=self.contact_listener)
        self.projectiles = []
        self.obstacles = []
        self.walls = []
        self.player_health = MAX_HEALTH
        self.enemy_health = MAX_HEALTH
        self.player_turret_angle = 0.0
        self.enemy_turret_angle = 0.0
        self.player_fire_cooldown = 0
        self.enemy_fire_cooldown = ENEMY_FIRE_COOLDOWN
        self.enemy_mode = "approach"
        self.enemy_mode_timer = int(rng.integers(50, 100))
        self.enemy_aim_offset = float(rng.normal(0.0, ENEMY_AIM_NOISE))
        self.enemy_stuck_steps = 0
        self.enemy_flank_heading = None
        self.player_hits = 0
        self.enemy_hits = 0
        self.steps = 0
        self.explosions = []
        self.hit_flash = {"player": 0, "enemy": 0}

        # Create walls (boundaries)
        wall_thickness = 0.5
        wall_defs = [
            ((WORLD_W / 2, -wall_thickness / 2), (WORLD_W / 2, wall_thickness / 2)),
            ((WORLD_W / 2, WORLD_H + wall_thickness / 2), (WORLD_W / 2, wall_thickness / 2)),
            ((-wall_thickness / 2, WORLD_H / 2), (wall_thickness / 2, WORLD_H / 2)),
            ((WORLD_W + wall_thickness / 2, WORLD_H / 2), (wall_thickness / 2, WORLD_H / 2)),
        ]
        for pos, half_size in wall_defs:
            wall = self.world.CreateStaticBody(position=pos)
            wall.CreatePolygonFixture(
                box=half_size, density=0, friction=0.3, categoryBits=CAT_WORLD
            )
            wall.userData = {"type": "wall", "color": COLOR_WALL, "z": Z_WALL}
            self.walls.append(wall)

        # Create obstacles: cover boxes in the upper/lower bands, never across the spawn line
        for ox, oy, hw, hh in self._sample_obstacles():
            obs = self.world.CreateStaticBody(position=(ox, oy))
            obs.CreatePolygonFixture(
                box=(hw, hh), density=0, friction=0.5, categoryBits=CAT_WORLD
            )
            obs.userData = {"type": "obstacle", "color": COLOR_OBSTACLE, "z": Z_OBSTACLE}
            self.obstacles.append(obs)

        # Create tanks (hull bodies only; turrets are angles drawn on top)
        self.player_body = self._create_tank(
            (5.0, WORLD_H / 2 + rng.uniform(-2.0, 2.0)),
            rng.uniform(-0.2, 0.2),
            "player_body",
            COLOR_PLAYER_BODY,
        )
        self.enemy_body = self._create_tank(
            (WORLD_W - 5.0, WORLD_H / 2 + rng.uniform(-2.0, 2.0)),
            math.pi + rng.uniform(-0.2, 0.2),
            "enemy_body",
            COLOR_ENEMY_BODY,
        )

        self._aim_potential = self._compute_aim_potential()
        obs = self._get_obs()
        return obs, self._get_info()

    def _sample_obstacles(self, num_obstacles=6):
        """Random cover boxes (centre x, y, half w, half h) in the bands above/below the spawn line."""
        rng = self.np_random
        placed = []
        attempts = 0
        while len(placed) < num_obstacles and attempts < 200:
            attempts += 1
            hw = float(rng.uniform(0.5, 1.5))
            hh = float(rng.uniform(0.5, 1.5))
            ox = float(rng.uniform(11.0, WORLD_W - 11.0))
            # Lower band top edge <= 11.5 m, upper band bottom edge >= 18.5 m; tanks spawn at y = 13..17
            if rng.random() < 0.5:
                oy = float(rng.uniform(4.5, 10.0))
            else:
                oy = float(rng.uniform(WORLD_H - 10.0, WORLD_H - 4.5))
            # Keep a tank-sized gap between obstacles so they never merge into a wall
            if any(
                abs(ox - px) < hw + pw + 3.0 and abs(oy - py) < hh + ph + 3.0
                for px, py, pw, ph in placed
            ):
                continue
            placed.append((ox, oy, hw, hh))
        return placed

    def _create_tank(self, position, angle, tank_type, color):
        body = self.world.CreateDynamicBody(
            position=(float(position[0]), float(position[1])),
            angle=float(angle),
            linearDamping=2.0,
            angularDamping=5.0,
        )
        body.CreatePolygonFixture(
            box=(HULL_HALF_L, HULL_HALF_W),
            density=5.0,
            friction=0.3,
            categoryBits=CAT_TANK,
            maskBits=CAT_WORLD | CAT_TANK | CAT_PROJECTILE,
        )
        body.userData = {
            "type": tank_type,
            "color": color,
            "z": Z_TANK_BODY,
        }
        return body

    def _drive_tank(self, body, throttle, turn, max_speed=TANK_MAX_SPEED):
        """Tread model: speed along the hull heading ramps toward throttle * max_speed, no sideways slip."""
        a = body.angle
        fwd_x, fwd_y = math.cos(a), math.sin(a)
        v = body.linearVelocity
        speed = v[0] * fwd_x + v[1] * fwd_y
        target = float(throttle) * max_speed
        speed += float(np.clip(target - speed, -TANK_ACCEL * DT, TANK_ACCEL * DT))
        body.linearVelocity = (speed * fwd_x, speed * fwd_y)
        body.angularVelocity = float(turn) * HULL_TURN_RATE

    def _line_of_sight(self, from_body, to_body, half_width=0.0):
        """True if the first solid fixture on the ray between the two hull centres belongs to to_body.
        half_width > 0 also checks two parallel rays offset sideways (a shell-wide corridor)."""
        fx, fy = from_body.position[0], from_body.position[1]
        tx, ty = to_body.position[0], to_body.position[1]
        dist = math.hypot(tx - fx, ty - fy)
        if dist < 1e-6:
            return True
        nx, ny = -(ty - fy) / dist, (tx - fx) / dist
        offsets = (0.0, -half_width, half_width) if half_width > 0 else (0.0,)
        for off in offsets:
            hit_type = self._ray_first_type(
                (fx + nx * off, fy + ny * off), (tx + nx * off, ty + ny * off),
                from_body.userData["type"],
            )
            if hit_type != to_body.userData["type"]:
                return False
        return True

    def _ray_first_type(self, p1, p2, ignore_type=None):
        """Type of the first solid (non-sensor) body hit on the segment p1 -> p2, or None if clear."""
        cb = MyRayCastCallback(ignore_type=ignore_type)
        self.world.RayCast(cb, (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        if not cb.hit or cb.fixture.body.userData is None:
            return None
        return cb.fixture.body.userData.get("type")

    def _enemy_flank_heading(self):
        """Pick a heading for the enemy that leads to a spot with line of sight to the player:
        probe 8 directions 4 m out, keep the clear ones, prefer probes that see the player."""
        ex, ey = self.enemy_body.position[0], self.enemy_body.position[1]
        px, py = self.player_body.position[0], self.player_body.position[1]
        best_score, best_heading = -np.inf, math.atan2(py - ey, px - ex)
        for k in range(8):
            heading = k * math.pi / 4
            qx = ex + math.cos(heading) * 4.0
            qy = ey + math.sin(heading) * 4.0
            if not (2.0 < qx < WORLD_W - 2.0 and 2.0 < qy < WORLD_H - 2.0):
                continue
            if self._ray_first_type((ex, ey), (qx, qy), "enemy_body") is not None:
                continue
            sees = self._ray_first_type((qx, qy), (px, py), "enemy_body") == "player_body"
            score = (
                10.0 * sees
                - 0.3 * abs(math.hypot(px - qx, py - qy) - 12.0)
                - 0.5 * abs(wrap_angle(heading - self.enemy_body.angle))
            )
            if score > best_score:
                best_score, best_heading = score, heading
        return best_heading

    def _fire_projectile(self, tank_body, turret_angle, is_player=True):
        """Spawns a sensor shell at the barrel tip (outside the hull). turret_angle is hull-relative."""
        owner = "player" if is_player else "enemy"
        if sum(1 for p in self.projectiles if p.userData["owner"] == owner) >= MAX_PROJECTILES:
            return False

        aim = tank_body.angle + turret_angle
        pos = tank_body.position
        muzzle = BARREL_END + PROJECTILE_RADIUS + 0.1
        start_x = pos[0] + math.cos(aim) * muzzle
        start_y = pos[1] + math.sin(aim) * muzzle

        proj = self.world.CreateDynamicBody(
            position=(float(start_x), float(start_y)),
            linearDamping=0.0,
            fixedRotation=True,
        )
        proj.CreateCircleFixture(
            radius=PROJECTILE_RADIUS,
            density=1.0,
            isSensor=True,
            categoryBits=CAT_PROJECTILE,
            maskBits=CAT_WORLD | CAT_TANK,
        )
        vx = math.cos(aim) * PROJECTILE_SPEED
        vy = math.sin(aim) * PROJECTILE_SPEED
        proj.linearVelocity = (float(vx), float(vy))
        proj.userData = {
            "type": "projectile",
            "owner": owner,
            "color": COLOR_PROJECTILE_PLAYER if is_player else COLOR_PROJECTILE_ENEMY,
            "z": Z_PROJECTILE,
            "life": PROJECTILE_LIFE,
        }
        self.projectiles.append(proj)
        return True

    def _enemy_ai(self):
        """Enemy AI: steer the hull (approach / strafe / hold / back off), track with the turret,
        fire only with a clear line of sight."""
        if self.enemy_body is None or self.player_body is None:
            return

        rng = self.np_random
        ex, ey = self.enemy_body.position
        px, py = self.player_body.position
        dist = math.hypot(px - ex, py - ey)
        bearing = math.atan2(py - ey, px - ex)
        los = self._line_of_sight(self.enemy_body, self.player_body, half_width=PROJECTILE_RADIUS)

        # Movement mode, re-picked every 1.5-3 s
        self.enemy_mode_timer -= 1
        if self.enemy_mode_timer <= 0:
            modes = ["approach", "strafe_left", "strafe_right", "hold"]
            self.enemy_mode = modes[int(rng.choice(4, p=[0.4, 0.2, 0.2, 0.2]))]
            self.enemy_mode_timer = int(rng.integers(75, 150))
        mode = self.enemy_mode

        if mode == "unstick":
            # Back out of a wall / obstacle while turning
            throttle, turn = -ENEMY_SPEED, 1.0
        else:
            throttle = ENEMY_SPEED
            if not los:
                # Blocked by cover: drive toward a spot with line of sight (re-planned every 10 steps)
                if self.steps % 10 == 0 or self.enemy_flank_heading is None:
                    self.enemy_flank_heading = self._enemy_flank_heading()
                desired = self.enemy_flank_heading
            elif mode == "strafe_left":
                desired = bearing + math.pi / 2
            elif mode == "strafe_right":
                desired = bearing - math.pi / 2
            else:
                desired = bearing
                if mode == "hold" or dist < 10.0:
                    throttle = 0.0
            if dist < 7.0:
                desired = bearing
                throttle = -ENEMY_SPEED  # back off
            # Wall avoidance: head for the arena centre when close to a wall
            margin = 4.0
            if ex < margin or ex > WORLD_W - margin or ey < margin or ey > WORLD_H - margin:
                desired = math.atan2(WORLD_H / 2 - ey, WORLD_W / 2 - ex)
                throttle = ENEMY_SPEED

            heading_err = wrap_angle(desired - self.enemy_body.angle)
            turn = float(np.clip(heading_err * 2.0, -1.0, 1.0))
            if throttle > 0:
                throttle *= max(0.0, math.cos(heading_err))

            # Stuck detection: commanded to move but barely moving for 0.5 s -> back out for 0.8 s
            v = self.enemy_body.linearVelocity
            if abs(throttle) > 0.2 and math.hypot(v[0], v[1]) < 0.3:
                self.enemy_stuck_steps += 1
            else:
                self.enemy_stuck_steps = 0
            if self.enemy_stuck_steps > 25:
                self.enemy_stuck_steps = 0
                self.enemy_flank_heading = None  # re-plan after backing out
                self.enemy_mode = "unstick"
                self.enemy_mode_timer = 40
        self._drive_tank(self.enemy_body, throttle, turn)

        # Turret tracks the player (hull-relative) with a per-shot aim error
        target_rel = wrap_angle(bearing + self.enemy_aim_offset - self.enemy_body.angle)
        angle_diff = wrap_angle(target_rel - self.enemy_turret_angle)
        self.enemy_turret_angle = wrap_angle(
            self.enemy_turret_angle
            + float(np.clip(angle_diff, -ENEMY_TURRET_RATE * DT, ENEMY_TURRET_RATE * DT))
        )

        # Fire when on target and the line of sight is clear
        if self.enemy_fire_cooldown <= 0 and abs(angle_diff) < 0.05 and los:
            if self._fire_projectile(self.enemy_body, self.enemy_turret_angle, is_player=False):
                self.enemy_fire_cooldown = ENEMY_FIRE_COOLDOWN + int(rng.integers(0, 21))
                self.enemy_aim_offset = float(rng.normal(0.0, ENEMY_AIM_NOISE))

        self.enemy_fire_cooldown = max(0, self.enemy_fire_cooldown - 1)

    def _aim_error(self):
        """Signed angle from the player's world turret direction to the bearing of the enemy."""
        ex, ey = self.enemy_body.position
        plx, ply = self.player_body.position
        angle_to_enemy = math.atan2(ey - ply, ex - plx)
        return wrap_angle(angle_to_enemy - (self.player_body.angle + self.player_turret_angle))

    def _compute_aim_potential(self):
        return AIM_SHAPING * math.cos(self._aim_error())

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        drive = float(action[0])
        hull_turn = float(action[1])
        turret_rot = float(action[2])
        fire_trigger = float(action[3])

        self.steps += 1

        # Treads: drive along the hull heading and turn the hull
        self._drive_tank(self.player_body, drive, hull_turn)

        # Turret rotation (relative to hull, kept wrapped)
        self.player_turret_angle = wrap_angle(
            self.player_turret_angle + turret_rot * TURRET_MAX_RATE * DT
        )

        # Fire
        if fire_trigger > 0.5 and self.player_fire_cooldown <= 0:
            if self._fire_projectile(self.player_body, self.player_turret_angle, is_player=True):
                self.player_fire_cooldown = FIRE_COOLDOWN
        self.player_fire_cooldown = max(0, self.player_fire_cooldown - 1)

        # Enemy AI
        self._enemy_ai()

        # Step physics (projectile contacts are recorded by the contact listener)
        self.contact_listener.events = []
        self.world.Step(DT, 6, 2)
        self.world.ClearForces()

        # Resolve projectile contacts: any contact destroys the shell, tank contacts deal damage
        # (Box2D proxies are not identity-stable, so shells are keyed by their userData dict)
        hits_dealt = 0
        hits_taken = 0
        dead = {}  # id(shell userData) -> type of the body it hit
        for proj_data, other_type in self.contact_listener.events:
            if id(proj_data) in dead:
                continue
            owner = proj_data["owner"]
            if other_type == owner + "_body":
                continue  # own hull (cannot normally happen: shells spawn outside it)
            if owner == "player" and other_type == "enemy_body":
                hits_dealt += 1
            elif owner == "enemy" and other_type == "player_body":
                hits_taken += 1
            dead[id(proj_data)] = other_type
        self.contact_listener.events = []

        self._age_effects()
        remaining = []
        for proj in self.projectiles:
            data = proj.userData
            data["life"] -= 1
            px, py = proj.position
            if id(data) in dead:
                vx, vy = proj.linearVelocity
                # Shells move 0.4 m per step: back up half a step to put the blast near the struck surface
                self._spawn_explosion(px - vx * DT * 0.5, py - vy * DT * 0.5, dead[id(data)])
            if (id(data) in dead or data["life"] <= 0
                    or px < -1 or px > WORLD_W + 1 or py < -1 or py > WORLD_H + 1):
                self.world.DestroyBody(proj)
            else:
                remaining.append(proj)
        self.projectiles = remaining

        self.player_hits += hits_dealt
        self.enemy_hits += hits_taken
        self.enemy_health = max(0.0, self.enemy_health - HIT_DAMAGE * hits_dealt)
        self.player_health = max(0.0, self.player_health - HIT_DAMAGE * hits_taken)

        # Compute reward
        terminated = False
        truncated = False

        # Per-hit rewards (paid once per hit)
        reward = REWARD_HIT * hits_dealt + REWARD_HURT * hits_taken

        # Potential-based aim shaping: telescopes to at most +-2 * AIM_SHAPING per episode
        potential = self._compute_aim_potential()
        reward += potential - self._aim_potential
        self._aim_potential = potential

        # Terminal conditions (simultaneous kill is a draw: no bonus)
        if self.enemy_health <= 0 and self.player_health <= 0:
            terminated = True
        elif self.enemy_health <= 0:
            reward += REWARD_WIN
            terminated = True
        elif self.player_health <= 0:
            reward += REWARD_LOSE
            terminated = True

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["hits_dealt"] = hits_dealt
        info["hits_taken"] = hits_taken

        return obs, float(reward), terminated, truncated, info

    def _get_info(self):
        return {
            "player_health": self.player_health,
            "enemy_health": self.enemy_health,
            "player_hits": self.player_hits,
            "enemy_hits": self.enemy_hits,
            "is_success": self.enemy_health <= 0 < self.player_health,
        }

    def _tank_obs(self, body, turret_angle, health, cooldown, cooldown_max):
        x, y = body.position
        vx, vy = body.linearVelocity
        a = body.angle
        return [
            x / WORLD_W * 2.0 - 1.0, y / WORLD_H * 2.0 - 1.0,
            math.sin(a), math.cos(a),
            vx / TANK_MAX_SPEED, vy / TANK_MAX_SPEED,
            math.sin(turret_angle), math.cos(turret_angle),
            health / MAX_HEALTH,
            cooldown / cooldown_max,
        ]

    def _get_obs(self):
        px, py = self.player_body.position
        ex, ey = self.enemy_body.position
        aim_err = self._aim_error()
        los = self._line_of_sight(self.player_body, self.enemy_body)

        obs = np.array(
            self._tank_obs(self.player_body, self.player_turret_angle, self.player_health,
                           self.player_fire_cooldown, FIRE_COOLDOWN)
            + self._tank_obs(self.enemy_body, self.enemy_turret_angle, self.enemy_health,
                             self.enemy_fire_cooldown, ENEMY_FIRE_COOLDOWN + 20)
            + [
                (ex - px) / WORLD_W, (ey - py) / WORLD_H,
                math.hypot(ex - px, ey - py) / math.hypot(WORLD_W, WORLD_H),
                math.sin(aim_err), math.cos(aim_err),
                1.0 if los else 0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _age_effects(self):
        """Advance explosions / hit flashes by one step and drop expired ones (visual state only)."""
        for e in self.explosions:
            e["age"] += 1
        self.explosions = [e for e in self.explosions if e["age"] < e["life"]]
        for k in self.hit_flash:
            self.hit_flash[k] = max(0, self.hit_flash[k] - 1)

    def _spawn_explosion(self, x, y, hit_type):
        """Record a fireball (shell hit a tank) or a small dust puff (wall / obstacle) at the impact point."""
        if hit_type in ("player_body", "enemy_body"):
            target = hit_type[:-len("_body")]
            self.explosions.append({
                "x": float(x), "y": float(y), "z": Z_EXPLOSION, "age": 0,
                "life": EXPLOSION_LIFE, "radius": EXPLOSION_RADIUS, "target": target,
            })
            self.hit_flash[target] = HIT_FLASH_STEPS
        else:
            self.explosions.append({
                "x": float(x), "y": float(y), "z": Z_PROJECTILE + 0.25, "age": 0,
                "life": PUFF_LIFE, "radius": PUFF_RADIUS, "target": hit_type or "wall",
            })

    @staticmethod
    def explosion_layers(e):
        """Fireball layers of one explosion record, back to front: list of (radius_m, rgba).
        The fireball grows (ease-out) from its impact radius to full size while the yellow core burns out
        first, the orange fire darkens and the outer smoke fades."""
        frac = e["age"] / float(e["life"])
        r0, r1 = e["radius"]
        r = r0 + (r1 - r0) * (1.0 - (1.0 - frac) ** 2)
        fade = 1.0 - frac
        layers = []
        if e["target"] in ("player", "enemy"):
            smoke = _mix_color((170, 50, 20), (60, 50, 45), frac)
            layers.append((r, smoke + (int(220 * fade ** 0.7),)))
            fire = _mix_color((255, 140, 20), (190, 50, 15), frac)
            layers.append((0.7 * r, fire + (int(240 * fade),)))
            if frac < 0.6:
                core = _mix_color((255, 250, 190), (255, 190, 50), frac / 0.6)
                layers.append((0.4 * r, core + (int(255 * (1.0 - frac / 0.6) ** 0.5),)))
        else:
            dust = _mix_color((220, 200, 150), (130, 115, 95), frac)
            layers.append((r, dust + (int(200 * fade),)))
            if frac < 0.5:
                layers.append((0.45 * r, (255, 220, 120, int(230 * (1.0 - frac / 0.5)))))
        return layers

    def get_explosion_layers(self):
        """Fireball layers of all live explosions, back to front: list of (x, y, z, radius_m, rgba)."""
        return [
            (e["x"], e["y"], e["z"], r, rgba)
            for e in self.explosions
            for r, rgba in self.explosion_layers(e)
        ]

    def render_color(self, data):
        """Display color of a body (userData dict), tinted toward COLOR_HIT_FLASH while its tank was just hit."""
        color = data.get("color", (200, 200, 200))
        owner = str(data.get("type", "")).split("_")[0]
        flash = self.hit_flash.get(owner, 0) if data.get("type") != "projectile" else 0
        if flash <= 0:
            return color
        return _mix_color(color, COLOR_HIT_FLASH, 0.75 * flash / HIT_FLASH_STEPS)

    def get_turret_polygons(self):
        """World-space turret block and barrel for both tanks: list of (verts_2d, color, type, z_base, z_top)."""
        polys = []
        for body, turret_angle, color, turret_type in (
            (self.player_body, self.player_turret_angle, COLOR_PLAYER_TURRET, "player_turret"),
            (self.enemy_body, self.enemy_turret_angle, COLOR_ENEMY_TURRET, "enemy_turret"),
        ):
            if body is None:
                continue
            color = self.render_color({"type": turret_type, "color": color})
            aim = body.angle + turret_angle
            c, s = math.cos(aim), math.sin(aim)
            cx, cy = body.position[0], body.position[1]

            def to_world(lx, ly):
                return (cx + lx * c - ly * s, cy + lx * s + ly * c)

            hx, hy = TURRET_HALF
            block = [to_world(-hx, -hy), to_world(hx, -hy), to_world(hx, hy), to_world(-hx, hy)]
            barrel = [
                to_world(BARREL_START, -BARREL_HALF_W), to_world(BARREL_END, -BARREL_HALF_W),
                to_world(BARREL_END, BARREL_HALF_W), to_world(BARREL_START, BARREL_HALF_W),
            ]
            polys.append((block, color, turret_type, Z_TANK_TURRET, Z_TANK_TURRET + 0.5))
            polys.append((barrel, color, turret_type, Z_TANK_TURRET + 0.15, Z_TANK_TURRET + 0.35))
        return polys

    def _get_all_bodies(self):
        """Get all bodies with their rendering info"""
        bodies = []
        for body in self.world.bodies:
            if body.userData is None:
                continue
            bodies.append(body)
        return bodies

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            else:
                self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_GROUND)

        # Draw all bodies
        for body in self.world.bodies:
            if body.userData is None:
                continue
            color = self.render_color(body.userData)
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    vertices = [body.transform * v for v in shape.vertices]
                    points = [
                        (int(v[0] * PPM), int(SCREEN_H - v[1] * PPM))
                        for v in vertices
                    ]
                    if len(points) >= 3:
                        pygame.draw.polygon(self.screen, color, points)
                        pygame.draw.polygon(self.screen, (0, 0, 0), points, 1)
                elif isinstance(shape, b2CircleShape):
                    pos = body.transform * shape.pos
                    center = (int(pos[0] * PPM), int(SCREEN_H - pos[1] * PPM))
                    radius = int(shape.radius * PPM)
                    pygame.draw.circle(self.screen, color, center, radius)
                    pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 1)

        # Draw turrets (angle-only, no physics body)
        for verts, color, _, _, _ in self.get_turret_polygons():
            points = [(int(v[0] * PPM), int(SCREEN_H - v[1] * PPM)) for v in verts]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (0, 0, 0), points, 1)

        # Draw shell impact explosions on top
        for x, y, _, r, rgba in self.get_explosion_layers():
            draw_fireball(self.screen, x * PPM, SCREEN_H - y * PPM, r * PPM, r * PPM, rgba)

        # Draw health bars
        # Player health
        bar_w = 100
        bar_h = 10
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, bar_w, bar_h))
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (10, 10, int(bar_w * max(0.0, self.player_health) / MAX_HEALTH), bar_h),
        )
        # Enemy health
        pygame.draw.rect(self.screen, (50, 50, 50), (SCREEN_W - 110, 10, bar_w, bar_h))
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (
                SCREEN_W - 110,
                10,
                int(bar_w * max(0.0, self.enemy_health) / MAX_HEALTH),
                bar_h,
            ),
        )

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class TankDuelOrthographicWrapper(gym.Wrapper):
    """
    Wraps TankDuelEnv to provide orthographic multi-view 3D rendering.
    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    The 3D FPV view is a third-person orbit camera trailing the player tank body (tank fully visible).
    """

    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.view_w = SCREEN_W
        self.view_h = SCREEN_H

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state

        # Follow/Orbit camera baseline offsets (relative to player tank body, 3 m x 2 m hull, ~1.5 m tall with turret)
        self.camera_dist = 4.5     # Distance behind the tank centre along its forward heading (stays inside the 5 m spawn-to-wall gap)
        self.camera_height = 2.0   # Vertical offset above the tank base (camera at world z = 2.5, level with the wall tops)

        # Orbit pivot distance (m) along the FPV optical axis: the tank itself, so the camera circles the player
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
        base_env = self.env.unwrapped
        if getattr(base_env, 'player_body', None) is None:
            return (0.0, 0.0, 0.0)

        px, py = base_env.player_body.position[0], base_env.player_body.position[1]
        player_angle = base_env.player_body.angle
        # Third-person: camera trails camera_dist behind the tank body along its heading, camera_height above it
        cam_x = px - math.cos(player_angle) * self.camera_dist
        cam_y = py - math.sin(player_angle) * self.camera_dist
        cam_z = Z_TANK_BODY + self.camera_height

        return (cam_x, cam_y, cam_z)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the orbited camera in world space."""
        base_env = self.env.unwrapped
        if getattr(base_env, 'player_body', None) is None:
            return (0.0, 0.0, 0.0)

        ax, ay, az = self._cam_anchor
        player_angle = base_env.player_body.angle
        d_depth, d_lat, d_vert = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)
        # Inverse of _project_fpv's mapping: depth axis = (cos a, sin a), lateral (right) axis = (sin a, -cos a)
        cos_a = math.cos(player_angle)
        sin_a = math.sin(player_angle)
        cam_x = ax + d_depth * cos_a + d_lat * sin_a
        cam_y = ay + d_depth * sin_a - d_lat * cos_a
        cam_z = az + d_vert

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

    def _get_3d_shapes(self):
        """Extract all bodies and extrude them into 3D shapes."""
        shapes_3d = []
        env = self.env

        for body in env.world.bodies:
            if body.userData is None:
                continue

            z_base = body.userData.get("z", 0.0)
            z_top = z_base + 0.5
            color = env.unwrapped.render_color(body.userData)
            no_fpv = body.userData.get("no_fpv", False)
            body_type = body.userData.get("type", "unknown")

            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    vertices_2d = [body.transform * v for v in shape.vertices]
                    shapes_3d.extend(
                        self._extrude_polygon(vertices_2d, z_base, z_top, color, no_fpv, body_type)
                    )

                elif isinstance(shape, b2CircleShape):
                    pos = body.transform * shape.pos
                    r = shape.radius
                    # Approximate circle with octagon
                    n_sides = 8
                    verts_bottom = []
                    verts_top = []
                    for i in range(n_sides):
                        angle = 2 * math.pi * i / n_sides
                        x = pos[0] + r * math.cos(angle)
                        y = pos[1] + r * math.sin(angle)
                        verts_bottom.append((x, y, z_base))
                        verts_top.append((x, y, z_top))

                    shapes_3d.append({
                        "verts": verts_top,
                        "color": color,
                        "z_sort": z_top,
                        "no_fpv": no_fpv,
                        "type": body_type,
                    })

                    for i in range(n_sides):
                        j = (i + 1) % n_sides
                        side_verts = [
                            verts_bottom[i],
                            verts_bottom[j],
                            verts_top[j],
                            verts_top[i],
                        ]
                        side_color = (
                            max(0, color[0] - 30),
                            max(0, color[1] - 30),
                            max(0, color[2] - 30),
                        )
                        shapes_3d.append({
                            "verts": side_verts,
                            "color": side_color,
                            "z_sort": z_base + 0.25,
                            "no_fpv": no_fpv,
                            "type": body_type,
                        })

        # Turrets have no physics body: extrude the env's turret block + barrel polygons
        for vertices_2d, color, turret_type, z_base, z_top in env.unwrapped.get_turret_polygons():
            shapes_3d.extend(
                self._extrude_polygon(vertices_2d, z_base, z_top, color, False, turret_type)
            )

        return shapes_3d

    def _extrude_polygon(self, vertices_2d, z_base, z_top, color, no_fpv, body_type):
        """Extrude a 2D world-space polygon into a 3D prism (top face + darkened side faces)."""
        faces = []
        verts_bottom = [(v[0], v[1], z_base) for v in vertices_2d]
        verts_top = [(v[0], v[1], z_top) for v in vertices_2d]

        # Top face
        faces.append({
            "verts": verts_top,
            "color": color,
            "z_sort": z_top,
            "no_fpv": no_fpv,
            "type": body_type,
        })

        # Side faces
        n = len(vertices_2d)
        for i in range(n):
            j = (i + 1) % n
            side_verts = [
                verts_bottom[i],
                verts_bottom[j],
                verts_top[j],
                verts_top[i],
            ]
            # Darken sides
            side_color = (
                max(0, color[0] - 30),
                max(0, color[1] - 30),
                max(0, color[2] - 30),
            )
            faces.append({
                "verts": side_verts,
                "color": side_color,
                "z_sort": (z_base + z_top) / 2,
                "no_fpv": no_fpv,
                "type": body_type,
            })
        return faces

    def _project_top_down(self, shapes, player_pos):
        """Top-down view: XY plane, centered on player."""
        surface = pygame.Surface((self.view_w, self.view_h))
        surface.fill(COLOR_GROUND)

        px, py = player_pos
        scale = 15.0  # pixels per meter

        # Sort by z (painter's algorithm)
        sorted_shapes = sorted(shapes, key=lambda s: s["z_sort"])

        for shape in sorted_shapes:
            verts = shape["verts"]
            # Project: x -> screen_x, y -> screen_y (top-down)
            points = []
            for v in verts:
                sx = int((v[0] - px) * scale + self.view_w / 2)
                sy = int(self.view_h / 2 - (v[1] - py) * scale)
                points.append((sx, sy))

            if len(points) >= 3:
                # Clip to screen
                clipped = self._clip_polygon_to_screen(points)
                if len(clipped) >= 3:
                    pygame.draw.polygon(surface, shape["color"], clipped)
                    pygame.gfxdraw.aapolygon(surface, clipped, (0, 0, 0))

        self._draw_explosions(
            surface,
            lambda x, y, z: ((x - px) * scale + self.view_w / 2, self.view_h / 2 - (y - py) * scale),
            scale, scale,
        )

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _draw_explosions(self, surface, to_screen, scale_x, scale_y):
        """Overlay the env's explosion fireballs on an orthographic view.
        to_screen maps world (x, y, z) to pixel (sx, sy); scale_x / scale_y are that view's pixels per metre."""
        for x, y, z, r, rgba in self.env.unwrapped.get_explosion_layers():
            sx, sy = to_screen(x, y, z)
            draw_fireball(surface, sx, sy, r * scale_x, r * scale_y, rgba)

    def _project_rear(self, shapes, player_pos, player_angle):
        """Rear view: looking from behind the player (XZ plane from behind)."""
        surface = pygame.Surface((self.view_w, self.view_h))
        surface.fill(COLOR_SKY)

        # Draw ground
        pygame.draw.rect(
            surface, COLOR_GROUND, (0, self.view_h // 2, self.view_w, self.view_h // 2)
        )

        px, py = player_pos
        # Rear view: camera behind player, looking forward
        cam_angle = player_angle + math.pi  # looking from behind

        scale = 30.0
        sorted_shapes = sorted(shapes, key=lambda s: s["z_sort"])

        for shape in sorted_shapes:
            verts = shape["verts"]
            points = []
            for v in verts:
                # Transform to player-relative coords
                dx = v[0] - px
                dy = v[1] - py
                # Rotate by -cam_angle to get view-space
                rx = dx * math.cos(-cam_angle) - dy * math.sin(-cam_angle)
                ry = dx * math.sin(-cam_angle) + dy * math.cos(-cam_angle)
                rz = v[2]

                # In rear view: rx is lateral, rz is vertical, ry is depth
                sx = int(rx * scale + self.view_w / 2)
                sy = int(self.view_h - rz * scale * 2 - self.view_h * 0.3)
                points.append((sx, sy))

            if len(points) >= 3:
                clipped = self._clip_polygon_to_screen(points)
                if len(clipped) >= 3:
                    pygame.draw.polygon(surface, shape["color"], clipped)
                    pygame.gfxdraw.aapolygon(surface, clipped, (0, 0, 0))

        def rear_to_screen(x, y, z):
            dx, dy = x - px, y - py
            rx = dx * math.cos(-cam_angle) - dy * math.sin(-cam_angle)
            return (rx * scale + self.view_w / 2, self.view_h - z * scale * 2 - self.view_h * 0.3)

        self._draw_explosions(surface, rear_to_screen, scale, scale * 2)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _project_side(self, shapes, player_pos, player_angle):
        """Side view: looking from the right side of the player (YZ plane)."""
        surface = pygame.Surface((self.view_w, self.view_h))
        surface.fill(COLOR_SKY)

        pygame.draw.rect(
            surface, COLOR_GROUND, (0, self.view_h // 2, self.view_w, self.view_h // 2)
        )

        px, py = player_pos
        # Side view: camera to the right, looking left
        cam_angle = player_angle - math.pi / 2

        scale = 30.0
        sorted_shapes = sorted(shapes, key=lambda s: s["z_sort"])

        for shape in sorted_shapes:
            verts = shape["verts"]
            points = []
            for v in verts:
                dx = v[0] - px
                dy = v[1] - py
                # Rotate
                rx = dx * math.cos(-cam_angle) - dy * math.sin(-cam_angle)
                ry = dx * math.sin(-cam_angle) + dy * math.cos(-cam_angle)
                rz = v[2]

                # Side view: ry is lateral (forward/back), rz is vertical
                sx = int(ry * scale + self.view_w / 2)
                sy = int(self.view_h - rz * scale * 2 - self.view_h * 0.3)
                points.append((sx, sy))

            if len(points) >= 3:
                clipped = self._clip_polygon_to_screen(points)
                if len(clipped) >= 3:
                    pygame.draw.polygon(surface, shape["color"], clipped)
                    pygame.gfxdraw.aapolygon(surface, clipped, (0, 0, 0))

        def side_to_screen(x, y, z):
            dx, dy = x - px, y - py
            ry = dx * math.sin(-cam_angle) + dy * math.cos(-cam_angle)
            return (ry * scale + self.view_w / 2, self.view_h - z * scale * 2 - self.view_h * 0.3)

        self._draw_explosions(surface, side_to_screen, scale, scale * 2)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _project_fpv(self, shapes, player_pos, player_angle):
        """3D FPV: third-person perspective projection from the orbit camera trailing the player tank."""
        surface = pygame.Surface((self.view_w, self.view_h))
        surface.fill(COLOR_SKY)

        # Draw ground
        pygame.draw.rect(
            surface, COLOR_GROUND, (0, self.view_h // 2, self.view_w, self.view_h // 2)
        )

        # Un-orbited camera anchor; orbit_camera_frame applies the orbit (cam_pos is the orbited position)
        cam_x, cam_y, cam_z = self._cam_anchor

        fov_scale = 400.0  # focal length in pixels

        # Transform and project shapes
        projected_shapes = []

        for shape in shapes:
            if shape.get("no_fpv", False):
                continue

            verts = shape["verts"]
            # Transform to camera space
            cam_verts = []
            for v in verts:
                dx = v[0] - cam_x
                dy = v[1] - cam_y
                dz = v[2] - cam_z

                # Rotate to camera frame (forward is player_angle)
                # Camera looks along player_angle direction
                cos_a = math.cos(player_angle)
                sin_a = math.sin(player_angle)
                # x_cam = right (sin a, -cos a), y_cam = forward/depth (cos a, sin a), z_cam = up
                x_cam = dx * sin_a - dy * cos_a
                y_cam = dx * cos_a + dy * sin_a
                z_cam = dz

                d_c, x_c, z_c = orbit_camera_frame(y_cam, x_cam, z_cam, self.cam_yaw, self.cam_pitch, self.pivot_depth)
                cam_verts.append((x_c, d_c, z_c))

            # Clip against near plane (y_cam > Z_NEAR)
            clipped_verts = self._clip_near_plane(cam_verts, Z_NEAR)
            if len(clipped_verts) < 3:
                continue

            # Project to screen
            points = []
            avg_depth = 0.0
            valid = True
            for cv in clipped_verts:
                if cv[1] <= 0:
                    valid = False
                    break
                sx = int(cv[0] / cv[1] * fov_scale + self.view_w / 2)
                sy = int(-cv[2] / cv[1] * fov_scale + self.view_h / 2)
                points.append((sx, sy))
                avg_depth += cv[1]

            if not valid or len(points) < 3:
                continue

            avg_depth /= len(clipped_verts)
            projected_shapes.append({
                "points": points,
                "color": shape["color"],
                "depth": avg_depth,
            })

        # Explosions: camera-facing fireball billboards (radius ~ 1 / depth). Each is sorted by the nearest
        # point of its full-grown sphere, so it envelops the struck hull / turret instead of hiding behind it.
        base_env = self.env.unwrapped
        cos_a = math.cos(player_angle)
        sin_a = math.sin(player_angle)
        for e in base_env.explosions:
            r_full = e["radius"][1]
            dx, dy, dz = e["x"] - cam_x, e["y"] - cam_y, e["z"] - cam_z
            d_c, x_c, z_c = orbit_camera_frame(
                dx * cos_a + dy * sin_a, dx * sin_a - dy * cos_a, dz, self.cam_yaw, self.cam_pitch, self.pivot_depth
            )
            if d_c <= Z_NEAR + r_full:
                continue  # fireball reaches the near plane (camera inside it): skip
            bx = x_c / d_c * fov_scale + self.view_w / 2
            by = -z_c / d_c * fov_scale + self.view_h / 2
            for r, rgba in base_env.explosion_layers(e):
                projected_shapes.append({
                    "billboard": (bx, by, r / d_c * fov_scale),
                    "color": rgba,
                    "depth": d_c - r_full,
                })

        # Sort by depth (far to near for painter's algorithm; stable, so fireball layers keep their order)
        projected_shapes.sort(key=lambda s: -s["depth"])

        for ps in projected_shapes:
            if "billboard" in ps:
                bx, by, br = ps["billboard"]
                draw_fireball(surface, bx, by, br, br, ps["color"])
                continue
            clipped = self._clip_polygon_to_screen(ps["points"])
            if len(clipped) >= 3:
                pygame.draw.polygon(surface, ps["color"], clipped)
                pygame.gfxdraw.aapolygon(surface, clipped, (0, 0, 0))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _clip_near_plane(self, verts, near):
        """Sutherland-Hodgman clipping against near plane (y >= near)."""
        if not verts:
            return []

        output = list(verts)
        # Clip against y >= near
        input_list = output
        output = []

        for i in range(len(input_list)):
            current = input_list[i]
            prev = input_list[i - 1]

            curr_inside = current[1] >= near
            prev_inside = prev[1] >= near

            if curr_inside:
                if not prev_inside:
                    # Entering
                    t = (near - prev[1]) / (current[1] - prev[1]) if (current[1] - prev[1]) != 0 else 0
                    t = max(0.0, min(1.0, t))
                    ix = prev[0] + t * (current[0] - prev[0])
                    iy = near
                    iz = prev[2] + t * (current[2] - prev[2])
                    output.append((ix, iy, iz))
                output.append(current)
            elif prev_inside:
                # Leaving
                t = (near - prev[1]) / (current[1] - prev[1]) if (current[1] - prev[1]) != 0 else 0
                t = max(0.0, min(1.0, t))
                ix = prev[0] + t * (current[0] - prev[0])
                iy = near
                iz = prev[2] + t * (current[2] - prev[2])
                output.append((ix, iy, iz))

        return output

    def _clip_polygon_to_screen(self, points):
        """Clip polygon to screen bounds using Sutherland-Hodgman."""
        def clip_edge(poly, edge_func):
            if not poly:
                return []
            output = []
            for i in range(len(poly)):
                current = poly[i]
                prev = poly[i - 1]
                curr_inside = edge_func(current)
                prev_inside = edge_func(prev)

                if curr_inside:
                    if not prev_inside:
                        output.append(intersect(prev, current, edge_func))
                    output.append(current)
                elif prev_inside:
                    # intersect() expects (outside, inside); passing (inside, outside) converged to the far
                    # off-screen point, which overflowed gfxdraw's int16 coords as stray lines
                    output.append(intersect(current, prev, edge_func))
            return output

        def intersect(p1, p2, edge_func):
            # Binary search for intersection (simple approach); p1 outside, p2 inside
            # For axis-aligned edges this is straightforward
            x1, y1 = p1
            x2, y2 = p2
            for _ in range(24):
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                if edge_func((mx, my)):
                    x2, y2 = mx, my
                else:
                    x1, y1 = mx, my
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))

        poly = list(points)
        # Clip left
        poly = clip_edge(poly, lambda p: p[0] >= 0)
        # Clip right
        poly = clip_edge(poly, lambda p: p[0] < self.view_w)
        # Clip top
        poly = clip_edge(poly, lambda p: p[1] >= 0)
        # Clip bottom
        poly = clip_edge(poly, lambda p: p[1] < self.view_h)

        return poly

    def render(self, camera_pose="Center"):
        """Returns tuple of 4 views: (top_down, rear, side, fpv)"""
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)

        if not pygame.get_init():
            pygame.init()

        env = self.env
        shapes = self._get_3d_shapes()

        player_pos = (env.player_body.position[0], env.player_body.position[1])

        im_top_down = self._project_top_down(shapes, player_pos)
        im_rear = self._project_rear(shapes, player_pos, env.player_body.angle)
        im_side = self._project_side(shapes, player_pos, env.player_body.angle)
        # Get dynamically calculated position from our property
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        im_fpv = self._project_fpv(shapes, player_pos, env.player_body.angle)

        return (im_top_down, im_rear, im_side, im_fpv)


# Main test
if __name__ == "__main__":
    pygame.init()

    env = TankDuelEnv(render_mode="rgb_array")
    wrapper = TankDuelOrthographicWrapper(env)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Create display for visualization
    display = pygame.display.set_mode((SCREEN_W * 2, SCREEN_H * 2))
    pygame.display.set_caption("Tank Duel - Orthographic Views")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0.0

    for step in range(500):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        # Random action
        action = env.action_space.sample()
        # Bias towards moving and firing
        action[0] = 0.5
        action[3] = 1.0 if step % 20 == 0 else -1.0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            views = wrapper.render()
            im_top, im_rear, im_side, im_fpv = views

            # Display all 4 views in a 2x2 grid
            surf_top = pygame.surfarray.make_surface(im_top.transpose(1, 0, 2))
            surf_rear = pygame.surfarray.make_surface(im_rear.transpose(1, 0, 2))
            surf_side = pygame.surfarray.make_surface(im_side.transpose(1, 0, 2))
            surf_fpv = pygame.surfarray.make_surface(im_fpv.transpose(1, 0, 2))

            display.blit(surf_top, (0, 0))
            display.blit(surf_rear, (SCREEN_W, 0))
            display.blit(surf_side, (0, SCREEN_H))
            display.blit(surf_fpv, (SCREEN_W, SCREEN_H))

            pygame.display.flip()
            clock.tick(30)

        if terminated or truncated:
            print(f"Episode ended at step {step}, total reward: {total_reward:.2f}")
            print(f"Player health: {info['player_health']}, Enemy health: {info['enemy_health']}")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
    pygame.quit()