try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
import math
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
from Box2D.b2 import (world, polygonShape, circleShape)

FPS = 50
SCALE = 30.0

# Arena dimensions in physics units (metres)
ARENA_W = 25.0   # X: 0 → 25
ARENA_H = 20.0   # Y: 0 → 20  (top-down, no gravity)
DRONE_ALT = 5.0  # Fixed 3D altitude for all drones (metres, used only in renderer)
WALL_HEIGHT = 3.0  # Visual height of 3D arena walls in renderer

# ==============================================================================
# 1. DRONE DOGFIGHT BASE ENVIRONMENT (Top-Down, No Gravity)
# ==============================================================================
class DroneDogfightEnv(gym.Env):
    """
    Drone Dogfight Task:
    Outmaneuver opposing quadrotor aerial units in a closed top-down arena.
    Drones move freely in the X-Y horizontal plane (no Z / altitude DOF).

    Actions (4D continuous):
      0: East-West Thrust   [-1..1] — +X = east, -X = west
      1: North-South Thrust [-1..1] — +Y = north, -Y = south
      2: Unused             (drones do not rotate)
      3: Fire Weapon        [>0.5]  — fires a bolt toward the opponent

    Observation (14D):
      0-3:   Player (X, Y, Vx, Vy)
      4-5:   Player Angle, Angular Velocity
      6-9:   Opponent (X, Y, Vx, Vy)
      10-11: Opponent Angle, Angular Velocity
      12:    Relative Distance to Opponent
      13:    Target Alignment Angle (rad)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        # Zero gravity — top-down horizontal arena
        self.world = world(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self.bullets = []
        self.boundaries = []
        self.max_steps = 600

    # ------------------------------------------------------------------
    def _make_drone(self, pos):
        """Create a quadrotor body with canopy + 4 diagonal arms. Rotation locked."""
        body = self.world.CreateDynamicBody(
            position=pos, angle=0.0,
            linearDamping=1.2, angularDamping=5.0,
            fixedRotation=True   # drones do not rotate
        )
        # Main canopy (oriented north-south in world frame)
        body.CreatePolygonFixture(box=(0.6, 0.25), density=1.5, friction=0.3)
        # 4 diagonal rotor arms (always at 45° in world frame)
        body.CreatePolygonFixture(box=(0.35, 0.06, ( 0.45,  0.30),  0.785), density=0.2)
        body.CreatePolygonFixture(box=(0.35, 0.06, (-0.45,  0.30), -0.785), density=0.2)
        body.CreatePolygonFixture(box=(0.35, 0.06, ( 0.45, -0.30), -0.785), density=0.2)
        body.CreatePolygonFixture(box=(0.35, 0.06, (-0.45, -0.30),  0.785), density=0.2)
        return body

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for body in self.bodies:
            self.world.DestroyBody(body)
        for b in self.bullets:
            self.world.DestroyBody(b['body'])
        self.bodies = []
        self.bullets = []
        self.boundaries = []

        self.step_count = 0
        self.opponent_hp = 100.0
        self.explosion   = None   # {'pos': (x,y), 'frames': int, 'radius': float}

        # Static arena walls (thin boxes on all 4 sides)
        wall_defs = [
            # (center_x, center_y, half_w, half_h)
            (ARENA_W / 2,       0.25,      ARENA_W / 2, 0.25),  # South wall
            (ARENA_W / 2, ARENA_H - 0.25, ARENA_W / 2, 0.25),  # North wall
            (0.25,        ARENA_H / 2,    0.25, ARENA_H / 2),   # West wall
            (ARENA_W - 0.25, ARENA_H / 2, 0.25, ARENA_H / 2),  # East wall
        ]
        for cx, cy, hw, hh in wall_defs:
            b = self.world.CreateStaticBody(
                position=(cx, cy),
                shapes=polygonShape(box=(hw, hh))
            )
            self.bodies.append(b)
            self.boundaries.append(b)

        # Player drone — west side; opponent — east side
        self.player   = self._make_drone((4.0, ARENA_H / 2))
        self.bodies.append(self.player)
        self.player_alive = True
        self.player_hp    = 20.0
        self.player_last_pos = (4.0, ARENA_H / 2)

        self.opponent = self._make_drone((ARENA_W - 4.0, ARENA_H / 2))
        self.bodies.append(self.opponent)
        self.opponent_alive = True
        self.opponent_hp    = 20.0
        self.opp_last_pos   = (ARENA_W - 4.0, ARENA_H / 2)

        # Explosion state per drone
        self.player_explosion = None   # {'pos','frames','radius'}
        self.opp_explosion    = None
        # Respawn countdown timers (frames)
        self.player_respawn_timer = 0
        self.opp_respawn_timer    = 0

        # Opponent bullets (fires back at player)
        self.opp_bullets = []

        self.score  = 0   # player kills
        self.deaths = 0   # player deaths

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _trigger_explosion(self, pos, owner):
        """Trigger an orange explosion at pos for the given owner ('player'/'opp')."""
        ex = {'pos': pos, 'frames': 14, 'radius': 2.0}
        if owner == 'player':
            self.player_explosion = ex
        else:
            self.opp_explosion = ex

    def _respawn_player(self):
        body = self._make_drone((4.0, ARENA_H / 2))
        self.bodies.append(body)
        self.player       = body
        self.player_alive = True
        self.player_hp    = 20.0

    def _respawn_opponent(self):
        import random
        spawn_x = float(np.random.uniform(ARENA_W * 0.55, ARENA_W - 3.0))
        spawn_y = float(np.random.uniform(2.0, ARENA_H - 2.0))
        body = self._make_drone((spawn_x, spawn_y))
        self.bodies.append(body)
        self.opponent       = body
        self.opponent_alive = True
        self.opponent_hp    = 20.0
        self.opp_last_pos   = (spawn_x, spawn_y)

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1
        reward = 0.0

        # ---- Decay explosions & countdown respawn timers ----
        for attr in ('player_explosion', 'opp_explosion'):
            ex = getattr(self, attr)
            if ex is not None:
                ex['frames'] -= 1
                if ex['frames'] <= 0:
                    setattr(self, attr, None)

        if not self.player_alive:
            self.player_respawn_timer -= 1
            if self.player_respawn_timer <= 0:
                self._respawn_player()

        if not self.opponent_alive:
            self.opp_respawn_timer -= 1
            if self.opp_respawn_timer <= 0:
                self._respawn_opponent()

        # ---- Player control (world-axis, no rotation) ----
        if self.player_alive:
            x_thrust = float(action[0]) * 25.0
            y_thrust = float(action[1]) * 25.0
            self.player.ApplyForceToCenter((x_thrust, y_thrust), wake=True)
            self.player_last_pos = (float(self.player.position.x), float(self.player.position.y))

        # ---- Opponent AI — sinusoidal evasion + fire back ----
        if self.opponent_alive:
            opp_t = self.step_count * 0.04
            target_x = 14.0 + 5.0 * math.cos(opp_t * 0.9)
            target_y = ARENA_H / 2 + 5.0 * math.sin(opp_t)
            dx_opp = float(target_x - self.opponent.position.x)
            dy_opp = float(target_y - self.opponent.position.y)
            self.opponent.ApplyForceToCenter((dx_opp * 10.0, dy_opp * 10.0), wake=True)
            self.opp_last_pos = (float(self.opponent.position.x), float(self.opponent.position.y))

            # Opponent fires west (-X) at player every 8 steps
            if self.player_alive and self.step_count % 8 == 0:
                ox = float(self.opponent.position.x - 1.2)
                oy = float(self.opponent.position.y)
                ob = self.world.CreateDynamicBody(position=(ox, oy))
                ob.CreateCircleFixture(radius=0.12, density=0.05)
                ob.linearVelocity = (-50.0, 0.0)   # fires west
                self.opp_bullets.append({'body': ob, 'life': 28})

        # ---- Player weapon — fires east (+X) ----
        if self.player_alive and float(action[3]) > 0.5 and self.step_count % 5 == 0:
            bx = float(self.player.position.x + 1.2)
            by = float(self.player.position.y)
            bullet = self.world.CreateDynamicBody(position=(bx, by))
            bullet.CreateCircleFixture(radius=0.12, density=0.05)
            bullet.linearVelocity = (50.0, 0.0)
            self.bullets.append({'body': bullet, 'life': 28})

        # ---- Player bullets vs opponent ----
        active_p = []
        for b in self.bullets:
            b['life'] -= 1
            if b['life'] > 0 and self.opponent_alive:
                bpos = np.array(b['body'].position)
                if float(np.linalg.norm(bpos - np.array(self.opponent.position))) < 1.0:
                    self.opponent_hp -= 20.0
                    reward += 15.0
                    hit_pos = (float(b['body'].position.x), float(b['body'].position.y))
                    self.world.DestroyBody(b['body'])
                    self._trigger_explosion(hit_pos, 'opp')
                    if self.opponent_hp <= 0:
                        # Opponent killed — destroy body, schedule respawn
                        self.world.DestroyBody(self.opponent)
                        self.bodies.remove(self.opponent)
                        self.opponent_alive    = False
                        self.opponent_hp       = 0.0
                        self.opp_respawn_timer = 40   # ~0.8s
                        reward += 50.0
                        self.score += 1
                        # Clear stale opp bullets
                        for ob in self.opp_bullets:
                            self.world.DestroyBody(ob['body'])
                        self.opp_bullets = []
                    continue   # bullet consumed
            elif b['life'] <= 0:
                self.world.DestroyBody(b['body'])
                continue
            active_p.append(b)
        self.bullets = active_p

        # ---- Opponent bullets vs player ----
        active_o = []
        for b in self.opp_bullets:
            b['life'] -= 1
            if b['life'] > 0 and self.player_alive:
                bpos = np.array(b['body'].position)
                if float(np.linalg.norm(bpos - np.array(self.player.position))) < 1.0:
                    self.player_hp -= 20.0
                    reward -= 5.0
                    hit_pos = (float(b['body'].position.x), float(b['body'].position.y))
                    self.world.DestroyBody(b['body'])
                    self._trigger_explosion(hit_pos, 'player')
                    if self.player_hp <= 0:
                        # Player killed — destroy body, schedule respawn
                        self.world.DestroyBody(self.player)
                        self.bodies.remove(self.player)
                        self.player_alive        = False
                        self.player_hp           = 0.0
                        self.player_respawn_timer = 40
                        reward -= 50.0
                        self.deaths += 1
                        # Clear stale player bullets
                        for pb in self.bullets:
                            self.world.DestroyBody(pb['body'])
                        self.bullets = []
                    continue
            elif b['life'] <= 0:
                self.world.DestroyBody(b['body'])
                continue
            active_o.append(b)
        self.opp_bullets = active_o

        self.world.Step(1.0 / FPS, 6, 2)
        obs = self._get_obs()

        # Proximity incentive (only when both alive)
        terminated = False
        if self.player_alive and self.opponent_alive:
            p_pos  = np.array(self.player.position)
            op_pos = np.array(self.opponent.position)
            dist   = float(np.linalg.norm(op_pos - p_pos)) + 1e-5
            reward += -0.001 * dist
            # Out of bounds
            if (p_pos[0] <= 0.5 or p_pos[0] >= ARENA_W - 0.5 or
                    p_pos[1] <= 0.5 or p_pos[1] >= ARENA_H - 0.5):
                reward -= 50.0
                terminated = True

        if self.step_count >= self.max_steps:
            terminated = True

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False,\
               {'score': self.score, 'deaths': self.deaths,
                'player_hp': self.player_hp, 'opp_hp': self.opponent_hp}

    # ------------------------------------------------------------------
    def _get_obs(self):
        # Player state
        if self.player_alive:
            p_pos = self.player.position
            px, py = float(p_pos.x), float(p_pos.y)
            pvx = float(self.player.linearVelocity.x)
            pvy = float(self.player.linearVelocity.y)
            pa  = float(self.player.angle)
            pav = float(self.player.angularVelocity)
        else:
            px, py, pvx, pvy, pa, pav = self.player_last_pos[0], self.player_last_pos[1], 0,0,0,0

        # Opponent state
        if self.opponent_alive:
            op_pos = self.opponent.position
            opx, opy = float(op_pos.x), float(op_pos.y)
            opvx = float(self.opponent.linearVelocity.x)
            opvy = float(self.opponent.linearVelocity.y)
            opa  = float(self.opponent.angle)
            opav = float(self.opponent.angularVelocity)
        else:
            opx, opy, opvx, opvy, opa, opav = self.opp_last_pos[0], self.opp_last_pos[1], 0,0,0,0

        rel  = np.array([opx - px, opy - py])
        dist = float(np.linalg.norm(rel)) + 1e-5
        align = float(np.arccos(np.clip(np.dot(np.array([1.0, 0.0]), rel / dist), -1.0, 1.0)))

        return np.array([
            px,  py,  pvx,  pvy,  pa,  pav,
            opx, opy, opvx, opvy, opa, opav,
            dist, align
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
        if self.player_alive:
            for f in self.player.fixtures:
                verts = [(int((self.player.transform * v)[0] * SCALE),
                          int(sh - (self.player.transform * v)[1] * SCALE))
                         for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, (0, 220, 255), verts)
        if self.opponent_alive:
            for f in self.opponent.fixtures:
                verts = [(int((self.opponent.transform * v)[0] * SCALE),
                          int(sh - (self.opponent.transform * v)[1] * SCALE))
                         for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, (255, 60, 60), verts)

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
      FPV       : perspective from drone nose, looking along heading in X-Y plane
    """

    DRONE_Z_LO = (DRONE_ALT - 0.6)   # bottom extrusion (metres)
    DRONE_Z_HI = (DRONE_ALT + 0.6)   # top extrusion   (metres)

    def __init__(self, env):
        super().__init__(env)
        self.SCALE = SCALE
        self.VIEWPORT_W = 800
        self.VIEWPORT_H = 600
        self.fov = 400.0

    # ------------------------------------------------------------------
    def _surf_to_img(self, surf):
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    # ------------------------------------------------------------------
    def render(self):
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
        horizon_y = int(VH / 2) if 'VH' in dir() else int(self.VIEWPORT_H / 2)
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
                b_verts = [
                    (bx - ux, by - uy, z_lo),
                    (bx + ux, by + uy, z_lo),
                    (bx + ux, by + uy, z_hi),
                    (bx - ux, by - uy, z_hi),
                ]
                polys_3d.append((b_verts, b_color, 3))

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
        # Camera — FPV at drone nose, looking along heading (in X-Y plane)
        p_angle = float(base.player.angle)
        p_x = base.player.position.x * S
        p_y = base.player.position.y * S
        p_z = DRONE_ALT * S    # fixed altitude

        nose_offset = 1.2 * S
        cam_x = p_x + nose_offset * math.cos(p_angle)
        cam_y = p_y + nose_offset * math.sin(p_angle)
        cam_z = p_z  # eye level at drone altitude

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

            # FPV — skip player self-body and env geometry (floor/walls)
            if is_player or no_fpv:
                continue

            # Sutherland-Hodgman near-plane clip (rx >= 1.0)
            rel_verts = []
            for vx, vy, vz in p_verts:
                dx, dy = vx - cam_x, vy - cam_y
                dz = vz - cam_z
                # Rotate into camera space (rotate around Z-axis)
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                rel_verts.append((rx, ry, dz))

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
            for rx, ry, rz in clipped:
                # Project: horizontal = -ry/rx*fov, vertical = -rz/rx*fov (Z is up)
                px_f = -(ry / rx) * self.fov + cx
                py_f = -(rz / rx) * self.fov + cy
                pts_fpv.append((px_f, py_f))
                df.append(rx)

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
