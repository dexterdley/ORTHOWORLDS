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
from Box2D.b2 import (world, polygonShape)

FPS      = 50
SCALE    = 28.0        # pixels per metre
GRAVITY  = -50.0       # strong gravity for snappy Mario feel

# Level dimensions
LEVEL_W  = 72.0        # metres wide
LEVEL_H  = 18.0        # metres tall

# Physics half-sizes of the player box
P_HW, P_HH = 0.38, 0.75   # half-width, half-height  → 0.76m × 1.5m character

# 3D extrusion depth (front-back, Z axis)
Z_CHAR_LO, Z_CHAR_HI   = -0.40,  0.40   # character depth
Z_PLAT_LO, Z_PLAT_HI   = -1.50,  1.50   # platform depth

# Colours — classic Mario palette
SKY_C    = (92, 148, 252)       # Mario-blue sky
GROUND_C = (101, 67, 33)        # brown ground body
GRASS_C  = (80, 180, 40)        # green grass top
PLAT_C   = (185, 122, 68)       # warm brick
PLAT_TOP = (210, 148, 90)       # brick highlight
FLAG_C   = (255, 215, 0)        # gold flagpole
GOAL_C   = (255, 215, 0)        # yellow goal flag

MARIO_HAT   = (205,   0,   0)
MARIO_FACE  = (250, 190, 130)
MARIO_SHIRT = (205,   0,   0)
MARIO_OVR   = (  0,  80, 200)
MARIO_SHOE  = (100,  60,  30)

# Ground segments and platforms as (cx, cy, hw, hh)
GROUND_SEGS = [
    (10.0,  -0.4, 10.0, 0.4),    # x = 0 .. 20
    (27.5,  -0.4,  5.5, 0.4),    # x = 22 .. 33   (pit at 20-22)
    (41.5,  -0.4,  6.5, 0.4),    # x = 35 .. 48
    (55.0,  -0.4,  5.0, 0.4),    # x = 50 .. 60
    (66.5,  -0.4,  5.5, 0.4),    # x = 61 .. 72   (exit area)
]

# Floating platforms (cx, cy, hw, hh)
PLATFORMS = [
    (19.0,  3.0,  2.0, 0.35),    # over/before pit 1
    (22.5,  5.0,  1.5, 0.35),    # platform bridge
    (25.0,  3.5,  1.5, 0.35),    # landing pad
    (34.0,  3.0,  1.5, 0.35),    # before pit 2
    (37.0,  5.5,  2.0, 0.35),    # mid-air
    (48.5,  3.5,  1.5, 0.35),    # over gap
    (52.0,  6.0,  2.0, 0.35),    # high platform
    (60.0,  4.0,  1.5, 0.35),    # before pit 4
    (63.0,  2.5,  1.5, 0.35),    # final step
]

FLAGPOLE_X = 70.0
FLAGPOLE_Y_TOP = 10.0    # top of pole

SPAWN_X, SPAWN_Y = 2.0, 1.0   # Mario spawn pos (body centre)
EXIT_DIST = 1.8                # metres from flagpole to win

# ==============================================================================
# 1. MARIO BASE ENVIRONMENT
# ==============================================================================
class MarioEscapeEnv(gym.Env):
    """
    Mario Escape — simplified side-scrolling platformer.

    Goal: Reach the flagpole at the far right of the level.

    Actions (2D continuous):
      0: Horizontal force  [-1..1]  (+1 = move right, -1 = move left)
      1: Jump              [>0.5]   fires a jump impulse when grounded

    Observation (10D):
      0-3: Player X, Y, Vx, Vy
      4  : Player angle (should stay ~0 with fixedRotation)
      5  : Is grounded (1.0 / 0.0)
      6  : X distance to flagpole
      7  : Y distance to flagpole top
      8  : Distance to nearest pit edge (approximate)
      9  : Step fraction (step_count / max_steps)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.screen      = None
        self.clock       = None

        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

        self.world   = world(gravity=(0, GRAVITY), doSleep=True)
        self.bodies  = []
        self.grounds = []     # ground segment bodies
        self.plats   = []     # platform bodies
        self.max_steps = 1200
        self.step_count = 0
        self.is_grounded = False

    # ------------------------------------------------------------------
    def _build_static(self, specs, color_tag):
        result = []
        for cx, cy, hw, hh in specs:
            b = self.world.CreateStaticBody(
                position=(cx, cy), shapes=polygonShape(box=(hw, hh)))
            b._color_tag = color_tag
            self.bodies.append(b)
            result.append(b)
        return result

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for b in self.bodies:
            self.world.DestroyBody(b)
        self.bodies  = []
        self.grounds = []
        self.plats   = []

        self.step_count  = 0
        self.is_grounded = False
        self.facing_right = True

        self.grounds = self._build_static(GROUND_SEGS, 'ground')
        self.plats   = self._build_static(PLATFORMS,   'platform')

        # Flagpole (thin vertical static body)
        pole = self.world.CreateStaticBody(
            position=(FLAGPOLE_X, FLAGPOLE_Y_TOP / 2),
            shapes=polygonShape(box=(0.1, FLAGPOLE_Y_TOP / 2)))
        pole._color_tag = 'flag'
        self.bodies.append(pole)
        self.flagpole = pole

        # West boundary wall (gray static body)
        west_w = self.world.CreateStaticBody(
            position=(-0.2, LEVEL_H / 2),
            shapes=polygonShape(box=(0.2, LEVEL_H / 2)))
        west_w._color_tag = 'wall'
        self.bodies.append(west_w)

        # Mario player body
        self.player = self.world.CreateDynamicBody(
            position=(SPAWN_X, SPAWN_Y),
            linearDamping=0.4,
            angularDamping=10.0,
            fixedRotation=True)
        self.player.CreatePolygonFixture(box=(P_HW, P_HH), density=1.2, friction=0.8)
        self.bodies.append(self.player)

        self.prev_x = float(self.player.position.x)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # --- Strict grounded detection: low vy + touching ground/platform from above (disables double jump)
        vy = float(self.player.linearVelocity.y)
        touching_ground = False
        if abs(vy) < 1.0:
            for ce in self.player.contacts:
                if ce.contact.touching:
                    other = ce.other
                    if getattr(other, '_color_tag', None) in ('ground', 'platform'):
                        if float(self.player.position.y) > float(other.position.y + 0.1):
                            touching_ground = True
                            break
        self.is_grounded = touching_ground

        # --- Horizontal movement
        h_force = float(action[0]) * 220.0
        self.player.ApplyForceToCenter((h_force, 0.0), wake=True)

        # --- Jump (impulse when grounded)
        if float(action[1]) > 0.5 and self.is_grounded:
            # Zero out downward velocity first, then apply upward impulse
            self.player.linearVelocity = (self.player.linearVelocity.x, 0.0)
            mass = self.player.mass
            self.player.ApplyLinearImpulse(
                (0.0, mass * 22.0),           # jump strength
                self.player.worldCenter, True)

        self.world.Step(1.0 / FPS, 8, 3)
        obs = self._get_obs()

        vx = float(self.player.linearVelocity.x)
        if vx > 0.15:
            self.facing_right = True
        elif vx < -0.15:
            self.facing_right = False

        p   = np.array(self.player.position)
        px, py = float(p[0]), float(p[1])

        # --- Reward
        progress = px - self.prev_x
        reward   = progress * 3.0 - 0.02    # forward progress – time cost
        self.prev_x = px

        terminated = False

        # Success: reached flagpole
        if abs(px - FLAGPOLE_X) < EXIT_DIST:
            reward += 200.0
            terminated = True

        # Failure: fell into pit
        elif py < -3.0:
            reward -= 80.0
            terminated = True

        # Timeout
        elif self.step_count >= self.max_steps:
            terminated = True

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        p  = self.player.position
        v  = self.player.linearVelocity
        dx = FLAGPOLE_X - float(p.x)
        dy = FLAGPOLE_Y_TOP - float(p.y)
        # Nearest pit: approximate as distance to nearest ground-segment edge
        # (simple: just use X distance to flagpole as proxy)
        frac = self.step_count / self.max_steps
        return np.array([
            float(p.x), float(p.y),
            float(v.x), float(v.y),
            float(self.player.angle),
            float(self.is_grounded),
            dx, dy,
            0.0,   # reserved
            frac
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def render(self):
        """2D debug render (side-scrolling view)."""
        if self.render_mode is None:
            return
        sw, sh = 900, 500
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((sw, sh))
            else:
                self.screen = pygame.Surface((sw, sh))
            self.clock = pygame.time.Clock()

        self.screen.fill(SKY_C)
        cx  = sw // 2
        p_x = float(self.player.position.x) * SCALE   # player screen X
        y_shift = 80   # vertical north shift in pixels

        def wx(world_x): return int(world_x * SCALE - p_x + cx)
        def wy(world_y): return int(sh - y_shift - world_y * SCALE)

        # Draw green grass floor below ground level (y=0)
        pygame.draw.rect(self.screen, GRASS_C, (0, wy(0), sw, sh - wy(0)))
        pygame.draw.line(self.screen, (100, 210, 60), (0, wy(0)), (sw, wy(0)), 3)

        for b in self.grounds + self.plats:
            tag = getattr(b, '_color_tag', 'ground')
            for f in b.fixtures:
                verts = [(wx((b.transform * v)[0]), wy((b.transform * v)[1]))
                         for v in f.shape.vertices]
                col = GROUND_C if tag == 'ground' else PLAT_C
                pygame.draw.polygon(self.screen, col, verts)

        # West Wall (translucent gray)
        ww_x, ww_y = wx(-0.4), wy(LEVEL_H)
        ww_w, ww_h = int(0.4 * SCALE), wy(0) - wy(LEVEL_H)
        if ww_w > 0 and ww_h > 0:
            w_surf = pygame.Surface((ww_w, ww_h), pygame.SRCALPHA)
            w_surf.fill((140, 145, 155, 120))   # translucent gray (alpha 120)
            self.screen.blit(w_surf, (ww_x, ww_y))
            pygame.draw.rect(self.screen, (90, 95, 105, 160), (ww_x, ww_y, ww_w, ww_h), 2)

        # Flagpole
        px_f, py_f = wx(FLAGPOLE_X), wy(FLAGPOLE_Y_TOP)
        pygame.draw.line(self.screen, (200, 200, 200), (px_f, wy(0)), (px_f, py_f), 4)
        pygame.draw.polygon(self.screen, GOAL_C, [
            (px_f, py_f), (px_f + 20, py_f + 10), (px_f, py_f + 20)])

        # Mario
        mx = wx(float(self.player.position.x))
        my = wy(float(self.player.position.y) + P_HH)
        vx = float(self.player.linearVelocity.x)
        facing_right = getattr(self, 'facing_right', True)
        swing = int(math.sin(self.step_count * 0.45) * 8.0) if (abs(vx) > 0.15 and self.is_grounded) else 0

        # Hat & Brim (facing East vs West)
        pygame.draw.rect(self.screen, MARIO_HAT, (mx - 14, my, 28, 10))
        if facing_right:
            pygame.draw.rect(self.screen, MARIO_HAT,  (mx + 4, my + 4, 14, 6))
            pygame.draw.rect(self.screen, MARIO_FACE, (mx - 8, my + 10, 18, 14))
        else:
            pygame.draw.rect(self.screen, MARIO_HAT,  (mx - 18, my + 4, 14, 6))
            pygame.draw.rect(self.screen, MARIO_FACE, (mx - 10, my + 10, 18, 14))

        # Arms (left & right sleeves + white gloves with animated swing)
        pygame.draw.rect(self.screen, MARIO_SHIRT, (mx - 18 - swing//2, my + 24 + swing, 6, 12))
        pygame.draw.rect(self.screen, (250,250,250), (mx - 18 - swing//2, my + 36 + swing, 6, 6))
        pygame.draw.rect(self.screen, MARIO_SHIRT, (mx + 12 + swing//2, my + 24 - swing, 6, 12))
        pygame.draw.rect(self.screen, (250,250,250), (mx + 12 + swing//2, my + 36 - swing, 6, 6))

        # Torso & Legs
        pygame.draw.rect(self.screen, MARIO_SHIRT, (mx - 12, my + 24,  24, 14))
        pygame.draw.rect(self.screen, MARIO_OVR,   (mx - 12, my + 38,  24, 18))
        pygame.draw.rect(self.screen, MARIO_SHOE,  (mx - 14, my + 52,  14,  8))
        pygame.draw.rect(self.screen, MARIO_SHOE,  (mx +  0, my + 52,  14,  8))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


# ==============================================================================
# 2. 3D ORTHOGRAPHIC MULTI-VIEW WRAPPER
# ==============================================================================
class MarioOrthographicWrapper(gym.Wrapper):
    """
    Extrudes the 2D side-scrolling Mario level into a 3D volumetric scene.

    Coordinate convention (side-scrolling game):
      physics X   → 3D X  (horizontal, left-right)
      physics Y   → 3D Y  (vertical,   up-down with gravity)
      extrusion Z → 3D Z  (depth, front-back)

    Views:
      Top-Down  : X-Z plan view  (looking down from above along -Y)
      Rear      : Z-Y elevation  (looking along +X from behind)
      Side      : X-Y elevation  → the natural 2D game view
      FPV       : perspective from slightly behind Mario looking right
    """

    def __init__(self, env):
        super().__init__(env)
        self.SCALE      = SCALE
        self.VIEWPORT_W = 800
        self.VIEWPORT_H = 600
        self.fov        = 280.0

    # ------------------------------------------------------------------
    def _surf_to_img(self, surf):
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    @staticmethod
    def _draw(proj_list, surf):
        proj_list.sort(key=lambda x: (x[3], -x[0]))
        for _, pts, color, _ in proj_list:
            ipts = [(int(round(px)), int(round(py))) for px, py in pts]
            if len(ipts) >= 3:
                try:
                    if len(color) == 4 and color[3] < 255:
                        # Render translucent polygon
                        xs = [p[0] for p in ipts]; ys = [p[1] for p in ipts]
                        min_x, max_x = max(0, min(xs)), min(surf.get_width() - 1, max(xs))
                        min_y, max_y = max(0, min(ys)), min(surf.get_height() - 1, max(ys))
                        if max_x > min_x and max_y > min_y:
                            pw, ph = max_x - min_x + 1, max_y - min_y + 1
                            s_poly = pygame.Surface((pw, ph), pygame.SRCALPHA)
                            rel_pts = [(px - min_x, py - min_y) for px, py in ipts]
                            gfxdraw.aapolygon(s_poly, rel_pts, color)
                            gfxdraw.filled_polygon(s_poly, rel_pts, color)
                            surf.blit(s_poly, (min_x, min_y))
                    else:
                        gfxdraw.aapolygon(surf, ipts, color[:3])
                        gfxdraw.filled_polygon(surf, ipts, color[:3])
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _add_box(self, polys, x0, x1, y0, y1, z0, z1,
                 c_front, c_top, c_side, layer, no_fpv=False):
        """Append all 6 faces of an axis-aligned box to polys."""
        S = self.SCALE
        # Front face (z=z0*S)
        polys.append(([
            (x0*S, y0*S, z0*S), (x1*S, y0*S, z0*S),
            (x1*S, y1*S, z0*S), (x0*S, y1*S, z0*S)
        ], c_front, layer, no_fpv))
        # Back face (z=z1*S)
        polys.append(([
            (x0*S, y0*S, z1*S), (x1*S, y0*S, z1*S),
            (x1*S, y1*S, z1*S), (x0*S, y1*S, z1*S)
        ], c_front, layer, no_fpv))
        # Top face
        polys.append(([
            (x0*S, y1*S, z0*S), (x1*S, y1*S, z0*S),
            (x1*S, y1*S, z1*S), (x0*S, y1*S, z1*S)
        ], c_top, layer, no_fpv))
        # Left face
        polys.append(([
            (x0*S, y0*S, z0*S), (x0*S, y0*S, z1*S),
            (x0*S, y1*S, z1*S), (x0*S, y1*S, z0*S)
        ], c_side, layer, no_fpv))
        # Right face
        polys.append(([
            (x1*S, y0*S, z0*S), (x1*S, y0*S, z1*S),
            (x1*S, y1*S, z1*S), (x1*S, y1*S, z0*S)
        ], c_side, layer, no_fpv))

    # ------------------------------------------------------------------
    def render(self):
        base = self.env.unwrapped

        VW, VH = self.VIEWPORT_W, self.VIEWPORT_H
        cx, cy = VW / 2.0, VH / 2.0
        S = self.SCALE

        top_surf  = pygame.Surface((VW, VH))
        rear_surf = pygame.Surface((VW, VH))
        side_surf = pygame.Surface((VW, VH))
        fpv_surf  = pygame.Surface((VW, VH))

        y_shift = 80   # pixels shifted north

        # Background colours
        top_surf.fill((70, 150, 50))      # top-down: green grass background
        rear_surf.fill(SKY_C)            # rear: sky
        pygame.draw.rect(rear_surf, GRASS_C, (0, VH - y_shift, VW, y_shift))
        pygame.draw.line(rear_surf, (100, 210, 60), (0, VH - y_shift), (VW, VH - y_shift), 2)

        side_surf.fill(SKY_C)            # side: sky
        pygame.draw.rect(side_surf, GRASS_C, (0, VH - y_shift, VW, y_shift))
        pygame.draw.line(side_surf, (100, 210, 60), (0, VH - y_shift), (VW, VH - y_shift), 2)

        # FPV: sky upper half, green grass lower half
        fpv_surf.fill(SKY_C)
        pygame.draw.rect(fpv_surf, GRASS_C, (0, VH // 2, VW, VH // 2))
        pygame.draw.line(fpv_surf, (100, 210, 60), (0, VH // 2), (VW, VH // 2), 2)

        if not hasattr(base, 'player'):
            return [np.zeros((VH, VW, 3), np.uint8)] * 4

        polys_3d = []

        # Player position (world metres, unscaled)
        p_x = float(base.player.position.x)
        p_y = float(base.player.position.y)
        p_x_s = p_x * S  # scaled
        p_y_s = p_y * S

        # ----------------------------------------------------------------
        # A. Sky + Ground backdrop strips for top-down & 3D  (no_fpv)
        # ----------------------------------------------------------------
        # Main lane (|z| <= 1.8) is brown; outer terrain is green grass
        tile = 2.0
        for xi, x0 in enumerate(np.arange(max(0, p_x-35), min(LEVEL_W, p_x+40), tile)):
            x1 = x0 + tile
            for zi in range(-15, 16):   # Wide terrain coverage
                z0, z1 = zi * tile, (zi + 1) * tile
                if abs((z0 + z1) / 2.0) <= 1.8:
                    c = (101, 67, 33) if (xi + zi) % 2 == 0 else (90, 58, 28)   # brown main lane
                else:
                    c = (70, 160, 45) if (xi + zi) % 2 == 0 else (60, 140, 38)   # green outer terrain
                polys_3d.append(([
                    (x0*S, 0, z0*S), (x1*S, 0, z0*S),
                    (x1*S, 0, z1*S), (x0*S, 0, z1*S)
                ], c, 0, False))

        # ----------------------------------------------------------------
        # B. West Boundary Wall (Translucent Gray extruded box at X=0)
        # ----------------------------------------------------------------
        self._add_box(polys_3d, -0.4, 0.0, 0.0, LEVEL_H,
                       Z_PLAT_LO - 1.0, Z_PLAT_HI + 1.0,
                       (140, 145, 155, 120), (160, 165, 175, 120), (110, 115, 125, 120), 2)

        # ----------------------------------------------------------------
        # C. Ground segments — extruded boxes (brown body & top)
        # ----------------------------------------------------------------
        for b in base.grounds:
            for f in b.fixtures:
                v2d = [(b.transform * v) for v in f.shape.vertices]
                xs  = [v[0] for v in v2d]; ys = [v[1] for v in v2d]
                x0b, x1b = min(xs), max(xs)
                y0b, y1b = min(ys), max(ys)
                # All brown for main lane
                self._add_box(polys_3d, x0b, x1b, y0b, y1b,
                               Z_PLAT_LO, Z_PLAT_HI,
                               GROUND_C, GROUND_C,
                               (80, 50, 20), 1, no_fpv=True)

        # ----------------------------------------------------------------
        # C. Floating Platforms — brick-coloured boxes
        # ----------------------------------------------------------------
        for b in base.plats:
            for f in b.fixtures:
                v2d = [(b.transform * v) for v in f.shape.vertices]
                xs  = [v[0] for v in v2d]; ys = [v[1] for v in v2d]
                x0b, x1b = min(xs), max(xs)
                y0b, y1b = min(ys), max(ys)
                self._add_box(polys_3d, x0b, x1b, y0b, y1b,
                               Z_PLAT_LO, Z_PLAT_HI,
                               PLAT_C, PLAT_TOP,
                               (160, 100, 55), 1)

        # ----------------------------------------------------------------
        # D. Flagpole + Flag
        # ----------------------------------------------------------------
        fx = FLAGPOLE_X
        self._add_box(polys_3d, fx-0.1, fx+0.1, 0.0, FLAGPOLE_Y_TOP,
                       -0.12, 0.12, (200, 200, 200), (220, 220, 220), (180, 180, 180), 2)
        # Flag triangle (front face only, approximate as quad)
        fq = [
            (fx*S,       FLAGPOLE_Y_TOP*S,        0),
            ((fx+2.5)*S, (FLAGPOLE_Y_TOP-1.0)*S,  0),
            ((fx+2.5)*S, (FLAGPOLE_Y_TOP-2.0)*S,  0),
            (fx*S,       (FLAGPOLE_Y_TOP-2.0)*S,  0),
        ]
        polys_3d.append((fq, GOAL_C, 3, False))

        # ----------------------------------------------------------------
        # E. Mario character — stacked coloured boxes
        # ----------------------------------------------------------------
        foot_y = p_y - P_HH    # Y at bottom of character
        mx     = p_x
        vx     = float(base.player.linearVelocity.x)
        facing_right = getattr(base, 'facing_right', True)
        is_moving = abs(vx) > 0.15 and base.is_grounded
        swing_off = math.sin(base.step_count * 0.45) * 0.18 if is_moving else 0.0

        def mario_box(y_lo, y_hi, x_pad_l, x_pad_r, color_f, color_t, layer):
            self._add_box(polys_3d,
                          mx - P_HW + x_pad_l,
                          mx + P_HW - x_pad_r,
                          foot_y + y_lo,
                          foot_y + y_hi,
                          Z_CHAR_LO, Z_CHAR_HI,
                          color_f, color_t, color_f, layer, no_fpv=True)

        mario_box(0.00, 0.22, 0.00, 0.00, MARIO_SHOE,  (120, 75, 40),  2)   # Shoes
        mario_box(0.22, 0.72, 0.00, 0.00, MARIO_OVR,   (20,  100, 230), 2)  # Overalls/legs
        mario_box(0.72, 1.08, 0.02, 0.02, MARIO_SHIRT, (230, 20,  20),  2)  # Shirt
        mario_box(1.08, 1.42, 0.04, 0.04, MARIO_FACE,  (255, 210, 150), 2)  # Face/head

        # Arms (left & right sleeves + white gloves with animated swing)
        # Left arm & glove
        self._add_box(polys_3d, mx - P_HW - 0.14 - swing_off, mx - P_HW + 0.02 - swing_off, foot_y + 0.74, foot_y + 1.04, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, MARIO_SHIRT, (230, 20, 20), MARIO_SHIRT, 3, no_fpv=True)
        self._add_box(polys_3d, mx - P_HW - 0.14 - swing_off, mx - P_HW + 0.02 - swing_off, foot_y + 0.60, foot_y + 0.74, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, (250, 250, 250), (250, 250, 250), (220, 220, 220), 3, no_fpv=True)
        # Right arm & glove
        self._add_box(polys_3d, mx + P_HW - 0.02 + swing_off, mx + P_HW + 0.14 + swing_off, foot_y + 0.74, foot_y + 1.04, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, MARIO_SHIRT, (230, 20, 20), MARIO_SHIRT, 3, no_fpv=True)
        self._add_box(polys_3d, mx + P_HW - 0.02 + swing_off, mx + P_HW + 0.14 + swing_off, foot_y + 0.60, foot_y + 0.74, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, (250, 250, 250), (250, 250, 250), (220, 220, 220), 3, no_fpv=True)

        # Cap — wider than head and slightly overhanging
        self._add_box(polys_3d,
                      mx - P_HW - 0.05,
                      mx + P_HW + 0.05,
                      foot_y + 1.30,
                      foot_y + 1.55,
                      Z_CHAR_LO - 0.05, Z_CHAR_HI + 0.05,
                      MARIO_HAT, (230, 10, 10), MARIO_HAT, 3, no_fpv=True)
        # Cap brim (front overhang depending on facing direction)
        if facing_right:
            self._add_box(polys_3d,
                          mx + P_HW - 0.05,
                          mx + P_HW + 0.18,
                          foot_y + 1.28,
                          foot_y + 1.38,
                          Z_CHAR_LO - 0.05, Z_CHAR_HI + 0.05,
                          MARIO_HAT, (230, 10, 10), MARIO_HAT, 3, no_fpv=True)
        else:
            self._add_box(polys_3d,
                          mx - P_HW - 0.18,
                          mx - P_HW + 0.05,
                          foot_y + 1.28,
                          foot_y + 1.38,
                          Z_CHAR_LO - 0.05, Z_CHAR_HI + 0.05,
                          MARIO_HAT, (230, 10, 10), MARIO_HAT, 3, no_fpv=True)

        # ----------------------------------------------------------------
        # F. Multi-View Projections
        # ----------------------------------------------------------------
        # FPV camera: placed at Mario's head/eyes, looking along facing direction
        cam_x = p_x_s
        cam_y = (p_y + P_HH - 0.15) * S   # eye height
        cam_z = 0.0                        # center depth

        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []

        for poly_item in polys_3d:
            if len(poly_item) == 4:
                p_verts, color, layer, no_fpv = poly_item
            else:
                p_verts, color, layer = poly_item
                no_fpv = False

            pts_t, pts_r, pts_s = [], [], []
            dt, dr, ds = [], [], []

            for vx, vy, vz in p_verts:
                sx = vx - p_x_s    # scroll to centre player

                # Side Elevation (the actual 2D Mario view): X horizontal, Y vertical (shifted north)
                pts_s.append((sx + cx, VH - y_shift - vy))
                ds.append(vz)

                # Top-Down (looking down -Y): X horizontal, Z depth
                pts_t.append((sx + cx, vz + cy))
                dt.append(-vy)

                # Rear (looking right +X): Z depth horizontal, Y vertical (shifted north)
                pts_r.append((vz + cx, VH - y_shift - vy))
                dr.append(sx)

            if len(pts_s) >= 3: proj_side.append((sum(ds)/len(ds), pts_s, color, layer))
            if len(pts_t) >= 3: proj_top.append( (sum(dt)/len(dt), pts_t, color, layer))
            if len(pts_r) >= 3: proj_rear.append((sum(dr)/len(dr), pts_r, color, layer))

            if no_fpv:
                continue

            # FPV: first person perspective looking along facing direction
            rel_verts = []
            for vx, vy, vz in p_verts:
                if facing_right:
                    dx = vx - cam_x    # forward (+X)
                    dy = vy - cam_y    # up/down relative to eye
                    dz = vz - cam_z    # left/right relative to eye
                else:
                    dx = cam_x - vx    # forward (-X when facing West)
                    dy = vy - cam_y    # up/down relative to eye
                    dz = cam_z - vz    # left/right relative to eye
                rel_verts.append((dx, dy, dz))

            clipped = []
            near = 0.5
            for i in range(len(rel_verts)):
                pc, pp = rel_verts[i], rel_verts[i - 1]
                if pc[0] >= near and pp[0] >= near:
                    clipped.append(pc)
                elif (pc[0] >= near) != (pp[0] >= near):
                    t = (near - pp[0]) / (pc[0] - pp[0] + 1e-9)
                    clipped.append((near,
                                    pp[1] + t*(pc[1]-pp[1]),
                                    pp[2] + t*(pc[2]-pp[2])))
                    if pc[0] >= near:
                        clipped.append(pc)

            pts_fpv, df = [], []
            for dx, dy, dz in clipped:
                px_f = -(dz / dx) * self.fov + cx
                py_f = -(dy / dx) * self.fov + cy
                pts_fpv.append((px_f, py_f))
                df.append(dx)

            if len(pts_fpv) >= 3:
                proj_fpv.append((sum(df)/len(df), pts_fpv, color, layer))

        # Draw all surfaces
        self._draw(proj_top,  top_surf)
        self._draw(proj_rear, rear_surf)
        self._draw(proj_side, side_surf)
        self._draw(proj_fpv,  fpv_surf)

        # ---- FPV HUD overlays ----        
        pygame.draw.rect(fpv_surf, (200, 200, 200), (10, VH - 20, 160, 10), 1)
        # Distance to goal
        dist = abs(FLAGPOLE_X - float(base.player.position.x))
        txt_col = GOAL_C if dist < 10 else (220, 220, 220)
        pygame.draw.rect(fpv_surf, txt_col,
                         (VW - 20 - int(min(dist/LEVEL_W, 1)*150), VH - 20,
                          int(min(dist/LEVEL_W, 1)*150), 10))

        return [self._surf_to_img(s)
                for s in [top_surf, rear_surf, side_surf, fpv_surf]]
