try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
import pygame
from pygame import gfxdraw
import math
from .camera_utils import orbit_camera_frame, orbit_camera_offset, rotate_camera_frame
from Box2D.b2 import (world, polygonShape, contactListener)

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

# Pits (x_start, x_end) = gaps between consecutive ground segments
PITS = [(a[0] + a[2], b[0] - b[2])
        for a, b in zip(GROUND_SEGS[:-1], GROUND_SEGS[1:])
        if b[0] - b[2] > a[0] + a[2] + 1e-6]

# Floating platforms (cx, cy, hw, hh)
# Low steps (top ~2.55m) are reachable with one ground jump; high steps (top ~4.95m)
# are reachable from a low step. None overhangs a pit take-off / landing zone, so a
# running jump over a pit never bonks its head on a platform underside.
PLATFORMS = [
    (13.5,  2.2,  1.5, 0.35),    # low step before pit 1
    (16.0,  4.6,  1.5, 0.35),    # high step before pit 1
    (27.5,  2.2,  1.5, 0.35),    # low step after pit 1
    (40.0,  2.2,  2.0, 0.35),    # low step after pit 2
    (44.0,  4.6,  1.5, 0.35),    # high step before pit 3
    (55.0,  2.2,  1.5, 0.35),    # low step after pit 3
    (57.5,  4.6,  1.0, 0.35),    # high step before pit 4
]

FLAGPOLE_X = 70.0
FLAGPOLE_Y_TOP = 10.0    # top of pole

SPAWN_X, SPAWN_Y = 2.0, 1.0   # Mario spawn pos (body centre)
SPAWN_JITTER = 1.0             # +- metres of random spawn X offset (seeded)
EXIT_DIST = 1.8                # metres from flagpole to win

# Movement (velocity-controlled, no unbounded force)
MAX_VX      = 7.0     # m/s top running speed
GROUND_ACC  = 45.0    # m/s^2 acceleration towards target speed on the ground
AIR_ACC     = 20.0    # m/s^2 acceleration towards target speed in the air
JUMP_VY     = 19.0    # m/s take-off speed → ~3.6m apex (≥1m clearance over low steps), ~0.76s air time, ~5.3m at top speed
PIT_OBS_CAP = 20.0    # metres; obs value when no pit lies ahead

# ==============================================================================
# 1. MARIO BASE ENVIRONMENT
# ==============================================================================
class OneWayPlatformListener(contactListener):
    """Makes floating platforms one-way: Mario passes up through the underside and lands on top."""
    def PreSolve(self, contact, old_manifold):
        fa, fb = contact.fixtureA, contact.fixtureB
        if fa.body.userData == 'player' and fb.body.userData == 'platform':
            player, plat = fa.body, fb
        elif fb.body.userData == 'player' and fa.body.userData == 'platform':
            player, plat = fb.body, fa
        else:
            return
        plat_top = float(plat.body.position.y) + max(float(v[1]) for v in plat.shape.vertices)
        feet = float(player.position.y) - P_HH
        if feet < plat_top - 0.05:   # feet below the top surface → pass through
            contact.enabled = False


class MarioEscapeEnv(gym.Env):
    """
    Mario Escape — simplified side-scrolling platformer.

    Goal: Reach the flagpole at the far right of the level, jumping over 4 pits.

    Actions (2D continuous, Box[-1, 1]):
      0: Run     [-1..1]  target horizontal speed = action * MAX_VX
                          (+1 = run right at full speed, -1 = run left, 0 = stop)
      1: Jump    [>0.5]   jump when standing on top of ground/platform (<=0.5 = no jump)

    Observation (8D):
      0-3: Player X, Y, Vx, Vy
      4  : Is grounded (1.0 / 0.0) — standing on the top surface of ground/platform
      5  : Signed X distance from player centre to the start of the next pit ahead
           (negative while above that pit; PIT_OBS_CAP if no pit ahead)
      6  : Width of that pit (0.0 if no pit ahead)
      7  : Step fraction (step_count / max_steps)

    Reward: 3 * forward progress (metres) - 0.02 per step,
            +200 on reaching the flagpole, -80 on falling into a pit.
    Episode ends (terminated) on goal or fall; truncated after max_steps.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.screen      = None
        self.clock       = None

        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        self.platform_listener = OneWayPlatformListener()
        self.world   = world(gravity=(0, GRAVITY), doSleep=True, contactListener=self.platform_listener)
        self.bodies  = []
        self.grounds = []     # ground segment bodies
        self.plats   = []     # platform bodies
        self.player  = None
        self.max_steps = 1200
        self.step_count = 0
        self.is_grounded = False
        self.facing_right = True

    # ------------------------------------------------------------------
    def _build_static(self, specs, color_tag):
        result = []
        for cx, cy, hw, hh in specs:
            b = self.world.CreateStaticBody(
                position=(cx, cy), shapes=polygonShape(box=(hw, hh)))
            b.userData = color_tag     # Box2D-persistent tag (ce.other returns a fresh wrapper)
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
        pole.userData = 'flag'
        self.bodies.append(pole)
        self.flagpole = pole

        # West boundary wall (gray static body)
        west_w = self.world.CreateStaticBody(
            position=(-0.2, LEVEL_H / 2),
            shapes=polygonShape(box=(0.2, LEVEL_H / 2)))
        west_w.userData = 'wall'
        self.bodies.append(west_w)

        # Mario player body (zero friction: speed is velocity-controlled, so no wall-sticking)
        spawn_x = SPAWN_X + float(self.np_random.uniform(-SPAWN_JITTER, SPAWN_JITTER))
        self.player = self.world.CreateDynamicBody(
            position=(spawn_x, SPAWN_Y),
            linearDamping=0.0,
            angularDamping=10.0,
            fixedRotation=True)
        self.player.CreatePolygonFixture(box=(P_HW, P_HH), density=1.2, friction=0.0)
        self.player.userData = 'player'
        self.bodies.append(self.player)

        # Settle onto the ground so the first obs reports grounded correctly
        for _ in range(5):
            self.world.Step(1.0 / FPS, 8, 3)
        self.is_grounded = self._check_grounded()

        self.prev_x = float(self.player.position.x)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _check_grounded(self):
        """True only if touching the TOP surface of ground/platform (contact normal points up)."""
        if float(self.player.linearVelocity.y) > 1.0:   # still rising from a jump
            return False
        # Geometric test (independent of Box2D manifold-normal / enabled-flag conventions):
        # touching a ground/platform block with the feet resting on its top face
        feet = float(self.player.position.y) - P_HH
        px = float(self.player.position.x)
        for ce in self.player.contacts:
            if not ce.contact.touching or ce.other.userData not in ('ground', 'platform'):
                continue
            other = ce.other
            verts = [other.transform * v for f in other.fixtures for v in f.shape.vertices]
            top = max(float(v[1]) for v in verts)
            left = min(float(v[0]) for v in verts) - P_HW
            right = max(float(v[0]) for v in verts) + P_HW
            if abs(feet - top) < 0.1 and left < px < right:
                return True
        return False

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.step_count += 1

        vx = float(self.player.linearVelocity.x)
        vy = float(self.player.linearVelocity.y)

        # --- Horizontal movement: accelerate towards target speed (capped at MAX_VX)
        target = float(action[0]) * MAX_VX
        acc = (GROUND_ACC if self.is_grounded else AIR_ACC) / FPS
        vx += float(np.clip(target - vx, -acc, acc))

        # --- Jump (only from a top surface; is_grounded reflects the current, post-step state)
        if float(action[1]) > 0.5 and self.is_grounded:
            vy = JUMP_VY
            self.is_grounded = False

        self.player.linearVelocity = (vx, vy)
        self.player.awake = True

        self.world.Step(1.0 / FPS, 8, 3)
        self.is_grounded = self._check_grounded()
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
        truncated  = False

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
            truncated = True

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def _next_pit(self, px):
        """(signed distance to start of next pit not yet passed, pit width)."""
        for x0, x1 in PITS:
            if px < x1:
                return x0 - px, x1 - x0
        return PIT_OBS_CAP, 0.0

    # ------------------------------------------------------------------
    def _get_obs(self):
        p  = self.player.position
        v  = self.player.linearVelocity
        pit_d, pit_w = self._next_pit(float(p.x))
        frac = self.step_count / self.max_steps
        return np.array([
            float(p.x), float(p.y),
            float(v.x), float(v.y),
            float(self.is_grounded),
            min(pit_d, PIT_OBS_CAP), pit_w,
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
            tag = b.userData or 'ground'
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
      FPV       : true 3D third-person perspective, orbit camera trailing Mario
                  (behind along -X, above) looking right (+X, level progression)

    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """

    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.SCALE      = SCALE
        self.VIEWPORT_W = 800
        self.VIEWPORT_H = 600
        self.fov        = 280.0

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state

        # Follow/Orbit camera baseline offsets (relative to Mario), in scaled px
        self.camera_dist = 2.5 * SCALE     # Distance behind Mario along -X (level progression axis)
        self.camera_height = 0.8 * SCALE   # Vertical offset above Mario's body centre

        # Orbit pivot: Mario himself (camera_dist ahead of the camera along +X)
        self.pivot_depth = self.camera_dist

        # Camera always follows Mario: its height eases toward his (so jumps read against the
        # ground/horizon), and a look-at tilt keeps Mario framed at the same screen spot
        self.follow_y = None
        self.follow_rate = 0.15   # fraction of the height gap closed per rendered frame
        self.look_pitch = 0.0

        # Last computed camera world position (valid before first render)
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        self.follow_y = None
        self.look_pitch = 0.0
        return self.env.reset(**kwargs)

    def _update_follow_y(self):
        """Eases the camera height toward Mario's and aims the camera at him (look-at tilt)."""
        base = self.env.unwrapped
        if not hasattr(base, 'player') or base.player is None:
            return
        p_y = float(base.player.position.y)
        if self.follow_y is None:
            self.follow_y = p_y
        self.follow_y += self.follow_rate * (p_y - self.follow_y)
        # Tilt so Mario sits where he would if the camera were exactly at his height
        self.look_pitch = math.atan2((p_y - self.follow_y) * self.SCALE, self.pivot_depth)

    @property
    def _cam_anchor(self):
        """Un-orbited FPV anchor: camera_dist behind (-X) and camera_height above Mario, looking +X.
        Trails along the level progression axis (does not flip with Mario's facing)."""
        base = self.env.unwrapped
        if not hasattr(base, 'player') or base.player is None:
            return None
        S = self.SCALE
        cam_x = float(base.player.position.x) * S - self.camera_dist   # trailing behind
        follow_y = self.follow_y if self.follow_y is not None else float(base.player.position.y)
        cam_y = follow_y * S + self.camera_height                      # above Mario's (eased) body centre
        cam_z = 0.0                                                    # center depth
        return (cam_x, cam_y, cam_z)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the camera in world space."""
        anchor = self._cam_anchor
        if anchor is None:
            return (0.0, 0.0, 0.0)

        # Orbit offset in the un-orbited camera frame (depth, lateral, vertical)
        d_off, lat_off, v_off = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)
        # Camera frame → world: depth = +X, lateral = -Z, vertical = Y (camera always looks +X)
        cam_x = anchor[0] + d_off
        cam_y = anchor[1] + v_off
        cam_z = anchor[2] - lat_off

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
    def render(self, camera_pose="Center"):
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)
        self._update_follow_y()

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

        # FPV: sky above, green grass below the horizon, which moves with the camera's net tilt
        # (look-at tilt up lowers it, orbit pitch looking down raises it)
        net_pitch = self.look_pitch - self.cam_pitch
        horizon = int(max(0, min(VH, cy + self.fov * math.tan(net_pitch))))
        fpv_surf.fill(SKY_C)
        pygame.draw.rect(fpv_surf, GRASS_C, (0, horizon, VW, VH - horizon))
        pygame.draw.line(fpv_surf, (100, 210, 60), (0, horizon), (VW, horizon), 2)

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
        # E. Mario character — stacked coloured boxes (visible in the third-person FPV)
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
                          color_f, color_t, color_f, layer)

        mario_box(0.00, 0.22, 0.00, 0.00, MARIO_SHOE,  (120, 75, 40),  2)   # Shoes
        mario_box(0.22, 0.72, 0.00, 0.00, MARIO_OVR,   (20,  100, 230), 2)  # Overalls/legs
        mario_box(0.72, 1.08, 0.02, 0.02, MARIO_SHIRT, (230, 20,  20),  2)  # Shirt
        mario_box(1.08, 1.42, 0.04, 0.04, MARIO_FACE,  (255, 210, 150), 2)  # Face/head

        # Arms (left & right sleeves + white gloves with animated swing)
        # Left arm & glove
        self._add_box(polys_3d, mx - P_HW - 0.14 - swing_off, mx - P_HW + 0.02 - swing_off, foot_y + 0.74, foot_y + 1.04, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, MARIO_SHIRT, (230, 20, 20), MARIO_SHIRT, 3)
        self._add_box(polys_3d, mx - P_HW - 0.14 - swing_off, mx - P_HW + 0.02 - swing_off, foot_y + 0.60, foot_y + 0.74, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, (250, 250, 250), (250, 250, 250), (220, 220, 220), 3)
        # Right arm & glove
        self._add_box(polys_3d, mx + P_HW - 0.02 + swing_off, mx + P_HW + 0.14 + swing_off, foot_y + 0.74, foot_y + 1.04, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, MARIO_SHIRT, (230, 20, 20), MARIO_SHIRT, 3)
        self._add_box(polys_3d, mx + P_HW - 0.02 + swing_off, mx + P_HW + 0.14 + swing_off, foot_y + 0.60, foot_y + 0.74, Z_CHAR_LO - 0.04, Z_CHAR_HI + 0.04, (250, 250, 250), (250, 250, 250), (220, 220, 220), 3)

        # Cap — wider than head and slightly overhanging
        self._add_box(polys_3d,
                      mx - P_HW - 0.05,
                      mx + P_HW + 0.05,
                      foot_y + 1.30,
                      foot_y + 1.55,
                      Z_CHAR_LO - 0.05, Z_CHAR_HI + 0.05,
                      MARIO_HAT, (230, 10, 10), MARIO_HAT, 3)
        # Cap brim (front overhang depending on facing direction)
        if facing_right:
            self._add_box(polys_3d,
                          mx + P_HW - 0.05,
                          mx + P_HW + 0.18,
                          foot_y + 1.28,
                          foot_y + 1.38,
                          Z_CHAR_LO - 0.05, Z_CHAR_HI + 0.05,
                          MARIO_HAT, (230, 10, 10), MARIO_HAT, 3)
        else:
            self._add_box(polys_3d,
                          mx - P_HW - 0.18,
                          mx - P_HW + 0.05,
                          foot_y + 1.28,
                          foot_y + 1.38,
                          Z_CHAR_LO - 0.05, Z_CHAR_HI + 0.05,
                          MARIO_HAT, (230, 10, 10), MARIO_HAT, 3)

        # ----------------------------------------------------------------
        # F. Multi-View Projections
        # ----------------------------------------------------------------
        # Get dynamically calculated position from our property (orbited, for logging)
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        # Un-orbited anchor for camera-relative vectors (orbit_camera_frame applies the orbit)
        anc_x, anc_y, anc_z = self._cam_anchor

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

            # FPV: third-person orbit camera trailing Mario, looking along +X
            rel_verts = []
            for vx, vy, vz in p_verts:
                dx = vx - anc_x    # forward (+X)
                dy = vy - anc_y    # up/down relative to camera
                dz = vz - anc_z    # left/right relative to camera
                # Orbit first so near-plane clipping runs on the orbited depth
                d_c, lat_c, dy_c = orbit_camera_frame(dx, -dz, dy, self.cam_yaw, self.cam_pitch, self.pivot_depth)
                # Look-at tilt keeps Mario framed while the camera height eases after him
                d_c, lat_c, dy_c = rotate_camera_frame(d_c, lat_c, dy_c, 0.0, self.look_pitch)
                rel_verts.append((d_c, dy_c, -lat_c))

            clipped = []
            near = 0.5 * S   # 0.5 m (scaled px): keeps projected coords within gfxdraw's 16-bit range
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
            for d_c, dy_c, dz_c in clipped:
                px_f = -(dz_c / d_c) * self.fov + cx
                py_f = -(dy_c / d_c) * self.fov + cy
                pts_fpv.append((px_f, py_f))
                df.append(d_c)

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
