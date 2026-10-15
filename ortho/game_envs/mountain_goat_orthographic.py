"""
mountain_goat_orthographic.py
-----------------------------
Mountain Goat 🐐 Environment & 3D Multi-View Orthographic Wrapper

Task:
  A quadruped mountain goat starts on a flat meadow at the East foot of a mountain and climbs
  West (-X direction) up a ledged slope (+Y) to the summit plateau. The agent controls a torso
  pitch torque plus the hip/knee motors of two diagonal leg pairs (trot gait) and must make
  uphill progress without tumbling, belly-flopping or walking off the meadow into the valley.

Views returned by MountainGoatOrthographicWrapper.render():
  1. Top-Down  : X-Z plan view (looking down from above, centered on goat)
  2. Rear      : Z-Y elevation (looking from behind the goat along +X)
  3. Side      : X-Y elevation (native 2D side-view of the mountain climb)
  4. 3DV / FPV : True 3D third-person perspective from behind/above the goat looking West (-X) up the slope

Fully compatible with Gym/Gymnasium and Box2D SWIG bindings.
"""

import math
import numpy as np
from .camera_utils import orbit_camera_frame, orbit_camera_offset
import pygame
from pygame import gfxdraw

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from Box2D.b2 import (
    world,
    polygonShape,
    circleShape,
    revoluteJointDef,
    weldJointDef,
    fixtureDef,
    rayCastCallback,
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
FPS = 50
SCALE = 28.0            # pixels per metre
GRAVITY = -15.0         # mountain gravity

# Mountain bounds (flat meadow East of X=60, slope climbs West to the summit plateau at X=PEAK)
MOUNTAIN_START_X = 60.0 # East foot of the slope (meadow spans START_X .. START_X + 5)
MOUNTAIN_PEAK_X  = 25.0 # West summit plateau edge (goal)
MOUNTAIN_BASE_H  = 2.0  # Meadow height
MOUNTAIN_MAX_H   = 12.0 # Summit plateau height (mean slope ~16 deg, steepest ledge faces ~28 deg)
TERRAIN_STEP     = 1.25 # Horizontal length of each terrain segment
LEDGE_A1, LEDGE_A2 = 0.30, 0.12  # Rocky ledge sine amplitudes (long / short wavelength)
VALLEY_H         = 0.5  # Height of the valley floor East of the meadow

# Goat physics sizes
TORSO_W, TORSO_H = 0.6, 0.35    # torso half-width, half-height (metres)
HEAD_R           = 0.22         # head radius
HORN_W, HORN_H   = 0.08, 0.30   # horn size
THIGH_LEN, SHIN_LEN = 0.45, 0.45
STAND_H          = TORSO_H + THIGH_LEN + SHIN_LEN  # torso centre height above ground when standing
GOAT_GROUP       = -1           # negative Box2D groupIndex: goat parts never collide with each other

# Control & episode settings
TORSO_TORQUE     = 60.0         # torso pitch torque (no reaction body: a deliberate balance aid)
MOTOR_SPEED      = 5.0          # rad/s at |action| = 1
HIP_TORQUE, KNEE_TORQUE = 80.0, 60.0
LEG_DENSITY      = 12.0         # heavy-ish legs keep the torso/leg mass ratio solver-friendly
TORSO_DENSITY    = 3.0
FAR_SERVO_GAIN   = 10.0         # far legs servo onto their diagonal partner's angle
MAX_EPISODE_STEPS = 2000
LIDAR_RANGE      = 6.0
N_LIDAR          = 10

# Reward settings
PROGRESS_X_W     = 3.0          # per metre moved West (-X)
PROGRESS_Y_W     = 2.0          # per metre gained in height
ENERGY_W         = 0.003
GOAL_BONUS       = 100.0
FAIL_PENALTY     = 10.0

# 3D Extrusion depths (Z axis)
Z_TERRAIN_LO, Z_TERRAIN_HI = -3.0, 3.0
Z_TORSO_LO,   Z_TORSO_HI   = -0.4, 0.4
Z_LEG_NEAR_LO, Z_LEG_NEAR_HI = 0.45, 0.70
Z_LEG_FAR_LO,  Z_LEG_FAR_HI  = -0.70, -0.45

# Color Palette — Alpine Mountain Theme
SKY_C         = (175, 215, 245)  # Crisp alpine sky
CLOUD_C       = (250, 252, 255)  # Fluffy white sky clouds
CLOUD_SHADE_C = (215, 225, 235)  # Cloud bottom shadow
FLOOR_C       = (200, 205, 210)  # Light gray floor outside mountain
SNOW_C        = (245, 248, 252)  # Mountain snow caps
ROCK_C        = (115, 75, 45)    # Rich warm mountain brown
ROCK_SIDE_C   = (90, 55, 30)     # Darker brown mountain rock
GRASS_C       = (60, 175, 50)    # Vibrant green mountain slopes & climbing lane
GOAT_FUR      = (235, 235, 235)  # White goat body
GOAT_FAR_FUR  = (180, 185, 190)  # Slightly shaded far limbs
GOAT_HORN     = (220, 180, 70)   # Golden horns
GOAT_HOOF     = (50, 45, 45)     # Dark hooves
JOINT_CYAN    = (0, 230, 255)    # Joint indicators
OUTLINE_C     = (30, 30, 35)     # Polygon outlines

# Left & Right Horn Z-Extrusions
Z_HORN_LEFT_LO,  Z_HORN_LEFT_HI  = 0.12, 0.32   # Left horn (positive Z)
Z_HORN_RIGHT_LO, Z_HORN_RIGHT_HI = -0.32, -0.12 # Right horn (negative Z)

# ==============================================================================
# BOX2D RAYCAST CALLBACK FOR LIDAR
# ==============================================================================
class LidarCallback(rayCastCallback):
    """Closest-hit ray callback that ignores every goat fixture (they share GOAT_GROUP)."""
    def __init__(self):
        super().__init__()
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.filterData.groupIndex == GOAT_GROUP:
            return -1.0  # Ignore goat body parts
        if fraction < self.fraction:
            self.fraction = fraction
        return fraction


# ==============================================================================
# 1. MOUNTAIN GOAT BASE ENVIRONMENT
# ==============================================================================
class MountainGoatEnv(gym.Env):
    """
    2D Physics Environment where a quadrupedal Mountain Goat climbs a ledged slope from East to West.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, max_episode_steps=MAX_EPISODE_STEPS):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.screen = None
        self.clock = None

        # Actions (5D), all in [-1..1]:
        # 0: Torso pitch torque (+ = counter-clockwise = pitch nose down, - = pitch nose up toward the slope)
        # 1: Pair A hip motor speed  (near-front + far-rear legs; + swings hoof East/back)
        # 2: Pair A knee motor speed (near-front + far-rear legs; + folds shin East/back)
        # 3: Pair B hip motor speed  (near-rear + far-front legs)
        # 4: Pair B knee motor speed (near-rear + far-front legs)
        # Far legs servo onto their diagonal partner's joint angles, giving a trot gait.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Obs (28D):
        # 0-1:   Torso Vx, Vy (/5)
        # 2-3:   Torso angle (rad), angular velocity (/5)
        # 4:     Torso height above terrain minus standing height (m)
        # 5:     Climb progress fraction (0 at slope foot, 1 at summit)
        # 6-13:  Near joint angle (rad) & speed (/MOTOR_SPEED): front hip, front knee, rear hip, rear knee
        # 14-17: Hoof ground contact flags: near front, near rear, far front, far rear
        # 18-27: Lidar terrain distance fractions, 10 rays from forward-up (-X) through down to backward (+X)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

        self.world = None
        self.torso = None
        self.bodies = []
        self.terrain_polys = []
        self.joints = []
        self.far_joints = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world = world(gravity=(0.0, GRAVITY), doSleep=True)
        self.bodies = []
        self.terrain_polys = []
        self.joints = []
        self.far_joints = []

        # Modest per-episode terrain variety (ledge phase / amplitude)
        self.ledge_phase = self.np_random.uniform(0.0, 2.0 * math.pi, size=2)
        self.ledge_amp = float(self.np_random.uniform(0.7, 1.0))

        # Generate Procedural Mountain (East meadow -> slope -> West summit plateau)
        self._generate_mountain()

        # Spawn Goat standing on the flat East meadow, hooves on the ground
        start_x = MOUNTAIN_START_X + 2.0 + float(self.np_random.uniform(-0.5, 0.5))
        start_y = MOUNTAIN_BASE_H + STAND_H + 0.02
        self._spawn_goat(start_x, start_y)

        self.prev_x = start_x
        self.prev_y = start_y
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def _get_mountain_y(self, x):
        """Returns height of mountain surface at X position."""
        # Linear incline from East foot (START_X, BASE_H) up to the West summit (PEAK_X, MAX_H)
        progress = (MOUNTAIN_START_X - x) / (MOUNTAIN_START_X - MOUNTAIN_PEAK_X)
        progress = float(np.clip(progress, 0.0, 1.0))
        base_h = MOUNTAIN_BASE_H + (MOUNTAIN_MAX_H - MOUNTAIN_BASE_H) * progress
        # Rocky ledges, faded out on the flat meadow and summit plateau
        fade = float(np.clip((MOUNTAIN_START_X - x) / 3.0, 0.0, 1.0) * np.clip((x - MOUNTAIN_PEAK_X) / 3.0, 0.0, 1.0))
        ph0, ph1 = getattr(self, 'ledge_phase', (0.0, 0.0))
        ledges = LEDGE_A1 * math.sin(x * 0.6 + ph0) + LEDGE_A2 * math.sin(x * 1.3 + ph1)
        return base_h + fade * getattr(self, 'ledge_amp', 1.0) * ledges

    def _generate_mountain(self):
        """Creates static Box2D bodies and 3D polygon structures for the mountain."""
        x_steps = np.arange(MOUNTAIN_START_X + 5.0, MOUNTAIN_PEAK_X - 6.0, -TERRAIN_STEP)

        ground_body = self.world.CreateStaticBody(position=(0, 0))
        self.ground = ground_body
        self.bodies.append(ground_body)

        # Light gray valley floor East of the meadow (falling in here ends the episode)
        floor_poly_east = [(MOUNTAIN_START_X + 5.0, -10.0), (100.0, -10.0), (100.0, VALLEY_H), (MOUNTAIN_START_X + 5.0, VALLEY_H)]
        ground_body.CreatePolygonFixture(vertices=floor_poly_east, friction=1.0)
        self.terrain_polys.append({
            'poly': floor_poly_east,
            'top_edge': ((MOUNTAIN_START_X + 5.0, VALLEY_H), (100.0, VALLEY_H)),
            'top_color': FLOOR_C,
            'body_color': FLOOR_C
        })

        snow_h = MOUNTAIN_BASE_H + 0.75 * (MOUNTAIN_MAX_H - MOUNTAIN_BASE_H)
        for i in range(len(x_steps) - 1):
            x0, x1 = x_steps[i], x_steps[i+1]
            y0, y1 = self._get_mountain_y(x0), self._get_mountain_y(x1)

            # Ground segment poly (top edge from (x0, y0) to (x1, y1), bottom down to y=-5)
            poly_2d = [(x0, -5.0), (x1, -5.0), (x1, y1), (x0, y0)]
            ground_body.CreatePolygonFixture(vertices=poly_2d, friction=1.2, restitution=0.0)

            # High altitude has snow top, lower has grass top
            is_snow = ((y0 + y1) / 2.0) > snow_h
            top_color = SNOW_C if is_snow else GRASS_C
            body_color = ROCK_C

            self.terrain_polys.append({
                'poly': poly_2d,
                'top_edge': ((x0, y0), (x1, y1)),
                'top_color': top_color,
                'body_color': body_color
            })

        # Procedural Sky Clouds (3D puff volumes)
        self.clouds = []
        for cx in np.arange(MOUNTAIN_START_X + 10.0, MOUNTAIN_PEAK_X - 10.0, -6.0):
            cy = MOUNTAIN_MAX_H + 6.0 + 3.0 * math.sin(cx * 0.4)
            cz = 4.0 * math.cos(cx * 0.7)
            cw, ch, cd = 4.5, 1.4, 2.5
            self.clouds.append({'x': cx, 'y': cy, 'z': cz, 'w': cw, 'h': ch, 'd': cd})

    def _spawn_goat(self, x, y):
        """Builds multi-body Mountain Goat with joints (torso level, legs straight down)."""
        # 1. Main Torso
        self.torso = self.world.CreateDynamicBody(position=(x, y), angle=0.0)
        self.torso.CreatePolygonFixture(box=(TORSO_W, TORSO_H), density=TORSO_DENSITY, friction=0.8, groupIndex=GOAT_GROUP)
        self.bodies.append(self.torso)

        # 2. Head & Left/Right Horns (facing West / -X side of torso)
        head_off = (-TORSO_W - 0.1, TORSO_H + 0.15)
        head_pos = (x + head_off[0], y + head_off[1])
        self.head = self.world.CreateDynamicBody(position=head_pos)
        self.head.CreateCircleFixture(radius=HEAD_R, density=2.0, groupIndex=GOAT_GROUP)
        self.bodies.append(self.head)

        # Weld head to torso (anchor at the head centre, matching the spawn pose)
        self.world.CreateJoint(weldJointDef(
            bodyA=self.torso, bodyB=self.head,
            localAnchorA=head_off, localAnchorB=(0, 0)
        ))

        # Left & Right Horns (curved backward/upward relative to head), welded on top of the head
        horn_poly = [(0, 0), (-0.15, HORN_H), (0.05, HORN_H * 0.85)]
        horn_pos = (head_pos[0] - 0.1, head_pos[1] + HEAD_R)

        self.horn_left = self.world.CreateDynamicBody(position=horn_pos)
        self.horn_left.CreatePolygonFixture(vertices=horn_poly, density=0.4, groupIndex=GOAT_GROUP)
        self.bodies.append(self.horn_left)
        self.world.CreateJoint(weldJointDef(
            bodyA=self.head, bodyB=self.horn_left,
            localAnchorA=(-0.1, HEAD_R), localAnchorB=(0, 0)
        ))

        self.horn_right = self.world.CreateDynamicBody(position=horn_pos)
        self.horn_right.CreatePolygonFixture(vertices=horn_poly, density=0.4, groupIndex=GOAT_GROUP)
        self.bodies.append(self.horn_right)
        self.world.CreateJoint(weldJointDef(
            bodyA=self.head, bodyB=self.horn_right,
            localAnchorA=(-0.1, HEAD_R), localAnchorB=(0, 0)
        ))

        # 3. Legs & Hooves (Front = West side / -X, Rear = East side / +X)
        # Order matters: the wrapper treats legs[0:4] as near legs and legs[4:8] as far legs.
        self.legs = []
        self.hooves = []
        self.joints = []
        self.far_joints = []

        leg_configs = [
            # (name, torso_anchor_x)
            ("front_near", -TORSO_W + 0.15),
            ("rear_near",   TORSO_W - 0.15),
            ("front_far",  -TORSO_W + 0.15),
            ("rear_far",    TORSO_W - 0.15),
        ]

        for name, ax in leg_configs:
            # Thigh
            thigh_pos = (x + ax, y - TORSO_H - THIGH_LEN/2)
            thigh = self.world.CreateDynamicBody(position=thigh_pos)
            thigh.CreatePolygonFixture(box=(0.08, THIGH_LEN/2), density=LEG_DENSITY, friction=0.9, groupIndex=GOAT_GROUP)
            self.bodies.append(thigh)

            # Shin / Hoof
            shin_pos = (x + ax, y - TORSO_H - THIGH_LEN - SHIN_LEN/2)
            shin = self.world.CreateDynamicBody(position=shin_pos)
            shin.CreatePolygonFixture(box=(0.07, SHIN_LEN/2), density=LEG_DENSITY, friction=1.5, groupIndex=GOAT_GROUP)
            self.bodies.append(shin)

            # Hip Joint
            hip_j = self.world.CreateJoint(revoluteJointDef(
                bodyA=self.torso, bodyB=thigh,
                localAnchorA=(ax, -TORSO_H), localAnchorB=(0, THIGH_LEN/2),
                enableMotor=True, maxMotorTorque=HIP_TORQUE, enableLimit=True,
                lowerAngle=-0.8, upperAngle=0.8
            ))

            # Knee Joint
            knee_j = self.world.CreateJoint(revoluteJointDef(
                bodyA=thigh, bodyB=shin,
                localAnchorA=(0, -THIGH_LEN/2), localAnchorB=(0, SHIN_LEN/2),
                enableMotor=True, maxMotorTorque=KNEE_TORQUE, enableLimit=True,
                lowerAngle=-1.2, upperAngle=0.3
            ))

            self.legs.extend([thigh, shin])
            self.hooves.append(shin)
            if "near" in name:
                self.joints.extend([hip_j, knee_j])
            else:
                self.far_joints.extend([hip_j, knee_j])

        # Diagonal trot coupling: far-front follows near-rear (pair B), far-rear follows near-front (pair A)
        self.far_partners = [self.joints[2], self.joints[3], self.joints[0], self.joints[1]]

    def _hoof_contacts(self):
        # Goat parts never collide with each other, so any touching contact is with the terrain
        return [1.0 if any(ce.contact.touching for ce in shin.contacts) else 0.0 for shin in self.hooves]

    def _body_on_ground(self):
        for body in (self.torso, self.head, self.horn_left, self.horn_right):
            if any(ce.contact.touching for ce in body.contacts):
                return True
        return False

    def step(self, action):
        assert self.torso is not None, "Call reset() before step()"
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)

        # 1. Apply Actions
        # Torso pitch torque
        self.torso.ApplyTorque(float(action[0] * TORSO_TORQUE), True)
        # Near leg joint motor speeds (front hip, front knee, rear hip, rear knee)
        for j, a in zip(self.joints, action[1:5]):
            j.motorSpeed = float(a * MOTOR_SPEED)
        # Far legs: feed-forward partner command + servo onto the partner's angle
        for fj, pj in zip(self.far_joints, self.far_partners):
            cmd = pj.motorSpeed + FAR_SERVO_GAIN * (pj.angle - fj.angle)
            fj.motorSpeed = float(np.clip(cmd, -2.0 * MOTOR_SPEED, 2.0 * MOTOR_SPEED))

        # Step physics
        self.world.Step(1.0 / FPS, 10, 10)

        obs = self._get_obs()

        # 2. Rewards
        curr_x = float(self.torso.position.x)
        curr_y = float(self.torso.position.y)

        # Telescoping Westward progress (-X toward the summit) + upward progress (+Y)
        westward_progress = self.prev_x - curr_x
        upward_progress   = curr_y - self.prev_y
        reward = float(westward_progress * PROGRESS_X_W + upward_progress * PROGRESS_Y_W)

        # Small energy penalty
        reward -= float(ENERGY_W * np.sum(action ** 2))

        self.prev_x = curr_x
        self.prev_y = curr_y

        terminated = False
        outcome = None

        # Victory: Reached the summit plateau
        if curr_x <= MOUNTAIN_PEAK_X + 1.0:
            reward += GOAL_BONUS
            terminated = True
            outcome = "summit"
        # Failure: tipped over, torso/head on the ground, or walked off the meadow into the valley
        elif abs(self.torso.angle) > 1.2 or self._body_on_ground():
            reward -= FAIL_PENALTY
            terminated = True
            outcome = "fell"
        elif curr_x > MOUNTAIN_START_X + 5.5 or curr_y < MOUNTAIN_BASE_H:
            reward -= FAIL_PENALTY
            terminated = True
            outcome = "valley"

        truncated = (not terminated) and self.step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        info = self._get_info()
        info["outcome"] = outcome
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        x = float(self.torso.position.x)
        return {
            "x": x,
            "y": float(self.torso.position.y),
            "progress": (MOUNTAIN_START_X - x) / (MOUNTAIN_START_X - MOUNTAIN_PEAK_X),
            "step": self.step_count,
        }

    def _get_obs(self):
        t_pos = self.torso.position
        t_vel = self.torso.linearVelocity

        obs = [
            float(t_vel.x) / 5.0, float(t_vel.y) / 5.0,
            float(self.torso.angle), float(self.torso.angularVelocity) / 5.0,
            float(t_pos.y - self._get_mountain_y(t_pos.x)) - STAND_H,
            (MOUNTAIN_START_X - float(t_pos.x)) / (MOUNTAIN_START_X - MOUNTAIN_PEAK_X),
        ]

        # 4 near (actuated) joint states
        for j in self.joints[:4]:
            obs.append(float(j.angle))
            obs.append(float(j.speed) / MOTOR_SPEED)

        # Hoof contacts
        obs.extend(self._hoof_contacts())

        # 10-ray Lidar fan in the torso frame: from 30 deg above forward (-X) down through vertical to backward (+X)
        for i in range(N_LIDAR):
            ang = self.torso.angle + math.radians(150.0) + i * (math.radians(210.0) / (N_LIDAR - 1))
            p1 = t_pos
            p2 = (t_pos.x + LIDAR_RANGE * math.cos(ang), t_pos.y + LIDAR_RANGE * math.sin(ang))
            cb = LidarCallback()
            self.world.RayCast(cb, p1, p2)
            obs.append(float(cb.fraction))

        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return None
        w, h = 800, 600
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((w, h)) if self.render_mode == "human" else pygame.Surface((w, h))
            self.clock = pygame.time.Clock()

        self.screen.fill(SKY_C)
        scroll_x = self.torso.position.x * SCALE - w / 2.0
        scroll_y = h / 2.0 - self.torso.position.y * SCALE

        # Draw terrain
        for t_info in self.terrain_polys:
            pts = [(v[0] * SCALE - scroll_x, h - (v[1] * SCALE + scroll_y)) for v in t_info['poly']]
            pygame.draw.polygon(self.screen, t_info['body_color'], pts)
            # Top edge
            p0, p1 = t_info['top_edge']
            sp0 = (p0[0] * SCALE - scroll_x, h - (p0[1] * SCALE + scroll_y))
            sp1 = (p1[0] * SCALE - scroll_x, h - (p1[1] * SCALE + scroll_y))
            pygame.draw.line(self.screen, t_info['top_color'], sp0, sp1, 5)

        # Draw goat bodies (far legs first so near legs overlap them)
        horns = (self.horn_left, self.horn_right)
        far_legs = self.legs[4:]
        draw_order = far_legs + [b for b in self.bodies[1:] if b not in far_legs]
        for body in draw_order:
            for f in body.fixtures:
                if isinstance(f.shape, circleShape):
                    pos = body.transform * f.shape.pos
                    cx = int(pos[0] * SCALE - scroll_x)
                    cy = int(h - (pos[1] * SCALE + scroll_y))
                    rad = int(f.shape.radius * SCALE)
                    pygame.draw.circle(self.screen, GOAT_FUR, (cx, cy), rad)
                elif hasattr(f.shape, 'vertices'):
                    pts = [(body.transform * v) * SCALE for v in f.shape.vertices]
                    spts = [(p[0] - scroll_x, h - (p[1] + scroll_y)) for p in pts]
                    if body in horns:
                        c = GOAT_HORN
                    elif body in far_legs:
                        c = GOAT_FAR_FUR
                    else:
                        c = GOAT_FUR
                    pygame.draw.polygon(self.screen, c, spts)
                    pygame.draw.polygon(self.screen, OUTLINE_C, spts, 1)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            self.screen = None


# ==============================================================================
# 2. THE ORTHOGRAPHIC MULTI-VIEW WRAPPER FOR MOUNTAIN GOAT
# ==============================================================================
class MountainGoatOrthographicWrapper(gym.Wrapper):
    """
    Extrudes 2D MountainGoatEnv into 3D multi-views:
      - Top-Down  : X-Z plan view looking down
      - Rear      : Z-Y elevation looking along +X from behind
      - Side      : X-Y elevation (the 2D mountain side view)
      - 3DV / FPV : 3D third-person orbit camera trailing the goat, looking West (-X) up the mountain slope
    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """
    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.SCALE = SCALE
        self.VW = 800
        self.VH = 600
        self.fov = 320.0

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state

        # Follow/Orbit camera baseline offsets (relative to torso)
        self.camera_dist = 3.0 * SCALE    # Distance behind the goat along +X (it climbs toward -X)
        self.camera_height = 1.0 * SCALE  # Vertical offset above the goat

        # Orbit pivot: the goat itself (camera circles the goat while still facing it)
        self.pivot_depth = self.camera_dist

        # Persistent camera world position (updated each render)
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        return self.env.reset(**kwargs)

    @property
    def _cam_anchor(self):
        """Un-orbited follow-camera anchor (behind and above the goat, looking West, -X) used for projection."""
        base = self.env.unwrapped
        if not hasattr(base, 'torso') or base.torso is None:
            return (0.0, 0.0, 0.0)

        # 3DV / FPV Camera: trails camera_dist behind the goat (East, +X) and camera_height above it,
        # looking West (-X direction) up the slope so the goat is fully visible
        cam_x = base.torso.position.x * SCALE + self.camera_dist
        cam_y = base.torso.position.y * SCALE + self.camera_height
        cam_z = 0.0

        return (cam_x, cam_y, cam_z)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the camera in world space."""
        base = self.env.unwrapped
        if not hasattr(base, 'torso') or base.torso is None:
            return (0.0, 0.0, 0.0)

        anchor_x, anchor_y, anchor_z = self._cam_anchor
        # Orbit displacement in camera frame: depth = -X, lateral = -Z, vertical = +Y
        off_d, off_lat, off_vert = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)
        cam_x = anchor_x - off_d
        cam_y = anchor_y + off_vert
        cam_z = anchor_z - off_lat

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

    def render(self, camera_pose="Center"):
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)

        base = self.env.unwrapped

        top_surf  = pygame.Surface((self.VW, self.VH))
        rear_surf = pygame.Surface((self.VW, self.VH))
        side_surf = pygame.Surface((self.VW, self.VH))
        fpv_surf  = pygame.Surface((self.VW, self.VH))

        top_surf.fill(ROCK_SIDE_C)
        rear_surf.fill(SKY_C)
        side_surf.fill(SKY_C)
        fpv_surf.fill(SKY_C)

        # Light gray floor ground backdrop rects for elevation views
        pygame.draw.rect(rear_surf, FLOOR_C, (0, self.VH // 2, self.VW, self.VH // 2))
        pygame.draw.rect(side_surf, FLOOR_C, (0, self.VH // 2, self.VW, self.VH // 2))
        pygame.draw.rect(fpv_surf,  FLOOR_C, (0, self.VH // 2, self.VW, self.VH // 2))

        if not hasattr(base, 'torso') or base.torso is None:
            dummy = np.zeros((self.VH, self.VW, 3), dtype=np.uint8)
            return dummy, dummy, dummy, dummy

        polys_3d = []

        # ----------------------------------------------------------------------
        # A. Extrude Sky Clouds (Fluffy 3D Cloud Puffs in the Sky)
        # ----------------------------------------------------------------------
        if hasattr(base, 'clouds'):
            for c in base.clouds:
                cx_s, cy_s, cz_s = c['x'] * SCALE, c['y'] * SCALE, c['z'] * SCALE
                cw_s, ch_s, cd_s = c['w'] * SCALE, c['h'] * SCALE, c['d'] * SCALE
                
                v0 = (cx_s - cw_s/2, cy_s - ch_s/2, cz_s - cd_s/2)
                v1 = (cx_s + cw_s/2, cy_s - ch_s/2, cz_s - cd_s/2)
                v2 = (cx_s + cw_s/2, cy_s + ch_s/2, cz_s - cd_s/2)
                v3 = (cx_s - cw_s/2, cy_s + ch_s/2, cz_s - cd_s/2)
                
                v4 = (cx_s - cw_s/2, cy_s - ch_s/2, cz_s + cd_s/2)
                v5 = (cx_s + cw_s/2, cy_s - ch_s/2, cz_s + cd_s/2)
                v6 = (cx_s + cw_s/2, cy_s + ch_s/2, cz_s + cd_s/2)
                v7 = (cx_s - cw_s/2, cy_s + ch_s/2, cz_s + cd_s/2)

                polys_3d.append(([v0, v1, v2, v3], CLOUD_C, 0))        # back
                polys_3d.append(([v4, v5, v6, v7], CLOUD_C, 0))        # front
                polys_3d.append(([v0, v1, v5, v4], CLOUD_SHADE_C, 0))  # bottom shadow
                polys_3d.append(([v3, v2, v6, v7], CLOUD_C, 0))        # top
                polys_3d.append(([v0, v3, v7, v4], CLOUD_C, 0))        # left
                polys_3d.append(([v1, v2, v6, v5], CLOUD_C, 0))        # right

        # ----------------------------------------------------------------------
        # B. Extrude Terrain (Brown Mountain & Green Climbing Lane)
        # ----------------------------------------------------------------------
        Z_LANE_LO, Z_LANE_HI = -1.2, 1.2  # Green climbing lane width along Z

        for t_info in base.terrain_polys:
            poly_2d = t_info['poly']
            v_back  = [(p[0] * SCALE, p[1] * SCALE, Z_TERRAIN_LO * SCALE) for p in poly_2d]
            v_front = [(p[0] * SCALE, p[1] * SCALE, Z_TERRAIN_HI * SCALE) for p in poly_2d]

            # Caps (brown rock)
            polys_3d.append((v_back,  ROCK_SIDE_C, 0))
            polys_3d.append((v_front, ROCK_SIDE_C, 0))

            # Top faces: Outer brown rock shoulders + Center green climbing lane
            p0, p1 = t_info['top_edge']

            # Outer back brown rock shoulder
            back_shoulder = [
                (p0[0] * SCALE, p0[1] * SCALE, Z_TERRAIN_LO * SCALE),
                (p1[0] * SCALE, p1[1] * SCALE, Z_TERRAIN_LO * SCALE),
                (p1[0] * SCALE, p1[1] * SCALE, Z_LANE_LO * SCALE),
                (p0[0] * SCALE, p0[1] * SCALE, Z_LANE_LO * SCALE)
            ]
            polys_3d.append((back_shoulder, ROCK_C, 1))

            # Center green mountain climbing lane
            center_green_lane = [
                (p0[0] * SCALE, p0[1] * SCALE, Z_LANE_LO * SCALE),
                (p1[0] * SCALE, p1[1] * SCALE, Z_LANE_LO * SCALE),
                (p1[0] * SCALE, p1[1] * SCALE, Z_LANE_HI * SCALE),
                (p0[0] * SCALE, p0[1] * SCALE, Z_LANE_HI * SCALE)
            ]
            polys_3d.append((center_green_lane, t_info['top_color'], 2))

            # Outer front brown rock shoulder
            front_shoulder = [
                (p0[0] * SCALE, p0[1] * SCALE, Z_LANE_HI * SCALE),
                (p1[0] * SCALE, p1[1] * SCALE, Z_LANE_HI * SCALE),
                (p1[0] * SCALE, p1[1] * SCALE, Z_TERRAIN_HI * SCALE),
                (p0[0] * SCALE, p0[1] * SCALE, Z_TERRAIN_HI * SCALE)
            ]
            polys_3d.append((front_shoulder, ROCK_C, 1))

            # Side walls (brown rock)
            num_v = len(v_back)
            for i in range(num_v):
                idx_c, idx_n = i, (i + 1) % num_v
                wall = [v_back[idx_c], v_back[idx_n], v_front[idx_n], v_front[idx_c]]
                polys_3d.append((wall, ROCK_C, 0))

        # ----------------------------------------------------------------------
        # C. Extrude Goat Body Parts (Torso, Head, Left/Right Horns, Legs)
        # ----------------------------------------------------------------------
        # (body_obj, z0, z1, color1, color2, no_fpv) -- goat is fully visible in the third-person FPV
        body_depths = [
            (base.torso, Z_TORSO_LO, Z_TORSO_HI, GOAT_FUR, GOAT_FUR, False),
            (base.head,  Z_TORSO_LO, Z_TORSO_HI, GOAT_FUR, GOAT_FUR, False),
        ]

        if hasattr(base, 'horn_left'):
            body_depths.append((base.horn_left,  Z_HORN_LEFT_LO,  Z_HORN_LEFT_HI,  GOAT_HORN, GOAT_HORN, False))
        if hasattr(base, 'horn_right'):
            body_depths.append((base.horn_right, Z_HORN_RIGHT_LO, Z_HORN_RIGHT_HI, GOAT_HORN, GOAT_HORN, False))

        # Add legs
        for idx, leg_body in enumerate(base.legs):
            if idx < 4:  # Near legs
                body_depths.append((leg_body, Z_LEG_NEAR_LO, Z_LEG_NEAR_HI, GOAT_FUR, GOAT_HOOF, False))
            else:        # Far legs
                body_depths.append((leg_body, Z_LEG_FAR_LO,  Z_LEG_FAR_HI,  GOAT_FAR_FUR, GOAT_HOOF, False))

        for body_obj, z0, z1, c1, c2, no_fpv in body_depths:
            for f in body_obj.fixtures:
                if isinstance(f.shape, circleShape):
                    pos = body_obj.transform * f.shape.pos
                    rad = f.shape.radius
                    angles = np.linspace(0, 2*math.pi, 8, endpoint=False)
                    v2d = [(pos[0] + rad*math.cos(a), pos[1] + rad*math.sin(a)) for a in angles]
                elif hasattr(f.shape, 'vertices'):
                    v2d = [(body_obj.transform * v) for v in f.shape.vertices]
                else:
                    continue

                v_back  = [(p[0] * SCALE, p[1] * SCALE, z0 * SCALE) for p in v2d]
                v_front = [(p[0] * SCALE, p[1] * SCALE, z1 * SCALE) for p in v2d]

                polys_3d.append((v_back,  c1, 3, no_fpv))
                polys_3d.append((v_front, c1, 3, no_fpv))

                num_v = len(v_back)
                for i in range(num_v):
                    idx_c, idx_n = i, (i + 1) % num_v
                    wall = [v_back[idx_c], v_back[idx_n], v_front[idx_n], v_front[idx_c]]
                    polys_3d.append((wall, c2 if i % 2 == 0 else c1, 3, no_fpv))

        # ----------------------------------------------------------------------
        # D. Projections to 4 Viewports
        # ----------------------------------------------------------------------
        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []
        cx, cy = self.VW / 2.0, self.VH / 2.0

        goat_x = base.torso.position.x * SCALE
        goat_y = base.torso.position.y * SCALE

        # Get dynamically calculated (orbited) position from our property; project from the anchor
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        anchor_x, anchor_y, anchor_z = self._cam_anchor

        for poly_item in polys_3d:
            if len(poly_item) == 4:
                p_verts, color, layer, no_fpv = poly_item
            else:
                p_verts, color, layer = poly_item
                no_fpv = False

            pts_t, pts_r, pts_s = [], [], []
            dt, dr, ds = [], [], []

            for vx, vy, vz in p_verts:
                rel_x = vx - goat_x
                rel_y = vy - goat_y

                # A. Top-Down: X horizontal (-X left), Z vertical
                pts_t.append((rel_x + cx, vz + cy))
                dt.append(-vy)

                # B. Rear: Z horizontal, Y vertical (looking along +X from East)
                pts_r.append((vz + cx, self.VH - (rel_y + cy)))
                dr.append(rel_x)

                # C. Side: X horizontal, Y vertical (native 2D side view)
                pts_s.append((rel_x + cx, self.VH - (rel_y + cy)))
                ds.append(vz)

            if len(pts_t) >= 3: proj_top.append((sum(dt)/len(dt), pts_t, color, layer))
            if len(pts_r) >= 3: proj_rear.append((sum(dr)/len(dr), pts_r, color, layer))
            if len(pts_s) >= 3: proj_side.append((sum(ds)/len(ds), pts_s, color, layer))

            if no_fpv:
                continue

            # D. 3DV / FPV: Perspective looking WEST (-X direction) up mountain with Sutherland-Hodgman near plane clipping
            # Camera-relative points are orbited first so the near plane clips on the true (orbited) depth
            rel_verts_fpv = []
            for vx, vy, vz in p_verts:
                dx = anchor_x - vx     # Positive when objects are West of the camera (in front)
                dy = vy - anchor_y     # Vertical relative to camera
                dz = anchor_z - vz     # Lateral relative to camera
                d_o, dz_o, dy_o = orbit_camera_frame(dx, dz, dy, self.cam_yaw, self.cam_pitch, self.pivot_depth)
                rel_verts_fpv.append((d_o, dy_o, dz_o))

            clipped_fpv = []
            near = 0.5 * SCALE  # 0.5 metres in front of camera
            num_v = len(rel_verts_fpv)
            for i in range(num_v):
                p_curr = rel_verts_fpv[i]
                p_prev = rel_verts_fpv[i - 1]

                if p_curr[0] >= near:
                    if p_prev[0] < near:
                        t = (near - p_prev[0]) / (p_curr[0] - p_prev[0])
                        y_int = p_prev[1] + t * (p_curr[1] - p_prev[1])
                        z_int = p_prev[2] + t * (p_curr[2] - p_prev[2])
                        clipped_fpv.append((near, y_int, z_int))
                    clipped_fpv.append(p_curr)
                elif p_prev[0] >= near:
                    t = (near - p_prev[0]) / (p_curr[0] - p_prev[0])
                    y_int = p_prev[1] + t * (p_curr[1] - p_prev[1])
                    z_int = p_prev[2] + t * (p_curr[2] - p_prev[2])
                    clipped_fpv.append((near, y_int, z_int))

            if len(clipped_fpv) >= 3:
                pts_fpv = []
                for d_c, dy_rot, dz_rot in clipped_fpv:
                    if d_c > 0.1:
                        px = (dz_rot / d_c) * self.fov + cx
                        py = self.VH - ((dy_rot / d_c) * self.fov + cy)
                        pts_fpv.append((px, py))
                depth_c = sum(c[0] for c in clipped_fpv) / len(clipped_fpv)
                if len(pts_fpv) >= 3:
                    proj_fpv.append((depth_c, pts_fpv, color, layer))

        # ----------------------------------------------------------------------
        # D. Z-Index Sorting & Rendering to Surfaces
        # ----------------------------------------------------------------------
        for proj_list, surf in [(proj_top, top_surf), (proj_rear, rear_surf),
                                (proj_side, side_surf), (proj_fpv, fpv_surf)]:
            proj_list.sort(key=lambda item: (item[3], -item[0]))
            for _, pts, color, _ in proj_list:
                safe_c = (int(color[0]), int(color[1]), int(color[2]))
                ipts = [(int(round(px)), int(round(py))) for px, py in pts]
                if len(ipts) >= 3:
                    try:
                        gfxdraw.aapolygon(surf, ipts, safe_c)
                        gfxdraw.filled_polygon(surf, ipts, safe_c)
                    except Exception:
                        pass

        # Draw glowing joint markers on side elevation
        for j in base.joints[:4]:
            jx = int((j.anchorA.x * SCALE - goat_x) + cx)
            jy = int(self.VH - ((j.anchorA.y * SCALE - goat_y) + cy))
            if 0 <= jx < self.VW and 0 <= jy < self.VH:
                gfxdraw.aacircle(side_surf, jx, jy, 4, JOINT_CYAN)
                gfxdraw.filled_circle(side_surf, jx, jy, 4, JOINT_CYAN)

        return [np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))
                for s in [top_surf, rear_surf, side_surf, fpv_surf]]
