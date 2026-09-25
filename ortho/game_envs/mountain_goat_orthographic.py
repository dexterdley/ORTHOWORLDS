"""
mountain_goat_orthographic.py
-----------------------------
Mountain Goat 🐐 Environment & 3D Multi-View Orthographic Wrapper

Task:
  A mountain goat climbs a steep, procedural mountain cliff moving East-to-West (-X direction, climbing upwards +Y).
  The agent controls torso pitch lean and leg joint motors to traverse rocky ledges without tumbling downhill.

Views returned by MountainGoatOrthographicWrapper.render():
  1. Top-Down  : X-Z plan view (looking down from above, centered on goat)
  2. Rear      : Z-Y elevation (looking from behind the goat along +X)
  3. Side      : X-Y elevation (native 2D side-view of the mountain climb)
  4. 3DV / FPV : True 3D perspective from the goat's head looking West (-X) up the mountain slope

Fully compatible with Gym/Gymnasium and Box2D SWIG bindings.
"""

import math
import numpy as np
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
    fixtureDef,
    rayCastCallback,
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
FPS = 50
SCALE = 28.0            # pixels per metre
GRAVITY = -15.0         # mountain gravity

# Mountain bounds (starts at X=60 on East, climbs West to X=0, Y from 2.0 to 25.0)
MOUNTAIN_START_X = 60.0 # East start
MOUNTAIN_PEAK_X  = 5.0  # West peak goal
MOUNTAIN_MAX_H   = 22.0 # Height at peak

# Goat physics sizes
TORSO_W, TORSO_H = 0.6, 0.35    # torso width, height (metres)
HEAD_R           = 0.22         # head radius
HORN_W, HORN_H   = 0.08, 0.30   # horn size

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
    def __init__(self, ignore_body=None):
        super().__init__()
        self.fraction = 1.0
        self.ignore_body = ignore_body

    def ReportFixture(self, fixture, point, normal, fraction):
        if self.ignore_body and fixture.body == self.ignore_body:
            return -1.0  # Ignore self
        if fraction < self.fraction:
            self.fraction = fraction
        return fraction


# ==============================================================================
# 1. MOUNTAIN GOAT BASE ENVIRONMENT
# ==============================================================================
class MountainGoatEnv(gym.Env):
    """
    2D Physics Environment where a quadrupedal Mountain Goat climbs a steep cliff from East to West.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Actions (5D):
        # 0: Torso lean torque [-1..1]
        # 1: Front hip motor speed [-1..1]
        # 2: Front knee motor speed [-1..1]
        # 3: Rear hip motor speed [-1..1]
        # 4: Rear knee motor speed [-1..1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Obs (22D):
        # 0-3: Torso X, Y, Vx, Vy
        # 4-5: Torso angle, angular velocity
        # 6-13: 4 Joint angles & speeds (front hip, front knee, rear hip, rear knee)
        # 14-21: 8 Lidar terrain distance sensors surrounding the goat
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)

        self.world = None
        self.bodies = []
        self.terrain_polys = []
        self.joints = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world = world(gravity=(0.0, GRAVITY), doSleep=True)
        self.bodies = []
        self.terrain_polys = []
        self.joints = []

        # Generate Procedural Mountain Cliff (East to West: X from 65 down to 0)
        self._generate_mountain()

        # Spawn Goat near East foot of mountain (X=58.0, Y calculated from slope)
        start_x = 58.0
        start_y = self._get_mountain_y(start_x) + 2.5
        self._spawn_goat(start_x, start_y)

        self.prev_dist = start_x  # Distance remaining to peak (Peak is at X=5.0)
        self.prev_height = start_y
        self.step_count = 0

        return self._get_obs(), {}

    def _get_mountain_y(self, x):
        """Returns height of mountain surface at X position."""
        # Steep incline climbing from East (x=60, y=2) up to West peak (x=5, y=22)
        progress = (MOUNTAIN_START_X - x) / (MOUNTAIN_START_X - MOUNTAIN_PEAK_X)
        progress = np.clip(progress, 0.0, 1.2)
        base_h = 2.0 + 20.0 * (progress ** 1.3)
        # Add jagged mountain ledges using sines
        ledges = 0.8 * math.sin(x * 0.5) + 0.4 * math.sin(x * 1.2)
        return max(1.0, base_h + ledges)

    def _generate_mountain(self):
        """Creates static Box2D bodies and 3D polygon structures for the mountain."""
        x_steps = np.arange(MOUNTAIN_START_X + 5.0, MOUNTAIN_PEAK_X - 5.0, -2.5)
        
        ground_body = self.world.CreateStaticBody(position=(0, 0))
        self.bodies.append(ground_body)

        # Light gray floor outside mountain (East of start and West of peak)
        floor_poly_east = [(MOUNTAIN_START_X + 5.0, -10.0), (100.0, -10.0), (100.0, 1.5), (MOUNTAIN_START_X + 5.0, 1.5)]
        ground_body.CreatePolygonFixture(vertices=floor_poly_east, friction=1.0)
        self.terrain_polys.append({
            'poly': floor_poly_east,
            'top_edge': ((MOUNTAIN_START_X + 5.0, 1.5), (100.0, 1.5)),
            'top_color': FLOOR_C,
            'body_color': FLOOR_C
        })

        for i in range(len(x_steps) - 1):
            x0, x1 = x_steps[i], x_steps[i+1]
            y0, y1 = self._get_mountain_y(x0), self._get_mountain_y(x1)

            # Ground segment poly (top edge from (x0, y0) to (x1, y1), bottom down to y=-5)
            poly_2d = [(x0, -5.0), (x1, -5.0), (x1, y1), (x0, y0)]
            ground_body.CreatePolygonFixture(vertices=poly_2d, friction=1.2, restitution=0.0)

            # High altitude has snow top, lower has grass top
            is_snow = ((y0 + y1) / 2.0) > 12.0
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
            cy = 24.0 + 3.0 * math.sin(cx * 0.4)
            cz = 4.0 * math.cos(cx * 0.7)
            cw, ch, cd = 4.5, 1.4, 2.5
            self.clouds.append({'x': cx, 'y': cy, 'z': cz, 'w': cw, 'h': ch, 'd': cd})

    def _spawn_goat(self, x, y):
        """Builds multi-body Mountain Goat with joints."""
        # 1. Main Torso
        self.torso = self.world.CreateDynamicBody(position=(x, y), angle=0.1)
        self.torso.CreatePolygonFixture(box=(TORSO_W, TORSO_H), density=6.0, friction=0.8)
        self.bodies.append(self.torso)

        # 2. Head & Left/Right Horns (facing West / -X side of torso)
        head_pos = (x - TORSO_W - 0.1, y + TORSO_H + 0.15)
        self.head = self.world.CreateDynamicBody(position=head_pos)
        self.head.CreateCircleFixture(radius=HEAD_R, density=2.0)
        self.bodies.append(self.head)

        # Weld head to torso
        self.world.CreateJoint(revoluteJointDef(
            bodyA=self.torso, bodyB=self.head,
            localAnchorA=(-TORSO_W, TORSO_H), localAnchorB=(0, 0),
            enableLimit=True, lowerAngle=-0.2, upperAngle=0.2
        ))

        # Left & Right Horns (curved backward/upward relative to head)
        horn_poly = [(0, 0), (-0.15, HORN_H), (0.05, HORN_H * 0.85)]
        
        self.horn_left = self.world.CreateDynamicBody(position=(head_pos[0] - 0.1, head_pos[1] + HEAD_R))
        self.horn_left.CreatePolygonFixture(vertices=horn_poly, density=0.4)
        self.bodies.append(self.horn_left)
        self.world.CreateJoint(revoluteJointDef(
            bodyA=self.head, bodyB=self.horn_left,
            localAnchorA=(-0.1, HEAD_R), localAnchorB=(0, 0),
            enableLimit=True, lowerAngle=-0.1, upperAngle=0.1
        ))

        self.horn_right = self.world.CreateDynamicBody(position=(head_pos[0] - 0.1, head_pos[1] + HEAD_R))
        self.horn_right.CreatePolygonFixture(vertices=horn_poly, density=0.4)
        self.bodies.append(self.horn_right)
        self.world.CreateJoint(revoluteJointDef(
            bodyA=self.head, bodyB=self.horn_right,
            localAnchorA=(-0.1, HEAD_R), localAnchorB=(0, 0),
            enableLimit=True, lowerAngle=-0.1, upperAngle=0.1
        ))

        # 3. Legs & Hooves (Front = West side / -X, Rear = East side / +X)
        self.legs = []
        self.joints = []

        leg_configs = [
            # (name, torso_anchor_x, thigh_len, shin_len, is_front)
            ("front_near", -TORSO_W + 0.15, 0.45, 0.45, True),
            ("rear_near",   TORSO_W - 0.15, 0.45, 0.45, False),
            ("front_far",  -TORSO_W + 0.15, 0.45, 0.45, True),
            ("rear_far",    TORSO_W - 0.15, 0.45, 0.45, False),
        ]

        for name, ax, thigh_h, shin_h, is_front in leg_configs:
            # Thigh
            thigh_pos = (x + ax, y - TORSO_H - thigh_h/2)
            thigh = self.world.CreateDynamicBody(position=thigh_pos)
            thigh.CreatePolygonFixture(box=(0.08, thigh_h/2), density=1.5, friction=0.9)
            self.bodies.append(thigh)

            # Shin / Hoof
            shin_pos = (x + ax, y - TORSO_H - thigh_h - shin_h/2)
            shin = self.world.CreateDynamicBody(position=shin_pos)
            shin.CreatePolygonFixture(box=(0.07, shin_h/2), density=1.5, friction=1.5)
            self.bodies.append(shin)

            # Hip Joint
            hip_j = self.world.CreateJoint(revoluteJointDef(
                bodyA=self.torso, bodyB=thigh,
                localAnchorA=(ax, -TORSO_H), localAnchorB=(0, thigh_h/2),
                enableMotor=True, maxMotorTorque=250.0, enableLimit=True,
                lowerAngle=-0.8, upperAngle=0.8
            ))

            # Knee Joint
            knee_j = self.world.CreateJoint(revoluteJointDef(
                bodyA=thigh, bodyB=shin,
                localAnchorA=(0, -thigh_h/2), localAnchorB=(0, shin_h/2),
                enableMotor=True, maxMotorTorque=200.0, enableLimit=True,
                lowerAngle=-1.2, upperAngle=0.3
            ))

            self.legs.extend([thigh, shin])
            if "near" in name:
                self.joints.extend([hip_j, knee_j])

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # 1. Apply Actions
        # Torso pitch lean
        self.torso.ApplyTorque(float(action[0] * 120.0), True)
        # Leg joint motor speeds
        if len(self.joints) >= 4:
            self.joints[0].motorSpeed = float(action[1] * 5.0)  # Front Hip
            self.joints[1].motorSpeed = float(action[2] * 5.0)  # Front Knee
            self.joints[2].motorSpeed = float(action[3] * 5.0)  # Rear Hip
            self.joints[3].motorSpeed = float(action[4] * 5.0)  # Rear Knee

        # Step physics
        self.world.Step(1.0 / FPS, 6, 2)

        obs = self._get_obs()

        # 2. Rewards
        curr_x = float(self.torso.position.x)
        curr_y = float(self.torso.position.y)

        # Westward progress (+reward for moving -X toward peak) + Upward progress (+Y)
        westward_progress = self.prev_dist - curr_x
        upward_progress   = curr_y - self.prev_height
        reward = float(westward_progress * 3.0 + upward_progress * 2.0)

        # Small energy penalty
        reward -= float(0.01 * np.sum(action ** 2))

        self.prev_dist = curr_x
        self.prev_height = curr_y

        terminated = False

        # Victory: Reached mountain peak zone (X <= 8.0, Y >= 18.0)
        if curr_x <= MOUNTAIN_PEAK_X + 3.0 and curr_y >= MOUNTAIN_MAX_H - 3.0:
            reward += 100.0
            terminated = True
        # Failure: Tipped over or fell below cliff
        elif abs(self.torso.angle) > 1.4 or curr_y < 0.5:
            reward -= 50.0
            terminated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        t_pos = self.torso.position
        t_vel = self.torso.linearVelocity

        obs = [
            float(t_pos.x), float(t_pos.y),
            float(t_vel.x), float(t_vel.y),
            float(self.torso.angle), float(self.torso.angularVelocity)
        ]

        # 4 active joint states
        for j in self.joints[:4]:
            obs.append(float(j.angle))
            obs.append(float(j.speed))

        # 8-ray Lidar scan around goat
        lidar_range = 8.0
        for i in range(8):
            ang = self.torso.angle + (-math.pi/2.0 + i * (math.pi / 7.0))
            p1 = t_pos
            p2 = (t_pos.x + lidar_range * math.cos(ang), t_pos.y + lidar_range * math.sin(ang))
            cb = LidarCallback(ignore_body=self.torso)
            self.world.RayCast(cb, p1, p2)
            obs.append(float(cb.fraction))

        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return
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

        # Draw goat bodies
        for body in self.bodies:
            if body == self.bodies[0]:  # terrain body
                continue
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
                    c = GOAT_HORN if body == self.horn else GOAT_FUR
                    pygame.draw.polygon(self.screen, c, spts)
                    pygame.draw.polygon(self.screen, OUTLINE_C, spts, 1)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


# ==============================================================================
# 2. THE ORTHOGRAPHIC MULTI-VIEW WRAPPER FOR MOUNTAIN GOAT
# ==============================================================================
class MountainGoatOrthographicWrapper(gym.Wrapper):
    """
    Extrudes 2D MountainGoatEnv into 3D multi-views:
      - Top-Down  : X-Z plan view looking down
      - Rear      : Z-Y elevation looking along +X from behind
      - Side      : X-Y elevation (the 2D mountain side view)
      - 3DV / FPV : 3D Perspective looking West (-X) up the mountain slope from the goat's head!
    """
    def __init__(self, env):
        super().__init__(env)
        self.SCALE = SCALE
        self.VW = 800
        self.VH = 600
        self.fov = 320.0

    def render(self):
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
        # (body_obj, z0, z1, color1, color2, no_fpv)
        body_depths = [
            (base.torso, Z_TORSO_LO, Z_TORSO_HI, GOAT_FUR, GOAT_FUR, True),   # no_fpv = True
            (base.head,  Z_TORSO_LO, Z_TORSO_HI, GOAT_FUR, GOAT_FUR, True),   # no_fpv = True
        ]

        if hasattr(base, 'horn_left'):
            body_depths.append((base.horn_left,  Z_HORN_LEFT_LO,  Z_HORN_LEFT_HI,  GOAT_HORN, GOAT_HORN, False))
        if hasattr(base, 'horn_right'):
            body_depths.append((base.horn_right, Z_HORN_RIGHT_LO, Z_HORN_RIGHT_HI, GOAT_HORN, GOAT_HORN, False))

        # Add legs (no_fpv = True)
        for idx, leg_body in enumerate(base.legs):
            if idx < 4:  # Near legs
                body_depths.append((leg_body, Z_LEG_NEAR_LO, Z_LEG_NEAR_HI, GOAT_FUR, GOAT_HOOF, True))
            else:        # Far legs
                body_depths.append((leg_body, Z_LEG_FAR_LO,  Z_LEG_FAR_HI,  GOAT_FAR_FUR, GOAT_HOOF, True))

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

        # 3DV / FPV Camera: Located at front of goat's head looking West (-X direction)
        cam_x = (base.head.position.x - 0.2 if hasattr(base, 'head') else base.torso.position.x) * SCALE
        cam_y = (base.head.position.y + 0.1 if hasattr(base, 'head') else base.torso.position.y) * SCALE
        cam_z = 0.0

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
            rel_verts_fpv = []
            for vx, vy, vz in p_verts:
                dx = cam_x - vx          # Positive when objects are West of the camera (in front)
                dy = vy - cam_y          # Vertical relative to head
                dz = cam_z - vz          # Lateral relative to head
                rel_verts_fpv.append((dx, dy, dz))

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
                for dx_c, dy_c, dz_c in clipped_fpv:
                    px = (dz_c / dx_c) * self.fov + cx
                    py = self.VH - ((dy_c / dx_c) * self.fov + cy)
                    pts_fpv.append((px, py))
                depth_c = sum(c[0] for c in clipped_fpv) / len(clipped_fpv)
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