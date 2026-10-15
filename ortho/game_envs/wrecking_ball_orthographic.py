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
import cv2
from .camera_utils import orbit_camera_frame, orbit_camera_offset
from Box2D.b2 import (world, polygonShape, circleShape, revoluteJointDef, ropeJointDef)

FPS = 50
SCALE = 30.0

# World layout (metres)
GROUND_TOP = 2.0
GROUND_HALF_W = 60.0
CHASSIS_HALF = (2.5, 1.0)
BOOM_PIVOT = (1.5, 1.0)          # Boom hinge on the chassis (chassis-local)
BOOM_HALF_LEN = 3.5
BOOM_TIP = 3.3                   # Hinge / cable sheave at +-BOOM_TIP along the boom
BOOM_INIT_ANGLE = 0.5            # Boom starts raised (rad above horizontal)
BALL_RADIUS = 0.7
CABLE_INIT, CABLE_MIN, CABLE_MAX = 3.0, 1.5, 5.5
BLOCK_HALF = (0.6, 0.7)
BLOCKS_PER_PILLAR = 3
SITE_X = 20.0                    # Nominal centre between the two pillars
BARRIER_X, BARRIER_HALF = 11.0, (0.25, 0.5)   # Low site barrier: stops the chassis, ball swings over it

# Actuator limits
DRIVE_SPEED = 4.0                # m/s target chassis speed at |a0| = 1
DRIVE_FORCE = 1500.0             # N, max drive (traction) force
BOOM_SPEED = 0.6                 # rad/s at |a1| = 1
WINCH_SPEED = 1.5                # m/s cable reel rate at |a2| = 1

# Reward
BLOCK_REWARD = 15.0
SUCCESS_BONUS = 50.0
FAIL_PENALTY = 20.0
APPROACH_COEF = 0.1


class WreckingBallEnv(gym.Env):
    """
    Wrecking Ball Task:
    Drive a crane, raise/lower its boom and reel its cable to swing a heavy
    cable-suspended wrecking ball over a low site barrier and knock down
    two pillars of stacked concrete blocks.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, max_episode_steps=1000):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.screen = None
        self.clock = None

        # Actions:
        # 0: Chassis drive, target speed (+ = forward towards the pillars, - = reverse)
        # 1: Boom luff speed (+ = raise boom, - = lower boom)
        # 2: Winch (+ = reel cable in / hoist ball up, - = pay cable out / lower ball)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # State observation:
        # 0-2: Chassis (X, Vx, tilt angle)
        # 3-4: Boom (angle above horizontal, angular speed)
        # 5: Cable length
        # 6-9: Ball relative to boom tip (dX, dY), ball velocity (Vx, Vy)
        # 10-11: Ball X relative to pillar site centre, ball Y
        # 12-13: Pillar 1 / Pillar 2 base X relative to chassis X
        # 14-15: Pillar 1 / Pillar 2 fraction of blocks still standing
        # 16: Fraction of all blocks knocked down
        # 17: Fraction of episode time remaining
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(18,), dtype=np.float32)

        self.world = None
        self.chassis = None
        self.bodies = []

    def _destroy(self):
        if self.world is None:
            return
        for body in self.bodies:
            self.world.DestroyBody(body)
        self.bodies = []
        self.world = None
        self.chassis = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        # Fresh world every episode: no leftover contacts / sleeping state between episodes
        self.world = world(gravity=(0, -10), doSleep=True)
        rng = self.np_random

        # 1. Terrain Ground (top surface at y = GROUND_TOP)
        self.terrain = self.world.CreateStaticBody(position=(0, 0), shapes=polygonShape(box=(GROUND_HALF_W, GROUND_TOP)))
        self.terrain.fixtures[0].friction = 0.8
        self.bodies.append(self.terrain)

        # Low concrete site barrier: the chassis cannot drive through the pillars, the ball swings over it
        self.barrier = self.world.CreateStaticBody(
            position=(BARRIER_X, GROUND_TOP + BARRIER_HALF[1]), shapes=polygonShape(box=BARRIER_HALF))
        self.bodies.append(self.barrier)

        # 2. Crane Chassis (Base Vehicle), resting on the ground. Crane parts share a negative
        # group index so the chassis, boom and ball never collide with each other.
        chassis_x = float(rng.uniform(-0.5, 0.5))
        self.chassis = self.world.CreateDynamicBody(position=(chassis_x, GROUND_TOP + CHASSIS_HALF[1]))
        self.chassis.CreatePolygonFixture(box=CHASSIS_HALF, density=10.0, friction=0.2, groupIndex=-1)
        self.bodies.append(self.chassis)

        # 3. Crane Boom (Arm), created at the hinge-consistent pose for BOOM_INIT_ANGLE
        ca, sa = math.cos(BOOM_INIT_ANGLE), math.sin(BOOM_INIT_ANGLE)
        pivot = (chassis_x + BOOM_PIVOT[0], GROUND_TOP + CHASSIS_HALF[1] + BOOM_PIVOT[1])
        self.boom = self.world.CreateDynamicBody(position=(pivot[0] + BOOM_TIP * ca, pivot[1] + BOOM_TIP * sa),
                                                 angle=BOOM_INIT_ANGLE)
        self.boom.CreatePolygonFixture(box=(BOOM_HALF_LEN, 0.3), density=1.5, friction=0.5, groupIndex=-1)
        self.bodies.append(self.boom)

        # Joint between Chassis and Boom (joint angle = boom angle above the chassis deck)
        self.boom_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.chassis, bodyB=self.boom, localAnchorA=BOOM_PIVOT, localAnchorB=(-BOOM_TIP, 0),
            referenceAngle=0.0, enableMotor=True, maxMotorTorque=6000,
            enableLimit=True, lowerAngle=-0.1, upperAngle=1.2
        ))

        # 4. Heavy Wrecking Ball, hanging straight below the boom tip on a taut cable
        tip = self.boom.GetWorldPoint((BOOM_TIP, 0.0))
        self.cable_length = CABLE_INIT
        self.ball = self.world.CreateDynamicBody(position=(tip.x, tip.y - self.cable_length),
                                                 linearDamping=0.02, angularDamping=0.1)
        self.ball.CreateCircleFixture(radius=BALL_RADIUS, density=6.0, friction=0.8, restitution=0.2,
                                      groupIndex=-1) # Heavy steel ball!
        self.bodies.append(self.ball)

        # 5. Cable: rope constraint (max distance) from boom tip sheave to the ball centre; the
        # winch changes its length. Rendered as a straight steel cable (see cable_points()).
        self.cable_joint = self.world.CreateJoint(ropeJointDef(
            bodyA=self.boom, bodyB=self.ball, localAnchorA=(BOOM_TIP, 0.0), localAnchorB=(0.0, 0.0),
            maxLength=self.cable_length
        ))

        # 6. Target Structural Pillars (2 Vertical Columns of Stacked Dynamic Concrete Blocks),
        # stacked in exact resting contact so nothing settles at the start
        self.site_x = SITE_X + float(rng.uniform(-0.5, 0.5))
        self.pillars = []
        self.pillar_x = []
        self.block_rest = []
        for side in (-1.0, 1.0):
            px = self.site_x + side * 0.9 + float(rng.uniform(-0.1, 0.1))
            self.pillar_x.append(px)
            column_blocks = []
            for h_idx in range(BLOCKS_PER_PILLAR): # 3 stacked blocks per pillar
                py = GROUND_TOP + BLOCK_HALF[1] + h_idx * 2 * BLOCK_HALF[1]
                block = self.world.CreateDynamicBody(position=(px, py))
                block.CreatePolygonFixture(box=BLOCK_HALF, density=2.0, friction=0.8, restitution=0.1)
                self.bodies.append(block)
                column_blocks.append(block)
                self.block_rest.append((px, py))
            self.pillars.append(column_blocks)
        self.blocks = [b for col in self.pillars for b in col]
        # Goal blocks: everything above the foundation row (pillars must be smashed down to their
        # bottom block, i.e. below TARGET_HEIGHT). Bottom blocks are tracked but not rewarded.
        self.is_target = [h_idx >= 1 for _ in self.pillars for h_idx in range(BLOCKS_PER_PILLAR)]
        self.num_targets = sum(self.is_target)
        self.knocked = [False] * len(self.blocks)

        self.steps = 0
        self.prev_potential = self._approach_potential()

        return self._get_obs(), self._get_info()

    def _update_knocked(self):
        """Latch blocks toppled / displaced from their stacks (monotone, rotation-aware): a block is down once
        it has slid off its stack by more than half its width, dropped by half its height, or tilted > ~30 deg.
        Returns the number of newly knocked goal blocks."""
        newly = 0
        for i, block in enumerate(self.blocks):
            if self.knocked[i]:
                continue
            rx, ry = self.block_rest[i]
            tilt = abs((block.angle + math.pi) % (2 * math.pi) - math.pi)
            if (abs(block.position.x - rx) > BLOCK_HALF[0] or ry - block.position.y > BLOCK_HALF[1]
                    or tilt > 0.5):
                self.knocked[i] = True
                newly += int(self.is_target[i])
        return newly

    def _targets_knocked(self, pillar=None):
        idx = range(len(self.blocks)) if pillar is None else \
            range(pillar * BLOCKS_PER_PILLAR, (pillar + 1) * BLOCKS_PER_PILLAR)
        return sum(1 for i in idx if self.is_target[i] and self.knocked[i])

    def _approach_potential(self):
        # Potential on ball distance to the pillar site (bounded, telescoping => not farmable)
        d = max(0.0, abs(self.ball.position.x - self.site_x) - 2.0)
        return -APPROACH_COEF * min(d, 30.0)

    def boom_tip(self):
        return self.boom.GetWorldPoint((BOOM_TIP, 0.0))

    def cable_points(self):
        """Cable polyline in world metres: boom tip sheave -> top of the ball (straight when taut, sags when slack)."""
        tip = self.boom_tip()
        bp = self.ball.position
        dx, dy = bp.x - tip.x, bp.y - tip.y
        dist = max(math.hypot(dx, dy), 1e-6)
        end = (bp.x - dx / dist * BALL_RADIUS, bp.y - dy / dist * BALL_RADIUS)
        span = max(dist - BALL_RADIUS, 0.0)
        slack = max(self.cable_length - dist, 0.0)
        sag = min(0.5 * math.sqrt(slack * max(span, 0.0)), 1.5)
        pts = []
        n = 6
        for k in range(n + 1):
            t = k / n
            pts.append((tip.x + (end[0] - tip.x) * t, tip.y + (end[1] - tip.y) * t - sag * 4 * t * (1 - t)))
        return pts

    def step(self, action):
        if self.world is None:
            self.reset()
        action = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(-1)[:3]), -1.0, 1.0)

        # Action execution
        # 0: drive -> traction force servoing chassis speed, applied at the track contact line
        mass = self.chassis.mass + self.boom.mass
        target_v = float(action[0]) * DRIVE_SPEED
        force = float(np.clip(12.0 * mass * (target_v - self.chassis.linearVelocity.x), -DRIVE_FORCE, DRIVE_FORCE))
        self.chassis.ApplyForce((force, 0.0), self.chassis.GetWorldPoint((0.0, -CHASSIS_HALF[1])), True)
        # 1: boom luff motor
        self.boom_joint.motorSpeed = float(action[1]) * BOOM_SPEED
        # 2: winch reels the cable (positive = hoist = shorter cable)
        self.cable_length = float(np.clip(self.cable_length - float(action[2]) * WINCH_SPEED / FPS, CABLE_MIN, CABLE_MAX))
        self.cable_joint.SetMaxLength(self.cable_length)
        self.ball.awake = True

        self.world.Step(1.0 / FPS, 8, 3)
        self.steps += 1

        # --- Reward Calculation ---
        # 1. Destruction: +BLOCK_REWARD per goal block newly knocked down (latched, so not farmable)
        newly = self._update_knocked()
        reward = BLOCK_REWARD * newly
        # 2. Approach shaping: potential-based on ball distance to the pillar site
        potential = self._approach_potential()
        reward += potential - self.prev_potential
        self.prev_potential = potential

        terminated = False

        # Success Condition: every goal block knocked down (both pillars smashed down to their foundations)
        if self._targets_knocked() == self.num_targets:
            reward += SUCCESS_BONUS
            terminated = True

        # Failure Condition: crane chassis tipped over or left the ground (fell off the world)
        elif (abs(self.chassis.angle) > np.pi / 3 or self.chassis.position.y < GROUND_TOP
              or abs(self.chassis.position.x) > GROUND_HALF_W):
            reward -= FAIL_PENALTY
            terminated = True

        truncated = (not terminated) and self.steps >= self.max_episode_steps
        obs = self._get_obs()

        if self.render_mode == "human": self.render()
        return obs, float(reward), terminated, truncated, self._get_info()

    def _get_info(self):
        n = self._targets_knocked()
        return {"blocks_knocked": int(n), "num_blocks": int(self.num_targets),
                "success": bool(n == self.num_targets), "cable_length": float(self.cable_length)}

    def _get_obs(self):
        tip = self.boom_tip()
        bp, bv = self.ball.position, self.ball.linearVelocity
        cx = self.chassis.position.x
        per_pillar = BLOCKS_PER_PILLAR - 1
        standing = [1.0 - self._targets_knocked(p) / per_pillar for p in range(2)]
        obs = np.array([
            cx, self.chassis.linearVelocity.x, self.chassis.angle,
            self.boom_joint.angle, self.boom_joint.speed,
            self.cable_length,
            bp.x - tip.x, bp.y - tip.y, bv.x, bv.y,
            bp.x - self.site_x, bp.y,
            self.pillar_x[0] - cx, self.pillar_x[1] - cx,
            standing[0], standing[1],
            self._targets_knocked() / self.num_targets,
            1.0 - self.steps / self.max_episode_steps,
        ], dtype=np.float32)
        return np.clip(np.nan_to_num(obs), -100.0, 100.0)

    def render(self):
        if self.render_mode is None or self.world is None: return
        screen_w, screen_h = 800, 600
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_w, screen_h)) if self.render_mode == "human" else pygame.Surface((screen_w, screen_h))
            self.clock = pygame.time.Clock()

        self.screen.fill((210, 235, 255))
        # Camera follows the chassis (chassis at ~1/4 of the screen width so the pillars stay in view)
        off_x = self.chassis.position.x * SCALE - screen_w * 0.25
        def to_screen(p):
            return (p[0] * SCALE - off_x, screen_h - p[1] * SCALE)

        # Color mapping
        colors = {
            self.terrain: (110, 190, 110),
            self.barrier: (140, 140, 150),
            self.chassis: (230, 160, 20),
            self.boom: (230, 160, 20),
            self.ball: (60, 60, 70),
        }
        for i, block in enumerate(self.blocks):
            colors[block] = (200, 150, 150) if self.knocked[i] else (160, 160, 170)

        for body in self.bodies:
            for fixture in body.fixtures:
                c = colors.get(body, (150, 150, 150))
                if isinstance(fixture.shape, circleShape):
                    pos = to_screen(body.transform * fixture.shape.pos)
                    pygame.draw.circle(self.screen, c, (int(pos[0]), int(pos[1])), int(fixture.shape.radius * SCALE))
                else:
                    vertices = [to_screen(body.transform * v) for v in fixture.shape.vertices]
                    pygame.draw.polygon(self.screen, c, vertices)
                    pygame.draw.polygon(self.screen, (0, 0, 0), vertices, 1)
        pygame.draw.lines(self.screen, (40, 40, 40), False, [to_screen(p) for p in self.cable_points()], 3)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            self.screen = None
        self._destroy()


# ==============================================================================
# 2. THE ORTHOGRAPHIC MULTI-VIEW WRAPPER FOR WRECKING BALL
# ==============================================================================
class WreckingBallOrthographicWrapper(gym.Wrapper):
    """
    Extrudes 2D Box2D Wrecking Ball components into 3D volumes and renders 4 views:
    1. Top-Down Plan View
    2. Orthographic Rear Elevation
    3. Orthographic Side Elevation
    4. True 3D Perspective third-person view: orbit camera trailing the crane chassis,
       crane fully visible (with Sutherland-Hodgman near-plane clipping)
    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """
    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.SCALE = SCALE
        self.VIEWPORT_W, self.VIEWPORT_H = 800, 600
        self.fov = 400.0

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state

        # Follow/Orbit camera baseline offsets (relative to chassis)
        self.camera_dist = 12.0 * self.SCALE   # Distance behind the crane chassis along -X
        self.camera_height = 4.5 * self.SCALE  # Vertical offset above the chassis centre

        # Orbit pivot: the chassis centre, camera_dist ahead of the trailing camera
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
        """Un-orbited 3D camera anchor (X, Y, Z) in world space, trailing behind the crane chassis."""
        base_env = self.env.unwrapped
        if not hasattr(base_env, 'chassis') or base_env.chassis is None:
            return None

        # Baseline trailing vector: camera_dist behind (-X) and camera_height above the chassis,
        # looking horizontally along +X towards the crane, ball and pillars
        cam_x = base_env.chassis.position.x * self.SCALE - self.camera_dist
        cam_y = base_env.chassis.position.y * self.SCALE + self.camera_height
        cam_z = 0.0

        return (cam_x, cam_y, cam_z)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the orbited camera in world space."""
        anchor = self._cam_anchor
        if anchor is None:
            return (0.0, 0.0, 0.0)

        # Camera frame (depth, lateral, vertical) -> world: depth = +x, lateral = -z, vertical = +y
        d_off, lat_off, vert_off = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)
        return (anchor[0] + d_off, anchor[1] + vert_off, anchor[2] - lat_off)

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

        base_env = self.env.unwrapped
        
        top_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        rear_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        side_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        fpv_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        
        sky_color = (210, 235, 255)
        for surf in [top_surf, rear_surf, side_surf, fpv_surf]: surf.fill(sky_color)
        if getattr(base_env, 'chassis', None) is None:
            return np.zeros((600, 800, 3)), np.zeros((600, 800, 3)), np.zeros((600, 800, 3)), np.zeros((600, 800, 3))

        polys_3d = []

        # 1. 3D Checkerboard Ground Grid
        for x in np.arange(-20.0, 40.0, 2.0):
            for z in np.arange(-15.0, 15.0, 2.0):
                c = (110, 190, 110) if (int(x/2.0) + int(z/2.0)) % 2 == 0 else (90, 170, 90)
                y = 2.0 * self.SCALE
                polys_3d.append(([(x*self.SCALE, y, z*self.SCALE), ((x+2)*self.SCALE, y, z*self.SCALE), 
                                  ((x+2)*self.SCALE, y, (z+2)*self.SCALE), (x*self.SCALE, y, (z+2)*self.SCALE)], c, 0))

        # 2. Extrude Objects into 3D Volumetric Mesh
        parts_z = {
            base_env.chassis: (-1.6, 1.6, (230, 160, 20), (180, 120, 15)), 
            base_env.boom: (-0.5, 0.5, (230, 160, 20), (180, 120, 15)),
            base_env.ball: (-0.7, 0.7, (60, 60, 70), (40, 40, 50)),
            base_env.barrier: (-3.0, 3.0, (150, 150, 155), (115, 115, 120)),
        }

        for col in base_env.pillars:
            for block in col:
                parts_z[block] = (-0.8, 0.8, (160, 160, 170), (120, 120, 130))

        def extrude(verts_2d, z_min, z_max, c1, c2):
            V_back = [(vx, vy, z_min * self.SCALE) for vx, vy in verts_2d]
            V_front = [(vx, vy, z_max * self.SCALE) for vx, vy in verts_2d]

            polys_3d.append((V_back, c1, 1))
            polys_3d.append((V_front, c1, 1))
            for i in range(len(V_back)):
                polys_3d.append(([V_back[i], V_back[(i+1)%len(V_back)], V_front[(i+1)%len(V_front)], V_front[i]], c2 if i % 2 == 0 else c1, 1))

        for body in base_env.bodies:
            if body in parts_z:
                z_min, z_max, c1, c2 = parts_z[body]
                for f in body.fixtures:
                    verts_2d = []
                    # Handle Polygon vs Circle shape extraction
                    if hasattr(f.shape, 'vertices'):
                        verts_2d = [(f.body.transform * v * self.SCALE) for v in f.shape.vertices]
                    elif hasattr(f.shape, 'pos'):
                        pos = f.body.transform * f.shape.pos
                        rad = f.shape.radius
                        verts_2d = [((pos[0] + rad*math.cos(a)) * self.SCALE, (pos[1] + rad*math.sin(a)) * self.SCALE)
                                    for a in np.linspace(0, 2*math.pi, 10, endpoint=False)]
                    extrude(verts_2d, z_min, z_max, c1, c2)

        # Steel cable: thin square-section prisms along the boom-tip -> ball polyline
        cable_pts = base_env.cable_points()
        hw = 0.07
        for (x0, y0), (x1, y1) in zip(cable_pts[:-1], cable_pts[1:]):
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 1e-4:
                continue
            nx, ny = -(y1 - y0) / seg_len * hw, (x1 - x0) / seg_len * hw
            verts_2d = [((x0 + nx) * self.SCALE, (y0 + ny) * self.SCALE), ((x1 + nx) * self.SCALE, (y1 + ny) * self.SCALE),
                        ((x1 - nx) * self.SCALE, (y1 - ny) * self.SCALE), ((x0 - nx) * self.SCALE, (y0 - ny) * self.SCALE)]
            extrude(verts_2d, -hw, hw, (40, 40, 40), (20, 20, 20))

        # 3. Multi-View Projections
        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []
        cy, cx = self.VIEWPORT_H / 2, self.VIEWPORT_W / 2
        
        track_x = base_env.chassis.position.x * self.SCALE
        # Get dynamically calculated (orbited) position from our property for logging;
        # projection uses the un-orbited anchor since orbit_camera_frame applies the orbit
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        anchor_x, anchor_y, anchor_z = self._cam_anchor

        for p_vertices, color, layer in polys_3d:
            pts_t, pts_r, pts_s = [], [], []
            depths_t, depths_r, depths_s = [], [], []
            
            for vx, vy, vz in p_vertices:
                vx_scrolled = vx - track_x
                pts_t.append((vx_scrolled + cx, vz + cy))
                depths_t.append(-vy) 
                pts_r.append((vz + cx, self.VIEWPORT_H - vy))
                depths_r.append(vx_scrolled) 
                pts_s.append((vx_scrolled + cx, self.VIEWPORT_H - vy))
                depths_s.append(vz) 

            # FPV Near-Plane Clipping (Sutherland-Hodgman), on the orbited camera depth
            clipped_pts = []
            rel_verts = [orbit_camera_frame(vx - anchor_x, -(vz - anchor_z), vy - anchor_y,
                                            self.cam_yaw, self.cam_pitch, self.pivot_depth)
                         for vx, vy, vz in p_vertices]
            # Near plane 0.5 m: keeps projected coords inside gfxdraw's int16 range for nearby ground tiles
            near = 0.5 * self.SCALE
            for i in range(len(rel_verts)):
                p_c, p_p = rel_verts[i], rel_verts[i - 1]
                if p_c[0] >= near and p_p[0] >= near: clipped_pts.append(p_c)
                elif (p_c[0] >= near) != (p_p[0] >= near):
                    t = (near - p_p[0]) / (p_c[0] - p_p[0])
                    clipped_pts.append((near, p_p[1] + t * (p_c[1] - p_p[1]), p_p[2] + t * (p_c[2] - p_p[2])))
                    if p_c[0] >= near: clipped_pts.append(p_c)

            pts_fpv, depths_fpv = [], []
            for d_c, lat_c, dy_c in clipped_pts:
                dz_c = -lat_c
                if d_c > 0.1:
                    pts_fpv.append((-(dz_c / d_c) * self.fov + cx, self.VIEWPORT_H - ((dy_c / d_c) * self.fov + cy)))
                    depths_fpv.append(d_c)
                
            if len(pts_t) >= 3: proj_top.append((sum(depths_t)/len(depths_t), pts_t, color, layer))
            if len(pts_r) >= 3: proj_rear.append((sum(depths_r)/len(depths_r), pts_r, color, layer))
            if len(pts_s) >= 3: proj_side.append((sum(depths_s)/len(depths_s), pts_s, color, layer))
            if len(pts_fpv) >= 3: proj_fpv.append((sum(depths_fpv)/len(depths_fpv), pts_fpv, color, layer))

        # 4. Z-Index Sorting (Painter's Algorithm)
        for proj_list, surf in [(proj_top, top_surf), (proj_rear, rear_surf), (proj_side, side_surf), (proj_fpv, fpv_surf)]:
            proj_list.sort(key=lambda x: (x[3], -x[0]))
            for _, pts, color, _ in proj_list:
                if len(pts) < 3:
                    continue
                safe_color = (int(color[0]), int(color[1]), int(color[2]))
                ipts = [(int(round(p[0])), int(round(p[1]))) for p in pts]
                try:
                    gfxdraw.aapolygon(surf, ipts, safe_color)
                    gfxdraw.filled_polygon(surf, ipts, safe_color)
                except Exception:
                    pass

        return [np.transpose(pygame.surfarray.array3d(s), (1, 0, 2)) for s in [top_surf, rear_surf, side_surf, fpv_surf]]
