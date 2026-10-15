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
from Box2D.b2 import (
    world as b2World,
    polygonShape as b2PolygonShape,
    circleShape as b2CircleShape,
    revoluteJointDef,
    vec2,
)

import pygame
import pygame.gfxdraw

# Module-level constants for 3D rendering
Z_NEAR = 0.1
Z_FAR = 100.0
Z_GROUND = 0.0
Z_STRUCTURE = 2.0
Z_CATAPULT = 3.0
Z_BOULDER = 4.0
Z_SKY = -1.0

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PPM = 10.0  # Pixels per meter
FPS = 50
TIME_STEP = 1.0 / FPS

# Colors
COLOR_SKY = (135, 206, 235)
COLOR_GROUND = (139, 119, 101)
COLOR_CATAPULT_BASE = (101, 67, 33)
COLOR_CATAPULT_ARM = (160, 82, 45)
COLOR_BOULDER = (128, 128, 128)
COLOR_ENEMY_STRUCTURE = (178, 34, 34)
COLOR_ENEMY_WALL = (139, 0, 0)
COLOR_ENEMY_TOPPLED = (90, 40, 40)
COLOR_PLAYER_STRUCTURE = (34, 139, 34)
COLOR_TRIGGER = (255, 215, 0)
COLOR_TERRAIN = (194, 178, 128)
COLOR_DARK_GROUND = (101, 67, 33)


class CatapultWarEnv(gym.Env):
    """
    Side-on siege game: a catapult at x~10 lobs boulders at a 5-block enemy tower at x~30-40.

    Action (Box, 3 dims, all in [-1, 1]):
      [0] winch:  >0 winds tension up, <0 lets it out (rate, 0.04 per step at |a|=1); tension in [0, 1]
                  sets the launch speed (LAUNCH_SPEED_MIN at 0 .. LAUNCH_SPEED_MAX at 1).
      [1] aim:    >0 raises the arm (steeper shot), <0 lowers it (flatter shot); rate-controlled,
                  the aim angle is kept in [AIM_MIN, AIM_MAX] (20-70 deg above horizontal).
      [2] fire:   >0.5 releases the loaded boulder along the arm's angle (tension resets to 0).
    Each episode has SHOT_BUDGET boulders. A shot ends when the boulder leaves the field, comes to
    rest or flies for MAX_SHOT_STEPS; the next boulder is then reloaded on the arm.
    """
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': FPS}

    # Launch / aim parameters
    LAUNCH_SPEED_MIN = 5.0
    LAUNCH_SPEED_MAX = 20.0
    TENSION_RATE = 0.04
    AIM_MIN = math.radians(20.0)
    AIM_MAX = math.radians(70.0)
    AIM_RATE = math.radians(60.0)  # rad/s at |aim action| = 1
    ARM_FOLLOW_ANGLE = math.radians(85.0)  # follow-through swing after release (visual)
    ARM_FOLLOW_STEPS = 12

    # Episode structure
    SHOT_BUDGET = 4
    MAX_SHOT_STEPS = 200
    SETTLE_STEPS = 40  # steps to let the tower settle after the last shot ends
    N_BLOCKS = 5

    # Reward
    REWARD_TOPPLE = 10.0
    REWARD_SUCCESS = 50.0
    REWARD_PROXIMITY = 2.0      # one-off per shot, scaled by closest approach to the tower
    PROXIMITY_RADIUS = 12.0

    # Catapult geometry
    PIVOT = (9.0, 3.5)
    ARM_HALF_LEN = 2.5
    ARM_PIVOT_OFFSET = 2.0  # arm centre lies this far from the pivot along the arm
    BOULDER_RADIUS = 0.5
    BOULDER_SEAT = (2.0, 0.2 + 0.5)  # boulder centre in arm-local coords (on top of the arm near the tip)

    def __init__(self, render_mode='rgb_array'):
        super().__init__()
        self.render_mode = render_mode

        # Action space: [winch_tension_rate, aim_rate, fire]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space (fixed slots):
        # [0] arm_angle, [1] arm_angular_vel, [2] aim_angle, [3] tension, [4] loaded,
        # [5] shots_left / SHOT_BUDGET, [6:10] boulder x, y, vx, vy, [10] wind, [11] tower_x,
        # [12 + 4*i : 16 + 4*i] block i: x, y, angle, toppled   (i = 0..4)
        obs_size = 12 + 4 * self.N_BLOCKS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.world = None
        self.ground = None
        self.catapult_base = None
        self.catapult_arm = None
        self.arm_joint = None
        self.boulder = None
        self.enemy_blocks = []
        self.block_spawn = []
        self.block_toppled = []
        self.tension = 0.0
        self.aim_angle = math.radians(45.0)
        self.trigger_locked = True
        self.boulder_launched = False
        self.step_count = 0
        self.max_steps = 1000  # shot budget (not time) normally ends the episode
        self.shots_left = self.SHOT_BUDGET
        self.shot_steps = 0
        self.rest_steps = 0
        self.min_dist = float('inf')
        self.follow_steps = 0
        self.settle_steps = -1
        self.tower_x = 35.0
        self.wind = 0.0

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world = b2World(gravity=(0, -9.81))
        self.tension = 0.0
        self.aim_angle = math.radians(45.0)
        self.trigger_locked = True
        self.boulder_launched = False
        self.step_count = 0
        self.shots_left = self.SHOT_BUDGET
        self.shot_steps = 0
        self.rest_steps = 0
        self.min_dist = float('inf')
        self.follow_steps = 0
        self.settle_steps = -1
        self.boulder = None
        self.enemy_blocks = []

        # Modest per-episode randomization: tower distance and a horizontal wind on the boulder
        self.tower_x = float(self.np_random.uniform(30.0, 40.0))
        self.wind = float(self.np_random.uniform(-1.5, 1.5))

        # Ground (top surface at y = 1)
        self.ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(box=(100, 1))
        )

        # Catapult base at left side
        self.catapult_base = self.world.CreateStaticBody(
            position=(10, 2),
            shapes=b2PolygonShape(box=(1.5, 1.0))
        )

        # Catapult arm (dynamic) - pivoted near its rear end, elevated at the aim angle
        px, py = self.PIVOT
        a = self.aim_angle
        self.catapult_arm = self.world.CreateDynamicBody(
            position=(px + self.ARM_PIVOT_OFFSET * math.cos(a), py + self.ARM_PIVOT_OFFSET * math.sin(a)),
            angle=a
        )
        self.catapult_arm.CreateFixture(
            shape=b2PolygonShape(box=(self.ARM_HALF_LEN, 0.2)),
            density=2.0,
            friction=0.3,
            groupIndex=-1  # never collides with the boulder (same negative group)
        )

        # Joint connecting arm to base (pivot at rear); joint angle == arm elevation
        jd = revoluteJointDef()
        jd.bodyA = self.catapult_base
        jd.bodyB = self.catapult_arm
        jd.localAnchorA = (px - 10.0, py - 2.0)
        jd.localAnchorB = (-self.ARM_PIVOT_OFFSET, 0.0)
        jd.referenceAngle = 0.0
        jd.enableMotor = True
        jd.motorSpeed = 0.0
        jd.maxMotorTorque = 2000.0
        jd.enableLimit = True
        jd.lowerAngle = self.AIM_MIN
        jd.upperAngle = self.ARM_FOLLOW_ANGLE
        self.arm_joint = self.world.CreateJoint(jd)

        # Boulder seated on the arm
        self._create_boulder()

        # Enemy structures on the right side
        self._create_enemy_structures()

        obs = self._get_obs()
        return obs, self._get_info()

    def _boulder_seat(self):
        p = self.catapult_arm.GetWorldPoint(self.BOULDER_SEAT)
        return (float(p[0]), float(p[1]))

    def _create_boulder(self):
        self.boulder = self.world.CreateDynamicBody(
            position=self._boulder_seat(),
            bullet=True,
            angularDamping=3.0  # rolling resistance so a landed boulder comes to rest
        )
        self.boulder.CreateFixture(
            shape=b2CircleShape(radius=self.BOULDER_RADIUS),
            density=5.0,
            friction=0.5,
            restitution=0.3,
            groupIndex=-1
        )
        self.boulder.gravityScale = 0.0  # held on the arm until released
        self.boulder_launched = False
        self.trigger_locked = True

    def _hold_boulder(self):
        """Keep a loaded boulder seated on the arm (no overlap, no drift)."""
        if self.boulder and not self.boulder_launched:
            self.boulder.position = self._boulder_seat()
            self.boulder.linearVelocity = (0, 0)
            self.boulder.angularVelocity = 0.0

    def _create_enemy_structures(self):
        base_x = self.tower_x
        g = 1.0  # ground top
        self.enemy_blocks = []

        # Central tower: 3 stacked blocks resting exactly on the ground / on each other
        for i in range(3):
            block = self.world.CreateDynamicBody(
                position=(base_x, g + 0.8 + i * 1.6)
            )
            block.CreateFixture(
                shape=b2PolygonShape(box=(0.8, 0.8)),
                density=1.0,
                friction=0.6,
                restitution=0.1
            )
            self.enemy_blocks.append(block)

        # Side pillars
        for dx in [-2.0, 2.0]:
            block = self.world.CreateDynamicBody(
                position=(base_x + dx, g + 1.0)
            )
            block.CreateFixture(
                shape=b2PolygonShape(box=(0.5, 1.0)),
                density=1.5,
                friction=0.6,
                restitution=0.1
            )
            self.enemy_blocks.append(block)

        self.block_spawn = [(float(b.position[0]), float(b.position[1])) for b in self.enemy_blocks]
        self.block_toppled = [False] * len(self.enemy_blocks)

    def _is_toppled(self, i):
        block = self.enemy_blocks[i]
        sx, sy = self.block_spawn[i]
        ang = (block.angle + math.pi) % (2 * math.pi) - math.pi
        return (abs(ang) > math.pi / 4
                or block.position[1] < sy - 0.6
                or abs(block.position[0] - sx) > 2.5)

    def _end_shot(self):
        """Finish the current shot: one-off proximity bonus, remove boulder, reload if any shots remain."""
        bonus = self.REWARD_PROXIMITY * max(0.0, 1.0 - self.min_dist / self.PROXIMITY_RADIUS)
        if self.boulder:
            self.world.DestroyBody(self.boulder)
        self.boulder = None
        self.boulder_launched = False
        self.shot_steps = 0
        self.rest_steps = 0
        self.min_dist = float('inf')
        if self.shots_left > 0:
            self._create_boulder()
        else:
            self.settle_steps = self.SETTLE_STEPS
        return bonus

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.step_count += 1
        reward = 0.0

        winch_tension = float(action[0])
        aim_adj = float(action[1])
        trigger_release = float(action[2])

        # Winch builds (or releases) stored tension
        self.tension = float(np.clip(self.tension + winch_tension * self.TENSION_RATE, 0.0, 1.0))

        # Aim: rate-controlled elevation within [AIM_MIN, AIM_MAX]
        self.aim_angle = float(np.clip(self.aim_angle + aim_adj * self.AIM_RATE * TIME_STEP,
                                       self.AIM_MIN, self.AIM_MAX))

        # Trigger release
        if (trigger_release > 0.5 and self.boulder and not self.boulder_launched
                and self.shots_left > 0):
            self.trigger_locked = False
            self.boulder_launched = True
            self.shots_left -= 1
            self.shot_steps = 0
            self.rest_steps = 0
            self.min_dist = float('inf')

            # Launch boulder along the arm with speed set by tension
            arm_angle = self.catapult_arm.angle
            speed = self.LAUNCH_SPEED_MIN + self.tension * (self.LAUNCH_SPEED_MAX - self.LAUNCH_SPEED_MIN)
            self._hold_boulder()
            self.boulder.gravityScale = 1.0
            self.boulder.linearVelocity = (float(speed * math.cos(arm_angle)), float(speed * math.sin(arm_angle)))
            self.boulder.awake = True
            self.tension = 0.0
            self.follow_steps = self.ARM_FOLLOW_STEPS

        # Arm motor: track the aim angle (or swing through after a release)
        target = self.ARM_FOLLOW_ANGLE if self.follow_steps > 0 else self.aim_angle
        self.follow_steps = max(0, self.follow_steps - 1)
        err = target - self.arm_joint.angle
        self.arm_joint.motorSpeed = float(np.clip(10.0 * err, -6.0, 6.0))

        # Wind pushes the boulder in flight
        if self.boulder and self.boulder_launched:
            self.boulder.ApplyForceToCenter((float(self.boulder.mass * self.wind), 0.0), True)

        # Step physics
        self._hold_boulder()
        self.world.Step(TIME_STEP, 8, 3)
        self.world.ClearForces()
        self._hold_boulder()

        # Toppled blocks (latched; bodies stay in the world and keep their obs slot)
        newly_toppled = 0
        for i in range(len(self.enemy_blocks)):
            if not self.block_toppled[i] and self._is_toppled(i):
                self.block_toppled[i] = True
                newly_toppled += 1
        reward += newly_toppled * self.REWARD_TOPPLE
        n_remaining = self.block_toppled.count(False)

        # Shot bookkeeping
        if self.boulder and self.boulder_launched:
            self.shot_steps += 1
            bx, by = self.boulder.position
            self.min_dist = min(self.min_dist, math.hypot(bx - self.tower_x, by - 3.4))
            speed = self.boulder.linearVelocity.length
            self.rest_steps = self.rest_steps + 1 if speed < 0.5 else 0
            out_of_bounds = by < -5.0 or bx > 80.0 or bx < -10.0
            if out_of_bounds or self.rest_steps >= 10 or self.shot_steps >= self.MAX_SHOT_STEPS:
                reward += self._end_shot()
        elif self.settle_steps > 0:
            self.settle_steps -= 1

        terminated = False
        truncated = False
        success = n_remaining == 0

        # Success: every enemy block toppled
        if success:
            reward += self.REWARD_SUCCESS
            terminated = True
        # Out of ammunition and the tower has settled
        elif self.settle_steps == 0:
            terminated = True

        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info['success'] = success

        return obs, float(reward), terminated, truncated, info

    def _get_info(self):
        return {
            'blocks_remaining': self.block_toppled.count(False),
            'blocks_toppled': self.block_toppled.count(True),
            'tension': self.tension,
            'aim_deg': math.degrees(self.aim_angle),
            'launched': self.boulder_launched,
            'shots_left': self.shots_left,
            'wind': self.wind,
            'tower_x': self.tower_x,
        }

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        if self.catapult_arm:
            obs[0] = self.catapult_arm.angle
            obs[1] = self.catapult_arm.angularVelocity
        obs[2] = self.aim_angle
        obs[3] = self.tension
        obs[4] = 1.0 if (self.boulder and not self.boulder_launched) else 0.0
        obs[5] = self.shots_left / float(self.SHOT_BUDGET)

        if self.boulder:
            obs[6] = self.boulder.position[0]
            obs[7] = self.boulder.position[1]
            obs[8] = self.boulder.linearVelocity[0]
            obs[9] = self.boulder.linearVelocity[1]

        obs[10] = self.wind
        obs[11] = self.tower_x

        # Enemy blocks: fixed slot per block
        for i, block in enumerate(self.enemy_blocks[:self.N_BLOCKS]):
            obs[12 + i * 4] = block.position[0]
            obs[13 + i * 4] = block.position[1]
            obs[14 + i * 4] = (block.angle + math.pi) % (2 * math.pi) - math.pi
            obs[15 + i * 4] = 1.0 if self.block_toppled[i] else 0.0

        return obs

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_SKY)

        # Camera centered on catapult area
        camera_x = 25.0
        camera_y = 10.0

        def world_to_screen(x, y):
            sx = int((x - camera_x + SCREEN_WIDTH / (2 * PPM)) * PPM)
            sy = int(SCREEN_HEIGHT - (y - camera_y + SCREEN_HEIGHT / (2 * PPM)) * PPM)
            return sx, sy

        # Draw ground
        gx1, gy1 = world_to_screen(-50, 1)
        gx2, gy2 = world_to_screen(100, -1)
        pygame.draw.rect(self.screen, COLOR_TERRAIN, (gx1, gy1, gx2 - gx1, gy2 - gy1))

        # Draw catapult base
        if self.catapult_base:
            self._draw_body(self.catapult_base, COLOR_CATAPULT_BASE, world_to_screen)

        # Draw catapult arm
        if self.catapult_arm:
            self._draw_body(self.catapult_arm, COLOR_CATAPULT_ARM, world_to_screen)

        # Draw boulder
        if self.boulder:
            bx, by = world_to_screen(self.boulder.position[0], self.boulder.position[1])
            radius = int(0.5 * PPM)
            if 0 <= bx < SCREEN_WIDTH and 0 <= by < SCREEN_HEIGHT:
                pygame.draw.circle(self.screen, COLOR_BOULDER, (bx, by), radius)
                pygame.gfxdraw.aacircle(self.screen, bx, by, radius, (80, 80, 80))

        # Draw enemy blocks
        for i, block in enumerate(self.enemy_blocks):
            color = COLOR_ENEMY_TOPPLED if self.block_toppled[i] else COLOR_ENEMY_STRUCTURE
            self._draw_body(block, color, world_to_screen)

        # Draw tension indicator
        tension_bar_width = int(self.tension * 100)
        pygame.draw.rect(self.screen, (255, 0, 0), (10, 10, tension_bar_width, 20))
        pygame.draw.rect(self.screen, (255, 255, 255), (10, 10, 100, 20), 2)

        # Draw trigger state
        trigger_color = COLOR_TRIGGER if self.trigger_locked else (100, 100, 100)
        pygame.draw.circle(self.screen, trigger_color, (130, 20), 10)

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(FPS)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        ).copy()

    def _draw_body(self, body, color, world_to_screen):
        for fixture in body.fixtures:
            shape = fixture.shape
            if isinstance(shape, b2PolygonShape):
                vertices = [(body.transform * v) for v in shape.vertices]
                screen_verts = [world_to_screen(v[0], v[1]) for v in vertices]
                if len(screen_verts) >= 3:
                    pygame.draw.polygon(self.screen, color, screen_verts)
                    pygame.gfxdraw.aapolygon(self.screen, screen_verts, (0, 0, 0))
            elif isinstance(shape, b2CircleShape):
                pos = body.transform * shape.pos
                sx, sy = world_to_screen(pos[0], pos[1])
                radius = int(shape.radius * PPM)
                if radius > 0:
                    pygame.draw.circle(self.screen, color, (sx, sy), radius)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class CatapultWarOrthographicWrapper(gym.Wrapper):
    """
    Wraps CatapultWarEnv to produce orthographic multi-view 3D renders plus a
    third-person true-3D view (orbit camera trailing the catapult).
    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """

    def __init__(self, env=None, render_mode='rgb_array', step_yaw_deg=8.0, step_pitch_deg=5.0):
        if env is None:
            env = CatapultWarEnv(render_mode=render_mode)
        super().__init__(env)
        self.view_width = SCREEN_WIDTH
        self.view_height = SCREEN_HEIGHT

        # Follow/Orbit camera baseline offsets (relative to catapult base)
        self.camera_dist = 6.5    # Orbit radius around the catapult (catapult is ~5 m long; FPV FOV is wide)
        self.camera_height = 2.0   # Vertical offset above the catapult base
        # Default downward look: the orbit is raised high enough that the sight line over the arm tip lands
        # short of the tower (x >= 30), so the tower shows above the catapult instead of hidden behind its arm
        self.base_pitch = math.radians(50.0)

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state
        self.pivot_depth = self.camera_dist  # Orbit focus on the catapult (camera circles the player)

        # Last computed camera position (updated in render)
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        return self.env.reset(**kwargs)

    @property
    def _cam_anchor(self):
        """Un-orbited trailing camera (behind/above the catapult, looking +X) used as origin for camera-relative vectors."""
        base_env = self.env.unwrapped
        if not hasattr(base_env, 'catapult_base'):
            return (0.0, 0.0, 0.0)

        # Player reference position (catapult)
        if base_env.catapult_base:
            player_x = base_env.catapult_base.position[0]
            player_y = base_env.catapult_base.position[1]
        else:
            player_x, player_y = 10.0, 2.0

        # Third-person trailing camera: camera_dist behind (-X) and camera_height above the catapult
        return (player_x - self.camera_dist, player_y + self.camera_height, 0.0)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the camera in world space."""
        base_env = self.env.unwrapped
        if not hasattr(base_env, 'catapult_base'):
            return (0.0, 0.0, 0.0)

        anchor_x, anchor_y, anchor_z = self._cam_anchor
        # Orbit displacement in camera frame; FPV maps depth->+X, lateral->+Z, vertical->+Y
        d_depth, d_lat, d_vert = orbit_camera_offset(self.cam_yaw, self.orbit_pitch, self.pivot_depth)
        cam_x = anchor_x + d_depth
        cam_y = anchor_y + d_vert
        cam_z = anchor_z + d_lat

        return (cam_x, cam_y, cam_z)

    @property
    def orbit_pitch(self):
        """Total orbit pitch: commanded cam_pitch on top of the default downward look, kept below overhead."""
        return min(self.cam_pitch + self.base_pitch, math.radians(75.0))

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
        """Returns tuple of 4 orthographic views: (top_down, rear, side, fpv)"""
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)

        # Ensure pygame is initialized for fonts and surfaces
        if not pygame.get_init():
            pygame.init()

        # Get game state
        env = self.env.unwrapped

        # Player reference position (catapult)
        if env.catapult_base:
            player_x = env.catapult_base.position[0]
            player_y = env.catapult_base.position[1]
        else:
            player_x, player_y = 10.0, 2.0

        # Collect 3D objects from 2D world with Z extrusion
        objects_3d = self._extrude_world(env)

        # Render four views
        im_top_down = self._render_top_down(objects_3d, player_x, player_y)
        im_rear = self._render_rear(objects_3d, player_x, player_y)
        im_side = self._render_side(objects_3d, player_x, player_y)
        # Get dynamically calculated position from our property
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        im_fpv = self._render_fpv(objects_3d, player_x, player_y)

        return (im_top_down, im_rear, im_side, im_fpv)

    def _extrude_world(self, env):
        """Convert 2D Box2D bodies into 3D extruded objects."""
        objects = []

        # Ground
        objects.append({
            'type': 'box',
            'vertices_2d': [(-50, -1), (100, -1), (100, 1), (-50, 1)],
            'z_min': -5.0,
            'z_max': 5.0,
            'color': COLOR_TERRAIN,
            'depth': 0.0,
            'no_fpv': True  # 3D view draws the ground as its backdrop plane
        })

        # Catapult base
        if env.catapult_base:
            for fixture in env.catapult_base.fixtures:
                verts = [(env.catapult_base.transform * v) for v in fixture.shape.vertices]
                objects.append({
                    'type': 'box',
                    'vertices_2d': [(v[0], v[1]) for v in verts],
                    'z_min': -1.0,
                    'z_max': 1.0,
                    'color': COLOR_CATAPULT_BASE,
                    'depth': Z_CATAPULT,
                    'player': True  # drawn as a solid extruded volume in the 3D view
                })

        # Catapult arm
        if env.catapult_arm:
            for fixture in env.catapult_arm.fixtures:
                verts = [(env.catapult_arm.transform * v) for v in fixture.shape.vertices]
                objects.append({
                    'type': 'box',
                    'vertices_2d': [(v[0], v[1]) for v in verts],
                    'z_min': -0.3,
                    'z_max': 0.3,
                    'color': COLOR_CATAPULT_ARM,
                    'depth': Z_CATAPULT + 0.5,
                    'player': True  # drawn as a solid extruded volume in the 3D view
                })

        # Boulder
        if env.boulder:
            bx = env.boulder.position[0]
            by = env.boulder.position[1]
            objects.append({
                'type': 'circle',
                'center': (bx, by),
                'radius': 0.5,
                'z_min': -0.5,
                'z_max': 0.5,
                'color': COLOR_BOULDER,
                'depth': Z_BOULDER
            })

        # Enemy blocks
        for i, block in enumerate(env.enemy_blocks):
            color = COLOR_ENEMY_TOPPLED if env.block_toppled[i] else COLOR_ENEMY_STRUCTURE
            for fixture in block.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    verts = [(block.transform * v) for v in shape.vertices]
                    objects.append({
                        'type': 'box',
                        'vertices_2d': [(v[0], v[1]) for v in verts],
                        'z_min': -0.8,
                        'z_max': 0.8,
                        'color': color,
                        'depth': Z_STRUCTURE
                    })

        return objects

    def _render_top_down(self, objects, player_x, player_y):
        """Top-down view: X-right, Z-up (Y into screen)"""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill((50, 50, 80))

        scale = 12.0

        # Sort by Y (depth into screen for top-down)
        sorted_objs = sorted(objects, key=lambda o: o.get('depth', 0))

        for obj in sorted_objs:
            if obj['type'] == 'box':
                verts = obj['vertices_2d']
                z_min = obj['z_min']
                z_max = obj['z_max']
                # Top-down: we see X and Z
                # Use average Y for the "top" face
                screen_verts = []
                for v in verts:
                    sx = int((v[0] - player_x) * scale + self.view_width / 2)
                    sy = int(self.view_height / 2 - (obj['z_max']) * scale)
                    screen_verts.append((sx, sy))

                # Draw the top face as the XZ footprint
                # For top-down, show the XZ extent
                if len(verts) >= 2:
                    min_x = min(v[0] for v in verts)
                    max_x = max(v[0] for v in verts)
                    rect_verts = [
                        (int((min_x - player_x) * scale + self.view_width / 2),
                         int(self.view_height / 2 - z_max * scale)),
                        (int((max_x - player_x) * scale + self.view_width / 2),
                         int(self.view_height / 2 - z_max * scale)),
                        (int((max_x - player_x) * scale + self.view_width / 2),
                         int(self.view_height / 2 - z_min * scale)),
                        (int((min_x - player_x) * scale + self.view_width / 2),
                         int(self.view_height / 2 - z_min * scale)),
                    ]
                    # Clip check
                    if any(0 <= v[0] < self.view_width and 0 <= v[1] < self.view_height for v in rect_verts):
                        try:
                            pygame.draw.polygon(surface, obj['color'], rect_verts)
                            pygame.gfxdraw.aapolygon(surface, rect_verts, (0, 0, 0))
                        except (ValueError, TypeError):
                            pass

            elif obj['type'] == 'circle':
                cx, cy = obj['center']
                r = obj['radius']
                sx = int((cx - player_x) * scale + self.view_width / 2)
                sy = int(self.view_height / 2)
                sr = int(r * scale)
                if 0 <= sx < self.view_width and 0 <= sy < self.view_height and sr > 0:
                    pygame.draw.circle(surface, obj['color'], (sx, sy), sr)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def _render_rear(self, objects, player_x, player_y):
        """Rear view: looking from behind (negative X direction)"""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_SKY)

        scale = 15.0

        # Sort by X (further objects drawn first)
        sorted_objs = sorted(objects, key=lambda o: -self._get_avg_x(o) + player_x)

        for obj in sorted_objs:
            if obj['type'] == 'box':
                verts = obj['vertices_2d']
                z_min = obj['z_min']
                z_max = obj['z_max']
                # Rear view: Z-horizontal, Y-vertical
                screen_verts = [
                    (int(z_min * scale + self.view_width / 2),
                     int(self.view_height - (min(v[1] for v in verts) - player_y + 2) * scale)),
                    (int(z_max * scale + self.view_width / 2),
                     int(self.view_height - (min(v[1] for v in verts) - player_y + 2) * scale)),
                    (int(z_max * scale + self.view_width / 2),
                     int(self.view_height - (max(v[1] for v in verts) - player_y + 2) * scale)),
                    (int(z_min * scale + self.view_width / 2),
                     int(self.view_height - (max(v[1] for v in verts) - player_y + 2) * scale)),
                ]
                if any(0 <= v[0] < self.view_width and 0 <= v[1] < self.view_height for v in screen_verts):
                    try:
                        pygame.draw.polygon(surface, obj['color'], screen_verts)
                        pygame.gfxdraw.aapolygon(surface, screen_verts, (0, 0, 0))
                    except (ValueError, TypeError):
                        pass

            elif obj['type'] == 'circle':
                cx, cy = obj['center']
                r = obj['radius']
                sx = int(self.view_width / 2)
                sy = int(self.view_height - (cy - player_y + 2) * scale)
                sr = int(r * scale)
                if sr > 0:
                    pygame.draw.circle(surface, obj['color'], (sx, sy), sr)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def _render_side(self, objects, player_x, player_y):
        """Side view: X-horizontal, Y-vertical"""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_SKY)

        scale = 12.0

        # Sort by Z (painter's algorithm)
        sorted_objs = sorted(objects, key=lambda o: o.get('z_min', 0))

        for obj in sorted_objs:
            if obj['type'] == 'box':
                verts = obj['vertices_2d']
                # Side view shows X and Y
                screen_verts = []
                for v in verts:
                    sx = int((v[0] - player_x) * scale + self.view_width / 2)
                    sy = int(self.view_height - (v[1] - player_y + 3) * scale)
                    screen_verts.append((sx, sy))

                if len(screen_verts) >= 3:
                    if any(0 <= v[0] < self.view_width and 0 <= v[1] < self.view_height for v in screen_verts):
                        try:
                            pygame.draw.polygon(surface, obj['color'], screen_verts)
                            pygame.gfxdraw.aapolygon(surface, screen_verts, (0, 0, 0))
                        except (ValueError, TypeError):
                            pass

            elif obj['type'] == 'circle':
                cx, cy = obj['center']
                r = obj['radius']
                sx = int((cx - player_x) * scale + self.view_width / 2)
                sy = int(self.view_height - (cy - player_y + 3) * scale)
                sr = int(r * scale)
                if sr > 0 and 0 <= sx < self.view_width and 0 <= sy < self.view_height:
                    pygame.draw.circle(surface, obj['color'], (sx, sy), sr)
                    pygame.gfxdraw.aacircle(surface, sx, sy, sr, (0, 0, 0))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def _render_fpv(self, objects, player_x, player_y):
        """Third-person 3D view: trailing camera behind the catapult looking toward the enemy"""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_SKY)

        # Draw ground plane up to the horizon, which rises on screen as the orbit looks further down
        horizon_y = int(self.view_height / 2 - 200.0 * math.tan(self.orbit_pitch))
        horizon_y = max(0, min(self.view_height, horizon_y))
        pygame.draw.rect(surface, COLOR_TERRAIN,
                         (0, horizon_y, self.view_width, self.view_height - horizon_y))

        # Simple perspective projection from un-orbited trailing camera; orbit_camera_frame applies the orbit
        eye_x, eye_y, eye_z = self._cam_anchor
        near = 0.1

        def to_cam(x, y, z):
            # Camera-relative (depth=+X, lateral=+Z, vertical=+Y), then orbited around the catapult
            return orbit_camera_frame(x - eye_x, z - eye_z, y - eye_y,
                                      self.cam_yaw, self.orbit_pitch, self.pivot_depth)

        # Build drawable items (orbited depth, kind, color, data). Every box is drawn as a solid
        # extruded volume so the tower and pillars stay visible when seen end-on from behind.
        items = []
        for obj in objects:
            if obj.get('no_fpv'):
                continue
            if obj['type'] == 'box':
                verts = obj['vertices_2d']
                z0 = obj.get('z_min', 0)
                z1 = obj.get('z_max', 0)
                side = tuple(int(c * 0.75) for c in obj['color'])
                faces = [(obj['color'], [(v[0], v[1], z0) for v in verts]),
                         (obj['color'], [(v[0], v[1], z1) for v in verts])]
                for i in range(len(verts)):
                    a, b = verts[i], verts[(i + 1) % len(verts)]
                    faces.append((side, [(a[0], a[1], z0), (b[0], b[1], z0),
                                         (b[0], b[1], z1), (a[0], a[1], z1)]))
                for color, face in faces:
                    cam_verts = [to_cam(*p) for p in face]
                    depth = sum(c[0] for c in cam_verts) / len(cam_verts)
                    items.append((depth, 'poly', color, cam_verts))
            elif obj['type'] == 'circle':
                cam_c = to_cam(obj['center'][0], obj['center'][1], 0.0)
                items.append((cam_c[0], 'circle', obj['color'], (cam_c, obj['radius'])))

        # Sort by orbited depth (far first)
        items.sort(key=lambda it: -it[0])

        for _, kind, color, data in items:
            if kind == 'poly':
                cam_verts = data

                # Clip polygon against the near plane on the orbited depth
                clipped = []
                n = len(cam_verts)
                for i in range(n):
                    a, b = cam_verts[i], cam_verts[(i + 1) % n]
                    a_in, b_in = a[0] >= near, b[0] >= near
                    if a_in:
                        clipped.append(a)
                    if a_in != b_in:
                        t = (near - a[0]) / (b[0] - a[0])
                        clipped.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(3)))

                # Project vertices
                screen_verts = []
                for d_c, z_c, dy_c in clipped:
                    proj_scale = 200.0 / d_c
                    sx = int(self.view_width / 2 + z_c * proj_scale)
                    sy = int(self.view_height / 2 - dy_c * proj_scale)
                    screen_verts.append((sx, sy))

                if len(screen_verts) >= 3:
                    if any(0 <= v[0] < self.view_width and 0 <= v[1] < self.view_height for v in screen_verts):
                        try:
                            pygame.draw.polygon(surface, color, screen_verts)
                            pygame.gfxdraw.aapolygon(surface, screen_verts, (0, 0, 0))
                        except (ValueError, TypeError):
                            pass

            elif kind == 'circle':
                (d_c, z_c, dy_c), r = data

                if d_c <= 0.5:
                    continue

                proj_scale = 200.0 / d_c
                sx = int(self.view_width / 2 + z_c * proj_scale)
                sy = int(self.view_height / 2 - dy_c * proj_scale)
                sr = max(1, int(r * proj_scale))

                if 0 <= sx < self.view_width and 0 <= sy < self.view_height:
                    pygame.draw.circle(surface, color, (sx, sy), sr)

        # Crosshair
        cx, cy = self.view_width // 2, self.view_height // 2
        pygame.draw.line(surface, (255, 0, 0), (cx - 10, cy), (cx + 10, cy), 2)
        pygame.draw.line(surface, (255, 0, 0), (cx, cy - 10), (cx, cy + 10), 2)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def _get_avg_x(self, obj):
        if obj['type'] == 'box':
            return sum(v[0] for v in obj['vertices_2d']) / max(len(obj['vertices_2d']), 1)
        elif obj['type'] == 'circle':
            return obj['center'][0]
        return 0.0


if __name__ == '__main__':
    pygame.init()

    # Test the environment
    env = CatapultWarEnv(render_mode='rgb_array')
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Test wrapper
    wrapper = CatapultWarOrthographicWrapper(env)

    total_reward = 0
    for step in range(700):
        # Wind up to ~0.6 tension at the default 45 deg aim, then release; repeat after each reload
        base_env = wrapper.unwrapped
        loaded = base_env.boulder is not None and not base_env.boulder_launched
        if loaded and base_env.tension < 0.6:
            action = np.array([1.0, 0.0, -1.0], dtype=np.float32)  # Build tension
        elif loaded:
            action = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Release!
        else:
            action = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Wait

        obs, reward, terminated, truncated, info = wrapper.step(action)
        total_reward += reward

        if step % 25 == 0:
            views = wrapper.render()
            print(f"Step {step}: reward={reward:.2f}, total={total_reward:.2f}, "
                  f"blocks={info['blocks_remaining']}, tension={info['tension']:.2f}")
            print(f"  View shapes: {[v.shape for v in views]}")

        if terminated or truncated:
            print(f"Episode ended at step {step}. Total reward: {total_reward:.2f}")
            break

    env.close()
    pygame.quit()
    print("Done!")