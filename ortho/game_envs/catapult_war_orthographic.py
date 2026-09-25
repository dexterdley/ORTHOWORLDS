import math
import numpy as np

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
COLOR_PLAYER_STRUCTURE = (34, 139, 34)
COLOR_TRIGGER = (255, 215, 0)
COLOR_TERRAIN = (194, 178, 128)
COLOR_DARK_GROUND = (101, 67, 33)


class CatapultWarEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': FPS}

    def __init__(self, render_mode='rgb_array'):
        super().__init__()
        self.render_mode = render_mode

        # Action space: [winch_tension, arm_angle_adjust, trigger_release]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space
        # [arm_angle, arm_angular_vel, boulder_x, boulder_y, boulder_vx, boulder_vy,
        #  tension, trigger_state, enemy_blocks_remaining (x5 positions)]
        obs_size = 18
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
        self.launched = False
        self.tension = 0.0
        self.trigger_locked = True
        self.step_count = 0
        self.max_steps = 500
        self.initial_enemy_count = 5
        self.boulder_launched = False
        self.prev_enemy_count = 0

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.world = b2World(gravity=(0, -9.81))
        self.launched = False
        self.tension = 0.0
        self.trigger_locked = True
        self.step_count = 0
        self.boulder_launched = False
        self.enemy_blocks = []

        # Ground
        self.ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(box=(100, 1))
        )

        # Catapult base at left side
        self.catapult_base = self.world.CreateStaticBody(
            position=(10, 2),
            shapes=b2PolygonShape(box=(1.5, 1.0))
        )

        # Catapult arm (dynamic) - pivoted at the left (rear) end, pointing right (East)
        self.catapult_arm = self.world.CreateDynamicBody(
            position=(11.0, 3.5),
            angle=0.0
        )
        self.catapult_arm.CreateFixture(
            shape=b2PolygonShape(box=(2.5, 0.2)),
            density=2.0,
            friction=0.3
        )

        # Joint connecting arm to base (Pivot at rear)
        jd = revoluteJointDef()
        jd.bodyA = self.catapult_base
        jd.bodyB = self.catapult_arm
        jd.localAnchorA = (-1.0, 1.5)  # Rear left of the base
        jd.localAnchorB = (-2.0, 0.0)  # Rear left of the arm
        jd.enableMotor = True
        jd.motorSpeed = 0.0
        jd.maxMotorTorque = 800.0
        jd.enableLimit = True
        jd.lowerAngle = 0.0
        jd.upperAngle = math.pi / 2.5
        self.arm_joint = self.world.CreateJoint(jd)

        # Boulder on the arm tip
        self._create_boulder()

        # Enemy structures on the right side
        self._create_enemy_structures()

        self.prev_enemy_count = len(self.enemy_blocks)

        obs = self._get_obs()
        return obs, {}

    def _create_boulder(self):
        arm_pos = self.catapult_arm.position
        arm_angle = self.catapult_arm.angle
        # Place boulder at the tip of the arm (right side)
        tip_x = arm_pos[0] + 2.5 * math.cos(arm_angle) - 0.5 * math.sin(arm_angle)
        tip_y = arm_pos[1] + 2.5 * math.sin(arm_angle) + 0.5 * math.cos(arm_angle)

        self.boulder = self.world.CreateDynamicBody(
            position=(tip_x, tip_y),
            bullet=True
        )
        self.boulder.CreateFixture(
            shape=b2CircleShape(radius=0.5),
            density=5.0,
            friction=0.5,
            restitution=0.3
        )
        self.boulder_launched = False

    def _create_enemy_structures(self):
        base_x = 35.0
        self.enemy_blocks = []

        # Create a simple tower structure
        for i in range(3):
            block = self.world.CreateDynamicBody(
                position=(base_x, 1.5 + i * 2.0)
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
                position=(base_x + dx, 1.5)
            )
            block.CreateFixture(
                shape=b2PolygonShape(box=(0.5, 1.0)),
                density=1.5,
                friction=0.6,
                restitution=0.1
            )
            self.enemy_blocks.append(block)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        winch_tension = action[0]
        arm_angle_adj = action[1]
        trigger_release = action[2]

        # Apply tension (builds up stored energy)
        self.tension = np.clip(self.tension + winch_tension * 0.05, 0.0, 1.0)

        # Adjust arm motor speed based on angle adjustment
        if self.trigger_locked:
            motor_speed = float(arm_angle_adj * 5.0)
            self.arm_joint.motorSpeed = motor_speed

            # Keep boulder attached to arm tip
            if self.boulder and not self.boulder_launched:
                arm_pos = self.catapult_arm.position
                arm_angle = self.catapult_arm.angle
                tip_x = arm_pos[0] + 2.5 * math.cos(arm_angle)
                tip_y = arm_pos[1] + 2.5 * math.sin(arm_angle)
                self.boulder.position = (tip_x, tip_y)
                self.boulder.linearVelocity = (0, 0)

        # Trigger release
        if trigger_release > 0.5 and self.trigger_locked and not self.boulder_launched:
            self.trigger_locked = False
            self.boulder_launched = True
            self.launched = True

            # Launch boulder with force proportional to tension
            arm_angle = self.catapult_arm.angle
            launch_force = self.tension * 800.0
            fx = float(launch_force * math.cos(arm_angle + math.pi / 4))
            fy = float(launch_force * math.sin(arm_angle + math.pi / 4))
            self.boulder.ApplyForceToCenter((fx, fy), True)

            # Also give the arm a kick
            self.arm_joint.motorSpeed = float(20.0)

        # Step physics
        self.world.Step(TIME_STEP, 8, 3)
        self.world.ClearForces()

        # Check enemy blocks status
        destroyed_count = 0
        remaining_blocks = []
        for block in self.enemy_blocks:
            # Consider block destroyed if it fell below ground or moved far
            if block.position[1] < -2.0 or abs(block.position[0] - 35.0) > 10.0:
                destroyed_count += 1
            else:
                remaining_blocks.append(block)

        # Compute reward
        reward = 0.0

        # Reward for destroying blocks
        blocks_just_destroyed = self.prev_enemy_count - len(remaining_blocks)
        if blocks_just_destroyed > 0:
            reward += blocks_just_destroyed * 20.0

        # Small reward for building tension
        if not self.boulder_launched:
            reward += self.tension * 0.01

        # Reward for boulder getting close to enemy
        if self.boulder and self.boulder_launched:
            dist_to_enemy = abs(self.boulder.position[0] - 35.0)
            reward += max(0, (25.0 - dist_to_enemy) * 0.1)

        # Terminal conditions
        terminated = False
        truncated = False

        # Success: all enemy blocks destroyed
        if len(remaining_blocks) == 0:
            reward += 100.0
            terminated = True

        # Failure: boulder out of bounds or timeout
        if self.boulder and self.boulder_launched:
            if (self.boulder.position[1] < -5.0 or
                    self.boulder.position[0] > 60.0 or
                    self.boulder.position[0] < -10.0):
                if len(remaining_blocks) > 0:
                    reward -= 10.0

        if self.step_count >= self.max_steps:
            truncated = True
            if len(remaining_blocks) > 0:
                reward -= 50.0

        self.prev_enemy_count = len(remaining_blocks)
        self.enemy_blocks = remaining_blocks

        obs = self._get_obs()
        info = {
            'blocks_remaining': len(self.enemy_blocks),
            'tension': self.tension,
            'launched': self.boulder_launched
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros(18, dtype=np.float32)

        if self.catapult_arm:
            obs[0] = self.catapult_arm.angle
            obs[1] = self.catapult_arm.angularVelocity

        if self.boulder:
            obs[2] = self.boulder.position[0]
            obs[3] = self.boulder.position[1]
            obs[4] = self.boulder.linearVelocity[0]
            obs[5] = self.boulder.linearVelocity[1]

        obs[6] = self.tension
        obs[7] = 1.0 if self.trigger_locked else 0.0

        # Enemy block positions (up to 5)
        for i, block in enumerate(self.enemy_blocks[:5]):
            obs[8 + i * 2] = block.position[0]
            obs[9 + i * 2] = block.position[1]

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
        for block in self.enemy_blocks:
            self._draw_body(block, COLOR_ENEMY_STRUCTURE, world_to_screen)

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
    """Wraps CatapultWarEnv to produce orthographic multi-view 3D renders."""

    def __init__(self, env=None, render_mode='rgb_array'):
        if env is None:
            env = CatapultWarEnv(render_mode=render_mode)
        super().__init__(env)
        self.view_width = SCREEN_WIDTH
        self.view_height = SCREEN_HEIGHT

    def render(self):
        """Returns tuple of 4 orthographic views: (top_down, rear, side, fpv)"""
        # Ensure pygame is initialized for fonts and surfaces
        if not pygame.get_init():
            pygame.init()

        # Get game state
        env = self.env

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
            'depth': 0.0
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
                    'depth': Z_CATAPULT
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
                    'depth': Z_CATAPULT + 0.5
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
        for block in env.enemy_blocks:
            for fixture in block.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    verts = [(block.transform * v) for v in shape.vertices]
                    objects.append({
                        'type': 'box',
                        'vertices_2d': [(v[0], v[1]) for v in verts],
                        'z_min': -0.8,
                        'z_max': 0.8,
                        'color': COLOR_ENEMY_STRUCTURE,
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
        """First-person view: looking from catapult toward enemy"""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_SKY)

        # Draw ground plane
        pygame.draw.rect(surface, COLOR_TERRAIN,
                         (0, self.view_height // 2, self.view_width, self.view_height // 2))

        # Simple perspective projection from catapult position
        eye_x = player_x
        eye_y = player_y + 3.0
        eye_z = 0.0

        # Sort by distance (far first)
        def get_dist(obj):
            if obj['type'] == 'box':
                avg_x = sum(v[0] for v in obj['vertices_2d']) / len(obj['vertices_2d'])
            else:
                avg_x = obj['center'][0]
            return -(avg_x - eye_x)

        sorted_objs = sorted(objects, key=get_dist)

        for obj in sorted_objs:
            if obj['type'] == 'box':
                verts = obj['vertices_2d']
                avg_x = sum(v[0] for v in verts) / len(verts)
                depth = avg_x - eye_x

                if depth <= 0.5:
                    continue

                # Project vertices
                screen_verts = []
                for v in verts:
                    dx = v[0] - eye_x
                    dy = v[1] - eye_y
                    if dx <= 0.5:
                        continue
                    # Perspective: z_screen maps to horizontal, y maps to vertical
                    proj_scale = 200.0 / dx
                    sz = obj.get('z_min', 0) * proj_scale
                    sx = int(self.view_width / 2 + sz)
                    sy = int(self.view_height / 2 - dy * proj_scale)
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
                dx = cx - eye_x
                dy = cy - eye_y

                if dx <= 0.5:
                    continue

                proj_scale = 200.0 / dx
                sx = int(self.view_width / 2)
                sy = int(self.view_height / 2 - dy * proj_scale)
                sr = max(1, int(r * proj_scale))

                if 0 <= sx < self.view_width and 0 <= sy < self.view_height:
                    pygame.draw.circle(surface, obj['color'], (sx, sy), sr)

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
    for step in range(200):
        # Build tension, then release
        if step < 50:
            action = np.array([0.8, 0.3, -1.0], dtype=np.float32)  # Build tension
        elif step == 50:
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