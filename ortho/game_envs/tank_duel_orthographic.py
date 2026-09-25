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
    revoluteJointDef as b2RevoluteJointDef,
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
FIRE_COOLDOWN = 15  # steps
TANK_MAX_SPEED = 5.0
TURRET_MAX_RATE = 2.0
MAX_HEALTH = 100.0


class MyRayCastCallback(b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.hit = False
        self.point = None
        self.normal = None
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.point = point
        self.normal = normal
        self.fraction = fraction
        return fraction


class TankDuelEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": FPS}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Action: [drive_velocity, turret_rotation, cannon_elevation, fire_trigger]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: player_x, player_y, player_angle, player_vx, player_vy,
        #              turret_angle, cannon_elevation, player_health,
        #              enemy_x, enemy_y, enemy_angle, enemy_vx, enemy_vy,
        #              enemy_turret_angle, enemy_health,
        #              relative_enemy_x, relative_enemy_y
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        self.world = None
        self.player_body = None
        self.player_turret = None
        self.enemy_body = None
        self.enemy_turret = None
        self.projectiles = []
        self.player_health = MAX_HEALTH
        self.enemy_health = MAX_HEALTH
        self.player_turret_angle = 0.0
        self.player_cannon_elevation = 0.0
        self.enemy_turret_angle = 0.0
        self.player_fire_cooldown = 0
        self.enemy_fire_cooldown = 0
        self.steps = 0
        self.max_steps = 1000
        self.obstacles = []
        self.walls = []

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.world = b2World(gravity=(0, 0))
        self.projectiles = []
        self.obstacles = []
        self.walls = []
        self.player_health = MAX_HEALTH
        self.enemy_health = MAX_HEALTH
        self.player_turret_angle = 0.0
        self.player_cannon_elevation = 0.0
        self.enemy_turret_angle = math.pi
        self.player_fire_cooldown = 0
        self.enemy_fire_cooldown = 0
        self.steps = 0

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
                box=half_size, density=0, friction=0.3
            )
            wall.userData = {"type": "wall", "color": COLOR_WALL, "z": Z_WALL}
            self.walls.append(wall)

        # Create obstacles
        num_obstacles = 5
        for _ in range(num_obstacles):
            ox = np.random.uniform(8, WORLD_W - 8)
            oy = np.random.uniform(8, WORLD_H - 8)
            ow = np.random.uniform(1.0, 3.0)
            oh = np.random.uniform(1.0, 3.0)
            obs = self.world.CreateStaticBody(position=(float(ox), float(oy)))
            obs.CreatePolygonFixture(
                box=(float(ow / 2), float(oh / 2)), density=0, friction=0.5
            )
            obs.userData = {"type": "obstacle", "color": COLOR_OBSTACLE, "z": Z_OBSTACLE}
            self.obstacles.append(obs)

        # Create player tank
        self.player_body = self.world.CreateDynamicBody(
            position=(5.0, WORLD_H / 2),
            angle=0.0,
            linearDamping=2.0,
            angularDamping=5.0,
        )
        self.player_body.CreatePolygonFixture(
            box=(1.5, 1.0), density=5.0, friction=0.3
        )
        self.player_body.userData = {
            "type": "player_body",
            "color": COLOR_PLAYER_BODY,
            "z": Z_TANK_BODY,
            "no_fpv": True,
        }

        # Player turret (visual only, tracked by angle)
        self.player_turret = self.world.CreateDynamicBody(
            position=(5.0, WORLD_H / 2),
            angle=0.0,
            linearDamping=2.0,
            angularDamping=5.0,
        )
        self.player_turret.CreatePolygonFixture(
            box=(1.0, 0.3), density=1.0, friction=0.1
        )
        self.player_turret.userData = {
            "type": "player_turret",
            "color": COLOR_PLAYER_TURRET,
            "z": Z_TANK_TURRET,
            "no_fpv": True,
        }

        # Joint connecting turret to body
        jd = b2RevoluteJointDef()
        jd.bodyA = self.player_body
        jd.bodyB = self.player_turret
        jd.localAnchorA = (0, 0)
        jd.localAnchorB = (-0.5, 0)
        jd.enableMotor = True
        jd.maxMotorTorque = 100.0
        jd.motorSpeed = 0.0
        self.player_turret_joint = self.world.CreateJoint(jd)

        # Create enemy tank
        self.enemy_body = self.world.CreateDynamicBody(
            position=(WORLD_W - 5.0, WORLD_H / 2),
            angle=math.pi,
            linearDamping=2.0,
            angularDamping=5.0,
        )
        self.enemy_body.CreatePolygonFixture(
            box=(1.5, 1.0), density=5.0, friction=0.3
        )
        self.enemy_body.userData = {
            "type": "enemy_body",
            "color": COLOR_ENEMY_BODY,
            "z": Z_TANK_BODY,
        }

        self.enemy_turret = self.world.CreateDynamicBody(
            position=(WORLD_W - 5.0, WORLD_H / 2),
            angle=math.pi,
            linearDamping=2.0,
            angularDamping=5.0,
        )
        self.enemy_turret.CreatePolygonFixture(
            box=(1.0, 0.3), density=1.0, friction=0.1
        )
        self.enemy_turret.userData = {
            "type": "enemy_turret",
            "color": COLOR_ENEMY_TURRET,
            "z": Z_TANK_TURRET,
        }

        jd2 = b2RevoluteJointDef()
        jd2.bodyA = self.enemy_body
        jd2.bodyB = self.enemy_turret
        jd2.localAnchorA = (0, 0)
        jd2.localAnchorB = (-0.5, 0)
        jd2.enableMotor = True
        jd2.maxMotorTorque = 100.0
        jd2.motorSpeed = 0.0
        self.enemy_turret_joint = self.world.CreateJoint(jd2)

        obs = self._get_obs()
        return obs, {}

    def _fire_projectile(self, tank_body, turret_angle, is_player=True):
        pos = tank_body.position
        dx = math.cos(turret_angle) * 2.5
        dy = math.sin(turret_angle) * 2.5
        start_x = pos[0] + dx
        start_y = pos[1] + dy

        proj = self.world.CreateDynamicBody(
            position=(float(start_x), float(start_y)),
            bullet=True,
            linearDamping=0.0,
        )
        proj.CreateCircleFixture(radius=0.2, density=1.0, friction=0.0)
        vx = math.cos(turret_angle) * PROJECTILE_SPEED
        vy = math.sin(turret_angle) * PROJECTILE_SPEED
        proj.linearVelocity = (float(vx), float(vy))
        proj.userData = {
            "type": "projectile",
            "owner": "player" if is_player else "enemy",
            "color": COLOR_PROJECTILE_PLAYER if is_player else COLOR_PROJECTILE_ENEMY,
            "z": Z_PROJECTILE,
            "life": 100,
        }
        self.projectiles.append(proj)

    def _enemy_ai(self):
        """Simple enemy AI: aim at player and fire"""
        if self.enemy_body is None or self.player_body is None:
            return

        ex, ey = self.enemy_body.position
        px, py = self.player_body.position
        target_angle = math.atan2(py - ey, px - ex)

        # Rotate turret towards player
        angle_diff = target_angle - self.enemy_turret_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        self.enemy_turret_angle += np.clip(angle_diff, -TURRET_MAX_RATE * DT, TURRET_MAX_RATE * DT)
        self.enemy_turret.angle = float(self.enemy_turret_angle)
        self.enemy_turret.position = self.enemy_body.position

        # Fire occasionally
        if self.enemy_fire_cooldown <= 0 and abs(angle_diff) < 0.2:
            self._fire_projectile(self.enemy_body, self.enemy_turret_angle, is_player=False)
            self.enemy_fire_cooldown = FIRE_COOLDOWN + np.random.randint(0, 20)

        # Simple movement - dodge or approach
        if self.steps % 100 < 50:
            force_x = math.cos(self.enemy_body.angle) * 20.0
            force_y = math.sin(self.enemy_body.angle) * 20.0
            self.enemy_body.ApplyForceToCenter((float(force_x), float(force_y)), True)
        else:
            # Random movement
            self.enemy_body.ApplyForceToCenter(
                (float(np.random.uniform(-10, 10)), float(np.random.uniform(-10, 10))), True
            )

        self.enemy_fire_cooldown -= 1

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        drive_vel = action[0]
        turret_rot = action[1]
        cannon_elev = action[2]
        fire_trigger = action[3]

        self.steps += 1

        # Apply drive force
        angle = self.player_body.angle
        force_mag = drive_vel * 50.0
        fx = math.cos(angle) * force_mag
        fy = math.sin(angle) * force_mag
        self.player_body.ApplyForceToCenter((float(fx), float(fy)), True)

        # Steering (use action[0] sign for turning too, or we can use part of drive)
        # Actually let's use a combination: drive forward/back + turn
        turn_torque = action[1] * 30.0
        self.player_body.ApplyTorque(float(turn_torque), True)

        # Turret rotation
        self.player_turret_angle += turret_rot * TURRET_MAX_RATE * DT
        self.player_turret.angle = float(self.player_turret_angle)
        self.player_turret.position = self.player_body.position

        # Cannon elevation (stored but affects projectile arc in 3D sense)
        self.player_cannon_elevation = np.clip(
            self.player_cannon_elevation + cannon_elev * 0.05, -0.5, 0.5
        )

        # Fire
        if fire_trigger > 0.5 and self.player_fire_cooldown <= 0:
            self._fire_projectile(self.player_body, self.player_turret_angle, is_player=True)
            self.player_fire_cooldown = FIRE_COOLDOWN
        self.player_fire_cooldown -= 1

        # Enemy AI
        self._enemy_ai()

        # Step physics
        self.world.Step(DT, 6, 2)
        self.world.ClearForces()

        # Check projectile collisions
        projectiles_to_remove = []
        for proj in self.projectiles:
            if proj.userData is None:
                projectiles_to_remove.append(proj)
                continue
            proj.userData["life"] -= 1
            if proj.userData["life"] <= 0:
                projectiles_to_remove.append(proj)
                continue

            px, py = proj.position
            # Out of bounds
            if px < -1 or px > WORLD_W + 1 or py < -1 or py > WORLD_H + 1:
                projectiles_to_remove.append(proj)
                continue

            # Check hit on player
            if proj.userData["owner"] == "enemy":
                dist = math.sqrt(
                    (px - self.player_body.position[0]) ** 2
                    + (py - self.player_body.position[1]) ** 2
                )
                if dist < 2.0:
                    self.player_health -= 10
                    projectiles_to_remove.append(proj)
                    continue

            # Check hit on enemy
            if proj.userData["owner"] == "player":
                dist = math.sqrt(
                    (px - self.enemy_body.position[0]) ** 2
                    + (py - self.enemy_body.position[1]) ** 2
                )
                if dist < 2.0:
                    self.enemy_health -= 10
                    projectiles_to_remove.append(proj)
                    continue

            # Check hit on obstacles/walls
            hit_obstacle = False
            for obs in self.obstacles + self.walls:
                ox, oy = obs.position
                dist = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                if dist < 2.5:
                    hit_obstacle = True
                    break
            if hit_obstacle:
                projectiles_to_remove.append(proj)

        for proj in projectiles_to_remove:
            if proj in self.projectiles:
                self.projectiles.remove(proj)
                self.world.DestroyBody(proj)

        # Compute reward
        reward = 0.0
        terminated = False
        truncated = False

        # Dense reward: aim accuracy (angle to enemy)
        ex, ey = self.enemy_body.position
        plx, ply = self.player_body.position
        angle_to_enemy = math.atan2(ey - ply, ex - plx)
        aim_diff = abs(self.player_turret_angle - angle_to_enemy)
        while aim_diff > math.pi:
            aim_diff = 2 * math.pi - aim_diff
        reward += 0.1 * (1.0 - aim_diff / math.pi)  # reward for good aim

        # Reward for damaging enemy
        if self.enemy_health < MAX_HEALTH:
            reward += (MAX_HEALTH - self.enemy_health) * 0.01

        # Penalty for taking damage
        reward -= (MAX_HEALTH - self.player_health) * 0.005

        # Terminal conditions
        if self.enemy_health <= 0:
            reward += 100.0
            terminated = True
        elif self.player_health <= 0:
            reward -= 50.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "player_health": self.player_health,
            "enemy_health": self.enemy_health,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        px, py = self.player_body.position
        pvx, pvy = self.player_body.linearVelocity
        pa = self.player_body.angle

        ex, ey = self.enemy_body.position
        evx, evy = self.enemy_body.linearVelocity
        ea = self.enemy_body.angle

        obs = np.array(
            [
                px, py, pa, pvx, pvy,
                self.player_turret_angle,
                self.player_cannon_elevation,
                self.player_health / MAX_HEALTH,
                ex, ey, ea, evx, evy,
                self.enemy_turret_angle,
                self.enemy_health / MAX_HEALTH,
                ex - px, ey - py,
            ],
            dtype=np.float32,
        )
        return obs

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
            color = body.userData.get("color", (200, 200, 200))
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

        # Draw health bars
        # Player health
        bar_w = 100
        bar_h = 10
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, bar_w, bar_h))
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (10, 10, int(bar_w * self.player_health / MAX_HEALTH), bar_h),
        )
        # Enemy health
        pygame.draw.rect(self.screen, (50, 50, 50), (SCREEN_W - 110, 10, bar_w, bar_h))
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (
                SCREEN_W - 110,
                10,
                int(bar_w * self.enemy_health / MAX_HEALTH),
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
    """Wraps TankDuelEnv to provide orthographic multi-view 3D rendering."""

    def __init__(self, env):
        super().__init__(env)
        self.view_w = SCREEN_W
        self.view_h = SCREEN_H

    def _get_3d_shapes(self):
        """Extract all bodies and extrude them into 3D shapes."""
        shapes_3d = []
        env = self.env

        for body in env.world.bodies:
            if body.userData is None:
                continue

            z_base = body.userData.get("z", 0.0)
            z_top = z_base + 0.5
            color = body.userData.get("color", (200, 200, 200))
            no_fpv = body.userData.get("no_fpv", False)
            body_type = body.userData.get("type", "unknown")

            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    vertices_2d = [body.transform * v for v in shape.vertices]
                    # Create 3D prism
                    verts_bottom = [(v[0], v[1], z_base) for v in vertices_2d]
                    verts_top = [(v[0], v[1], z_top) for v in vertices_2d]

                    # Top face
                    shapes_3d.append({
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
                        shapes_3d.append({
                            "verts": side_verts,
                            "color": side_color,
                            "z_sort": z_base + 0.25,
                            "no_fpv": no_fpv,
                            "type": body_type,
                        })

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

        return shapes_3d

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

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

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

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _project_fpv(self, shapes, player_pos, player_angle):
        """First Person View: perspective projection from player's viewpoint."""
        surface = pygame.Surface((self.view_w, self.view_h))
        surface.fill(COLOR_SKY)

        # Draw ground
        pygame.draw.rect(
            surface, COLOR_GROUND, (0, self.view_h // 2, self.view_w, self.view_h // 2)
        )

        px, py = player_pos
        # Camera slightly forward and up from player
        cam_forward = 2.0
        cam_x = px + math.cos(player_angle) * cam_forward
        cam_y = py + math.sin(player_angle) * cam_forward
        cam_z = 1.5  # camera height

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
                cos_a = math.cos(-player_angle)
                sin_a = math.sin(-player_angle)
                # After rotation: x_cam = right, y_cam = forward (depth), z_cam = up
                x_cam = dx * cos_a - dy * sin_a
                y_cam = dx * sin_a + dy * cos_a
                z_cam = dz

                cam_verts.append((x_cam, y_cam, z_cam))

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
                sx = int(-cv[0] / cv[1] * fov_scale + self.view_w / 2)
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

        # Sort by depth (far to near for painter's algorithm)
        projected_shapes.sort(key=lambda s: -s["depth"])

        for ps in projected_shapes:
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
                    output.append(intersect(prev, current, edge_func))
            return output

        def intersect(p1, p2, edge_func):
            # Binary search for intersection (simple approach)
            # For axis-aligned edges this is straightforward
            x1, y1 = p1
            x2, y2 = p2
            for _ in range(10):
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

    def render(self):
        """Returns tuple of 4 views: (top_down, rear, side, fpv)"""
        if not pygame.get_init():
            pygame.init()

        env = self.env
        shapes = self._get_3d_shapes()

        player_pos = (env.player_body.position[0], env.player_body.position[1])
        player_angle = env.player_turret_angle  # Use turret angle for FPV direction

        im_top_down = self._project_top_down(shapes, player_pos)
        im_rear = self._project_rear(shapes, player_pos, env.player_body.angle)
        im_side = self._project_side(shapes, player_pos, env.player_body.angle)
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