"""
Robot Sumo - A Box2D physics-based sumo wrestling game with orthographic multi-view rendering.
Two robots compete to push each other out of a circular ring.

Conventions:
- A robot's forward (plow side) is its local +y axis, i.e. world heading (-sin a, cos a) for body angle a.
- The ring is visual only (no walls): a robot whose centre goes beyond RING_RADIUS is out.
- Action (player): [drive, turn, plow], each in [-1, 1].
    drive: +1 full throttle forward (plow first), -1 full reverse.
    turn:  +1 turn left (counter-clockwise), -1 turn right.
    plow:  target plow yaw; +1 swing plow fully left, -1 fully right, 0 straight.
"""

import math
import numpy as np
from .camera_utils import orbit_camera_frame, orbit_camera_offset

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from Box2D.b2 import (
    world as b2World,
    revoluteJointDef as b2RevoluteJointDef,
)

import pygame
import pygame.gfxdraw

# Module-level constants
Z_NEAR = 0.1
Z_FAR = 50.0
GROUND_Z = 0.0
ROBOT_Z_BASE = 0.0
ROBOT_Z_TOP = 1.5
PLOW_Z_BASE = 0.0
PLOW_Z_TOP = 0.8
RING_Z = -0.1
RING_BORDER_Z = 0.2
WHEEL_Z_BASE = 0.0
WHEEL_Z_TOP = 0.5

# Ring parameters
RING_RADIUS = 5.0
RING_SEGMENTS = 48

# Robot parameters
ROBOT_WIDTH = 0.8
ROBOT_LENGTH = 1.0
ROBOT_MASS = 5.0
WHEEL_RADIUS = 0.25
PLOW_WIDTH = 1.2
PLOW_LENGTH = 0.3
PLOW_ANCHOR_Y = ROBOT_LENGTH * 0.6       # Plow hinge on the robot (local +y = front)
PLOW_HINGE_Y = -PLOW_LENGTH * 0.3        # Hinge point in the plow's own frame
PLOW_MAX_ANGLE = 0.5                     # Plow yaw limit (rad)

# Control parameters
DRIVE_FORCE = 40.0      # Max forward/reverse force (N) along the robot's heading
TURN_TORQUE = 8.0       # Max turning torque (N*m)
LATERAL_GRIP = 0.3      # Fraction of sideways slip removed per step (wheels resist sliding sideways)
PLOW_GAIN = 8.0         # Plow servo gain (motor speed per rad of angle error)
OPPONENT_POWER_RANGE = (0.55, 0.75)  # Opponent drive strength relative to the player (sampled per episode)

# Start pose randomization
START_RADIUS = 2.5
START_RADIUS_JITTER = 0.3
START_ANGLE_JITTER = 0.35   # Jitter of the robots' bearing around the ring (rad)
START_HEADING_JITTER = 0.3  # Jitter of each robot's heading relative to facing the centre (rad)

# Reward parameters
REWARD_WIN = 10.0
REWARD_LOSS = -10.0
SHAPING_EDGE = 1.0      # Weight of the (opp_dist - player_dist) potential
SHAPING_APPROACH = 0.2  # Weight of the (-separation) potential
SEPARATION_CAP = 2.0 * RING_RADIUS

# Colors (RGB)
COLOR_RING = (200, 180, 140)  # Sandy/tan
COLOR_RING_BORDER = (139, 69, 19)  # Brown border
COLOR_RING_LINE = (255, 255, 255)  # White line
COLOR_PLAYER = (30, 144, 255)  # Dodger blue
COLOR_PLAYER_ACCENT = (0, 100, 200)
COLOR_OPPONENT = (220, 50, 50)  # Red
COLOR_OPPONENT_ACCENT = (180, 20, 20)
COLOR_PLOW_PLAYER = (100, 200, 255)  # Light blue
COLOR_PLOW_OPPONENT = (255, 100, 100)  # Light red
COLOR_WHEEL = (50, 50, 50)  # Dark gray
COLOR_GROUND = (80, 60, 40)  # Dark brown ground
COLOR_SKY = (20, 20, 40)  # Dark sky


def heading_vector(angle):
    """World-space forward (plow side) unit vector for a robot with body angle `angle` (local +y)."""
    return -math.sin(angle), math.cos(angle)


def heading_to_angle(dx, dy):
    """Body angle whose forward (local +y) axis points along world direction (dx, dy)."""
    return math.atan2(-dx, dy)


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class RobotSumoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None

        # Action space (player): [drive, turn, plow]
        #   drive: +1 forward (plow first) / -1 reverse, force along the robot's heading
        #   turn:  +1 turn left (counter-clockwise) / -1 turn right
        #   plow:  target plow yaw, +1 fully left / -1 fully right / 0 straight
        # Opponent uses a simple scripted AI
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation (20,):
        #   [0:7]   player:   x, y, sin(angle), cos(angle), vx, vy, omega
        #   [7:14]  opponent: x, y, sin(angle), cos(angle), vx, vy, omega
        #   [14:16] plow_angle_player, plow_angle_opponent
        #   [16:18] dist_to_center_player, dist_to_center_opponent
        #   [18:20] opponent position relative to the player, in the player's frame (forward, left)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        self.world = None
        self.player_body = None
        self.opponent_body = None
        self.player_plow = None
        self.opponent_plow = None
        self.player_plow_joint = None
        self.opponent_plow_joint = None
        self.player_wheels = []
        self.opponent_wheels = []
        self.step_count = 0
        self.max_steps = 500
        self.opponent_power = 1.0
        self._potential = 0.0

    def _create_robot(self, position, angle):
        """Create a robot body with (visual, free-spinning) side wheels and a front plow (local +y)."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def to_world(lx, ly):
            return (position[0] + lx * cos_a - ly * sin_a,
                    position[1] + lx * sin_a + ly * cos_a)

        # Main body
        body = self.world.CreateDynamicBody(
            position=position,
            angle=angle,
            linearDamping=0.5,
            angularDamping=2.0,
        )
        body.CreatePolygonFixture(
            box=(ROBOT_WIDTH / 2, ROBOT_LENGTH / 2),
            density=ROBOT_MASS / (ROBOT_WIDTH * ROBOT_LENGTH),
            friction=0.8,
            restitution=0.3,
        )

        # Wheels (left and right, on local -x / +x). Zero gravity, so they give no traction:
        # they are free-spinning visuals; sideways grip is modelled in _apply_robot_actions.
        wheels = []
        for side in [-1, 1]:
            wheel = self.world.CreateDynamicBody(
                position=to_world(side * ROBOT_WIDTH * 0.6, 0),
                angle=angle,
            )
            wheel.CreateCircleFixture(
                radius=WHEEL_RADIUS,
                density=2.0,
                friction=1.5,
            )
            jd = b2RevoluteJointDef(
                bodyA=body,
                bodyB=wheel,
                localAnchorA=(side * ROBOT_WIDTH * 0.6, 0),
                localAnchorB=(0, 0),
            )
            joint = self.world.CreateJoint(jd)
            wheels.append((wheel, joint))

        # Plow (front, local +y), spawned exactly where the hinge places it
        plow = self.world.CreateDynamicBody(
            position=to_world(0, PLOW_ANCHOR_Y - PLOW_HINGE_Y),
            angle=angle,
        )
        plow.CreatePolygonFixture(
            box=(PLOW_WIDTH / 2, PLOW_LENGTH / 2),
            density=3.0,
            friction=0.5,
            restitution=0.2,
        )

        plow_jd = b2RevoluteJointDef(
            bodyA=body,
            bodyB=plow,
            localAnchorA=(0, PLOW_ANCHOR_Y),
            localAnchorB=(0, PLOW_HINGE_Y),
            enableMotor=True,
            enableLimit=True,
            lowerAngle=-PLOW_MAX_ANGLE,
            upperAngle=PLOW_MAX_ANGLE,
            maxMotorTorque=30.0,
            motorSpeed=0.0,
        )
        plow_joint = self.world.CreateJoint(plow_jd)

        return body, wheels, plow, plow_joint

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.player_wheels = []
        self.opponent_wheels = []
        self.step_count = 0

        # Ring is visual only (no walls): leaving it is checked in step().

        # Randomized start: robots on opposite sides of the centre, roughly facing it
        bearing = rng.uniform(-math.pi, math.pi)
        poses = []
        for b in (bearing, bearing + math.pi + rng.uniform(-START_ANGLE_JITTER, START_ANGLE_JITTER)):
            r = START_RADIUS + rng.uniform(-START_RADIUS_JITTER, START_RADIUS_JITTER)
            x, y = r * math.cos(b), r * math.sin(b)
            heading = heading_to_angle(-x, -y) + rng.uniform(-START_HEADING_JITTER, START_HEADING_JITTER)
            poses.append(((x, y), heading))

        # Create player robot
        self.player_body, self.player_wheels, self.player_plow, self.player_plow_joint = (
            self._create_robot(*poses[0])
        )

        # Create opponent robot
        self.opponent_body, self.opponent_wheels, self.opponent_plow, self.opponent_plow_joint = (
            self._create_robot(*poses[1])
        )
        self.opponent_power = float(rng.uniform(*OPPONENT_POWER_RANGE))

        player_dist, opponent_dist = self._center_dists()
        self._potential = self._compute_potential(player_dist, opponent_dist)

        obs = self._get_obs()
        return obs, {"player_dist": player_dist, "opponent_dist": opponent_dist, "step": 0}

    def _center_dists(self):
        p = self.player_body.position
        o = self.opponent_body.position
        return math.sqrt(p.x ** 2 + p.y ** 2), math.sqrt(o.x ** 2 + o.y ** 2)

    def _compute_potential(self, player_dist, opponent_dist):
        """Shaping potential: opponent near the edge, player near the centre, robots close together."""
        p = self.player_body.position
        o = self.opponent_body.position
        separation = min(math.sqrt((p.x - o.x) ** 2 + (p.y - o.y) ** 2), SEPARATION_CAP)
        return SHAPING_EDGE * (opponent_dist - player_dist) - SHAPING_APPROACH * separation

    def _get_obs(self):
        p = self.player_body
        o = self.opponent_body

        player_dist, opponent_dist = self._center_dists()

        plow_angle_player = self.player_plow_joint.angle if self.player_plow_joint else 0.0
        plow_angle_opponent = self.opponent_plow_joint.angle if self.opponent_plow_joint else 0.0

        # Opponent in the player's frame: forward = local +y, left = local -x
        fx, fy = heading_vector(p.angle)
        dx = o.position.x - p.position.x
        dy = o.position.y - p.position.y
        rel_forward = dx * fx + dy * fy
        rel_left = -dx * fy + dy * fx

        obs = np.array([
            p.position.x, p.position.y, math.sin(p.angle), math.cos(p.angle),
            p.linearVelocity.x, p.linearVelocity.y, p.angularVelocity,
            o.position.x, o.position.y, math.sin(o.angle), math.cos(o.angle),
            o.linearVelocity.x, o.linearVelocity.y, o.angularVelocity,
            plow_angle_player, plow_angle_opponent,
            player_dist, opponent_dist,
            rel_forward, rel_left,
        ], dtype=np.float32)
        return obs

    def _apply_robot_actions(self, body, plow_joint, actions, power=1.0):
        """Apply drive force along the heading, turning torque, sideways grip and plow servo."""
        drive = float(actions[0]) * DRIVE_FORCE * power
        turn = float(actions[1]) * TURN_TORQUE * power
        plow_target = float(actions[2]) * PLOW_MAX_ANGLE

        # Drive along the robot's forward (plow) axis
        fx, fy = heading_vector(body.angle)
        body.ApplyForceToCenter((drive * fx, drive * fy), True)

        # Turning torque (+ = counter-clockwise = left)
        body.ApplyTorque(turn, True)

        # Wheels resist sideways sliding: remove part of the lateral velocity
        rx, ry = fy, -fx  # robot's right (local +x)
        v = body.linearVelocity
        v_lat = v.x * rx + v.y * ry
        impulse = -LATERAL_GRIP * body.mass * v_lat
        body.ApplyLinearImpulse((impulse * rx, impulse * ry), body.worldCenter, True)

        # Plow servo toward the target yaw
        if plow_joint:
            plow_joint.motorSpeed = PLOW_GAIN * (plow_target - plow_joint.angle)

    def _opponent_ai(self):
        """Simple AI for opponent: turn its plow toward the player and push."""
        p = self.player_body
        o = self.opponent_body

        # Direction to player
        dx = p.position.x - o.position.x
        dy = p.position.y - o.position.y
        target_angle = heading_to_angle(dx, dy)

        # Heading error (opponent's forward is its local +y)
        angle_diff = wrap_angle(target_angle - o.angle)

        # Simple proportional control
        turn = np.clip(angle_diff * 2.0, -1.0, 1.0)
        forward = np.clip(1.0 - abs(angle_diff) / (math.pi / 2), 0.2, 1.0)

        return np.array([forward, turn, 0.0], dtype=np.float32)

    def step(self, action):
        if self.world is None:
            raise RuntimeError("RobotSumoEnv.step() called before reset()")
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.step_count += 1

        # Apply player actions
        self._apply_robot_actions(self.player_body, self.player_plow_joint, action)

        # Apply opponent AI actions
        opp_actions = self._opponent_ai()
        self._apply_robot_actions(
            self.opponent_body, self.opponent_plow_joint, opp_actions, self.opponent_power
        )

        # Step physics
        self.world.Step(1.0 / 50.0, 8, 3)
        self.world.ClearForces()

        player_dist, opponent_dist = self._center_dists()

        # Potential-based shaping (telescopes over an episode, so it cannot be farmed by idling)
        potential = self._compute_potential(player_dist, opponent_dist)
        reward = potential - self._potential
        self._potential = potential

        terminated = False
        truncated = False
        winner = None

        # Ring-out: a robot whose centre leaves the ring loses
        if player_dist > RING_RADIUS:
            reward += REWARD_LOSS
            terminated = True
            winner = "opponent"
        elif opponent_dist > RING_RADIUS:
            reward += REWARD_WIN
            terminated = True
            winner = "player"
        elif self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "player_dist": player_dist,
            "opponent_dist": opponent_dist,
            "step": self.step_count,
            "winner": winner,
            "is_success": winner == "player",
        }

        return obs, float(reward), terminated, truncated, info

    def _world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        scale = 40.0  # pixels per meter
        sx = int(self.screen_width / 2 + x * scale)
        sy = int(self.screen_height / 2 - y * scale)
        return sx, sy

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_GROUND)

        scale = 40.0

        # Draw ring
        center = self._world_to_screen(0, 0)
        ring_radius_px = int(RING_RADIUS * scale)
        pygame.draw.circle(self.screen, COLOR_RING, center, ring_radius_px)
        pygame.draw.circle(self.screen, COLOR_RING_BORDER, center, ring_radius_px, 3)

        # Draw inner circle line
        inner_radius = int(RING_RADIUS * 0.7 * scale)
        pygame.draw.circle(self.screen, COLOR_RING_LINE, center, inner_radius, 1)

        # Draw robots
        self._draw_robot(self.player_body, self.player_plow, self.player_wheels,
                         COLOR_PLAYER, COLOR_PLAYER_ACCENT, COLOR_PLOW_PLAYER, scale)
        self._draw_robot(self.opponent_body, self.opponent_plow, self.opponent_wheels,
                         COLOR_OPPONENT, COLOR_OPPONENT_ACCENT, COLOR_PLOW_OPPONENT, scale)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(50)
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            ).copy()

    def _draw_robot(self, body, plow, wheels, color, accent, plow_color, scale):
        """Draw a robot on the screen."""
        # Draw main body
        angle = body.angle
        cx, cy = body.position.x, body.position.y

        # Body corners
        hw = ROBOT_WIDTH / 2
        hl = ROBOT_LENGTH / 2
        corners = [
            (-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)
        ]
        rotated = []
        for lx, ly in corners:
            rx = cx + lx * math.cos(angle) - ly * math.sin(angle)
            ry = cy + lx * math.sin(angle) + ly * math.cos(angle)
            rotated.append(self._world_to_screen(rx, ry))

        if len(rotated) >= 3:
            pygame.draw.polygon(self.screen, color, rotated)
            pygame.draw.polygon(self.screen, accent, rotated, 2)

        # Draw plow
        if plow:
            plow_angle = plow.angle
            pcx, pcy = plow.position.x, plow.position.y
            phw = PLOW_WIDTH / 2
            phl = PLOW_LENGTH / 2
            plow_corners = [
                (-phw, -phl), (phw, -phl), (phw, phl), (-phw, phl)
            ]
            plow_rotated = []
            for lx, ly in plow_corners:
                rx = pcx + lx * math.cos(plow_angle) - ly * math.sin(plow_angle)
                ry = pcy + lx * math.sin(plow_angle) + ly * math.cos(plow_angle)
                plow_rotated.append(self._world_to_screen(rx, ry))

            if len(plow_rotated) >= 3:
                pygame.draw.polygon(self.screen, plow_color, plow_rotated)
                pygame.draw.polygon(self.screen, (255, 255, 255), plow_rotated, 1)

        # Draw wheels
        for wheel, joint in wheels:
            wx, wy = wheel.position.x, wheel.position.y
            screen_pos = self._world_to_screen(wx, wy)
            wheel_r = int(WHEEL_RADIUS * scale)
            pygame.draw.circle(self.screen, COLOR_WHEEL, screen_pos, wheel_r)

        # Draw direction indicator (forward = local +y, plow side)
        hx, hy = heading_vector(angle)
        front_x = cx + ROBOT_LENGTH * 0.3 * hx
        front_y = cy + ROBOT_LENGTH * 0.3 * hy
        pygame.draw.circle(self.screen, (255, 255, 0),
                           self._world_to_screen(front_x, front_y), 3)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class RobotSumoOrthographicWrapper(gym.Wrapper):
    """
    Wrapper that provides orthographic multi-view 3D rendering.
    The 3D FPV view is a third-person orbit camera trailing the player robot (robot visible).
    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """

    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.view_width = 800
        self.view_height = 600
        self.surfaces = {}

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state

        # Follow/Orbit camera baseline offsets (relative to the player robot)
        self.camera_dist = 2.0     # Distance behind the robot along its heading (plow side = forward)
        self.camera_height = 2.0   # Height of the camera above the ground (robot top is 1.5)

        # Orbit focus distance along the FPV depth axis (the player robot, camera_dist ahead)
        self.pivot_depth = self.camera_dist

        # Last computed FPV camera position (world space)
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        return self.env.reset(**kwargs)

    @property
    def _cam_anchor(self):
        """Un-orbited third-person camera anchor (camera_dist behind the player robot along its heading,
        camera_height above the ground) used for camera-relative coordinates."""
        base_env = self.env.unwrapped
        if getattr(base_env, 'player_body', None) is None:
            return (0.0, 0.0, 0.0)
        body = base_env.player_body
        angle = body.angle
        # FPV depth (forward) axis in world space: (-sin a, cos a)
        return (body.position.x + self.camera_dist * math.sin(angle),
                body.position.y - self.camera_dist * math.cos(angle),
                self.camera_height)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the camera in world space."""
        base_env = self.env.unwrapped
        if getattr(base_env, 'player_body', None) is None:
            return (0.0, 0.0, 0.0)

        anchor_x, anchor_y, anchor_z = self._cam_anchor
        angle = base_env.player_body.angle

        # Orbit displacement in the un-orbited camera frame (depth, lateral, vertical)
        d_dep, d_lat, d_vert = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)

        # Map back to world axes (inverse of the FPV mapping: lateral = (cos a, sin a), depth = (-sin a, cos a))
        cam_x = anchor_x + d_lat * math.cos(angle) - d_dep * math.sin(angle)
        cam_y = anchor_y + d_lat * math.sin(angle) + d_dep * math.cos(angle)
        cam_z = anchor_z + d_vert

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

    def _get_3d_objects(self):
        """Extract 3D objects from the 2D environment by extruding shapes."""
        objects = []
        env = self.env

        # Ring floor
        for i in range(RING_SEGMENTS):
            a1 = 2 * math.pi * i / RING_SEGMENTS
            a2 = 2 * math.pi * (i + 1) / RING_SEGMENTS
            x1 = RING_RADIUS * math.cos(a1)
            y1 = RING_RADIUS * math.sin(a1)
            x2 = RING_RADIUS * math.cos(a2)
            y2 = RING_RADIUS * math.sin(a2)
            # Ring border segments as 3D prisms
            objects.append({
                'type': 'edge',
                'vertices_3d': [
                    (x1, y1, RING_Z), (x2, y2, RING_Z),
                    (x2, y2, RING_BORDER_Z), (x1, y1, RING_BORDER_Z),
                ],
                'color': COLOR_RING_BORDER,
                'z_center': RING_BORDER_Z / 2,
            })

        # Ring surface (simplified as a large polygon approximation)
        objects.append({
            'type': 'circle_floor',
            'center': (0, 0),
            'radius': RING_RADIUS,
            'z': RING_Z,
            'color': COLOR_RING,
            'z_center': RING_Z,
        })

        # Player robot body
        if env.player_body:
            objects.extend(self._extrude_robot(
                env.player_body, env.player_plow, env.player_wheels,
                COLOR_PLAYER, COLOR_PLOW_PLAYER, ROBOT_Z_BASE, ROBOT_Z_TOP, True
            ))

        # Opponent robot body
        if env.opponent_body:
            objects.extend(self._extrude_robot(
                env.opponent_body, env.opponent_plow, env.opponent_wheels,
                COLOR_OPPONENT, COLOR_PLOW_OPPONENT, ROBOT_Z_BASE, ROBOT_Z_TOP, False
            ))

        return objects

    def _extrude_robot(self, body, plow, wheels, color, plow_color, z_base, z_top, is_player):
        """Extrude a robot into 3D prisms."""
        objects = []
        angle = body.angle
        cx, cy = body.position.x, body.position.y

        # Main body prism
        hw = ROBOT_WIDTH / 2
        hl = ROBOT_LENGTH / 2
        corners_2d = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        world_corners = []
        for lx, ly in corners_2d:
            rx = cx + lx * math.cos(angle) - ly * math.sin(angle)
            ry = cy + lx * math.sin(angle) + ly * math.cos(angle)
            world_corners.append((rx, ry))

        # Top face
        top_face = [(x, y, z_top) for x, y in world_corners]
        objects.append({
            'type': 'polygon',
            'vertices_3d': top_face,
            'color': color,
            'z_center': z_top,
        })

        # Side faces
        for i in range(4):
            j = (i + 1) % 4
            side = [
                (world_corners[i][0], world_corners[i][1], z_base),
                (world_corners[j][0], world_corners[j][1], z_base),
                (world_corners[j][0], world_corners[j][1], z_top),
                (world_corners[i][0], world_corners[i][1], z_top),
            ]
            # Darken side color
            side_color = tuple(max(0, c - 40) for c in color)
            objects.append({
                'type': 'polygon',
                'vertices_3d': side,
                'color': side_color,
                'z_center': (z_base + z_top) / 2,
            })

        # Plow
        if plow:
            plow_angle = plow.angle
            pcx, pcy = plow.position.x, plow.position.y
            phw = PLOW_WIDTH / 2
            phl = PLOW_LENGTH / 2
            plow_corners_2d = [(-phw, -phl), (phw, -phl), (phw, phl), (-phw, phl)]
            plow_world = []
            for lx, ly in plow_corners_2d:
                rx = pcx + lx * math.cos(plow_angle) - ly * math.sin(plow_angle)
                ry = pcy + lx * math.sin(plow_angle) + ly * math.cos(plow_angle)
                plow_world.append((rx, ry))

            plow_top = [(x, y, PLOW_Z_TOP) for x, y in plow_world]
            objects.append({
                'type': 'polygon',
                'vertices_3d': plow_top,
                'color': plow_color,
                'z_center': PLOW_Z_TOP,
            })

            for i in range(4):
                j = (i + 1) % 4
                side = [
                    (plow_world[i][0], plow_world[i][1], PLOW_Z_BASE),
                    (plow_world[j][0], plow_world[j][1], PLOW_Z_BASE),
                    (plow_world[j][0], plow_world[j][1], PLOW_Z_TOP),
                    (plow_world[i][0], plow_world[i][1], PLOW_Z_TOP),
                ]
                side_color = tuple(max(0, c - 30) for c in plow_color)
                objects.append({
                    'type': 'polygon',
                    'vertices_3d': side,
                    'color': side_color,
                    'z_center': (PLOW_Z_BASE + PLOW_Z_TOP) / 2,
                })

        # Wheels
        for wheel, joint in wheels:
            wx, wy = wheel.position.x, wheel.position.y
            objects.append({
                'type': 'cylinder',
                'center': (wx, wy),
                'radius': WHEEL_RADIUS,
                'z_base': WHEEL_Z_BASE,
                'z_top': WHEEL_Z_TOP,
                'color': COLOR_WHEEL,
                'z_center': (WHEEL_Z_BASE + WHEEL_Z_TOP) / 2,
            })

        return objects

    def _project_top_down(self, x, y, z, player_x, player_y):
        """Top-down projection (looking down Z axis)."""
        scale = 50.0
        sx = self.view_width / 2 + (x - player_x) * scale
        sy = self.view_height / 2 - (y - player_y) * scale
        return int(sx), int(sy)

    def _project_rear(self, x, y, z, player_x, player_y, player_angle):
        """Rear view projection (looking from behind the player)."""
        # Transform to player-relative coordinates
        dx = x - player_x
        dy = y - player_y
        # Rotate to player's frame
        rx = dx * math.cos(-player_angle) - dy * math.sin(-player_angle)
        ry = dx * math.sin(-player_angle) + dy * math.cos(-player_angle)

        scale = 50.0
        # Rear view: X is left-right, Z is up-down, Y is depth
        sx = self.view_width / 2 + rx * scale
        sy = self.view_height / 2 - z * scale * 2 + 100
        depth = ry
        return int(sx), int(sy), depth

    def _project_side(self, x, y, z, player_x, player_y, player_angle):
        """Side view projection (looking from the side)."""
        dx = x - player_x
        dy = y - player_y
        rx = dx * math.cos(-player_angle) - dy * math.sin(-player_angle)
        ry = dx * math.sin(-player_angle) + dy * math.cos(-player_angle)

        scale = 50.0
        # Side view: Y is forward-back (horizontal), Z is up-down
        sx = self.view_width / 2 + ry * scale
        sy = self.view_height / 2 - z * scale * 2 + 100
        depth = -rx
        return int(sx), int(sy), depth

    def _project_fpv(self, x, y, z, player_x, player_y, player_angle):
        """First-person view projection."""
        dx = x - player_x
        dy = y - player_y
        # Rotate to player's frame
        rx = dx * math.cos(-player_angle) - dy * math.sin(-player_angle)
        ry = dx * math.sin(-player_angle) + dy * math.cos(-player_angle)

        # Perspective-like projection from player's viewpoint
        depth = ry + 0.5  # Forward distance
        if depth < 0.1:
            depth = 0.1

        scale = 200.0
        sx = self.view_width / 2 + (rx / depth) * scale
        sy = self.view_height / 2 - ((z - 0.7) / depth) * scale
        return int(sx), int(sy), depth

    def _render_top_down(self, objects):
        """Render top-down view."""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_GROUND)

        player_x = self.env.player_body.position.x
        player_y = self.env.player_body.position.y

        # Sort by Z (painter's algorithm)
        sorted_objects = sorted(objects, key=lambda o: o['z_center'])

        for obj in sorted_objects:
            if obj['type'] == 'circle_floor':
                cx, cy = obj['center']
                sx, sy = self._project_top_down(cx, cy, obj['z'], player_x, player_y)
                radius_px = int(obj['radius'] * 50.0)
                pygame.draw.circle(surface, obj['color'], (sx, sy), radius_px)
                pygame.draw.circle(surface, COLOR_RING_BORDER, (sx, sy), radius_px, 2)
            elif obj['type'] == 'polygon' or obj['type'] == 'edge':
                points = []
                for v in obj['vertices_3d']:
                    px, py = self._project_top_down(v[0], v[1], v[2], player_x, player_y)
                    points.append((px, py))
                if len(points) >= 3:
                    try:
                        pygame.draw.polygon(surface, obj['color'], points)
                        pygame.gfxdraw.aapolygon(surface, points, (255, 255, 255))
                    except (ValueError, OverflowError):
                        pass
            elif obj['type'] == 'cylinder':
                cx, cy = obj['center']
                sx, sy = self._project_top_down(cx, cy, obj['z_top'], player_x, player_y)
                radius_px = int(obj['radius'] * 50.0)
                pygame.draw.circle(surface, obj['color'], (sx, sy), max(1, radius_px))

        # Draw crosshair at center (player position)
        cx, cy = self.view_width // 2, self.view_height // 2
        pygame.draw.line(surface, (255, 255, 0), (cx - 10, cy), (cx + 10, cy), 1)
        pygame.draw.line(surface, (255, 255, 0), (cx, cy - 10), (cx, cy + 10), 1)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def _render_rear(self, objects):
        """Render rear view."""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_SKY)

        # Draw ground
        pygame.draw.rect(surface, COLOR_GROUND,
                         (0, self.view_height // 2 + 50, self.view_width, self.view_height // 2))

        player_x = self.env.player_body.position.x
        player_y = self.env.player_body.position.y
        player_angle = self.env.player_body.angle

        # Sort by depth (far to near)
        render_items = []
        for obj in objects:
            if obj['type'] == 'polygon' or obj['type'] == 'edge':
                points = []
                total_depth = 0
                valid = True
                for v in obj['vertices_3d']:
                    sx, sy, depth = self._project_rear(
                        v[0], v[1], v[2], player_x, player_y, player_angle
                    )
                    points.append((sx, sy))
                    total_depth += depth
                avg_depth = total_depth / max(len(obj['vertices_3d']), 1)
                if len(points) >= 3:
                    render_items.append((avg_depth, points, obj['color']))
            elif obj['type'] == 'cylinder':
                cx, cy = obj['center']
                sx, sy, depth = self._project_rear(
                    cx, cy, obj['z_top'], player_x, player_y, player_angle
                )
                radius_px = int(obj['radius'] * 50.0)
                render_items.append((depth, ('circle', sx, sy, radius_px), obj['color']))

        # Sort far to near
        render_items.sort(key=lambda x: -x[0])

        for item in render_items:
            depth, geom, color = item
            if isinstance(geom, tuple) and geom[0] == 'circle':
                _, sx, sy, r = geom
                if 0 <= sx <= self.view_width and 0 <= sy <= self.view_height:
                    pygame.draw.circle(surface, color, (sx, sy), max(1, r))
            else:
                points = geom
                # Clip to screen bounds roughly
                clipped = [(max(-100, min(self.view_width + 100, p[0])),
                            max(-100, min(self.view_height + 100, p[1]))) for p in points]
                try:
                    pygame.draw.polygon(surface, color, clipped)
                    pygame.gfxdraw.aapolygon(surface, clipped, (200, 200, 200))
                except (ValueError, OverflowError):
                    pass

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def _render_side(self, objects):
        """Render side view."""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_SKY)

        pygame.draw.rect(surface, COLOR_GROUND,
                         (0, self.view_height // 2 + 50, self.view_width, self.view_height // 2))

        player_x = self.env.player_body.position.x
        player_y = self.env.player_body.position.y
        player_angle = self.env.player_body.angle

        render_items = []
        for obj in objects:
            if obj['type'] == 'polygon' or obj['type'] == 'edge':
                points = []
                total_depth = 0
                for v in obj['vertices_3d']:
                    sx, sy, depth = self._project_side(
                        v[0], v[1], v[2], player_x, player_y, player_angle
                    )
                    points.append((sx, sy))
                    total_depth += depth
                avg_depth = total_depth / max(len(obj['vertices_3d']), 1)
                if len(points) >= 3:
                    render_items.append((avg_depth, points, obj['color']))
            elif obj['type'] == 'cylinder':
                cx, cy = obj['center']
                sx, sy, depth = self._project_side(
                    cx, cy, obj['z_top'], player_x, player_y, player_angle
                )
                radius_px = int(obj['radius'] * 50.0)
                render_items.append((depth, ('circle', sx, sy, radius_px), obj['color']))

        render_items.sort(key=lambda x: -x[0])

        for item in render_items:
            depth, geom, color = item
            if isinstance(geom, tuple) and geom[0] == 'circle':
                _, sx, sy, r = geom
                if 0 <= sx <= self.view_width and 0 <= sy <= self.view_height:
                    pygame.draw.circle(surface, color, (sx, sy), max(1, r))
            else:
                points = geom
                clipped = [(max(-100, min(self.view_width + 100, p[0])),
                            max(-100, min(self.view_height + 100, p[1]))) for p in points]
                try:
                    pygame.draw.polygon(surface, color, clipped)
                    pygame.gfxdraw.aapolygon(surface, clipped, (200, 200, 200))
                except (ValueError, OverflowError):
                    pass

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def _clip_polygon_near(self, poly_3d):
        """Sutherland-Hodgman clipping against near plane (depth >= 0.1)"""
        near_z = 0.1
        clipped = []
        if not poly_3d:
            return clipped

        for i in range(len(poly_3d)):
            p1 = poly_3d[i]
            p2 = poly_3d[(i + 1) % len(poly_3d)]

            # Check if points are inside (depth >= near_z)
            # p = (d_c, lat_c, vert_c) in the orbited camera frame; depth is d_c
            d1 = p1[0]
            d2 = p2[0]

            inside1 = d1 >= near_z
            inside2 = d2 >= near_z

            if inside1:
                clipped.append(p1)
            
            if inside1 != inside2:
                # Intersect with near_z plane
                t = (near_z - d1) / (d2 - d1)
                ix = p1[0] + t * (p2[0] - p1[0])
                iy = p1[1] + t * (p2[1] - p1[1])
                iz = p1[2] + t * (p2[2] - p1[2])
                clipped.append((ix, iy, iz))
        return clipped

    def _render_fpv(self, objects):
        """Render third-person 3D view (orbit camera trailing the player robot)."""
        surface = pygame.Surface((self.view_width, self.view_height))
        surface.fill(COLOR_SKY)

        # Ground plane
        pygame.draw.rect(surface, COLOR_RING,
                         (0, self.view_height // 2, self.view_width, self.view_height // 2))

        player_x, player_y, anchor_z = self._cam_anchor
        player_angle = self.env.player_body.angle

        render_items = []
        for obj in objects:
            if obj.get('no_fpv', False):
                continue

            if obj['type'] == 'polygon' or obj['type'] == 'edge':
                # Transform vertices to the (orbited) camera frame, then clip on orbited depth
                cam_space_verts = []
                for v in obj['vertices_3d']:
                    dx = v[0] - player_x
                    dy = v[1] - player_y
                    rx = dx * math.cos(-player_angle) - dy * math.sin(-player_angle)
                    ry = dx * math.sin(-player_angle) + dy * math.cos(-player_angle)
                    cam_space_verts.append(orbit_camera_frame(ry, rx, v[2] - anchor_z, self.cam_yaw, self.cam_pitch, self.pivot_depth))

                clipped_verts = self._clip_polygon_near(cam_space_verts)
                if len(clipped_verts) < 3:
                    continue

                points = []
                total_depth = 0
                for v in clipped_verts:
                    scale = 200.0
                    d_c, lat_c, vert_c = v
                    if d_c < 0.1:
                        d_c = 0.1
                    sx = self.view_width / 2 + (lat_c / d_c) * scale
                    sy = self.view_height / 2 - (vert_c / d_c) * scale
                    points.append((sx, sy))
                    total_depth += d_c
                
                avg_depth = total_depth / len(clipped_verts)
                render_items.append((avg_depth, points, obj['color']))

            elif obj['type'] == 'cylinder':
                cx, cy = obj['center']
                dx = cx - player_x
                dy = cy - player_y
                rx = dx * math.cos(-player_angle) - dy * math.sin(-player_angle)
                ry = dx * math.sin(-player_angle) + dy * math.cos(-player_angle)
                d_c, lat_c, vert_c = orbit_camera_frame(ry, rx, obj['z_top'] - anchor_z, self.cam_yaw, self.cam_pitch, self.pivot_depth)
                if d_c < 0.1:
                    continue
                scale = 200.0
                sx = self.view_width / 2 + (lat_c / d_c) * scale
                sy = self.view_height / 2 - (vert_c / d_c) * scale
                radius_px = int(obj['radius'] * 200.0 / d_c)
                render_items.append((d_c, ('circle', sx, sy, radius_px), obj['color']))

        render_items.sort(key=lambda x: -x[0])

        for item in render_items:
            depth, geom, color = item
            if isinstance(geom, tuple) and geom[0] == 'circle':
                _, sx, sy, r = geom
                if -100 <= sx <= self.view_width + 100 and -100 <= sy <= self.view_height + 100:
                    pygame.draw.circle(surface, color, (int(sx), int(sy)), max(1, min(r, 200)))
            else:
                points = geom
                clipped = [(max(-2000, min(self.view_width + 2000, p[0])),
                            max(-2000, min(self.view_height + 2000, p[1]))) for p in points]
                try:
                    pygame.draw.polygon(surface, color, clipped)
                    pygame.gfxdraw.aapolygon(surface, clipped, (180, 180, 180))
                except (ValueError, OverflowError):
                    pass

        # Draw HUD crosshair
        cx, cy = self.view_width // 2, self.view_height // 2
        pygame.draw.line(surface, (0, 255, 0), (cx - 15, cy), (cx + 15, cy), 2)
        pygame.draw.line(surface, (0, 255, 0), (cx, cy - 15), (cx, cy + 15), 2)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def render(self, camera_pose="Center"):
        """Return tuple of 4 orthographic views."""
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)

        if not pygame.get_init():
            pygame.init()
            pygame.font.init()

        # Get 3D objects
        objects = self._get_3d_objects()

        # Render all four views
        im_top_down = self._render_top_down(objects)
        im_rear = self._render_rear(objects)
        im_side = self._render_side(objects)
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        im_fpv = self._render_fpv(objects)

        return (im_top_down, im_rear, im_side, im_fpv)

    def close(self):
        self.env.close()


# Main entry point for testing
if __name__ == "__main__":
    pygame.init()
    pygame.font.init()

    env = RobotSumoEnv(render_mode="rgb_array")
    wrapper = RobotSumoOrthographicWrapper(env)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Run a few steps
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            views = wrapper.render()
            print(f"Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            print(f"  View shapes: {[v.shape for v in views]}")
            print(f"  Player dist: {info['player_dist']:.2f}, Opponent dist: {info['opponent_dist']:.2f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}. Total reward: {total_reward:.3f}")
            obs, info = env.reset()
            total_reward = 0

    # Display combined view
    screen = pygame.display.set_mode((1600, 1200))
    pygame.display.set_caption("Robot Sumo - Orthographic Views")

    obs, info = env.reset()
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

        views = wrapper.render()

        # Compose views into 2x2 grid
        for i, view in enumerate(views):
            surf = pygame.surfarray.make_surface(view.transpose(1, 0, 2))
            x = (i % 2) * 800
            y = (i // 2) * 600
            screen.blit(surf, (x, y))

        pygame.display.flip()
        clock.tick(30)

    env.close()
    pygame.quit()