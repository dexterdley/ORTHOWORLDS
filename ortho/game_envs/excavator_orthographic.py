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
from Box2D.b2 import (world, polygonShape, circleShape, revoluteJointDef)

FPS = 50
SCALE = 30.0

# World / layout constants (metres, Box2D units)
GROUND_Y = 2.0            # Top surface of the terrain slab
WORLD_HALF_W = 50.0       # Terrain spans x in [-50, 50]
CHASSIS_START_X = 8.0
BIN_X = 17.0              # Hopper bin between the excavator and the rock face; the rock must be lifted over its walls
BIN_INNER_HALF_W = 1.7
BIN_WALL_TOP = GROUND_Y + 1.8
FACE_X = 24.0             # Near side of the rock face (static cliff) that ends the work area
FACE_H = 3.0
PAYLOAD_RADIUS = 0.35     # Octagonal rock: flat faces so it rests instead of rolling away
PAYLOAD_GAP = 1.0         # Rock starts up to this far from the rock face (seeded); the face backs the scoop

# Arm geometry: anchors are (local point on parent, local point on child)
SHOULDER_ANCHOR = (1.0, 1.0)              # On the chassis (top front)
BOOM_ANCHORS = ((-2.8, 0.0), (2.8, 0.0))  # Boom root / boom tip  -> length 5.6
DIPPER_ANCHORS = ((-1.8, 0.0), (1.8, 0.0))  # Dipper root / dipper tip -> length 3.6
BUCKET_ANCHOR = (-0.9, 0.0)               # Bucket pivot (just behind its back wall)
BUCKET_CUP = (0.0, -0.45)                 # Centre of the bucket's cup, in bucket coordinates

# Initial relative joint angles (rad): boom raised, dipper folded down, bucket cup facing up
INIT_BOOM, INIT_DIPPER, INIT_BUCKET = 0.9, -2.0, 1.1

DRIVE_SPEED = 4.0         # Track speed (m/s) at |drive| = 1
DRIVE_MAX_FORCE = 1500.0
JOINT_SPEED = 2.0         # Hydraulic joint speed (rad/s) at |action| = 1

SUCCESS_BONUS = 100.0
FAIL_PENALTY = 50.0       # Tipping over / payload or machine leaving the world
SETTLE_STEPS = 10         # Payload must rest in the bin this many consecutive steps

ARM_GROUP = -1            # Arm parts and chassis never collide with each other


class ExcavatorEnv(gym.Env):
    """
    Excavator task: drive up to a hopper bin, reach over it to scoop a loose rock lying at the foot of a
    rock face beyond the bin (the face backs the scoop), lift the rock back over the bin wall and dump it
    so it comes to rest inside the bin.
    Layout along x: excavator (start x=8) -> bin (x=17, walls 1.8 high) -> rock (x~23-23.6) -> rock face (x>=24).

    Action (Box(-1, 1, (4,))):
        0 drive  : track speed, +1 = forward (+x, toward bin and rock), -1 = reverse, 0 = brake/hold
        1 boom   : boom joint speed, +1 = raise boom, -1 = lower boom
        2 dipper : dipper (stick) joint speed, +1 = swing stick up/outward, -1 = fold stick down/in
        3 bucket : bucket joint speed, +1 = curl bucket up (mouth tilts up, holds load), -1 = tip forward (dump)

    Observation (23 floats):
        0-5   chassis x, y, vx, vy, angle, angular velocity
        6-11  boom angle, boom speed, dipper angle, dipper speed, bucket angle, bucket speed (relative joint angles)
        12-14 bucket cup centre x, y and bucket world angle
        15-18 payload x, y, vx, vy
        19-20 payload minus bucket cup (dx, dy)
        21-22 payload x minus bin centre x, payload height above the ground

    Reward: potential-based shaping r = Phi(s') - Phi(s) with
        Phi = -1.0 * min(dist(bucket cup, payload), 10) - 2.0 * |payload x - bin x| + 1.0 * lift
        (lift = payload height above ground capped at the bin wall height, counted as full once the
        payload is horizontally inside the bin), +100 when the payload rests inside the bin (below the
        wall top, between the walls, slow, not touching the bucket, for 10 steps), -50 and terminate
        when the machine tips over or the payload / chassis leaves the world. Truncated at max_episode_steps.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, max_episode_steps=1000):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        self.world = world(gravity=(0, -10), doSleep=True)
        self.bodies = []

    @staticmethod
    def _place(body_anchor_world, local_anchor, angle):
        """Body origin such that local_anchor (rotated by angle) lands on body_anchor_world."""
        c, s = math.cos(angle), math.sin(angle)
        return (body_anchor_world[0] - (c * local_anchor[0] - s * local_anchor[1]),
                body_anchor_world[1] - (s * local_anchor[0] + c * local_anchor[1]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Fresh world each episode so the same seed replays identically (no warm-started contacts)
        self.world = world(gravity=(0, -10), doSleep=True)
        self.bodies = []
        self.steps = 0
        self.settle_count = 0

        # 1. Terrain
        self.terrain = self.world.CreateStaticBody(position=(0, 0), shapes=polygonShape(box=(WORLD_HALF_W, GROUND_Y)))
        self.terrain.fixtures[0].friction = 0.8
        self.bodies.append(self.terrain)

        # 2. Goal hopper bin (ahead of the excavator, walls taller than the rock)
        wall_h = (BIN_WALL_TOP - GROUND_Y - 0.2) / 2.0
        self.goal_bin = self.world.CreateStaticBody(position=(BIN_X, GROUND_Y))
        self.goal_bin.CreatePolygonFixture(box=(BIN_INNER_HALF_W + 0.3, 0.1, (0, 0.1), 0), friction=0.8)                    # Floor
        self.goal_bin.CreatePolygonFixture(box=(0.15, wall_h, (-BIN_INNER_HALF_W - 0.15, 0.2 + wall_h), 0), friction=0.8)  # Left wall
        self.goal_bin.CreatePolygonFixture(box=(0.15, wall_h, (BIN_INNER_HALF_W + 0.15, 0.2 + wall_h), 0), friction=0.8)   # Right wall
        self.bodies.append(self.goal_bin)

        # Rock face beyond the bin: a loose rock can only be scooped when backed against something
        self.rock_face = self.world.CreateStaticBody(position=(FACE_X + 1.5, GROUND_Y + FACE_H / 2.0),
                                                     shapes=polygonShape(box=(1.5, FACE_H / 2.0)))
        self.rock_face.fixtures[0].friction = 0.8
        self.bodies.append(self.rock_face)

        # 3. Dynamic Payload: octagonal rock at the foot of the rock face (rests on a face, damped so it can be scooped and carried)
        half_w = PAYLOAD_RADIUS * math.cos(math.pi / 8)
        px = FACE_X - half_w - 0.02 - float(self.np_random.uniform(0.0, PAYLOAD_GAP))
        octagon = [(PAYLOAD_RADIUS * math.cos(a), PAYLOAD_RADIUS * math.sin(a))
                   for a in np.linspace(0, 2 * math.pi, 8, endpoint=False) + math.pi / 8]
        self.payload = self.world.CreateDynamicBody(position=(px, GROUND_Y + half_w),
                                                    linearDamping=0.2, angularDamping=2.0)
        self.payload.CreatePolygonFixture(vertices=octagon, density=2.0, friction=0.9, restitution=0.0)
        self.bodies.append(self.payload)

        # 4. Excavator Kinematic Chain (bodies created at joint-consistent poses)
        self.chassis = self.world.CreateDynamicBody(position=(CHASSIS_START_X, GROUND_Y + 1.0))
        self.chassis.CreatePolygonFixture(box=(2.0, 1.0), density=10.0, friction=0.3, groupIndex=ARM_GROUP)
        self.bodies.append(self.chassis)

        shoulder = self.chassis.GetWorldPoint(SHOULDER_ANCHOR)
        a_boom = INIT_BOOM
        self.boom = self.world.CreateDynamicBody(position=self._place(shoulder, BOOM_ANCHORS[0], a_boom), angle=a_boom)
        self.boom.CreatePolygonFixture(box=(3.0, 0.4), density=1.0, groupIndex=ARM_GROUP)
        self.bodies.append(self.boom)

        elbow = self.boom.GetWorldPoint(BOOM_ANCHORS[1])
        a_dipper = a_boom + INIT_DIPPER
        self.dipper = self.world.CreateDynamicBody(position=self._place(elbow, DIPPER_ANCHORS[0], a_dipper), angle=a_dipper)
        self.dipper.CreatePolygonFixture(box=(2.0, 0.3), density=1.0, groupIndex=ARM_GROUP)
        self.bodies.append(self.dipper)

        # BUCKET: back wall + wedge-edged floor; open mouth toward +x (loader-style scoop, curl to hold)
        wrist = self.dipper.GetWorldPoint(DIPPER_ANCHORS[1])
        a_bucket = a_dipper + INIT_BUCKET
        self.bucket = self.world.CreateDynamicBody(position=self._place(wrist, BUCKET_ANCHOR, a_bucket), angle=a_bucket)
        # Smooth steel (low friction) so the cutting edge slides under the rock
        self.bucket.CreatePolygonFixture(box=(0.08, 0.55, (-0.8, -0.35), 0), density=1.0, friction=0.2, groupIndex=ARM_GROUP)  # Back wall
        self.bucket.CreatePolygonFixture(vertices=[(-0.88, -0.92), (1.05, -0.92), (0.55, -0.8), (-0.88, -0.8)],
                                         density=1.0, friction=0.2, groupIndex=ARM_GROUP)                                   # Floor + cutting edge
        self.bodies.append(self.bucket)

        # Joints (joint angle = child angle - parent angle, so the reset pose reads INIT_BOOM / INIT_DIPPER / INIT_BUCKET)
        self.boom_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.chassis, bodyB=self.boom, localAnchorA=SHOULDER_ANCHOR, localAnchorB=BOOM_ANCHORS[0],
            referenceAngle=0.0, enableMotor=True, maxMotorTorque=1200, enableLimit=True,
            lowerAngle=-0.6, upperAngle=1.3
        ))
        self.dipper_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.boom, bodyB=self.dipper, localAnchorA=BOOM_ANCHORS[1], localAnchorB=DIPPER_ANCHORS[0],
            referenceAngle=0.0, enableMotor=True, maxMotorTorque=800, enableLimit=True,
            lowerAngle=-2.6, upperAngle=0.6
        ))
        self.bucket_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.dipper, bodyB=self.bucket, localAnchorA=DIPPER_ANCHORS[1], localAnchorB=BUCKET_ANCHOR,
            referenceAngle=0.0, enableMotor=True, maxMotorTorque=400, enableLimit=True,
            lowerAngle=-1.0, upperAngle=2.8
        ))

        self.prev_potential = self._potential()
        return self._get_obs(), {}

    # --- Helpers ---
    def _bucket_cup(self):
        return self.bucket.GetWorldPoint(BUCKET_CUP)

    def _payload_in_bin(self):
        p = self.payload.position
        return abs(p.x - BIN_X) < BIN_INNER_HALF_W and p.y < BIN_WALL_TOP

    def _touching(self, a, b):
        for ce in a.contacts:
            if ce.other == b and ce.contact.touching:
                return True
        return False

    def _potential(self):
        cup = self._bucket_cup()
        p = self.payload.position
        d_reach = min(math.hypot(cup[0] - p.x, cup[1] - p.y), 10.0)
        d_bin = min(abs(p.x - BIN_X), 30.0)
        lift_cap = BIN_WALL_TOP - GROUND_Y
        if abs(p.x - BIN_X) < BIN_INNER_HALF_W:
            lift = lift_cap
        else:
            lift = float(np.clip(p.y - (GROUND_Y + PAYLOAD_RADIUS), 0.0, lift_cap))
        return -1.0 * d_reach - 2.0 * d_bin + 1.0 * lift

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.steps += 1

        # Tracks: velocity-tracking drive with a force limit (0 = brake / hold position)
        target_v = float(action[0]) * DRIVE_SPEED
        force = float(np.clip(self.chassis.mass * 20.0 * (target_v - self.chassis.linearVelocity.x), -DRIVE_MAX_FORCE, DRIVE_MAX_FORCE))
        self.chassis.ApplyForceToCenter((force, 0.0), wake=True)
        self.boom_joint.motorSpeed = float(action[1] * JOINT_SPEED)
        self.dipper_joint.motorSpeed = float(action[2] * JOINT_SPEED)
        self.bucket_joint.motorSpeed = float(action[3] * JOINT_SPEED)

        self.world.Step(1.0 / FPS, 6, 2)
        obs = self._get_obs()

        # --- Potential-based shaping ---
        potential = self._potential()
        reward = float(potential - self.prev_potential)
        self.prev_potential = potential

        terminated = False
        info = {"success": False}
        p, c = self.payload.position, self.chassis.position

        # Success: payload resting inside the bin, released from the bucket
        if self._payload_in_bin() and self.payload.linearVelocity.length < 0.3 and not self._touching(self.payload, self.bucket):
            self.settle_count += 1
        else:
            self.settle_count = 0

        if self.settle_count >= SETTLE_STEPS:
            reward += SUCCESS_BONUS
            terminated = True
            info["success"] = True
        # Failure: tipped over, or payload / machine left the world
        elif (abs(self.chassis.angle) > np.pi / 2
              or abs(p.x) > WORLD_HALF_W or p.y < 0.0
              or abs(c.x) > WORLD_HALF_W or c.y < 0.0):
            reward -= FAIL_PENALTY
            terminated = True

        truncated = (not terminated) and self.steps >= self.max_episode_steps
        if self.render_mode == "human": self.render()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        cup = self._bucket_cup()
        p = self.payload.position
        obs = np.array([
            self.chassis.position.x, self.chassis.position.y,
            self.chassis.linearVelocity.x, self.chassis.linearVelocity.y,
            self.chassis.angle, self.chassis.angularVelocity,
            self.boom_joint.angle, self.boom_joint.speed,
            self.dipper_joint.angle, self.dipper_joint.speed,
            self.bucket_joint.angle, self.bucket_joint.speed,
            cup[0], cup[1], self.bucket.angle,
            p.x, p.y,
            self.payload.linearVelocity.x, self.payload.linearVelocity.y,
            p.x - cup[0], p.y - cup[1],
            p.x - BIN_X, p.y - GROUND_Y
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def close(self):
        if self.screen is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            self.screen = None
            self.clock = None

    def render(self):
        if self.render_mode is None: return
        screen_w, screen_h = 800, 600
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_w, screen_h)) if self.render_mode == "human" else pygame.Surface((screen_w, screen_h))
            self.clock = pygame.time.Clock()

        self.screen.fill((200, 230, 255)) 
        colors = {
            self.terrain: (100, 200, 100), self.chassis: (255, 200, 0), self.boom: (255, 200, 0), 
            self.dipper: (255, 200, 0), self.bucket: (150, 150, 150), self.goal_bin: (100, 100, 200),
            self.payload: (200, 100, 50), self.rock_face: (140, 120, 100)
        }

        for body in self.bodies:
            for fixture in body.fixtures:
                c = colors.get(body, (150, 150, 150))
                # Handle Circles (Payload) vs Polygons
                if isinstance(fixture.shape, circleShape):
                    pos = body.transform * fixture.shape.pos
                    pygame.draw.circle(self.screen, c, (int(pos[0] * SCALE), int(screen_h - pos[1] * SCALE)), int(fixture.shape.radius * SCALE))
                else:
                    vertices = [(body.transform * v) * SCALE for v in fixture.shape.vertices]
                    vertices = [(v[0], screen_h - v[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, c, vertices)
                    pygame.draw.polygon(self.screen, (0, 0, 0), vertices, 1) 

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


# ==========================================
# 2. THE ORTHOGRAPHIC MULTI-VIEW WRAPPER (UPDATED)
# ==========================================
class ExcavatorOrthographicWrapper(gym.Wrapper):
    """
    Extrudes 2D Box2D Excavator into 3D, now supporting Circles & Multi-Fixtures.
    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    The 3D FPV view is a third-person orbit camera trailing behind the excavator (whole machine visible).
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
        self.camera_dist = 10.0 * self.SCALE   # Distance behind the excavator along X-axis
        self.camera_height = 4.0 * self.SCALE  # Vertical offset above the chassis centre (sees over the bin and chassis)

        # Orbit pivot = the chassis, so the camera circles the excavator while facing it
        self.pivot_depth = self.camera_dist

        # Last computed camera position (valid even before the first full render)
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        return self.env.reset(**kwargs)

    @property
    def _cam_anchor(self):
        """Un-orbited 3D camera anchor (camera_dist behind, camera_height above the chassis) in world space; the projection is relative to this."""
        base_env = self.env.unwrapped
        if not hasattr(base_env, 'chassis') or base_env.chassis is None:
            return (0.0, 0.0, 0.0)

        track_x = base_env.chassis.position.x * self.SCALE
        cam_x = track_x - self.camera_dist
        cam_y = base_env.chassis.position.y * self.SCALE + self.camera_height
        cam_z = 0.0
        return (cam_x, cam_y, cam_z)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the orbited FPV camera in world space."""
        base_env = self.env.unwrapped
        if not hasattr(base_env, 'chassis') or base_env.chassis is None:
            return (0.0, 0.0, 0.0)

        anchor_x, anchor_y, anchor_z = self._cam_anchor
        # Camera frame -> world: depth = +x, lateral = -z, vertical = +y
        off_d, off_lat, off_v = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.pivot_depth)
        return (anchor_x + off_d, anchor_y + off_v, anchor_z - off_lat)

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
        
        sky_color = (200, 230, 255)
        for surf in [top_surf, rear_surf, side_surf, fpv_surf]: surf.fill(sky_color)
        if not hasattr(base_env, 'chassis'): return np.zeros((600, 800, 3)), np.zeros((600, 800, 3)), np.zeros((600, 800, 3)), np.zeros((600, 800, 3))

        polys_3d = []
        
        # 1. Grid
        for x in np.arange(-10.0, 40.0, 2.0):
            for z in np.arange(-15.0, 15.0, 2.0):
                c = (100, 200, 100) if (int(x/2.0) + int(z/2.0)) % 2 == 0 else (80, 180, 80)
                y = 2.0 * self.SCALE
                polys_3d.append(([(x*self.SCALE, y, z*self.SCALE), ((x+2)*self.SCALE, y, z*self.SCALE), 
                                  ((x+2)*self.SCALE, y, (z+2)*self.SCALE), (x*self.SCALE, y, (z+2)*self.SCALE)], c, 0))

        # 2. Extrude Objects
        parts_z = {
            base_env.chassis: (-1.5, 1.5, (255, 200, 0), (200, 150, 0)), 
            base_env.boom: (-0.6, 0.6, (255, 200, 0), (200, 150, 0)),
            base_env.dipper: (-0.5, 0.5, (255, 200, 0), (200, 150, 0)),
            base_env.bucket: (-0.8, 0.8, (150, 150, 150), (100, 100, 100)),
            base_env.goal_bin: (-2.0, 2.0, (100, 100, 200), (80, 80, 150)),
            base_env.payload: (-0.4, 0.4, (200, 100, 50), (150, 80, 40)),
            base_env.rock_face: (-4.0, 4.0, (140, 120, 100), (110, 95, 80))
        }

        for body in base_env.bodies:
            if body in parts_z:
                z_min, z_max, c1, c2 = parts_z[body]
                for f in body.fixtures:
                    verts_2d = []
                    # Handle Polygon vs Circle Extraction
                    if hasattr(f.shape, 'vertices'):
                        verts_2d = [(f.body.transform * v * self.SCALE) for v in f.shape.vertices]
                    elif hasattr(f.shape, 'pos'):
                        pos = f.body.transform * f.shape.pos
                        rad = f.shape.radius
                        verts_2d = [((pos[0] + rad*math.cos(a)) * self.SCALE, (pos[1] + rad*math.sin(a)) * self.SCALE) 
                                    for a in np.linspace(0, 2*math.pi, 8, endpoint=False)]
                                    
                    V_back = [(vx, vy, z_min * self.SCALE) for vx, vy in verts_2d]
                    V_front = [(vx, vy, z_max * self.SCALE) for vx, vy in verts_2d]
                    
                    polys_3d.append((V_back, c1, 1))
                    polys_3d.append((V_front, c1, 1))
                    for i in range(len(V_back)):
                        polys_3d.append(([V_back[i], V_back[(i+1)%len(V_back)], V_front[(i+1)%len(V_back)], V_front[i]], c2 if i % 2 == 0 else c1, 1))

        # 3. Apply Projections
        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []
        cy, cx = self.VIEWPORT_H / 2, self.VIEWPORT_W / 2
        
        track_x = base_env.chassis.position.x * self.SCALE
        # Get dynamically calculated position from our property (orbited, for logging)
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        # Projection stays relative to the un-orbited anchor; orbit_camera_frame applies the orbit
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

            # FPV Clipping (Sutherland-Hodgman) on the orbited depth, so the trailing orbit camera clips correctly.
            # Near plane at 1 world unit keeps projected ground coords inside gfxdraw's int16 range.
            near = 1.0 * self.SCALE
            clipped_pts = []
            rel_verts = [orbit_camera_frame(vx - anchor_x, -(vz - anchor_z), vy - anchor_y, self.cam_yaw, self.cam_pitch, self.pivot_depth)
                         for vx, vy, vz in p_vertices]
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

        # 4. Z-Index Sorting
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