import gym
import numpy as np
import pygame
from pygame import gfxdraw
import math 
import cv2
from .camera_utils import apply_camera_pose
from gymnasium import spaces
from Box2D.b2 import (world, polygonShape, circleShape, revoluteJointDef)

FPS = 50
SCALE = 30.0   

class ExcavatorEnv(gym.Env):
    """
    Excavator task: Scoop a dynamic payload and drive it backward to drop it in a bin.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # State space increased to 14 to include Payload X, Y, Vx, Vy
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        self.world = world(gravity=(0, -10), doSleep=True)
        self.bodies = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        for body in self.bodies:
            self.world.DestroyBody(body)
        self.bodies = []
        
        # 1. Terrain
        self.terrain = self.world.CreateStaticBody(position=(0, 0), shapes=polygonShape(box=(50, 2)))
        self.bodies.append(self.terrain)
        
        # 2. Goal Bin (Behind the excavator at X=2)
        self.goal_bin = self.world.CreateStaticBody(position=(2.0, 2.0))
        self.goal_bin.CreatePolygonFixture(box=(2.0, 0.2, (0, 0), 0))       # Bottom
        self.goal_bin.CreatePolygonFixture(box=(0.2, 1.0, (-2.0, 1.0), 0))  # Left wall
        self.goal_bin.CreatePolygonFixture(box=(0.2, 1.0, (2.0, 1.0), 0))   # Right wall
        self.bodies.append(self.goal_bin)

        # 3. Dynamic Payload (The rock to pick up, placed in front of excavator)
        self.payload = self.world.CreateDynamicBody(position=(17.0, 2.5))
        self.payload.CreateCircleFixture(radius=0.4, density=0.5, friction=0.8)
        self.bodies.append(self.payload)

        # 4. Excavator Kinematic Chain
        self.chassis = self.world.CreateDynamicBody(position=(10, 3.0))
        self.chassis.CreatePolygonFixture(box=(2.0, 1.0), density=5.0, friction=0.8)
        self.bodies.append(self.chassis)
        
        self.boom = self.world.CreateDynamicBody(position=(11.5, 4.5))
        self.boom.CreatePolygonFixture(box=(3.0, 0.4), density=1.0)
        self.bodies.append(self.boom)
        
        self.dipper = self.world.CreateDynamicBody(position=(14.5, 4.5))
        self.dipper.CreatePolygonFixture(box=(2.0, 0.3), density=1.0)
        self.bodies.append(self.dipper)
        
        # CONCAVE BUCKET: Composed of 3 separate fixtures
        self.bucket = self.world.CreateDynamicBody(position=(16.5, 4.5))
        self.bucket.CreatePolygonFixture(box=(0.6, 0.1, (0, -0.5), 0), density=1.0)    # Bottom plate
        self.bucket.CreatePolygonFixture(box=(0.1, 0.5, (-0.5, 0), 0), density=1.0)    # Back wall
        self.bucket.CreatePolygonFixture(box=(0.1, 0.3, (0.5, -0.2), 0), density=1.0)  # Front lip
        self.bodies.append(self.bucket)
        
        # Joints
        self.boom_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.chassis, bodyB=self.boom, localAnchorA=(1.0, 1.0), localAnchorB=(-2.8, 0),
            enableMotor=True, maxMotorTorque=800, enableLimit=True, lowerAngle=-np.pi/6, upperAngle=np.pi/2
        ))
        self.dipper_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.boom, bodyB=self.dipper, localAnchorA=(2.8, 0), localAnchorB=(-1.8, 0),
            enableMotor=True, maxMotorTorque=500, enableLimit=True, lowerAngle=-np.pi/2, upperAngle=np.pi/2
        ))
        self.bucket_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.dipper, bodyB=self.bucket, localAnchorA=(1.8, 0), localAnchorB=(-0.8, 0),
            enableMotor=True, maxMotorTorque=300, enableLimit=True, lowerAngle=-np.pi/1.2, upperAngle=np.pi/3
        ))
        
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        self.chassis.ApplyForceToCenter((float(action[0] * 200.0), 0.0), wake=True)
        self.boom_joint.motorSpeed = float(action[1] * 2.0)
        self.dipper_joint.motorSpeed = float(action[2] * 2.0)
        self.bucket_joint.motorSpeed = float(action[3] * 2.0)
        
        self.world.Step(1.0 / FPS, 6, 2)
        obs = self._get_obs()
        
        # --- Dense Reward Function ---
        dist_bucket_to_payload = np.linalg.norm([self.bucket.position.x - self.payload.position.x, self.bucket.position.y - self.payload.position.y])
        dist_payload_to_goal = np.linalg.norm([self.payload.position.x - self.goal_bin.position.x, self.payload.position.y - (self.goal_bin.position.y + 1.0)])
        
        # 1. Small penalty for bucket being far from payload
        # 2. Large penalty for payload being far from bin
        reward = float(- (dist_bucket_to_payload * 0.1) - (dist_payload_to_goal * 1.0))
        
        terminated = False
        
        # Success Condition: Payload is inside the bin
        if dist_payload_to_goal < 1.5:
            reward += 100.0
            terminated = True
            
        # Failure Condition: Tipped over
        elif abs(self.chassis.angle) > np.pi / 2:
            reward -= 50.0
            terminated = True
        
        if self.render_mode == "human": self.render()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        return np.array([
            self.chassis.position.x, self.chassis.linearVelocity.x,
            self.chassis.position.y, self.chassis.linearVelocity.y,
            self.boom_joint.angle, self.boom_joint.speed,
            self.dipper_joint.angle, self.dipper_joint.speed,
            self.bucket_joint.angle, self.bucket_joint.speed,
            self.payload.position.x, self.payload.position.y,
            self.payload.linearVelocity.x, self.payload.linearVelocity.y
        ], dtype=np.float32)

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
            self.payload: (200, 100, 50)
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
    """Extrudes 2D Box2D Excavator into 3D, now supporting Circles & Multi-Fixtures."""
    def __init__(self, env):
        super().__init__(env)
        self.SCALE = SCALE
        self.VIEWPORT_W, self.VIEWPORT_H = 800, 600
        self.fov = 400.0
        
    def render(self, camera_pose="Center"):
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
            base_env.payload: (-0.4, 0.4, (200, 100, 50), (150, 80, 40))
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
        cam_x, cam_y, cam_z = track_x - (0.5 * self.SCALE), base_env.chassis.position.y * self.SCALE + (1.5 * self.SCALE), 1.0 * self.SCALE 
        
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

            # FPV Clipping (Sutherland-Hodgman)
            clipped_pts = []
            rel_verts = [(vx - cam_x, vy - cam_y, vz - cam_z) for vx, vy, vz in p_vertices]
            for i in range(len(rel_verts)):
                p_c, p_p = rel_verts[i], rel_verts[i - 1]
                if p_c[0] >= 1.0 and p_p[0] >= 1.0: clipped_pts.append(p_c)
                elif (p_c[0] >= 1.0) != (p_p[0] >= 1.0):
                    t = (1.0 - p_p[0]) / (p_c[0] - p_p[0])
                    clipped_pts.append((1.0, p_p[1] + t * (p_c[1] - p_p[1]), p_p[2] + t * (p_c[2] - p_p[2])))
                    if p_c[0] >= 1.0: clipped_pts.append(p_c)

            pts_fpv, depths_fpv = [], []
            for dx, dy, dz in clipped_pts:
                d_c, lat_c, dy_c = apply_camera_pose(dx, -dz, dy, camera_pose=camera_pose)
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