try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
import math 
import cv2
from .camera_utils import apply_camera_pose
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
from Box2D.b2 import (world, polygonShape, circleShape, revoluteJointDef)

FPS = 50
SCALE = 30.0   

class WreckingBallEnv(gym.Env):
    """
    Wrecking Ball Task:
    Control a crane to swing a heavy cable-suspended wrecking ball 
    and smash target structural pillars down to a specified target height.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Actions: 
        # 0: Chassis Drive Force (-1..1)
        # 1: Boom Swing Motor Speed (-1..1)
        # 2: Winch Reel Speed (-1..1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # State observation:
        # 0-3: Chassis (X, Vx, Y, Vy)
        # 4-5: Boom Joint (angle, speed)
        # 6-7: Cable Winch / Joint (angle, speed)
        # 8-11: Wrecking Ball (X, Y, Vx, Vy)
        # 12-14: Target Pillar Heights (Pillar 1 top Y, Pillar 2 top Y, Max Top Y)
        # 15: Target Goal Height Threshold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        self.world = world(gravity=(0, -10), doSleep=True)
        self.bodies = []
        self.target_height = 3.5  # Goal height threshold to smash pillars below
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        for body in self.bodies:
            self.world.DestroyBody(body)
        self.bodies = []
        
        # 1. Terrain Ground
        self.terrain = self.world.CreateStaticBody(position=(0, 0), shapes=polygonShape(box=(50, 2)))
        self.bodies.append(self.terrain)
        
        # 2. Crane Chassis (Base Vehicle)
        self.chassis = self.world.CreateDynamicBody(position=(6.0, 3.0))
        self.chassis.CreatePolygonFixture(box=(2.5, 1.0), density=6.0, friction=0.9)
        self.bodies.append(self.chassis)
        
        # 3. Crane Boom (Arm)
        self.boom = self.world.CreateDynamicBody(position=(8.5, 5.0))
        self.boom.CreatePolygonFixture(box=(3.5, 0.3), density=1.5, friction=0.5)
        self.bodies.append(self.boom)
        
        # Joint between Chassis and Boom
        self.boom_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.chassis, bodyB=self.boom, localAnchorA=(1.2, 0.8), localAnchorB=(-3.3, 0),
            enableMotor=True, maxMotorTorque=1200, enableLimit=True, lowerAngle=-np.pi/6, upperAngle=np.pi/2.5
        ))
        
        # 4. Multi-Segment Cable Chain
        self.cable_segments = []
        num_links = 4
        prev_body = self.boom
        prev_anchor = (3.3, 0.0)
        
        for i in range(num_links):
            link_pos = (11.5 + i * 0.6, 5.0 - i * 0.5)
            link = self.world.CreateDynamicBody(position=link_pos)
            link.CreatePolygonFixture(box=(0.3, 0.08), density=0.8, friction=0.2)
            self.bodies.append(link)
            self.cable_segments.append(link)
            
            joint = self.world.CreateJoint(revoluteJointDef(
                bodyA=prev_body, bodyB=link, localAnchorA=prev_anchor, localAnchorB=(-0.3, 0),
                enableMotor=True if i == 0 else False, maxMotorTorque=400
            ))
            if i == 0:
                self.winch_joint = joint
                
            prev_body = link
            prev_anchor = (0.3, 0.0)
            
        # 5. Heavy Wrecking Ball
        last_link = self.cable_segments[-1]
        self.ball = self.world.CreateDynamicBody(position=(14.0, 3.0))
        self.ball.CreateCircleFixture(radius=0.7, density=10.0, friction=0.8, restitution=0.4) # Heavy steel ball!
        self.bodies.append(self.ball)
        
        self.ball_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=last_link, bodyB=self.ball, localAnchorA=(0.3, 0.0), localAnchorB=(0, 0.6)
        ))
        
        # 6. Target Structural Pillars (2 Vertical Columns of Stacked Dynamic Concrete Blocks)
        self.pillars = []
        pillar_x_coords = [16.5, 18.5]
        for px in pillar_x_coords:
            column_blocks = []
            for h_idx in range(3): # 3 stacked blocks per pillar
                py = 2.0 + 0.8 + h_idx * 1.4
                block = self.world.CreateDynamicBody(position=(px, py))
                block.CreatePolygonFixture(box=(0.6, 0.7), density=2.0, friction=0.8, restitution=0.1)
                self.bodies.append(block)
                column_blocks.append(block)
            self.pillars.append(column_blocks)
            
        self.initial_max_height = self._get_max_pillar_height()
        self.prev_max_height = self.initial_max_height
        
        return self._get_obs(), {}

    def _get_max_pillar_height(self):
        max_h = 2.0
        for col in self.pillars:
            for block in col:
                # Top edge of block
                top_y = block.position.y + 0.7
                if top_y > max_h:
                    max_h = top_y
        return float(max_h)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # Action execution
        self.chassis.ApplyForceToCenter((float(action[0] * 350.0), 0.0), wake=True)
        self.boom_joint.motorSpeed = float(action[1] * 2.0)
        self.winch_joint.motorSpeed = float(action[2] * 2.5)
        
        self.world.Step(1.0 / FPS, 6, 2)
        obs = self._get_obs()
        
        current_max_h = self._get_max_pillar_height()
        
        # --- Reward Calculation ---
        ball_pos = self.ball.position
        pillar_center_x = 17.5
        dist_ball_to_pillars = abs(ball_pos.x - pillar_center_x)
        
        # 1. Guidance: penalty for being far from pillars
        reward = float(-0.05 * dist_ball_to_pillars)
        
        # 2. Kinetic Energy Reward: encourage high speed when close to pillars
        ball_speed = float(np.linalg.norm(self.ball.linearVelocity))
        if dist_ball_to_pillars < 3.0:
            reward += float(0.02 * (ball_speed ** 2))
            
        # 3. Destruction Progress Reward: positive reward when pillar heights drop
        height_reduction = self.prev_max_height - current_max_h
        if height_reduction > 0:
            reward += float(height_reduction * 15.0)
        self.prev_max_height = current_max_h
        
        terminated = False
        
        # Success Condition: All pillars smashed below target threshold
        if current_max_h <= self.target_height:
            reward += 100.0
            terminated = True
            
        # Failure Condition: Crane chassis tipped over
        elif abs(self.chassis.angle) > np.pi / 2.2:
            reward -= 50.0
            terminated = True
            
        if self.render_mode == "human": self.render()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        p1_max_h = max([b.position.y + 0.7 for b in self.pillars[0]])
        p2_max_h = max([b.position.y + 0.7 for b in self.pillars[1]])
        max_h = max(p1_max_h, p2_max_h)
        
        return np.array([
            self.chassis.position.x, self.chassis.linearVelocity.x,
            self.chassis.position.y, self.chassis.linearVelocity.y,
            self.boom_joint.angle, self.boom_joint.speed,
            self.winch_joint.angle, self.winch_joint.speed,
            self.ball.position.x, self.ball.position.y,
            self.ball.linearVelocity.x, self.ball.linearVelocity.y,
            p1_max_h, p2_max_h, max_h, self.target_height
        ], dtype=np.float32)

    def render(self):
        if self.render_mode is None: return
        screen_w, screen_h = 800, 600
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_w, screen_h)) if self.render_mode == "human" else pygame.Surface((screen_w, screen_h))
            self.clock = pygame.time.Clock()

        self.screen.fill((210, 235, 255)) 
        
        # Color mapping
        colors = {
            self.terrain: (110, 190, 110),
            self.chassis: (230, 160, 20),
            self.boom: (230, 160, 20),
            self.ball: (60, 60, 70),
        }
        for seg in self.cable_segments:
            colors[seg] = (40, 40, 40)
        for col in self.pillars:
            for block in col:
                colors[block] = (160, 160, 170)

        for body in self.bodies:
            for fixture in body.fixtures:
                c = colors.get(body, (150, 150, 150))
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


# ==============================================================================
# 2. THE ORTHOGRAPHIC MULTI-VIEW WRAPPER FOR WRECKING BALL
# ==============================================================================
class WreckingBallOrthographicWrapper(gym.Wrapper):
    """
    Extrudes 2D Box2D Wrecking Ball components into 3D volumes and renders 4 views:
    1. Top-Down Plan View
    2. Orthographic Rear Elevation
    3. Orthographic Side Elevation
    4. True 3D Perspective FPV (with Sutherland-Hodgman near-plane clipping)
    """
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
        
        sky_color = (210, 235, 255)
        for surf in [top_surf, rear_surf, side_surf, fpv_surf]: surf.fill(sky_color)
        if not hasattr(base_env, 'chassis'): 
            return np.zeros((600, 800, 3)), np.zeros((600, 800, 3)), np.zeros((600, 800, 3)), np.zeros((600, 800, 3))

        polys_3d = []
        
        # 1. 3D Checkerboard Ground Grid
        for x in np.arange(-10.0, 40.0, 2.0):
            for z in np.arange(-15.0, 15.0, 2.0):
                c = (110, 190, 110) if (int(x/2.0) + int(z/2.0)) % 2 == 0 else (90, 170, 90)
                y = 2.0 * self.SCALE
                polys_3d.append(([(x*self.SCALE, y, z*self.SCALE), ((x+2)*self.SCALE, y, z*self.SCALE), 
                                  ((x+2)*self.SCALE, y, (z+2)*self.SCALE), (x*self.SCALE, y, (z+2)*self.SCALE)], c, 0))

        # 2. Extrude Objects into 3D Volumetric Mesh
        parts_z = {
            base_env.chassis: (-1.6, 1.6, (230, 160, 20), (180, 120, 15)), 
            base_env.boom: (-0.5, 0.5, (230, 160, 20), (180, 120, 15)),
            base_env.ball: (-0.7, 0.7, (60, 60, 70), (40, 40, 50))
        }
        
        for seg in base_env.cable_segments:
            parts_z[seg] = (-0.15, 0.15, (40, 40, 40), (20, 20, 20))
            
        for col in base_env.pillars:
            for block in col:
                parts_z[block] = (-0.8, 0.8, (160, 160, 170), (120, 120, 130))

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
                                    
                    V_back = [(vx, vy, z_min * self.SCALE) for vx, vy in verts_2d]
                    V_front = [(vx, vy, z_max * self.SCALE) for vx, vy in verts_2d]
                    
                    polys_3d.append((V_back, c1, 1))
                    polys_3d.append((V_front, c1, 1))
                    for i in range(len(V_back)):
                        polys_3d.append(([V_back[i], V_back[(i+1)%len(V_back)], V_front[(i+1)%len(V_front)], V_front[i]], c2 if i % 2 == 0 else c1, 1))

        # 3. Multi-View Projections
        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []
        cy, cx = self.VIEWPORT_H / 2, self.VIEWPORT_W / 2
        
        track_x = base_env.chassis.position.x * self.SCALE
        cam_x, cam_y, cam_z = track_x - (0.5 * self.SCALE), base_env.chassis.position.y * self.SCALE + (2.0 * self.SCALE), 1.2 * self.SCALE 
        
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

            # FPV Near-Plane Clipping (Sutherland-Hodgman)
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
