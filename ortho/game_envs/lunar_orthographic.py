import gym
import numpy as np
import pygame
from pygame import gfxdraw
import math
from .camera_utils import apply_camera_pose

class LunarOrthographicWrapper(gym.Wrapper):
    """
    Wraps LunarLander to extrude 2D Box2D shapes into 3D volumes,
    rendering Top-Down, Orthographic Rear, Orthographic Side, and True 3D FPV.
    """
    def __init__(self, env):
        super().__init__(env)
        self.SCALE = 30.0
        self.VIEWPORT_W = 600
        self.VIEWPORT_H = 400
        self.fov = 400.0
        
        # Camera offset for the 3D Perspective View
        self.camera_z_offset = 12.0 
        self.camera_y_offset = 3.0
        
    def render(self, camera_pose="Center"):
        base_env = self.env.unwrapped
        
        # 1. Setup Surfaces
        top_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        rear_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        side_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        fpv_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        
        sky_color = (10, 10, 20) # Deep space black/blue
        for surf in [top_surf, rear_surf, side_surf, fpv_surf]:
            surf.fill(sky_color)
            
        if base_env.lander is None:
            dummy = np.zeros((self.VIEWPORT_H, self.VIEWPORT_W, 3), dtype=np.uint8)
            return dummy, dummy, dummy, dummy
            
        lander_x, lander_y = base_env.lander.position
        
        polys_3d = []
        
        # 2. Extract and Extrude Box2D Bodies
        for body in base_env.world.bodies:
            is_particle = False
            
            # Assign Z-Depth Volume & Color per object type
            if body == base_env.lander:
                z_min, z_max = -1.5, 1.5
                c1, c2 = (128, 102, 230), (76, 76, 127) # Purple Hull
            elif body in base_env.legs:
                is_left = (body == base_env.legs[0])
                z_min, z_max = (1.6, 2.0) if is_left else (-2.0, -1.6)
                c1, c2 = (150, 150, 150), (100, 100, 100) # Grey Legs
            elif body.type == 0: 
                z_min, z_max = -10.0, 10.0
                c1, c2 = (200, 200, 200), (120, 120, 120) # Moon surface
            else: 
                # Completely ignore/remove engine particles
                continue
                
            for f in body.fixtures:
                verts_2d = []
                if hasattr(f.shape, 'vertices'):
                    verts_2d = [body.transform * v for v in f.shape.vertices]
                elif hasattr(f.shape, 'pos'): # Particles use circles
                    pos = body.transform * f.shape.pos
                    rad = f.shape.radius
                    verts_2d = [(pos[0] + rad*math.cos(a), pos[1] + rad*math.sin(a)) 
                                for a in np.linspace(0, 2*math.pi, 6, endpoint=False)]
                    
                    # FIX: Make the 3D depth exactly match the 2D radius
                    # This prevents the sparks from being stretched into massive 1-meter thick bricks
                    if is_particle:
                        z_min, z_max = -rad, rad
                                
                if len(verts_2d) < 2: continue
                
                # Convert 1D terrain edges into solid 2D blocks descending to Y=0
                if len(verts_2d) == 2 and body.type == 0:
                    v1, v2 = verts_2d
                    verts_2d = [v1, v2, (v2[0], 0.0), (v1[0], 0.0)]
                    
                scaled_verts = [(vx * self.SCALE, vy * self.SCALE) for vx, vy in verts_2d]
                
                # Extrude into Z-depths
                V_back = [(vx, vy, z_min * self.SCALE) for vx, vy in scaled_verts]
                V_front = [(vx, vy, z_max * self.SCALE) for vx, vy in scaled_verts]
                
                polys_3d.append((V_back, c1))
                polys_3d.append((V_front, c1))
                
                # Build connecting walls to form 3D blocks
                num_v = len(V_back)
                for i in range(num_v):
                    idx_curr = i
                    idx_next = (i + 1) % num_v
                    edge_poly = [
                        V_back[idx_curr], V_back[idx_next],
                        V_front[idx_next], V_front[idx_curr]
                    ]
                    c = c2 if i % 2 == 0 else c1
                    polys_3d.append((edge_poly, c))

        # --- For Landing FLAGS ---
        if hasattr(base_env, 'helipad_x1') and hasattr(base_env, 'helipad_y'):
            flag_y1 = base_env.helipad_y
            flag_y2 = flag_y1 + 50 / self.SCALE
            
            # Check if lander is resting safely
            lander_landed = not base_env.lander.awake
            
            # Dynamic Cloth Colors: Green if landed, Red if flying/crashing
            cloth_c1 = (50, 200, 50) if lander_landed else (230, 51, 0)
            cloth_c2 = (30, 150, 30) if lander_landed else (180, 40, 0)

            for x in [base_env.helipad_x1, base_env.helipad_x2]:
                pole_w = 1.0 / self.SCALE
                pole_verts = [(x, flag_y1), (x + pole_w, flag_y1), (x + pole_w, flag_y2), (x, flag_y2)]
                cloth_verts = [(x, flag_y2), (x, flag_y2 - 10 / self.SCALE), (x + 25 / self.SCALE, flag_y2 - 5 / self.SCALE)]
                
                z_min, z_max = -0.5, 0.5 
                
                for verts, is_cloth in [(pole_verts, False), (cloth_verts, True)]:

                    c1, c2 = (cloth_c1, cloth_c2) if is_cloth else ((200, 200, 200), (150, 150, 150))
                    scaled_verts = [(vx * self.SCALE, vy * self.SCALE) for vx, vy in verts]
                    V_back = [(vx, vy, z_min * self.SCALE) for vx, vy in scaled_verts]
                    V_front = [(vx, vy, z_max * self.SCALE) for vx, vy in scaled_verts]
                    
                    polys_3d.append((V_back, c1))
                    polys_3d.append((V_front, c1))
                    
                    num_v = len(V_back)
                    for i in range(num_v):
                        idx_curr = i
                        idx_next = (i + 1) % num_v
                        edge_poly = [V_back[idx_curr], V_back[idx_next], V_front[idx_next], V_front[idx_curr]]
                        c = c2 if i % 2 == 0 else c1
                        polys_3d.append((edge_poly, c))
                    
        # 3. Apply Multi-View Projections
        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []
        
        cx, cy = self.VIEWPORT_W / 2, self.VIEWPORT_H / 2
        
        # Center the cameras on the lander
        cam_x = lander_x * self.SCALE
        cam_y = lander_y * self.SCALE + (self.camera_y_offset * self.SCALE)
        cam_z = self.camera_z_offset * self.SCALE
        
        for p_vertices, color in polys_3d:
            pts_t, pts_r, pts_s, pts_f = [], [], [], []
            depths_t, depths_r, depths_s, depths_f = [], [], [], []
            valid_fpv = True
            
            for vx, vy, vz in p_vertices:
                # A. Plan (Top-Down): Tracking X, projecting Z
                rx = vx - cam_x
                pts_t.append((rx + cx, vz + cy))
                depths_t.append(-vy) # Height is depth from top
                
                # B. Orthographic Rear: Tracking X, projecting Y
                pts_r.append((rx + cx, self.VIEWPORT_H - vy))
                depths_r.append(-vz) # Looking from +Z to -Z
                
                # C. Orthographic Side: Projecting Z and Y
                pts_s.append((vz + cx, self.VIEWPORT_H - vy))
                depths_s.append(rx) # Lateral distance is depth from side
                
                # D. True 3D Perspective (FPV)
                dx = vx - cam_x
                dy = vy - cam_y
                depth_cam = cam_z - vz # Camera sits at +Z looking forward

                d_c, dx_c, dy_c = apply_camera_pose(depth_cam, dx, dy, camera_pose=camera_pose)
                
                if d_c < 1.0:
                    valid_fpv = False
                else:
                    px = (dx_c / d_c) * self.fov + cx
                    py = self.VIEWPORT_H - ((dy_c / d_c) * self.fov + cy)
                    pts_f.append((px, py))
                    depths_f.append(d_c)
                    
            if len(pts_t) >= 3: proj_top.append((sum(depths_t)/len(depths_t), pts_t, color))
            if len(pts_r) >= 3: proj_rear.append((sum(depths_r)/len(depths_r), pts_r, color))
            if len(pts_s) >= 3: proj_side.append((sum(depths_s)/len(depths_s), pts_s, color))
            if valid_fpv and len(pts_f) >= 3: proj_fpv.append((sum(depths_f)/len(depths_f), pts_f, color))
            
        # 4. Painter's Algorithm Sorting (Draw furthest first)
        proj_top.sort(key=lambda x: -x[0])
        proj_rear.sort(key=lambda x: -x[0])
        proj_side.sort(key=lambda x: -x[0])
        proj_fpv.sort(key=lambda x: -x[0])
        
        # 5. Render Polygons to Surfaces
        for surf, p_list in [(top_surf, proj_top), (rear_surf, proj_rear), 
                             (side_surf, proj_side), (fpv_surf, proj_fpv)]:
            for _, pts, color in p_list:
                if len(pts) < 3:
                    continue
                safe_color = (int(color[0]), int(color[1]), int(color[2]))
                ipts = [(int(round(p[0])), int(round(p[1]))) for p in pts]
                try:
                    gfxdraw.aapolygon(surf, ipts, safe_color)
                    gfxdraw.filled_polygon(surf, ipts, safe_color)
                except Exception:
                    pass
                
        # 6. Convert to Numpy Arrays
        out_imgs = []
        for surf in [top_surf, rear_surf, side_surf, fpv_surf]:
            img = pygame.surfarray.array3d(surf)
            img = np.transpose(img, (1, 0, 2))
            out_imgs.append(img)
            
        return out_imgs[0], out_imgs[1], out_imgs[2], out_imgs[3]