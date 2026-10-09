import gym
import numpy as np
import pygame
from pygame import gfxdraw
import math 
import cv2
from .camera_utils import apply_camera_pose

class MultiCarOrthographicWrapper(gym.Wrapper):
    """
    Wraps the MultiCarRacing environment to return four views:
    1. Top-Down
    2. Orthographic Rear
    3. Orthographic Side
    4. True 3D Perspective (First-Person)
    """
    def __init__(self, env):
        super().__init__(env)
        self.WINDOW_W = 1000
        self.WINDOW_H = 800
        
        # Camera Parameters
        self.fov = 500.0          # Used for 3D Perspective
        self.scale_factor = 25.0  # Used for Orthographic
        self.camera_h = 10.0      
        self.camera_z = 8.0       
        
    def render(self, camera_pose="Center"):
        top_down_img = self.env.render()
        if top_down_img is None:
            return None, None, None, None

        base_env = self.env.unwrapped
        
        # 1. Setup surfaces for the three custom views
        rear_surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
        side_surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
        fpv_surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
        
        sky_color = (135, 206, 235)
        ground_rect = (0, self.WINDOW_H // 2, self.WINDOW_W, self.WINDOW_H // 2)
        
        for surf in [rear_surf, side_surf, fpv_surf]:
            surf.fill(sky_color)
            pygame.draw.rect(surf, base_env.bg_color, ground_rect)

        if base_env.car is None or not base_env.road_poly:
            return top_down_img, top_down_img, top_down_img, top_down_img

        # Ego car acts as the camera origin
        car_x, car_y = base_env.car.hull.position
        car_angle = base_env.car.hull.angle
        
        rot_angle = -car_angle
        cos_a = math.cos(rot_angle)
        sin_a = math.sin(rot_angle)

        polys = []
        
        # --- LAYER 0: Grass ---
        PLAYFIELD = 2000 / 6.0
        GRASS_DIM = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                p = [
                    (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                    (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                    (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                    (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                ]
                polys.append((p, base_env.grass_color, 0)) 
                
        # --- LAYER 1: Road (Now with 3D Thickness) ---
        ROAD_THICKNESS = 0.25
        for p, c in base_env.road_poly:
            # Top surface of the road
            V_top = [(x, y, 0.0) for x, y in p]
            polys.append((V_top, c, 1)) 
            
            # Side surfaces to give the road depth
            r, g, b = c
            side_c = (int(r * 0.7), int(g * 0.7), int(b * 0.7)) # Darker curb
            
            num_v = len(p)
            for i in range(num_v):
                idx_curr = i
                idx_next = (i + 1) % num_v
                side_poly = [
                    (p[idx_curr][0], p[idx_curr][1], 0.0),
                    (p[idx_next][0], p[idx_next][1], 0.0),
                    (p[idx_next][0], p[idx_next][1], -ROAD_THICKNESS),
                    (p[idx_curr][0], p[idx_curr][1], -ROAD_THICKNESS)
                ]
                polys.append((side_poly, side_c, 1))

        # --- LAYER 2: 3D Cars (Red & Blue) ---
        cars_to_draw = []
        if base_env.car is not None:
            cars_to_draw.append((base_env.car, (200, 40, 40))) 
            
        if hasattr(base_env, 'opponent_car') and base_env.opponent_car is not None:
            cars_to_draw.append((base_env.opponent_car, (20, 50, 200))) 
            
        CAR_HEIGHT = 2.5 
        
        for car_obj, base_color in cars_to_draw:
            r, g, b = base_color
            color_top = (int(r*0.9), int(g*0.9), int(b*0.9))
            color_s1 = (int(r*0.75), int(g*0.75), int(b*0.75))
            color_s2 = (int(r*0.85), int(g*0.85), int(b*0.85))

            for fixture in car_obj.hull.fixtures:
                shape = fixture.shape
                shape_vertices = list(shape.vertices)
                
                V_bottom = []
                V_top = []
                
                for v in shape_vertices:
                    world_v = car_obj.hull.transform * v
                    V_bottom.append((world_v[0], world_v[1], 0.0))
                
                for vb in V_bottom:
                    V_top.append((vb[0], vb[1], CAR_HEIGHT))
                
                polys.append((V_top, color_top, 2)) 
                
                num_v = len(V_bottom)
                for i in range(num_v):
                    idx_curr = i
                    idx_next = (i + 1) % num_v
                    side_poly = [
                        V_bottom[idx_curr], V_bottom[idx_next],
                        V_top[idx_next], V_top[idx_curr]
                    ]
                    color = color_s1 if i % 2 == 0 else color_s2
                    polys.append((side_poly, color, 2))

                polys.append((V_bottom, color_s1, 2)) 
                
            # --- OVERHAULED 3D WHEELS ---
            WHEEL_SCALE = 0.8
            WHEEL_HEIGHT = 1.0 
            tire_c = (40, 40, 40)
            tire_side_c = (25, 25, 25)

            for w in car_obj.wheels:
                wheel_body = w if hasattr(w, 'fixtures') else w.wheel
                for fixture in wheel_body.fixtures:
                    shape = fixture.shape
                    world_verts = [(wheel_body.transform * v) for v in shape.vertices]
                    
                    # Find geometric center to scale outward
                    cx = sum([v[0] for v in world_verts]) / len(world_verts)
                    cy = sum([v[1] for v in world_verts]) / len(world_verts)
                    
                    W_bottom = []
                    W_top = []
                    for vx, vy in world_verts:
                        sx = cx + (vx - cx) * WHEEL_SCALE
                        sy = cy + (vy - cy) * WHEEL_SCALE
                        W_bottom.append((sx, sy, 0.0))
                        W_top.append((sx, sy, WHEEL_HEIGHT))
                        
                    # Top face of the wheel
                    polys.append((W_top, tire_c, 2))
                    
                    # Side walls of the wheel
                    num_v = len(W_bottom)
                    for i in range(num_v):
                        idx_curr = i
                        idx_next = (i + 1) % num_v
                        side_poly = [
                            W_bottom[idx_curr], W_bottom[idx_next],
                            W_top[idx_next], W_top[idx_curr]
                        ]
                        polys.append((side_poly, tire_side_c, 2))

        # 2. Apply Projections (Orthographic & Perspective)
        proj_polys_rear, proj_polys_side, proj_polys_fpv = [], [], []
        
        for p_vertices, color, layer in polys:
            pts_r, pts_s, pts_f = [], [], []
            depths_rear, depths_side, depths_fpv = [], [], []
            valid_cam = True
            valid_fpv = True
            
            for v in p_vertices:
                if len(v) == 3: vx, vy, vh = v
                else: vx, vy, vh = v[0], v[1], 0.0
                    
                dx = vx - car_x
                dy = vy - car_y
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                zh = self.camera_h - vh
                
                depth_cam = ry + self.camera_z
                if depth_cam < 2.5: 
                    valid_cam = False
                
                # A. Orthographic Rear
                px_r = (rx * self.scale_factor) + self.WINDOW_W / 2
                py_r = (zh * self.scale_factor) + self.WINDOW_H / 2
                pts_r.append((px_r, py_r))
                depths_rear.append(depth_cam)
                
                # B. Orthographic Side
                px_s = (ry * self.scale_factor) + self.WINDOW_W / 2
                py_s = (zh * self.scale_factor) + self.WINDOW_H / 2
                pts_s.append((px_s, py_s))
                depths_side.append(-rx) # Lateral axis becomes side view depth
                
                # C. True Perspective 3D (FPV)
                d_c, rx_c, vert_c = apply_camera_pose(depth_cam, rx, -zh, camera_pose=camera_pose)
                zh_c = -vert_c
                if d_c < 2.0:
                    valid_fpv = False
                else:
                    px_f = (rx_c / d_c) * self.fov + self.WINDOW_W / 2
                    py_f = (zh_c / d_c) * self.fov + self.WINDOW_H / 2
                    pts_f.append((px_f, py_f))
                    depths_fpv.append(d_c)
                
            if valid_cam and len(pts_r) >= 3: 
                avg_d = sum(depths_rear) / len(depths_rear)
                proj_polys_rear.append((avg_d, pts_r, color, layer))

            if valid_fpv and len(pts_f) >= 3:
                avg_df = sum(depths_fpv) / len(depths_fpv)
                proj_polys_fpv.append((avg_df, pts_f, color, layer))
                
            if len(pts_s) >= 3:
                avg_ds = sum(depths_side) / len(depths_side)
                proj_polys_side.append((avg_ds, pts_s, color, layer))
                
        # 3. Z-Index Sort for Painter's Algorithm
        proj_polys_rear.sort(key=lambda x: (x[3], -x[0]))
        proj_polys_side.sort(key=lambda x: (x[3], -x[0]))
        proj_polys_fpv.sort(key=lambda x: (x[3], -x[0]))
        
        # 4. Draw to surfaces
        draw_lists = [(rear_surf, proj_polys_rear), 
                      (side_surf, proj_polys_side), 
                      (fpv_surf, proj_polys_fpv)]
                      
        for surf, p_list in draw_lists:
            for _, pts, color, _ in p_list:
                if len(pts) < 3:
                    continue
                safe_color = (int(color[0]), int(color[1]), int(color[2]))
                ipts = [(int(round(p[0])), int(round(p[1]))) for p in pts]
                try:
                    gfxdraw.aapolygon(surf, ipts, safe_color)
                    gfxdraw.filled_polygon(surf, ipts, safe_color)
                except Exception:
                    pass
            
        # 5. Convert and resize to match Gym's standard output
        target_shape = (top_down_img.shape[1], top_down_img.shape[0])
        out_imgs = []
        
        for surf in [rear_surf, side_surf, fpv_surf]:
            img = pygame.surfarray.array3d(surf)
            img = np.transpose(img, (1, 0, 2))
            img_resized = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
            out_imgs.append(img_resized)
            
        return top_down_img, out_imgs[0], out_imgs[1], out_imgs[2]