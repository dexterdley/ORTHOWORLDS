import gym
import numpy as np
import pygame
from pygame import gfxdraw
from .camera_utils import apply_camera_pose

class BipedalOrthographicWrapper(gym.Wrapper):
    """
    Wraps BipedalWalker to extrude 2D Box2D shapes into 3D volumes,
    rendering Top-Down, Orthographic Rear, Orthographic Side, and True 3D FPV.
    """
    def __init__(self, env):
        super().__init__(env)
        self.SCALE = 30.0
        self.VIEWPORT_W = 600
        self.VIEWPORT_H = 400
        self.fov = 400.0
        
    def render(self, camera_pose="Center"):
        base_env = self.env.unwrapped
        
        if base_env.screen is None and base_env.render_mode == "human":
            return base_env.render()
            
        top_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        front_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        side_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        fpv_surf = pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H))
        
        sky_color = (215, 215, 255)
        for surf in [top_surf, front_surf, side_surf, fpv_surf]:
            surf.fill(sky_color)

        if base_env.hull is None:
            dummy = np.zeros((self.VIEWPORT_H, self.VIEWPORT_W, 3), dtype=np.uint8)
            return dummy, dummy, dummy, dummy

        scroll_x = base_env.scroll * self.SCALE
        polys_3d = []
        
        # --- 1. Extrude Terrain ---
        TERRAIN_Z_MIN, TERRAIN_Z_MAX = -10.0, 10.0
        for poly, color in base_env.terrain_poly:
            V_back = [(p[0] * self.SCALE, p[1] * self.SCALE, TERRAIN_Z_MIN * self.SCALE) for p in poly]
            V_front = [(p[0] * self.SCALE, p[1] * self.SCALE, TERRAIN_Z_MAX * self.SCALE) for p in poly]
            
            polys_3d.append((V_front, color, 0))
            polys_3d.append((V_back, color, 0))
            
            num_v = len(V_back)
            for i in range(num_v):
                idx_curr, idx_next = i, (i + 1) % num_v
                connect_poly = [V_back[idx_curr], V_back[idx_next], V_front[idx_next], V_front[idx_curr]]
                polys_3d.append((connect_poly, color, 0))

        # --- 2. Extrude Missing Start Flag ---
        TERRAIN_STEP = 14.0 / self.SCALE
        flag_x = 3 * TERRAIN_STEP
        flag_y1 = (self.VIEWPORT_H / self.SCALE) / 4 
        flag_y2 = flag_y1 + (50.0 / self.SCALE)
        
        if base_env.game_over:
            cloth_c1, cloth_c2 = (200, 30, 30), (150, 20, 20) 
        elif base_env.hull.position.x > flag_x:
            cloth_c1, cloth_c2 = (50, 220, 50), (30, 180, 30) 
        else:
            cloth_c1, cloth_c2 = (230, 51, 0), (180, 40, 0)   
            
        pole_w = 1.0 / self.SCALE
        pole_verts = [(flag_x, flag_y1), (flag_x + pole_w, flag_y1), (flag_x + pole_w, flag_y2), (flag_x, flag_y2)]
        cloth_verts = [(flag_x, flag_y2), (flag_x, flag_y2 - 10 / self.SCALE), (flag_x + 25 / self.SCALE, flag_y2 - 5 / self.SCALE)]
                       
        for verts, is_cloth in [(pole_verts, False), (cloth_verts, True)]:
            c1, c2 = (cloth_c1, cloth_c2) if is_cloth else ((40, 40, 40), (20, 20, 20))
            V_back = [(vx * self.SCALE, vy * self.SCALE, -0.5 * self.SCALE) for vx, vy in verts]
            V_front = [(vx * self.SCALE, vy * self.SCALE, 0.5 * self.SCALE) for vx, vy in verts]
            
            polys_3d.append((V_back, c1, 1))
            polys_3d.append((V_front, c1, 1))
            num_v = len(V_back)
            for i in range(num_v):
                idx_curr, idx_next = i, (i + 1) % num_v
                edge_poly = [V_back[idx_curr], V_back[idx_next], V_front[idx_next], V_front[idx_curr]]
                polys_3d.append((edge_poly, c2 if i % 2 == 0 else c1, 1))

        # --- 3. Extrude Robot Parts ---
        parts_z = {
            base_env.hull: (-1.0, 1.0),                  
            base_env.legs[0]: (1.1, 1.5),                
            base_env.legs[1]: (1.1, 1.5),                
            base_env.legs[2]: (-1.5, -1.1),              
            base_env.legs[3]: (-1.5, -1.1)               
        }

        for obj in base_env.drawlist:
            if obj in parts_z:
                z_min, z_max = parts_z[obj]
                for f in obj.fixtures:
                    if hasattr(f.shape, 'vertices'):
                        verts_2d = [(f.body.transform * v * self.SCALE) for v in f.shape.vertices]
                        V_back = [(vx, vy, z_min * self.SCALE) for vx, vy in verts_2d]
                        V_front = [(vx, vy, z_max * self.SCALE) for vx, vy in verts_2d]
                        
                        polys_3d.append((V_back, obj.color1, 2))
                        polys_3d.append((V_front, obj.color1, 2))
                        
                        num_v = len(V_back)
                        for i in range(num_v):
                            idx_curr, idx_next = i, (i + 1) % num_v
                            edge_poly = [V_back[idx_curr], V_back[idx_next], V_front[idx_next], V_front[idx_curr]]
                            polys_3d.append((edge_poly, obj.color2 if i % 2 == 0 else obj.color1, 2))

        # --- 4. Apply Projections ---
        proj_top, proj_front, proj_side, proj_fpv = [], [], [], []
        cy, cx = self.VIEWPORT_H / 2, self.VIEWPORT_W / 2
        
        # FPV Camera setup (Behind the walker, looking forward down the X-axis)
        hull_pos = base_env.hull.position
        cam_x = hull_pos.x * self.SCALE - 250.0  
        cam_y = hull_pos.y * self.SCALE + 50.0
        cam_z = 0.0
        
        for p_vertices, color, layer in polys_3d:
            pts_t, pts_f, pts_s, pts_fpv = [], [], [], []
            depths_t, depths_f, depths_s, depths_fpv = [], [], [], []
            valid_fpv = True
            
            for vx, vy, vz in p_vertices:
                vx_scrolled = vx - scroll_x
                
                # A. Top-Down
                pts_t.append((vx_scrolled, vz + cy))
                depths_t.append(-vy) 
                
                # B. Front Elevation
                pts_f.append((vz + cx, self.VIEWPORT_H - vy))
                depths_f.append(vx_scrolled) 
                
                # C. Side Elevation (Native)
                pts_s.append((vx_scrolled, self.VIEWPORT_H - vy))
                depths_s.append(vz) 
                
                # D. True 3D Perspective (FPV)
                dx = vx - cam_x
                dy = vy - cam_y
                dz = vz - cam_z

                dx_c, dz_c, dy_c = apply_camera_pose(dx, dz, dy, camera_pose=camera_pose)
                
                if dx_c < 10.0:  # Camera clipping plane
                    valid_fpv = False
                else:
                    px = (dz_c / dx_c) * self.fov + cx
                    py = self.VIEWPORT_H - ((dy_c / dx_c) * self.fov + cy)
                    pts_fpv.append((px, py))
                    depths_fpv.append(dx_c)
                
            if len(pts_t) >= 3: proj_top.append((sum(depths_t)/len(depths_t), pts_t, color, layer))
            if len(pts_f) >= 3: proj_front.append((sum(depths_f)/len(depths_f), pts_f, color, layer))
            if len(pts_s) >= 3: proj_side.append((sum(depths_s)/len(depths_s), pts_s, color, layer))
            if valid_fpv and len(pts_fpv) >= 3: proj_fpv.append((sum(depths_fpv)/len(depths_fpv), pts_fpv, color, layer))

        # --- 5. Z-Index Sorting & Draw ---
        for proj_list, surf in [(proj_top, top_surf), (proj_front, front_surf), 
                                (proj_side, side_surf), (proj_fpv, fpv_surf)]:
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

        out_imgs = []
        for surf in [top_surf, front_surf, side_surf, fpv_surf]:
            img = pygame.surfarray.array3d(surf)
            out_imgs.append(np.transpose(img, (1, 0, 2)))
            
        return out_imgs[0], out_imgs[1], out_imgs[2], out_imgs[3]