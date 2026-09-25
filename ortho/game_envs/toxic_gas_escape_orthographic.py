try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
import math
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
from Box2D.b2 import (world, polygonShape)

FPS      = 50
SCALE    = 20.0     # pixels per metre
ARENA_W  = 30.0     # metres
ARENA_H  = 22.0     # metres
WALL_H   = 3.0      # visual 3D height of maze walls (metres)
CAR_Z_LO = 0.0      # bottom of car extrusion (metres)
CAR_Z_HI = 0.9      # top  of car extrusion  (metres)
EXIT_POS = (28.0, 19.0)   # exit archway centre

# ==============================================================================
# 1. TOXIC GAS ESCAPE BASE ENVIRONMENT
# ==============================================================================
class ToxicGasEscapeEnv(gym.Env):
    """
    Toxic Gas Escape:
    Drive a car through a maze corridor to the exit before the toxic gas cloud
    advances from the left and catches you.

    Actions (2D):
      0: Drive force   [-1..1]  (forward/reverse along heading)
      1: Steer torque  [-1..1]  (yaw left/right)

    Observation (10D):
      0-1: Player (X, Y)
      2-3: Player velocity (Vx, Vy)
      4-5: Player angle, angular velocity
      6  : Gas front X position
      7  : Gas expansion speed
      8  : Distance to exit portal
      9  : Distance to gas front (positive = safe)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    # ------------------------------------------------------------------
    # Maze wall specs: (centre_x, centre_y, half_w, half_h)
    WALL_DEFS = [
        # Outer boundary
        (ARENA_W/2,       0.4,      ARENA_W/2, 0.4),   # South
        (ARENA_W/2, ARENA_H-0.4,   ARENA_W/2, 0.4),   # North
        (0.4,       ARENA_H/2,     0.4,  ARENA_H/2),   # West
        (ARENA_W-0.4, ARENA_H/2,   0.4,  ARENA_H/2),   # East
        # --- Internal maze walls ---
        # Row-1 divider: east portion at Y=7, gap at X=17..22
        (9.5,  7.0, 9.0, 0.5),   # W1a
        (26.5, 7.0, 4.5, 0.5),   # W1b
        # Row-2 divider: west portion at Y=14, gap at X=9..14
        (5.0,  14.0, 4.5, 0.5),  # W2a
        (21.5, 14.0, 8.0, 0.5),  # W2b
        # Vertical connector at X=9, from Y=7 to Y=14
        (9.0,  10.5, 0.5, 3.5),  # Vc1
        # Vertical connector at X=22, from Y=7 to Y=14
        (22.0, 10.5, 0.5, 3.5),  # Vc2
        # Dead-end nub top-left to force early right turn
        (3.5,  17.5, 3.0, 0.5),  # Dnub
    ]

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.screen      = None
        self.clock       = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.world    = world(gravity=(0, 0), doSleep=True)
        self.bodies   = []
        self.walls    = []
        self.gas_speed = 0.04   # metres per step at gas front
        self.max_steps = 800

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for b in self.bodies:
            self.world.DestroyBody(b)
        self.bodies = []
        self.walls  = []

        self.step_count = 0
        self.gas_x      = 0.8   # gas front X (metres)

        # Build maze walls
        for cx, cy, hw, hh in self.WALL_DEFS:
            b = self.world.CreateStaticBody(
                position=(cx, cy), shapes=polygonShape(box=(hw, hh)))
            self.bodies.append(b)
            self.walls.append(b)

        # Player car — starts bottom-left corridor, facing east
        self.player = self.world.CreateDynamicBody(
            position=(2.0, 3.5), angle=0.0,
            linearDamping=1.5, angularDamping=3.0)
        # Chassis
        self.player.CreatePolygonFixture(box=(0.85, 0.45), density=2.0, friction=0.6)
        # Cabin (rear-set)
        self.player.CreatePolygonFixture(box=(0.35, 0.30, (-0.15, 0), 0), density=0.4, friction=0.1)
        # 4 Wheels
        self.player.CreatePolygonFixture(box=(0.22, 0.09, ( 0.60,  0.42), 0), density=0.15, friction=0.9)
        self.player.CreatePolygonFixture(box=(0.22, 0.09, ( 0.60, -0.42), 0), density=0.15, friction=0.9)
        self.player.CreatePolygonFixture(box=(0.22, 0.09, (-0.60,  0.42), 0), density=0.15, friction=0.9)
        self.player.CreatePolygonFixture(box=(0.22, 0.09, (-0.60, -0.42), 0), density=0.15, friction=0.9)
        self.bodies.append(self.player)

        self.prev_dist_exit = float(np.linalg.norm(
            np.array(self.player.position) - np.array(EXIT_POS)))
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # --- Car dynamics ---
        angle        = float(self.player.angle)
        fwd          = (math.cos(angle), math.sin(angle))
        drive_force  = float(action[0]) * 160.0
        steer_torque = float(action[1]) * 35.0

        self.player.ApplyForceToCenter(
            (float(fwd[0] * drive_force), float(fwd[1] * drive_force)), wake=True)
        self.player.ApplyTorque(float(steer_torque), wake=True)

        # --- Advance toxic gas ---
        self.gas_x += self.gas_speed

        self.world.Step(1.0 / FPS, 6, 2)
        obs = self._get_obs()

        p    = np.array(self.player.position)
        dist_exit = float(np.linalg.norm(p - np.array(EXIT_POS)))
        dist_gas  = float(p[0] - self.gas_x)   # positive = ahead of gas

        # Reward
        progress = self.prev_dist_exit - dist_exit
        reward   = progress * 15.0 - 0.05        # progress bonus – time cost
        self.prev_dist_exit = dist_exit
        if dist_gas < 3.0:
            reward -= 0.8 * (3.0 - dist_gas)    # penalty for being close to gas

        terminated = False
        if dist_exit < 1.8:
            reward += 200.0
            terminated = True
        elif dist_gas <= 0.0:
            reward -= 80.0
            terminated = True
        elif self.step_count >= self.max_steps:
            terminated = True

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, {'gas_x': self.gas_x, 'dist_exit': dist_exit}

    # ------------------------------------------------------------------
    def _get_obs(self):
        p   = self.player.position
        dex = float(np.linalg.norm(np.array(p) - np.array(EXIT_POS)))
        dga = float(p.x - self.gas_x)
        return np.array([
            p.x, p.y,
            self.player.linearVelocity.x, self.player.linearVelocity.y,
            self.player.angle, self.player.angularVelocity,
            self.gas_x, self.gas_speed,
            dex, dga
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def render(self):
        """2D pygame debug render (top-down view)."""
        if self.render_mode is None:
            return
        sw, sh = int(ARENA_W * SCALE), int(ARENA_H * SCALE)
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((sw, sh))
            else:
                self.screen = pygame.Surface((sw, sh))
            self.clock = pygame.time.Clock()

        self.screen.fill((45, 50, 45))  # dark road

        # Gas overlay
        gx = int(self.gas_x * SCALE)
        if gx > 0:
            gs = pygame.Surface((gx, sh), pygame.SRCALPHA)
            gs.fill((30, 220, 80, 110))
            self.screen.blit(gs, (0, 0))

        # Exit
        ex = int(EXIT_POS[0] * SCALE); ey = int(sh - EXIT_POS[1] * SCALE)
        pygame.draw.circle(self.screen, (255, 215, 0), (ex, ey), int(1.6 * SCALE))
        pygame.draw.circle(self.screen, (255, 255, 200), (ex, ey), int(0.7 * SCALE))

        # Walls
        for b in self.walls:
            for f in b.fixtures:
                verts = [(int((b.transform * v)[0] * SCALE),
                          int(sh - (b.transform * v)[1] * SCALE))
                         for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, (120, 130, 145), verts)

        # Car
        for f in self.player.fixtures:
            verts = [(int((self.player.transform * v)[0] * SCALE),
                      int(sh - (self.player.transform * v)[1] * SCALE))
                     for v in f.shape.vertices]
            pygame.draw.polygon(self.screen, (255, 120, 20), verts)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


# ==============================================================================
# 2. 3D ORTHOGRAPHIC MULTI-VIEW WRAPPER
# ==============================================================================
class ToxicGasEscapeOrthographicWrapper(gym.Wrapper):
    """
    Extrudes the 2D top-down maze into a 3D volumetric scene.

    Coordinate convention:
      physics X   → 3D X  (East-West along maze)
      physics Y   → 3D Y  (North-South along maze)
      extrusion Z → 3D Z  (height above floor, 0 = ground)

    Views:
      Top-Down : X-Y plan view  (bird's eye maze map)
      Rear     : X-Z elevation  (looking from South wall toward North)
      Side     : Y-Z elevation  (looking from West wall toward East)
      FPV      : perspective from car roof looking forward along heading
    """

    def __init__(self, env):
        super().__init__(env)
        self.SCALE      = SCALE
        self.VIEWPORT_W = 800
        self.VIEWPORT_H = 600
        self.fov        = 300.0

    # ------------------------------------------------------------------
    def _surf_to_img(self, surf):
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_polys(proj_list, surf):
        proj_list.sort(key=lambda x: (x[3], -x[0]))
        for _, pts, color, _ in proj_list:
            ipts = [(int(round(px)), int(round(py))) for px, py in pts]
            if len(ipts) >= 3:
                try:
                    gfxdraw.aapolygon(surf, ipts, color)
                    gfxdraw.filled_polygon(surf, ipts, color)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def render(self):
        base = self.env.unwrapped

        VW, VH = self.VIEWPORT_W, self.VIEWPORT_H
        cx, cy = VW / 2.0, VH / 2.0
        S = self.SCALE

        top_surf  = pygame.Surface((VW, VH))
        rear_surf = pygame.Surface((VW, VH))
        side_surf = pygame.Surface((VW, VH))
        fpv_surf  = pygame.Surface((VW, VH))

        ROAD_BG  = (38, 44, 38)
        SKY_COL  = (100, 140, 100)

        for s in [top_surf, rear_surf, side_surf]:
            s.fill(ROAD_BG)

        # FPV: sky top-half, dark ground bottom-half
        fpv_surf.fill(SKY_COL)
        pygame.draw.rect(fpv_surf, (30, 35, 30), (0, VH // 2, VW, VH // 2))
        pygame.draw.line(fpv_surf, (80, 110, 80), (0, VH // 2), (VW, VH // 2), 2)

        if not hasattr(base, 'player'):
            return [np.zeros((VH, VW, 3), np.uint8)] * 4

        polys_3d = []   # (vertices, color, sort_layer [, no_fpv])

        WS = WALL_H * S   # wall height in scaled pixels

        # ----------------------------------------------------------------
        # A. Asphalt Road Floor Grid  (X-Y plane at Z=0)
        # ----------------------------------------------------------------
        tile = 2.0
        for xi, x0 in enumerate(np.arange(0.0, ARENA_W, tile)):
            x1 = min(x0 + tile, ARENA_W)
            for yi, y0 in enumerate(np.arange(0.0, ARENA_H, tile)):
                y1 = min(y0 + tile, ARENA_H)
                c = (50, 56, 50) if (xi + yi) % 2 == 0 else (42, 48, 42)
                polys_3d.append((
                    [(x0*S, y0*S, 0), (x1*S, y0*S, 0),
                     (x1*S, y1*S, 0), (x0*S, y1*S, 0)],
                    c, 0, True))   # no_fpv=True

        # ----------------------------------------------------------------
        # B. Maze Walls  (extruded from Z=0 to Z=WS)
        # ----------------------------------------------------------------
        for w in base.walls:
            for f in w.fixtures:
                v2d = [(w.transform * v) for v in f.shape.vertices]
                vlo = [(v[0]*S, v[1]*S, 0)   for v in v2d]
                vhi = [(v[0]*S, v[1]*S, WS)  for v in v2d]
                n = len(vlo)
                polys_3d.append((vlo, (100, 110, 125), 1, True))
                polys_3d.append((vhi, (140, 152, 168), 1, True))
                for i in range(n):
                    sc = (120, 132, 148) if i % 2 == 0 else (105, 116, 132)
                    polys_3d.append(([vlo[i], vlo[(i+1)%n], vhi[(i+1)%n], vhi[i]], sc, 1, True))

        # ----------------------------------------------------------------
        # C. Toxic Gas Slab  (X from 0 to gas_x, full Y, Z 0 to WS)
        # ----------------------------------------------------------------
        gx = base.gas_x * S
        if gx > 0:
            gas_layers = [
                # (z_lo, z_hi, alpha_colour)
                (0,    WS*0.35, (30, 200, 70)),
                (WS*0.30, WS*0.70, (20, 230, 90)),
                (WS*0.65, WS,      (15, 210, 75)),
            ]
            AH = ARENA_H * S
            for z0, z1, gc in gas_layers:
                slab = [
                    (0,  0,  z0), (gx, 0,  z0),
                    (gx, AH, z0), (0,  AH, z0)
                ]
                polys_3d.append((slab, gc, 2, True))
                slab_top = [
                    (0,  0,  z1), (gx, 0,  z1),
                    (gx, AH, z1), (0,  AH, z1)
                ]
                polys_3d.append((slab_top, gc, 2, True))

        # ----------------------------------------------------------------
        # D. Exit Archway Portal
        # ----------------------------------------------------------------
        ex, ey = EXIT_POS[0] * S, EXIT_POS[1] * S
        arch_h = WS * 0.85
        # Left pillar
        polys_3d.append(([
            (ex-1.2*S, ey-0.15*S, 0), (ex-0.8*S, ey-0.15*S, 0),
            (ex-0.8*S, ey-0.15*S, arch_h), (ex-1.2*S, ey-0.15*S, arch_h)
        ], (255, 215, 0), 3, True))
        # Right pillar
        polys_3d.append(([
            (ex-1.2*S, ey+0.15*S, 0), (ex-0.8*S, ey+0.15*S, 0),
            (ex-0.8*S, ey+0.15*S, arch_h), (ex-1.2*S, ey+0.15*S, arch_h)
        ], (255, 215, 0), 3, True))
        # Top beam
        polys_3d.append(([
            (ex-1.2*S, ey-0.3*S, arch_h*0.85), (ex-0.8*S, ey-0.3*S, arch_h*0.85),
            (ex-0.8*S, ey+0.3*S, arch_h), (ex-1.2*S, ey+0.3*S, arch_h)
        ], (255, 240, 80), 3, True))
        # Glow core
        nc = 8
        core = [(ex-1.0*S + 0.25*S*math.cos(a), ey + 0.25*S*math.sin(a), arch_h*0.5)
                for a in np.linspace(0, 2*math.pi, nc, endpoint=False)]
        polys_3d.append((core, (255, 255, 200), 4, True))

        # ----------------------------------------------------------------
        # E. Car Geometry  (chassis, cabin, wheels)
        # ----------------------------------------------------------------
        tf = base.player.transform
        p_angle = float(base.player.angle)

        fix0 = base.player.fixtures[0]   # chassis
        fix1 = base.player.fixtures[1]   # cabin
        wheel_fixes = base.player.fixtures[2:]

        def extrude(fix, z0, z1, c_top, c_side, is_self=False):
            v2d = [(tf * v) for v in fix.shape.vertices]
            vlo = [(v[0]*S, v[1]*S, z0*S) for v in v2d]
            vhi = [(v[0]*S, v[1]*S, z1*S) for v in v2d]
            n = len(vlo)
            tag = (c_top, 2, False) if not is_self else (c_top, 2, True)
            polys_3d.append((vlo, c_top, 2))
            polys_3d.append((vhi, c_top, 2))
            for i in range(n):
                sc = c_side[i % len(c_side)]
                polys_3d.append(([vlo[i], vlo[(i+1)%n], vhi[(i+1)%n], vhi[i]], sc, 2))

        # Chassis
        extrude(fix0, CAR_Z_LO, CAR_Z_HI,
                (255, 120, 20),
                [(200, 80, 10), (220, 100, 15), (200, 80, 10), (220, 100, 15)])
        # Cabin
        extrude(fix1, CAR_Z_HI * 0.55, CAR_Z_HI * 1.05,
                (35, 55, 85),
                [(25, 40, 65)] * 4)
        # Wheels (dark rubber, slight z-offset per side)
        for idx, wf in enumerate(wheel_fixes):
            z_lo = CAR_Z_LO
            z_hi = CAR_Z_LO + 0.18
            extrude(wf, z_lo, z_hi, (25, 25, 30), [(18, 18, 22)] * 4)

        # Headlights
        hl = (tf * (0.86, 0.0))
        hl_poly = [
            (hl[0]*S,                 hl[1]*S - 0.15*S,  CAR_Z_HI*0.3*S),
            (hl[0]*S + 0.2*S,         hl[1]*S - 0.15*S,  CAR_Z_HI*0.3*S),
            (hl[0]*S + 0.2*S,         hl[1]*S + 0.15*S,  CAR_Z_HI*0.5*S),
            (hl[0]*S,                 hl[1]*S + 0.15*S,  CAR_Z_HI*0.5*S),
        ]
        polys_3d.append((hl_poly, (255, 255, 150), 3))

        # ----------------------------------------------------------------
        # F. Multi-View Projections
        # ----------------------------------------------------------------
        # Player position in scaled coords
        p_x = base.player.position.x * S
        p_y = base.player.position.y * S

        # FPV camera: slightly behind car at roof height, looking along heading
        cam_offset = -1.5 * S
        cam_h      = CAR_Z_HI * S * 1.2
        cam_x  = p_x + cam_offset * math.cos(p_angle)
        cam_y  = p_y + cam_offset * math.sin(p_angle)
        cam_z  = cam_h

        cos_a = math.cos(-p_angle)
        sin_a = math.sin(-p_angle)

        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []

        for poly_item in polys_3d:
            if len(poly_item) == 4:
                p_verts, color, layer, no_fpv = poly_item
            else:
                p_verts, color, layer = poly_item
                no_fpv = False

            pts_t, pts_r, pts_s = [], [], []
            dt, dr, ds = [], [], []

            for vx, vy, vz in p_verts:
                sx = vx - p_x   # scrolled X (centred on car)
                sy = vy - p_y   # scrolled Y

                # Top-Down: X-Y maze map
                pts_t.append((sx + cx, cy - sy))
                dt.append(vz)

                # Rear (looking North, +Y): shows X left-right, Z up
                pts_r.append((sx + cx, VH - vz))
                dr.append(sy)

                # Side (looking East, +X): shows Y left-right, Z up
                pts_s.append((sy + cx, VH - vz))
                ds.append(-sx)

            if len(pts_t) >= 3: proj_top.append( (sum(dt)/len(dt),  pts_t, color, layer))
            if len(pts_r) >= 3: proj_rear.append((sum(dr)/len(dr),  pts_r, color, layer))
            if len(pts_s) >= 3: proj_side.append((sum(ds)/len(ds),  pts_s, color, layer))

            if no_fpv:
                continue

            # FPV — perspective projection
            rel_verts = []
            for vx, vy, vz in p_verts:
                dx, dy = vx - cam_x, vy - cam_y
                dz = vz - cam_z
                # Rotate into camera-forward space
                rfwd  =  dx * cos_a - dy * sin_a   # depth (forward)
                rright = dx * sin_a + dy * cos_a   # lateral
                rup   = -dz                         # up (screen y inverted)
                rel_verts.append((rfwd, rright, rup))

            clipped = []
            near = 1.0
            for i in range(len(rel_verts)):
                pc, pp = rel_verts[i], rel_verts[i - 1]
                if pc[0] >= near and pp[0] >= near:
                    clipped.append(pc)
                elif (pc[0] >= near) != (pp[0] >= near):
                    t = (near - pp[0]) / (pc[0] - pp[0])
                    clipped.append((near,
                                    pp[1] + t*(pc[1]-pp[1]),
                                    pp[2] + t*(pc[2]-pp[2])))
                    if pc[0] >= near:
                        clipped.append(pc)

            pts_fpv, df = [], []
            for rfwd, rright, rup in clipped:
                px_f = -(rright / rfwd) * self.fov + cx
                py_f =  (rup    / rfwd) * self.fov + cy
                pts_fpv.append((px_f, py_f))
                df.append(rfwd)

            if len(pts_fpv) >= 3:
                proj_fpv.append((sum(df)/len(df), pts_fpv, color, layer))

        # ----------------------------------------------------------------
        # G. Draw all views
        # ----------------------------------------------------------------
        self._draw_polys(proj_top,  top_surf)
        self._draw_polys(proj_rear, rear_surf)
        self._draw_polys(proj_side, side_surf)
        self._draw_polys(proj_fpv,  fpv_surf)

        # FPV HUD: speed bar + reticle
        spd = float(np.linalg.norm([base.player.linearVelocity.x, base.player.linearVelocity.y]))
        bar_w = int(min(spd / 8.0, 1.0) * 200)
        pygame.draw.rect(fpv_surf, (255, 100, 20), (10, VH - 22, bar_w, 12))
        pygame.draw.rect(fpv_surf, (200, 200, 200), (10, VH - 22, 200, 12), 1)
        # Gas proximity warning
        dist_gas = float(base.player.position.x - base.gas_x)
        if dist_gas < 5.0:
            warn_alpha = int(200 * max(0, (5.0 - dist_gas) / 5.0))
            warn = pygame.Surface((VW, VH), pygame.SRCALPHA)
            warn.fill((30, 200, 70, warn_alpha // 3))
            fpv_surf.blit(warn, (0, 0))
            pygame.draw.rect(fpv_surf, (30, 220, 80), (0, 0, VW, VH), 4)

        return [self._surf_to_img(s) for s in [top_surf, rear_surf, side_surf, fpv_surf]]
