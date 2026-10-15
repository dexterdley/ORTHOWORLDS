try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import pygame
from pygame import gfxdraw
import math
import heapq
from collections import deque
from .camera_utils import orbit_camera_frame, orbit_camera_offset
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
from Box2D.b2 import (world, polygonShape)
try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
    from scipy.ndimage import distance_transform_edt
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

FPS      = 50
SCALE    = 20.0     # pixels per metre
ARENA_W  = 30.0     # metres
ARENA_H  = 22.0     # metres
WALL_H   = 3.0      # visual 3D height of maze walls (metres)
CAR_Z_LO = 0.0      # bottom of car extrusion (metres)
CAR_Z_HI = 0.9      # top  of car extrusion  (metres)
EXIT_R   = 1.8      # reaching within this radius of the exit = escaped
EXIT_INSET = 1.4    # exit archway X: this far West of the East wall's inner face

# --- Random maze (regenerated every reset) ---
MAZE_COLS  = 6        # coarse cell grid (cells ~4.7 x 5.1 m -> corridors ~3.9-4.7 m wide)
MAZE_ROWS  = 4
OUTER_T    = 0.8      # outer boundary wall thickness (metres)
WALL_T     = 0.8      # internal wall thickness (metres)
MAZE_LOOPS = (1, 3)   # extra internal walls knocked out after the spanning tree (adds loops)

# --- Car model (top-down kinematic-ish car on Box2D) ---
ENGINE_ACCEL  = 10.0    # m/s^2 at full throttle (top speed = ENGINE_ACCEL / LIN_DAMP)
REVERSE_SCALE = 0.6     # reverse / braking force fraction
LIN_DAMP      = 1.2     # rolling + air drag (Box2D linear damping)
LAT_GRIP      = 0.9     # fraction of sideways slip cancelled per step (tyre grip)
WHEELBASE     = 1.2     # metres
MAX_STEER     = math.radians(35.0)
MAX_YAW_RATE  = 3.0     # rad/s
V_MAX         = ENGINE_ACCEL / LIN_DAMP

# --- Toxic gas front (a vertical sheet sweeping East along +X) ---
GAS_SPEED     = 0.035   # metres per step (1.75 m/s) once released
GAS_X0        = -2.0    # front X at reset (behind the West wall)
GAS_DELAY     = 60      # base head start (steps) on top of the per-maze route allowance
GAS_REF_SPEED = 5.0     # m/s: average route speed the per-maze head start is sized for
GAS_DELAY_MAX = 400     # cap on the release delay (also the obs countdown normaliser);
                        # generated mazes need <= ~350 (mean ~60)
GAS_DANGER    = 3.0     # metres: proximity penalty zone ahead of the front

# --- Geodesic distance field over the maze ---
GEO_RES       = 0.25    # grid cell size (metres)
GEO_INFLATE   = 0.4     # wall inflation ~ car half-width (metres)

# ==============================================================================
# 1. TOXIC GAS ESCAPE BASE ENVIRONMENT
# ==============================================================================
class ToxicGasEscapeEnv(gym.Env):
    """
    Toxic Gas Escape:
    Drive a car through a maze to the exit before the toxic gas cloud, sweeping in
    from the West (left, -X) after a delay, catches you.
    Every reset builds a NEW RANDOM MAZE from the episode RNG (same seed -> same maze):
    a recursive-backtracker spanning tree on a MAZE_COLS x MAZE_ROWS cell grid (corridors
    ~4 m wide), plus a few knocked-out walls for loops. The car starts in a random cell of
    the West column, the exit is in a random cell of the East column; the maze is always
    solvable. The gas head start is sized per maze from the start->exit route (so a car
    averaging ~GAS_REF_SPEED along the route stays ahead of the front, with GAS_DELAY spare).

    Actions (2D, Box[-1, 1]):
      0: Throttle  [-1..1]  +1 = full throttle forward, -1 = brake / reverse
      1: Steering  [-1..1]  +1 = steer left (counter-clockwise), -1 = steer right;
                            the turn rate scales with forward speed (no turning when stopped,
                            reversed when driving backwards, like a real car)

    Observation (13D, all in [-1, 1]):
      0-1 : Car position (X, Y), normalised to the arena
      2-3 : Car velocity in the car frame (forward, leftward) / V_MAX
      4-5 : Heading sin, cos
      6   : Yaw rate / MAX_YAW_RATE
      7-8 : Unit direction of the shortest maze path to the exit, in the car frame (forward, leftward)
      9   : Shortest maze-path (geodesic) distance to the exit / longest path in this episode's maze
      10  : Gap to the gas front (car X - gas X) / 10 m, clipped (<= 0 means caught)
      11  : Gas release countdown (steps until it starts moving / GAS_DELAY_MAX)
      12  : Episode time fraction (step / max_steps)

    Reward:
      + 2.0 * decrease of geodesic (maze-path) distance to the exit   (potential-based)
      - 0.01 per step
      - 0.05 * (3 - gap) per step while within 3 m of the gas front
      + 100 on escaping (within 1.8 m of the exit, terminated)
      - 100 when the gas front reaches the car (terminated)
    The episode is truncated at max_steps (2000 steps = 40 s); by then the gas front has
    normally swept the whole arena, so idling always ends in being caught.

    Per-episode maze data: wall_defs (list of (cx, cy, half_w, half_h) boxes), exit_pos,
    route (cell-centre polyline from the start cell to the exit), geo_field / geo_max.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, max_steps=2000, route_start_prob=0.25):
        self.render_mode = render_mode
        self.screen      = None
        self.clock       = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)

        self.world    = world(gravity=(0, 0), doSleep=True)
        self.bodies   = []
        self.walls    = []
        self.player   = None
        self.gas_speed = GAS_SPEED   # metres per step at gas front (once released)
        self.max_steps = max_steps
        # Probability of spawning part-way along the route (start-state variety / exploration)
        self.route_start_prob = route_start_prob

        self.step_count = 0
        self.gas_x0     = GAS_X0
        self.gas_delay  = GAS_DELAY
        self.gas_x      = GAS_X0

        # Per-episode maze (built in reset)
        self.wall_defs = []
        self.exit_pos  = (ARENA_W - OUTER_T - EXIT_INSET, ARENA_H / 2)
        self.route     = []
        self.geo_field, self.geo_max = None, 1.0

    @property
    def WALL_DEFS(self):
        """Backward-compatible alias for this episode's wall boxes."""
        return self.wall_defs

    # ------------------------------------------------------------------
    @staticmethod
    def _cell_bounds(n, length):
        """Interior [lo, hi] spans of n maze cells across `length` (outer walls OUTER_T thick,
        internal walls WALL_T thick centred on the cell-grid lines)."""
        lines = OUTER_T + np.arange(n + 1) * (length - 2 * OUTER_T) / n
        lo = [lines[k] + (WALL_T / 2 if k > 0 else 0.0) for k in range(n)]
        hi = [lines[k + 1] - (WALL_T / 2 if k < n - 1 else 0.0) for k in range(n)]
        return lines, lo, hi

    def _generate_maze(self, rng):
        """Random perfect maze (iterative recursive backtracker) on a MAZE_COLS x MAZE_ROWS grid,
        then MAZE_LOOPS extra internal walls removed. Sets wall_defs, exit_pos and route."""
        C, R = MAZE_COLS, MAZE_ROWS
        v_open = np.zeros((C - 1, R), dtype=bool)   # passage between cells (i, j) and (i+1, j)
        h_open = np.zeros((C, R - 1), dtype=bool)   # passage between cells (i, j) and (i, j+1)

        def link(a, b, val=None):
            (i0, j0), (i1, j1) = sorted((a, b))
            arr, k = (v_open, (i0, j0)) if j0 == j1 else (h_open, (i0, j0))
            if val is not None:
                arr[k] = val
            return arr[k]

        def nbrs(i, j):
            return [(a, b) for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)) if 0 <= a < C and 0 <= b < R]

        seen = np.zeros((C, R), dtype=bool)
        cur = (int(rng.integers(C)), int(rng.integers(R)))
        seen[cur] = True
        stack = [cur]
        while stack:
            cand = [n for n in nbrs(*stack[-1]) if not seen[n]]
            if not cand:
                stack.pop()
                continue
            nxt = cand[int(rng.integers(len(cand)))]
            link(stack[-1], nxt, True)
            seen[nxt] = True
            stack.append(nxt)

        # Loops: knock out a few remaining internal walls
        closed = [((i, j), (i + 1, j)) for i, j in zip(*np.nonzero(~v_open))] + \
                 [((i, j), (i, j + 1)) for i, j in zip(*np.nonzero(~h_open))]
        n_loops = min(int(rng.integers(MAZE_LOOPS[0], MAZE_LOOPS[1] + 1)), len(closed))
        for k in rng.choice(len(closed), size=n_loops, replace=False):
            link(*closed[int(k)], True)

        # Start cell on the West edge, exit cell on the East edge
        start = (0, int(rng.integers(R)))
        goal  = (C - 1, int(rng.integers(R)))

        # Wall boxes: outer boundary + internal walls merged into maximal straight runs
        xl, xlo, xhi = self._cell_bounds(C, ARENA_W)
        yl, ylo, yhi = self._cell_bounds(R, ARENA_H)
        t2, o2 = WALL_T / 2, OUTER_T / 2
        walls = [(ARENA_W / 2, o2, ARENA_W / 2, o2), (ARENA_W / 2, ARENA_H - o2, ARENA_W / 2, o2),
                 (o2, ARENA_H / 2, o2, ARENA_H / 2), (ARENA_W - o2, ARENA_H / 2, o2, ARENA_H / 2)]

        def runs(closed_flags):
            k, n = 0, len(closed_flags)
            while k < n:
                if closed_flags[k]:
                    k0 = k
                    while k < n and closed_flags[k]:
                        k += 1
                    yield k0, k            # cells k0 .. k-1
                else:
                    k += 1

        for i in range(1, C):              # vertical walls on x = xl[i]
            for j0, j1 in runs(~v_open[i - 1]):
                y0, y1 = max(yl[j0] - t2, OUTER_T), min(yl[j1] + t2, ARENA_H - OUTER_T)
                walls.append((float(xl[i]), (y0 + y1) / 2, t2, (y1 - y0) / 2))
        for j in range(1, R):              # horizontal walls on y = yl[j]
            for i0, i1 in runs(~h_open[:, j - 1]):
                x0, x1 = max(xl[i0] - t2, OUTER_T), min(xl[i1] + t2, ARENA_W - OUTER_T)
                walls.append(((x0 + x1) / 2, float(yl[j]), (x1 - x0) / 2, t2))
        self.wall_defs = walls

        def centre(c):
            return (0.5 * (xlo[c[0]] + xhi[c[0]]), 0.5 * (ylo[c[1]] + yhi[c[1]]))

        self.exit_pos = (ARENA_W - OUTER_T - EXIT_INSET, centre(goal)[1])

        # Route: BFS over open passages from the start cell to the exit cell (cell-centre polyline)
        prev = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                break
            for n in nbrs(*c):
                if n not in prev and link(c, n):
                    prev[n] = c
                    q.append(n)
        cells, c = [], goal
        while c is not None:
            cells.append(c)
            c = prev[c]
        pts = [centre(c) for c in reversed(cells)] + [self.exit_pos]
        # drop collinear interior points (keep the corners)
        route = [pts[0]]
        for k in range(1, len(pts) - 1):
            a, b, d = route[-1], pts[k], pts[k + 1]
            if abs((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])) > 1e-6:
                route.append(b)
        route.append(pts[-1])
        self.route = route

    # ------------------------------------------------------------------
    def _geodesic_field(self):
        """Shortest maze-path distance to exit_pos on a GEO_RES grid (8-connected Dijkstra, no
        corner cutting), walls inflated by GEO_INFLATE. Recomputed for every maze (scipy csgraph,
        ~10 ms; pure-Python heap fallback). Blocked cells get the distance of the nearest free
        cell plus the straight hop, so lookups stay smooth right next to walls.
        Returns (dist_field, d_max) with d_max the largest reachable distance."""
        nx, ny = int(round(ARENA_W / GEO_RES)), int(round(ARENA_H / GEO_RES))
        xs = (np.arange(nx) + 0.5) * GEO_RES
        ys = (np.arange(ny) + 0.5) * GEO_RES
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        free = np.ones((nx, ny), dtype=bool)
        for cx, cy, hw, hh in self.wall_defs:
            free &= ~((np.abs(X - cx) <= hw + GEO_INFLATE) & (np.abs(Y - cy) <= hh + GEO_INFLATE))
        ei, ej = int(self.exit_pos[0] / GEO_RES), int(self.exit_pos[1] / GEO_RES)
        free[ei, ej] = True

        if _HAVE_SCIPY:
            idx = np.arange(nx * ny).reshape(nx, ny)
            src, dst, wts = [], [], []
            for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
                j0, j1 = max(0, -dj), ny - max(0, dj)
                a = (slice(0, nx - di), slice(j0, j1))
                b = (slice(di, nx), slice(j0 + dj, j1 + dj))
                ok = free[a] & free[b]
                if di and dj:   # no corner cutting through blocked cells on diagonals
                    ok &= free[di:nx, j0:j1] & free[0:nx - di, j0 + dj:j1 + dj]
                src.append(idx[a][ok]); dst.append(idx[b][ok])
                wts.append(np.full(int(ok.sum()), GEO_RES * math.hypot(di, dj)))
            g = coo_matrix((np.concatenate(wts), (np.concatenate(src), np.concatenate(dst))),
                           shape=(nx * ny, nx * ny)).tocsr()
            dist = _sp_dijkstra(g, directed=False, indices=ei * ny + ej).reshape(nx, ny)
            reach = np.isfinite(dist)
            d_max = float(dist[reach].max())
            hop, (ni, nj) = distance_transform_edt(~reach, return_indices=True)
            dist = np.where(reach, dist, dist[ni, nj] + hop * GEO_RES)
            return dist, d_max

        nbrs = [(di, dj, GEO_RES * math.hypot(di, dj))
                for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        dist = np.full((nx, ny), np.inf)

        def dijkstra(heap, passable, corner_check=True):
            while heap:
                d, i, j = heapq.heappop(heap)
                if d > dist[i, j]:
                    continue
                for di, dj, c in nbrs:
                    a, b = i + di, j + dj
                    if 0 <= a < nx and 0 <= b < ny and passable[a, b]:
                        if corner_check and di and dj and not (passable[i + di, j] and passable[i, j + dj]):
                            continue
                        if d + c < dist[a, b]:
                            dist[a, b] = d + c
                            heapq.heappush(heap, (d + c, a, b))

        dist[ei, ej] = 0.0
        dijkstra([(0.0, ei, ej)], free)
        d_max = float(dist[np.isfinite(dist)].max())
        # Fill blocked (and unreachable) cells outward from the reachable region, which keeps its values
        reach = np.isfinite(dist)
        heap = [(dist[i, j], i, j) for i, j in zip(*np.nonzero(reach))]
        heapq.heapify(heap)
        dijkstra(heap, ~reach, corner_check=False)
        return dist, d_max

    def geodesic_dist(self, x, y):
        """Bilinearly interpolated maze-path distance (metres) from (x, y) to the exit."""
        f = self.geo_field
        fx = min(max(x / GEO_RES - 0.5, 0.0), f.shape[0] - 1.001)
        fy = min(max(y / GEO_RES - 0.5, 0.0), f.shape[1] - 1.001)
        i, j = int(fx), int(fy)
        tx, ty = fx - i, fy - j
        return float((f[i, j] * (1 - tx) + f[i + 1, j] * tx) * (1 - ty) +
                     (f[i, j + 1] * (1 - tx) + f[i + 1, j + 1] * tx) * ty)

    def path_direction(self, x, y):
        """Unit world-frame direction of steepest descent of the geodesic distance (the way to go)."""
        h = 0.3
        gx = self.geodesic_dist(x + h, y) - self.geodesic_dist(x - h, y)
        gy = self.geodesic_dist(x, y + h) - self.geodesic_dist(x, y - h)
        n = math.hypot(gx, gy)
        if n < 1e-6:
            dx, dy = self.exit_pos[0] - x, self.exit_pos[1] - y
            n = math.hypot(dx, dy) + 1e-6
            return dx / n, dy / n
        return -gx / n, -gy / n

    # ------------------------------------------------------------------
    def _route_length(self):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.route, self.route[1:]))

    def _sample_route_pose(self, frac):
        """Point (x, y, heading) at fraction `frac` of the route polyline length."""
        R = self.route
        segs = [(R[i], R[i + 1]) for i in range(len(R) - 1)]
        lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs]
        s = frac * sum(lens)
        for (a, b), L in zip(segs, lens):
            if s <= L:
                t = s / L
                return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), math.atan2(b[1] - a[1], b[0] - a[0])
            s -= L
        a, b = segs[-1]
        return b[0], b[1], math.atan2(b[1] - a[1], b[0] - a[0])

    def _route_gas_allowance(self, s0):
        """Release delay (steps) such that a car leaving route arc-length s0 at reset and averaging
        GAS_REF_SPEED along the rest of the route never has the gas front reach its X."""
        need, s_acc = 0.0, 0.0
        for a, b in zip(self.route, self.route[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            for t in np.linspace(0.0, 1.0, max(2, int(L / 0.5) + 1)):
                s = s_acc + t * L
                if s >= s0:
                    x = a[0] + t * (b[0] - a[0])
                    need = max(need, (s - s0) / GAS_REF_SPEED * FPS - (x - self.gas_x0) / self.gas_speed)
            s_acc += L
        return need

    def _gas_front(self, step):
        return self.gas_x0 + self.gas_speed * max(0, step - self.gas_delay)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}

        for b in self.bodies:
            self.world.DestroyBody(b)
        self.bodies = []
        self.walls  = []

        self.step_count = 0

        # Fresh random maze + its geodesic distance field
        self._generate_maze(rng)
        self.geo_field, self.geo_max = self._geodesic_field()
        for cx, cy, hw, hh in self.wall_defs:
            b = self.world.CreateStaticBody(
                position=(cx, cy), shapes=polygonShape(box=(hw, hh)))
            self.bodies.append(b)
            self.walls.append(b)

        # Start pose: the West-edge start cell facing along the route (jittered), or
        # occasionally part-way along the route
        route_start = options.get('route_start', rng.random() < self.route_start_prob)
        if route_start:
            frac = float(rng.uniform(0.15, 0.8))
            sx, sy, sa = self._sample_route_pose(frac)
            sx += float(rng.uniform(-0.5, 0.5)); sy += float(rng.uniform(-0.5, 0.5))
            sa += float(rng.uniform(-0.3, 0.3))
            s0 = frac * self._route_length()
        else:
            (x0, y0), (x1, y1) = self.route[0], self.route[1]
            sx = x0 + float(rng.uniform(-0.3, 0.3))
            sy = y0 + float(rng.uniform(-0.5, 0.5))
            sa = math.atan2(y1 - y0, x1 - x0) + float(rng.uniform(-0.25, 0.25))
            s0 = 0.0

        # Gas front: starts behind the West wall (and >= 8 m behind a route spawn); released after
        # a head start sized for this maze's remaining route (+ GAS_DELAY spare, jittered)
        self.gas_x0    = min(GAS_X0 + float(rng.uniform(-0.5, 0.5)), sx - 8.0)
        allowance      = self._route_gas_allowance(s0)
        self.gas_delay = int(np.clip(allowance + GAS_DELAY + rng.integers(-20, 21), 0, GAS_DELAY_MAX))
        self.gas_x     = self._gas_front(0)

        # Player car
        self.player = self.world.CreateDynamicBody(
            position=(sx, sy), angle=sa,
            linearDamping=LIN_DAMP, angularDamping=3.0)
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

        p = self.player.position
        self.prev_dist_exit = self.geodesic_dist(p.x, p.y)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {'gas_x': self.gas_x, 'dist_exit': self.prev_dist_exit,
                                 'route_len': self._route_length(), 'gas_delay': self.gas_delay}

    # ------------------------------------------------------------------
    def _apply_car_controls(self, throttle, steer):
        """Top-down car: tyre grip cancels sideways slip, engine pushes along the heading,
        and the yaw rate follows a bicycle model (proportional to forward speed)."""
        body  = self.player
        angle = float(body.angle)
        fwd   = (math.cos(angle), math.sin(angle))
        left  = (-math.sin(angle), math.cos(angle))
        v     = body.linearVelocity
        v_fwd = v.x * fwd[0] + v.y * fwd[1]
        v_lat = v.x * left[0] + v.y * left[1]
        m     = float(body.mass)

        # Lateral friction: remove most of the sideways velocity (no hovercraft drift)
        imp = -LAT_GRIP * m * v_lat
        body.ApplyLinearImpulse((imp * left[0], imp * left[1]), body.worldCenter, True)

        # Engine / brake along the heading
        drive = throttle * ENGINE_ACCEL * m * (1.0 if throttle >= 0.0 else REVERSE_SCALE)
        body.ApplyForceToCenter((drive * fwd[0], drive * fwd[1]), True)

        # Steering: no turning in place; turn rate grows with forward speed
        yaw_rate = v_fwd * math.tan(steer * MAX_STEER) / WHEELBASE
        body.angularVelocity = float(np.clip(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))

    # ------------------------------------------------------------------
    def step(self, action):
        if self.player is None:
            raise RuntimeError("ToxicGasEscapeEnv: call reset() before step().")
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        self.step_count += 1

        # --- Car dynamics ---
        self._apply_car_controls(float(action[0]), float(action[1]))

        # --- Advance toxic gas ---
        self.gas_x = self._gas_front(self.step_count)

        self.world.Step(1.0 / FPS, 6, 2)
        obs = self._get_obs()

        p         = self.player.position
        dist_exit = self.geodesic_dist(p.x, p.y)          # maze-path distance
        dist_gas  = float(p.x - self.gas_x)              # positive = ahead of gas
        escaped   = math.hypot(p.x - self.exit_pos[0], p.y - self.exit_pos[1]) < EXIT_R

        # Reward: potential-based progress along the maze path, time cost, gas proximity
        reward = 2.0 * (self.prev_dist_exit - dist_exit) - 0.01
        self.prev_dist_exit = dist_exit
        if dist_gas < GAS_DANGER:
            reward -= 0.05 * (GAS_DANGER - dist_gas)

        terminated, truncated = False, False
        if escaped:
            reward += 100.0
            terminated = True
        elif dist_gas <= 0.0:
            reward -= 100.0
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()
        info = {'gas_x': self.gas_x, 'dist_exit': dist_exit,
                'escaped': bool(escaped), 'caught': bool(dist_gas <= 0.0 and not escaped)}
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        p     = self.player.position
        angle = float(self.player.angle)
        ca, sa = math.cos(angle), math.sin(angle)
        v     = self.player.linearVelocity
        v_fwd = v.x * ca + v.y * sa
        v_lat = -v.x * sa + v.y * ca
        dx, dy = self.path_direction(p.x, p.y)
        obs = np.array([
            p.x / ARENA_W * 2.0 - 1.0, p.y / ARENA_H * 2.0 - 1.0,
            v_fwd / V_MAX, v_lat / V_MAX,
            sa, ca,
            self.player.angularVelocity / MAX_YAW_RATE,
            dx * ca + dy * sa, -dx * sa + dy * ca,
            self.geodesic_dist(p.x, p.y) / self.geo_max,
            (p.x - self.gas_x) / 10.0,
            max(0, self.gas_delay - self.step_count) / GAS_DELAY_MAX,
            self.step_count / self.max_steps,
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------
    def render(self):
        """2D pygame debug render (top-down view)."""
        if self.render_mode is None or self.player is None:
            return
        sw, sh = int(ARENA_W * SCALE), int(ARENA_H * SCALE)
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((sw, sh))
            else:
                self.screen = pygame.Surface((sw, sh))
            self.clock = pygame.time.Clock()
        if self.render_mode == "human":
            pygame.event.pump()

        self.screen.fill((45, 50, 45))  # dark road

        # Gas overlay
        gx = int(self.gas_x * SCALE)
        if gx > 0:
            gs = pygame.Surface((min(gx, sw), sh), pygame.SRCALPHA)
            gs.fill((30, 220, 80, 110))
            self.screen.blit(gs, (0, 0))

        # Exit
        ex = int(self.exit_pos[0] * SCALE); ey = int(sh - self.exit_pos[1] * SCALE)
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

    def close(self):
        if self.screen is not None and self.render_mode == "human":
            pygame.display.quit()
        self.screen = None


# ==============================================================================
# 2. 3D ORTHOGRAPHIC MULTI-VIEW WRAPPER
# ==============================================================================
class ToxicGasEscapeOrthographicWrapper(gym.Wrapper):
    """
    Extrudes the 2D top-down maze into a 3D volumetric scene. The maze is regenerated on every
    reset, so all geometry (walls, exit archway, gas-front gaps, spring-arm wall hits) is
    rebuilt each frame from the base env's per-episode wall_defs / exit_pos.

    Coordinate convention:
      physics X   → 3D X  (East-West along maze)
      physics Y   → 3D Y  (North-South along maze)
      extrusion Z → 3D Z  (height above floor, 0 = ground)

    Views:
      Top-Down : X-Y plan view  (bird's eye maze map)
      Rear     : X-Z elevation  (looking North, +Y; East to the right)
      Side     : Y-Z elevation  (looking West, -X; North to the right)
                 Rear and Side are zoomed section views centred on the car: geometry on the
                 viewer's side of the car (South of it / East of it) is cut away so maze walls
                 never hide the car, which is always drawn on top. The gas front is shown as a
                 translucent green volume (Rear: left of its X; Side: a sheet behind the car).
      FPV      : third-person chase camera camera_dist behind / camera_height above the car,
                 tilted down to keep the car centred; orbits the car with cam_yaw / cam_pitch.
                 A spring arm pulls the camera in when a maze wall is between it and the car.

    Maintains persistent camera xyz position, roll, yaw, and pitch state.
    """

    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.SCALE      = SCALE
        self.VIEWPORT_W = 800
        self.VIEWPORT_H = 600
        self.fov        = 420.0
        # Rear / Side section views: extra zoom over SCALE (7 -> 140 px/m, ~5.7 m across)
        # and the ground line height as a fraction of the viewport
        self.ortho_zoom   = 7.0
        self.ortho_ground = 0.78

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0 # Added roll state

        # Chase camera baseline offsets (scaled pixels): ~4 car lengths behind, ~1.6 m above the roof
        self.camera_dist   = 6.5 * SCALE
        self.camera_height = (CAR_Z_HI + 1.6) * SCALE   # absolute height above the floor
        self.target_height = 0.5 * CAR_Z_HI * SCALE     # aim point: car body centre
        self.min_arm       = 1.8 * SCALE                # spring-arm minimum length

        # Orbit pivot = the car (refreshed with the spring arm each render)
        self.pivot_depth = self.camera_dist

        # Camera world position (refreshed each render from cam_pos)
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    def _arm(self):
        """Spring-arm camera geometry for the current car pose and orbit angles.
        Returns (horizontal arm length, tilt angle, pivot depth) in scaled pixels / radians:
        the arm is shortened when a maze wall lies between the car and the (orbited) camera."""
        base = self.env.unwrapped
        S = self.SCALE
        p_angle = float(base.player.angle)
        px, py = base.player.position.x, base.player.position.y
        # Horizontal direction from the car to the orbited camera (behind, swung by cam_yaw)
        cy_, sy_ = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        ux = -cy_ * math.cos(p_angle) + sy_ * math.sin(p_angle)
        uy = -cy_ * math.sin(p_angle) - sy_ * math.cos(p_angle)
        reach = self.camera_dist / S * max(math.cos(self.cam_pitch), 0.0)
        hit = reach
        for cx, cy, hw, hh in base.wall_defs:
            # Slab test of the ray (px,py)+t*(ux,uy) against the wall AABB (padded 0.3 m)
            t0, t1 = 0.0, hit
            for o, u, lo, hi in ((px, ux, cx - hw - 0.3, cx + hw + 0.3), (py, uy, cy - hh - 0.3, cy + hh + 0.3)):
                if abs(u) < 1e-9:
                    if o < lo or o > hi:
                        t0, t1 = 1.0, 0.0
                        break
                else:
                    ta, tb = (lo - o) / u, (hi - o) / u
                    t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 <= t1 and t0 > 0.0:
                hit = min(hit, t0)
        scale = 1.0 if reach <= 0.0 else hit / reach
        arm = max(self.camera_dist * scale, self.min_arm)
        dh = self.camera_height - self.target_height
        return arm, math.atan2(dh, arm), math.hypot(arm, dh)

    @property
    def cam_pos(self):
        """Dynamically computes the exact X, Y, Z coordinates of the orbited camera in world space."""
        base = self.env.unwrapped
        if getattr(base, 'player', None) is None:
            return (0.0, 0.0, 0.0)

        ax, ay, az = self._cam_anchor
        _, tilt, pivot = self._arm()
        p_angle = float(base.player.angle)
        # Orbit offset in the un-orbited camera frame (depth along heading, lateral to the right, vertical up);
        # the baseline chase tilt is part of the orbit pitch, so cam_yaw always swings about the vertical
        od, ol, ov = orbit_camera_offset(self.cam_yaw, max(tilt + self.cam_pitch, 0.05), pivot)
        cam_x = ax + od * math.cos(p_angle) + ol * math.sin(p_angle)
        cam_y = ay + od * math.sin(p_angle) - ol * math.cos(p_angle)
        cam_z = az + ov

        return (cam_x, cam_y, cam_z)

    @property
    def _cam_anchor(self):
        """Un-orbited chase camera anchor used for projection: level with the aim point (car body centre),
        pivot_depth behind it. The orbit pitch (baseline tilt + cam_pitch) lifts it to camera_height."""
        base = self.env.unwrapped
        if getattr(base, 'player', None) is None:
            return (0.0, 0.0, 0.0)

        S = self.SCALE
        p_angle = float(base.player.angle)
        p_x = base.player.position.x * S
        p_y = base.player.position.y * S
        _, _, pivot = self._arm()

        cam_x  = p_x - pivot * math.cos(p_angle)
        cam_y  = p_y - pivot * math.sin(p_angle)
        cam_z  = self.target_height

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

    # ------------------------------------------------------------------
    def _surf_to_img(self, surf):
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_polys(proj_list, surf, key=None):
        proj_list.sort(key=key or (lambda x: (x[3], -x[0])))
        for _, pts, color, _ in proj_list:
            ipts = [(int(round(px)), int(round(py))) for px, py in pts]
            if len(ipts) >= 3:
                try:
                    gfxdraw.aapolygon(surf, ipts, color)
                    gfxdraw.filled_polygon(surf, ipts, color)
                except Exception:
                    pass

    @staticmethod
    def _area2d(pts):
        """Absolute area of a projected 2D polygon (edge-on faces project to ~0)."""
        n = len(pts)
        return 0.5 * abs(sum(pts[i][0] * pts[(i+1) % n][1] - pts[(i+1) % n][0] * pts[i][1]
                             for i in range(n)))

    # ------------------------------------------------------------------
    def render(self, camera_pose="Center"):
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)

        base = self.env.unwrapped

        VW, VH = self.VIEWPORT_W, self.VIEWPORT_H
        cx, cy = VW / 2.0, VH / 2.0
        S = self.SCALE

        if getattr(base, 'player', None) is None:
            return [np.zeros((VH, VW, 3), np.uint8) for _ in range(4)]

        top_surf  = pygame.Surface((VW, VH))
        rear_surf = pygame.Surface((VW, VH))
        side_surf = pygame.Surface((VW, VH))
        fpv_surf  = pygame.Surface((VW, VH))

        ROAD_BG  = (38, 44, 38)
        SKY_COL  = (100, 140, 100)

        for s in [top_surf, rear_surf, side_surf]:
            s.fill(ROAD_BG)

        # Chase camera geometry (spring arm + downward tilt toward the car)
        arm, tilt, pivot = self._arm()
        self.pivot_depth = pivot
        orbit_pitch = max(tilt + self.cam_pitch, 0.05)   # keep the camera above the floor

        # FPV: sky above the horizon, dark ground below (horizon moves with the total camera pitch)
        horizon = int(round(cy - math.tan(min(orbit_pitch, 1.4)) * self.fov))
        fpv_surf.fill(SKY_COL)
        if horizon < VH:
            pygame.draw.rect(fpv_surf, (30, 35, 30), (0, max(horizon, 0), VW, VH - max(horizon, 0)))
        if 0 <= horizon < VH:
            pygame.draw.line(fpv_surf, (80, 110, 80), (0, horizon), (VW, horizon), 2)

        # Each item: (vertices, color, layer, views) with views 'all' | 'ortho' | 'fpv';
        # an optional 5th element True enables back-face culling in the FPV.
        polys_3d = []

        WS = WALL_H * S   # wall height in scaled pixels
        gx_m = base.gas_x  # gas front X (metres)

        # ----------------------------------------------------------------
        # A. Asphalt Road Floor Grid  (X-Y plane at Z=0); tiles behind the gas front are tinted
        # ----------------------------------------------------------------
        tile = 2.0
        for xi, x0 in enumerate(np.arange(0.0, ARENA_W, tile)):
            x1 = min(x0 + tile, ARENA_W)
            for yi, y0 in enumerate(np.arange(0.0, ARENA_H, tile)):
                y1 = min(y0 + tile, ARENA_H)
                c = (50, 56, 50) if (xi + yi) % 2 == 0 else (42, 48, 42)
                cg = (40, 120, 60) if (xi + yi) % 2 == 0 else (34, 108, 54)
                for (a0, a1, col) in ((x0, min(x1, gx_m), cg), (max(x0, gx_m), x1, c)):
                    if a1 - a0 > 1e-3:
                        polys_3d.append((
                            [(a0*S, y0*S, 0), (a1*S, y0*S, 0),
                             (a1*S, y1*S, 0), (a0*S, y1*S, 0)],
                            col, 0, 'all'))

        # ----------------------------------------------------------------
        # B. Maze Walls  (this episode's random maze, base.wall_defs, rebuilt every frame;
        #    extruded from Z=0 to Z=WS); faces split into <= 2 m chunks so the FPV
        #    painter's sort stays correct along long corridors
        # ----------------------------------------------------------------
        for wcx, wcy, hw, hh in base.wall_defs:
            x0, x1, y0, y1 = wcx - hw, wcx + hw, wcy - hh, wcy + hh
            nxs = max(1, int(math.ceil((x1 - x0) / 2.0)))
            nys = max(1, int(math.ceil((y1 - y0) / 2.0)))
            xs = np.linspace(x0, x1, nxs + 1)
            ys = np.linspace(y0, y1, nys + 1)
            # Top (CCW seen from above)
            for i in range(nxs):
                for j in range(nys):
                    polys_3d.append(([(xs[i]*S, ys[j]*S, WS), (xs[i+1]*S, ys[j]*S, WS),
                                      (xs[i+1]*S, ys[j+1]*S, WS), (xs[i]*S, ys[j+1]*S, WS)],
                                     (140, 152, 168), 1, 'all', True))
            # Sides (outward-facing winding)
            for i in range(nxs):
                a, b = xs[i]*S, xs[i+1]*S
                polys_3d.append(([(a, y0*S, 0), (b, y0*S, 0), (b, y0*S, WS), (a, y0*S, WS)],
                                 (120, 132, 148), 1, 'all', True))   # South face
                polys_3d.append(([(b, y1*S, 0), (a, y1*S, 0), (a, y1*S, WS), (b, y1*S, WS)],
                                 (120, 132, 148), 1, 'all', True))   # North face
            for j in range(nys):
                a, b = ys[j]*S, ys[j+1]*S
                polys_3d.append(([(x1*S, a, 0), (x1*S, b, 0), (x1*S, b, WS), (x1*S, a, WS)],
                                 (105, 116, 132), 1, 'all', True))   # East face
                polys_3d.append(([(x0*S, b, 0), (x0*S, a, 0), (x0*S, a, WS), (x0*S, b, WS)],
                                 (105, 116, 132), 1, 'all', True))   # West face

        # ----------------------------------------------------------------
        # C. Toxic Gas  (X from 0 to gas_x, full Y, Z 0 to WS)
        # ----------------------------------------------------------------
        gx = gx_m * S
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
                polys_3d.append((slab, gc, 2, 'ortho'))
                slab_top = [
                    (0,  0,  z1), (gx, 0,  z1),
                    (gx, AH, z1), (0,  AH, z1)
                ]
                polys_3d.append((slab_top, gc, 2, 'ortho'))

            # FPV: translucent vertical gas front, only across open space (between walls), in <= 2 m pieces
            if gx_m < ARENA_W:
                blocked = sorted((wcy - hh, wcy + hh) for wcx, wcy, hw, hh in base.wall_defs
                                 if abs(gx_m - wcx) <= hw)
                y_cur, spans = 0.0, []
                for b0, b1 in blocked:
                    if b0 > y_cur:
                        spans.append((y_cur, b0))
                    y_cur = max(y_cur, b1)
                if y_cur < ARENA_H:
                    spans.append((y_cur, ARENA_H))
                for s0, s1 in spans:
                    n = max(1, int(math.ceil((s1 - s0) / 2.0)))
                    for k in range(n):
                        a = (s0 + (s1 - s0) * k / n) * S
                        b = (s0 + (s1 - s0) * (k + 1) / n) * S
                        polys_3d.append(([(gx, a, 0), (gx, b, 0), (gx, b, WS), (gx, a, WS)],
                                         (25, 215, 80, 170), 2, 'fpv'))

        # ----------------------------------------------------------------
        # D. Exit Archway Portal  (at this episode's exit on the East edge)
        # ----------------------------------------------------------------
        ex, ey = base.exit_pos[0] * S, base.exit_pos[1] * S
        arch_h = WS * 0.85
        # Floor pad marking the exit zone
        pad = [(ex + 1.6*S*math.cos(a), ey + 1.6*S*math.sin(a), 0.02*S)
               for a in np.linspace(0, 2*math.pi, 12, endpoint=False)]
        polys_3d.append((pad, (200, 170, 20), 0.5, 'all'))
        # Left pillar
        polys_3d.append(([
            (ex-1.2*S, ey-0.15*S, 0), (ex-0.8*S, ey-0.15*S, 0),
            (ex-0.8*S, ey-0.15*S, arch_h), (ex-1.2*S, ey-0.15*S, arch_h)
        ], (255, 215, 0), 3, 'all'))
        # Right pillar
        polys_3d.append(([
            (ex-1.2*S, ey+0.15*S, 0), (ex-0.8*S, ey+0.15*S, 0),
            (ex-0.8*S, ey+0.15*S, arch_h), (ex-1.2*S, ey+0.15*S, arch_h)
        ], (255, 215, 0), 3, 'all'))
        # Top beam
        polys_3d.append(([
            (ex-1.2*S, ey-0.3*S, arch_h*0.85), (ex-0.8*S, ey-0.3*S, arch_h*0.85),
            (ex-0.8*S, ey+0.3*S, arch_h), (ex-1.2*S, ey+0.3*S, arch_h)
        ], (255, 240, 80), 3, 'all'))
        # Glow core
        nc = 8
        core = [(ex-1.0*S + 0.25*S*math.cos(a), ey + 0.25*S*math.sin(a), arch_h*0.5)
                for a in np.linspace(0, 2*math.pi, nc, endpoint=False)]
        polys_3d.append((core, (255, 255, 200), 4, 'all'))

        # ----------------------------------------------------------------
        # E. Car Geometry  (chassis, cabin, wheels)
        # ----------------------------------------------------------------
        tf = base.player.transform
        p_angle = float(base.player.angle)
        car_first = len(polys_3d)   # polys from here on belong to the car (always on top in Rear / Side)

        fix0 = base.player.fixtures[0]   # chassis
        fix1 = base.player.fixtures[1]   # cabin
        wheel_fixes = base.player.fixtures[2:]

        def extrude(fix, z0, z1, c_top, c_side):
            v2d = [(tf * v) for v in fix.shape.vertices]
            vlo = [(v[0]*S, v[1]*S, z0*S) for v in v2d]
            vhi = [(v[0]*S, v[1]*S, z1*S) for v in v2d]
            n = len(vlo)
            polys_3d.append((vlo, c_top, 2, 'ortho'))
            polys_3d.append((vhi, c_top, 2, 'all', True))
            for i in range(n):
                sc = c_side[i % len(c_side)]
                polys_3d.append(([vlo[i], vlo[(i+1)%n], vhi[(i+1)%n], vhi[i]], sc, 2, 'all', True))

        # Chassis
        extrude(fix0, CAR_Z_LO, CAR_Z_HI,
                (255, 120, 20),
                [(200, 80, 10), (220, 100, 15), (200, 80, 10), (220, 100, 15)])
        # Cabin
        extrude(fix1, CAR_Z_HI * 0.55, CAR_Z_HI * 1.05,
                (35, 55, 85),
                [(25, 40, 65)] * 4)
        # Wheels (dark rubber)
        for idx, wf in enumerate(wheel_fixes):
            z_lo = CAR_Z_LO
            z_hi = CAR_Z_LO + 0.18
            extrude(wf, z_lo, z_hi, (25, 25, 30), [(18, 18, 22)] * 4)

        # Headlights (on the front face, rotated with the car)
        hl_poly = []
        for lx, ly, lz in ((0.86, -0.3, 0.3), (0.86, 0.3, 0.3), (0.86, 0.3, 0.55), (0.86, -0.3, 0.55)):
            w = tf * (lx, ly)
            hl_poly.append((w[0]*S, w[1]*S, CAR_Z_HI*lz*S))
        polys_3d.append((hl_poly, (255, 255, 150), 3, 'all'))

        # ----------------------------------------------------------------
        # F. Multi-View Projections
        # ----------------------------------------------------------------
        # Player position in scaled coords
        p_x = base.player.position.x * S
        p_y = base.player.position.y * S

        # Get dynamically calculated position from our property
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        # Projection uses the un-orbited anchor; tilt, then orbit_camera_frame applies the orbit
        anc_x, anc_y, anc_z = self._cam_anchor

        cos_h = math.cos(p_angle)
        sin_h = math.sin(p_angle)
        near = 0.5 * S   # 0.5 m on the orbited depth (keeps projected coords within gfxdraw's 16-bit range)

        proj_top, proj_rear, proj_side, proj_fpv = [], [], [], []

        # Rear / Side section views: zoomed, centred on the car, ground line at ORTHO_GY
        OZ = self.ortho_zoom
        ORTHO_GY = VH * self.ortho_ground

        for idx, poly_item in enumerate(polys_3d):
            p_verts, color, layer, views = poly_item[:4]
            cull = len(poly_item) > 4 and poly_item[4]
            is_car = idx >= car_first

            if views != 'fpv':
                pts_t, pts_r, pts_s = [], [], []
                dt, dr, ds = [], [], []

                for vx, vy, vz in p_verts:
                    sx = vx - p_x   # scrolled X (centred on car)
                    sy = vy - p_y   # scrolled Y

                    # Top-Down: X-Y maze map
                    pts_t.append((sx + cx, cy - sy))
                    dt.append(vz)

                    # Rear (looking North, +Y): shows X left-right, Z up; depth = Y beyond the car
                    pts_r.append((sx * OZ + cx, ORTHO_GY - vz * OZ))
                    dr.append(sy)

                    # Side (looking West, -X): shows Y left-right, Z up; depth = X behind the car
                    pts_s.append((sy * OZ + cx, ORTHO_GY - vz * OZ))
                    ds.append(-sx)

                oc = color[:3]
                if len(pts_t) >= 3: proj_top.append( (sum(dt)/len(dt),  pts_t, oc, layer))
                # Section cut through the car: drop geometry entirely on the viewer's side of the car
                # (it would hide the car) and edge-on faces (floor, roofs, gas slabs project to lines).
                # Layer slot: 1 = car (drawn last), 0 = scenery (painter-sorted far to near).
                # Scenery fades toward the background with depth so near and far walls read apart.
                for pts, dd, plist in ((pts_r, dr, proj_rear), (pts_s, ds, proj_side)):
                    if len(pts) >= 3 and (is_car or max(dd) >= 0.0) and self._area2d(pts) > 1.0:
                        d_avg = sum(dd) / len(dd)
                        f = 0.0 if is_car else min(max(d_avg / S, 0.0) / 14.0, 0.55)
                        fc = tuple(int(c + (b - c) * f) for c, b in zip(oc, ROAD_BG))
                        plist.append((d_avg, pts, fc, int(is_car)))

            if views == 'ortho':
                continue

            # FPV — perspective projection from the chase camera.
            # Camera-relative (depth along heading, lateral right, vertical up), orbited around the car
            # (pitch = baseline tilt + cam_pitch); near-plane clipping runs on the orbited depth.
            rel_verts = []
            for vx, vy, vz in p_verts:
                dx, dy, dz = vx - anc_x, vy - anc_y, vz - anc_z
                d0   =  dx * cos_h + dy * sin_h
                lat0 =  dx * sin_h - dy * cos_h
                rel_verts.append(orbit_camera_frame(d0, lat0, dz, self.cam_yaw, orbit_pitch, pivot))

            if cull:
                # Back-face culling in a right-handed (depth, left, up) frame, camera at the origin
                nx_ = ny_ = nz_ = 0.0
                m = len(rel_verts)
                for i in range(m):
                    a0, a1, a2 = rel_verts[i][0], -rel_verts[i][1], rel_verts[i][2]
                    b0, b1, b2 = rel_verts[(i+1) % m][0], -rel_verts[(i+1) % m][1], rel_verts[(i+1) % m][2]
                    nx_ += (a1 - b1) * (a2 + b2)
                    ny_ += (a2 - b2) * (a0 + b0)
                    nz_ += (a0 - b0) * (a1 + b1)
                c0 = sum(v[0] for v in rel_verts) / m
                c1 = -sum(v[1] for v in rel_verts) / m
                c2 = sum(v[2] for v in rel_verts) / m
                if nx_ * c0 + ny_ * c1 + nz_ * c2 >= 0.0:
                    continue

            clipped = []
            for i in range(len(rel_verts)):
                pc, pp = rel_verts[i], rel_verts[i - 1]
                if pc[0] >= near and pp[0] >= near:
                    clipped.append(pc)
                elif (pc[0] >= near) != (pp[0] >= near):
                    t = (near - pp[0]) / (pc[0] - pp[0] + 1e-9)
                    clipped.append((near,
                                    pp[1] + t*(pc[1]-pp[1]),
                                    pp[2] + t*(pc[2]-pp[2])))
                    if pc[0] >= near:
                        clipped.append(pc)

            pts_fpv, df = [], []
            for d_c, lat_c, vert_c in clipped:
                pts_fpv.append((cx + (lat_c / d_c) * self.fov, cy - (vert_c / d_c) * self.fov))
                df.append(d_c)

            if len(pts_fpv) >= 3:
                proj_fpv.append((sum(df)/len(df), pts_fpv, color, layer))

        # ----------------------------------------------------------------
        # G. Draw all views
        # ----------------------------------------------------------------
        self._draw_polys(proj_top,  top_surf)

        # Rear / Side: checkered ground band below the ground line (2 m tiles, scrolls with the car)
        gy_i = int(round(ORTHO_GY))
        t_px = tile * S * OZ
        for surf, c_m in ((rear_surf, base.player.position.x), (side_surf, base.player.position.y)):
            pygame.draw.rect(surf, (30, 35, 30), (0, gy_i, VW, VH - gy_i))
            k0 = int(math.floor((c_m - (cx / (S * OZ))) / tile))
            for k in range(k0, k0 + int(VW / t_px) + 2):
                if k % 2 == 0:
                    x_l = (k * tile - c_m) * S * OZ + cx
                    pygame.draw.rect(surf, (44, 50, 44), (int(x_l), gy_i, int(t_px) + 1, VH - gy_i))
            pygame.draw.line(surf, (90, 100, 90), (0, gy_i), (VW, gy_i), 2)

        # Gas front: Rear = translucent volume left of the front (drawn over scenery, under the car);
        # Side = translucent sheet behind the car at the front's depth, denser as it closes in
        gap = float(base.player.position.x - gx_m)
        g_top = ORTHO_GY - WS * OZ
        if gx > 0:
            gxs = min((gx - p_x) * OZ + cx, VW + 10.0)
            if gxs > 0:
                proj_rear.append((-1e9, [(-10.0, g_top), (gxs, g_top), (gxs, VH + 10.0), (-10.0, VH + 10.0)],
                                  (25, 215, 80, 150), 0))
            g_alpha = int(70 + 110 * min(max(1.0 - gap / 10.0, 0.0), 1.0))
            proj_side.append((p_x - gx, [(-10.0, g_top), (VW + 10.0, g_top), (VW + 10.0, ORTHO_GY), (-10.0, ORTHO_GY)],
                              (25, 215, 80, g_alpha), 0))

        self._draw_polys(proj_rear, rear_surf)
        self._draw_polys(proj_side, side_surf)
        # FPV: floor (tiles, then decals) first, then everything else far-to-near
        self._draw_polys(proj_fpv,  fpv_surf,
                         key=lambda x: (0, x[3], -x[0]) if x[3] < 1 else (1, 0, -x[0]))

        # FPV HUD: speed bar
        spd = float(np.linalg.norm([base.player.linearVelocity.x, base.player.linearVelocity.y]))
        bar_w = int(min(spd / V_MAX, 1.0) * 200)
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
