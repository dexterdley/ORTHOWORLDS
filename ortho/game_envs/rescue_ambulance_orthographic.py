"""
rescue_ambulance_orthographic.py — Rescue Ambulance.

Held-out test env for occlusion / obstacle transfer. The agent drives gymnasium's CarRacing car
(`car_dynamics.Car`, identical physics, painted as an ambulance) through a city grid of roads,
buildings and street obstacles to reach a victim. Same action space as MultiCar Racing
(CarRacing-v3): Box([-1, 0, 0], [1, 1, 1]) = (steer, gas, brake). Holding brake (with no gas)
once the car is stopped engages reverse, so the ambulance can back out of dead ends.

World layout (metres = Box2D units, same scale as CarRacing; the car is ~5 m long, ~2.5 m wide):
  * N_BLOCKS x N_BLOCKS city blocks separated by two-lane roads (ROAD_W = 12 m, LANE_W = 6 m).
    Intersection (i, j), i, j in 0..N_BLOCKS, sits at RescueAmbulanceEnv.intersection(i, j);
    the city is centred on the origin, +X = East, +Y = North.
  * Road tiles are Box2D sensors carrying `road_friction` (1.0) exactly like CarRacing's track tiles,
    sidewalks carry 0.8; anything else (grass) falls back to car_dynamics' 0.6 grass grip.
  * Buildings: solid static boxes filling each block inside its sidewalk ring (1-4 per block,
    4.5-7.5 m tall, concrete / brick colours).
  * Street obstacles (solid static boxes): "roadblock" (striped barrier across one lane),
    "bus" (11 m parked bus, 4 m tall - hides a small box behind it from the chase camera),
    "debris" (small Tank-Duel-cover-like box).
  * Victim: flat red disc at the curb of a road (Box2D sensor) plus a thin striped beacon pole.
"""
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import math
import heapq
import numpy as np
import pygame
from pygame import gfxdraw
from Box2D.b2 import (world as b2World, polygonShape, circleShape, fixtureDef,
                      contactListener, rayCastCallback)
try:
    from gymnasium.envs.box2d.car_dynamics import Car
except ImportError:
    from gym.envs.box2d.car_dynamics import Car
try:
    import cv2
except ImportError:
    cv2 = None
from .camera_utils import orbit_camera_frame, orbit_camera_offset

FPS = 50

# --- City grid ---
N_BLOCKS     = 4          # blocks per side -> (N_BLOCKS + 1)^2 intersections
ROAD_W       = 12.0       # full road width (two lanes)
ROAD_R       = ROAD_W / 2
LANE_W       = 6.0        # ~2 car widths per lane
SIDEWALK_W   = 2.0
BLOCK_W      = 30.0       # road edge to road edge (sidewalk ring included)
PITCH        = BLOCK_W + ROAD_W               # intersection spacing (42 m)
CITY_HALF    = N_BLOCKS / 2 * PITCH + ROAD_R  # outer edge of the ring road (90 m)
PLAYFIELD    = CITY_HALF + 30.0               # leaving |x| or |y| > PLAYFIELD ends the episode (like CarRacing)
ALLEY_W      = 1.0        # gap between buildings that share a block
BUILDING_H   = (4.5, 7.5) # building heights (below the 8 m chase camera, so it never sits inside one)
ROAD_FRICTION     = 1.0
SIDEWALK_FRICTION = 0.8   # grass = car_dynamics default (0.6)

# --- Reverse gear (brake held while stopped, so the action space stays CarRacing's) ---
REVERSE_ENGAGE_SPEED = 0.5   # m/s forward speed below which brake (with no gas) drives backwards
REVERSE_GAS          = 0.3   # rear-wheel gas magnitude while reversing
REVERSE_MAX_SPEED    = 8.0   # m/s backwards at full brake (scales with brake)
REVERSE_HOLD_BRAKE   = 0.05  # light brake applied while above the reverse target speed

# --- Street obstacles: default full sizes (w along the body's local x, h along local y) and heights ---
OBSTACLE_SPECS = {
    "roadblock": {"w": LANE_W - 0.6, "h": 0.8, "height": 1.2},
    "bus":       {"w": 11.0,         "h": 2.8, "height": 4.0},
    "debris":    {"w": 1.5,          "h": 1.5, "height": 1.4},
}
OBSTACLE_KINDS = tuple(OBSTACLE_SPECS)
CAR_PASS_W   = 3.2        # free lateral gap a car needs to get past obstacles on a road segment
CAR_LEN      = 5.0

# --- Victim / episode ---
VICTIM_R     = 1.6        # marker disc radius
ARRIVE_R     = 4.0        # hull centre within this distance of the victim = arrived
CURB_OFFSET  = ROAD_R - 1.5
MAX_STEPS    = 1000

# --- Reward ---
PROGRESS_K        = 1.0    # per metre of road-graph distance gained
STEP_COST         = 0.1
ARRIVE_BONUS      = 100.0
COLLISION_PENALTY = 10.0   # each new contact with a building / obstacle
CONTACT_COST      = 0.2    # per step while touching one
OUT_PENALTY       = 100.0  # leaving the playfield (as CarRacing)

# --- Observation ---
LIDAR_N      = 12
LIDAR_RANGE  = 40.0
V_REF        = 100.0       # m/s normaliser (CarRacing car reaches ~60 m/s after 6 s of full gas)
ANG_REF      = 5.0         # rad/s
DIST_REF     = 2 * CITY_HALF
ROUTE_REF    = 2 * N_BLOCKS * PITCH
OBS_DIM      = 14 + LIDAR_N

# --- Rendering ---
CAR_HEIGHT   = 2.5         # same extrusion as the MultiCar wrapper
WHEEL_SCALE  = 0.8
WHEEL_HEIGHT = 1.0
TOP_ZOOM     = 2.7 * 6.0   # CarRacing ZOOM * SCALE px/m
VIDEO_W, VIDEO_H = 600, 400  # output size (CarRacing rgb_array size, like MultiCar)
SKY_COLOR      = (135, 206, 235)
BG_COLOR       = (102, 204, 102)   # CarRacing bg / grass colours
GRASS_COLOR    = (102, 230, 102)
GRASS_DIM      = 15.0
ROAD_COLOR     = (102, 102, 102)
SIDEWALK_COLOR = (170, 168, 160)
MARK_COLOR     = (235, 235, 235)
AMBULANCE_WHITE = (240, 240, 240)
AMBULANCE_RED   = (215, 30, 30)
TIRE_COLOR      = (40, 40, 40)
VICTIM_COLOR    = (220, 25, 25)
BUILDING_PALETTE = [(150, 150, 145), (176, 172, 164), (126, 130, 138), (192, 186, 170),   # concrete
                    (152, 76, 60), (166, 96, 72), (128, 62, 52), (182, 122, 92)]        # brick
LIGHT_DIR = (0.45, 0.89)  # plan-view light direction used to shade side faces

DIRS = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}


# ==============================================================================
# Geometry helpers
# ==============================================================================
def _box_corners(x, y, angle, w, h):
    """CCW corners of a w x h box centred at (x, y), its local x axis rotated by angle."""
    c, s = math.cos(angle), math.sin(angle)
    return [(x + lx * c - ly * s, y + lx * s + ly * c)
            for lx, ly in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _convex_hull(points):
    """CCW convex hull (monotone chain)."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _clip_halfplane(poly, a, b, c):
    """Sutherland-Hodgman: keep the part of a 2D polygon with a*x + b*y + c >= 0."""
    out = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k - 1], poly[k]
        fp = a * p[0] + b * p[1] + c
        fq = a * q[0] + b * q[1] + c
        if fq >= 0.0:
            if fp < 0.0:
                t = fp / (fp - fq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            out.append(q)
        elif fp >= 0.0:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _clip_near(verts, near):
    """Clip camera-space (depth, lateral, vertical) polygon against depth >= near."""
    out = []
    n = len(verts)
    for k in range(n):
        p, q = verts[k - 1], verts[k]
        if q[0] >= near:
            if p[0] < near:
                t = (near - p[0]) / (q[0] - p[0])
                out.append((near, p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
            out.append(q)
        elif p[0] >= near:
            t = (near - p[0]) / (q[0] - p[0])
            out.append((near, p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
    return out


def _clip_screen(pts, W, H, margin=400.0):
    """Clip a projected polygon to a padded screen rect (keeps gfxdraw's int16 coordinates in range)."""
    for a, b, c in ((1, 0, margin), (-1, 0, W + margin), (0, 1, margin), (0, -1, H + margin)):
        if len(pts) < 3:
            return []
        pts = _clip_halfplane(pts, a, b, c)
    return pts


def _shade(color, k):
    return tuple(max(0, min(255, int(ch * k))) for ch in color)


def _mix(c0, c1, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))


def _separation_order(PA, PB, viewer):
    """Occlusion order of two vertical prisms standing on the ground, from their plan footprints.
    Finds a separating axis (SAT); the prism on the viewer's side of the separating line may hide the
    other but never vice versa (the plan projection of any view ray is a straight line).
    viewer = ("point", x, y) for the perspective camera, ("dir", vx, vy) for an orthographic view
    looking along -v (v points toward the viewer).
    Returns +1 (A in front), -1 (B in front), 0 (cannot overlap), None (footprints intersect)."""
    for poly in (PA, PB):
        m = len(poly)
        for k in range(m):
            p, q = poly[k], poly[(k + 1) % m]
            nx, ny = q[1] - p[1], -(q[0] - p[0])
            if abs(nx) + abs(ny) < 1e-9:
                continue
            a = [nx * x + ny * y for x, y in PA]
            b = [nx * x + ny * y for x, y in PB]
            a0, a1, b0, b1 = min(a), max(a), min(b), max(b)
            eps = 1e-6 * (abs(nx) + abs(ny))
            if a1 <= b0 + eps:
                lo_is_a, lo_max, hi_min = True, a1, b0
            elif b1 <= a0 + eps:
                lo_is_a, lo_max, hi_min = False, b1, a0
            else:
                continue
            if viewer[0] == "point":
                c = nx * viewer[1] + ny * viewer[2]
                if c <= lo_max:
                    lo_front = True
                elif c >= hi_min:
                    lo_front = False
                else:
                    return 0
            else:
                s = nx * viewer[1] + ny * viewer[2]
                if abs(s) < 1e-9:
                    return 0
                lo_front = s < 0.0
            return 1 if lo_front == lo_is_a else -1
    return None


def _order_objects(items, viewer):
    """Back-to-front draw order of prism objects (list of dicts with 'foot', 'bbox', 'key';
    larger key = farther). Topological sort of the pairwise occlusion relation between objects whose
    screen boxes overlap (Kahn, farthest-first); cycles fall back to the farthest remaining object."""
    n = len(items)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        bi = items[i]["bbox"]
        for j in range(i + 1, n):
            bj = items[j]["bbox"]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            r = _separation_order(items[i]["foot"], items[j]["foot"], viewer)
            if r is None:   # overlapping footprints: nearer centroid in front
                r = 1 if items[i]["key"] < items[j]["key"] else -1
            if r == 1:      # i in front -> draw j first
                succ[j].append(i); indeg[i] += 1
            elif r == -1:
                succ[i].append(j); indeg[j] += 1
    heap = [(-items[k]["key"], k) for k in range(n) if indeg[k] == 0]
    heapq.heapify(heap)
    done = [False] * n
    order = []
    while len(order) < n:
        if not heap:  # cycle: release the farthest remaining object
            k = max((k for k in range(n) if not done[k]), key=lambda k: items[k]["key"])
            indeg[k] = 0
            heap.append((-items[k]["key"], k))
        _, k = heapq.heappop(heap)
        if done[k]:
            continue
        done[k] = True
        order.append(k)
        for m in succ[k]:
            indeg[m] -= 1
            if indeg[m] == 0 and not done[m]:
                heapq.heappush(heap, (-items[m]["key"], m))
    return order


def _prism_faces(pr, side_ok, top_ok):
    """Visible faces [(verts3d, color)] of a vertical prism.
    pr: {"foot": CCW plan polygon, "z0", "z1", "top": color, "side": color,
         "bands": optional [(za, zb, color or (colA, colB) stripes)], "stripe_len": stripe period}"""
    foot, z0, z1 = pr["foot"], pr["z0"], pr["z1"]
    faces = []
    n = len(foot)
    bands = pr.get("bands") or [(z0, z1, pr["side"])]
    for k in range(n):
        p, q = foot[k], foot[(k + 1) % n]
        ex, ey = q[0] - p[0], q[1] - p[1]
        L = math.hypot(ex, ey)
        if L < 1e-6:
            continue
        nx, ny = ey / L, -ex / L   # outward normal of a CCW edge
        if not side_ok(p, nx, ny):
            continue
        k_light = 0.8 + 0.2 * (nx * LIGHT_DIR[0] + ny * LIGHT_DIR[1])
        for za, zb, col in bands:
            if isinstance(col[0], tuple):   # alternating stripes along the edge
                m = max(1, int(round(L / pr.get("stripe_len", 0.8))))
                for s in range(m):
                    t0, t1 = s / m, (s + 1) / m
                    a = (p[0] + ex * t0, p[1] + ey * t0)
                    b = (p[0] + ex * t1, p[1] + ey * t1)
                    faces.append(([(a[0], a[1], za), (b[0], b[1], za), (b[0], b[1], zb), (a[0], a[1], zb)],
                                  _shade(col[s % 2], k_light)))
            else:
                faces.append(([(p[0], p[1], za), (q[0], q[1], za), (q[0], q[1], zb), (p[0], p[1], zb)],
                              _shade(col, k_light)))
    if top_ok(z1):
        faces.append(([(x, y, z1) for x, y in foot], pr["top"]))
    return faces


def _make_object(tag, prisms, decals=(), scenery=True):
    pts = [v for pr in prisms for v in pr["foot"]]
    return {"tag": tag, "prisms": list(prisms), "decals": list(decals), "scenery": scenery,
            "foot": _convex_hull(pts)}


def _car_object(car):
    """Ambulance: CarRacing hull fixtures extruded to CAR_HEIGHT (white, red side stripe, red roof cross,
    blue light bar) and wheels scaled / extruded like the MultiCar wrapper."""
    prisms = []
    for fixture in car.hull.fixtures:
        foot = [tuple(car.hull.transform * v) for v in fixture.shape.vertices]
        prisms.append({"foot": foot, "z0": 0.0, "z1": CAR_HEIGHT, "top": AMBULANCE_WHITE,
                       "side": AMBULANCE_WHITE,
                       "bands": [(0.0, 1.0, (225, 225, 225)), (1.0, 1.5, AMBULANCE_RED),
                                 (1.5, CAR_HEIGHT, AMBULANCE_WHITE)]})
    for w in car.wheels:
        for fixture in w.fixtures:
            verts = [tuple(w.transform * v) for v in fixture.shape.vertices]
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)
            foot = [(cx + (vx - cx) * WHEEL_SCALE, cy + (vy - cy) * WHEEL_SCALE) for vx, vy in verts]
            prisms.append({"foot": foot, "z0": 0.0, "z1": WHEEL_HEIGHT, "top": TIRE_COLOR,
                           "side": (25, 25, 25)})
    # Roof decals (car-local metres; forward = local +y)
    tf = car.hull.transform
    zd = CAR_HEIGHT + 0.02
    decals = []
    for (x0, y0, x1, y1), col in (((-0.2, -1.5, 0.2, 0.1), AMBULANCE_RED),
                                  ((-0.8, -0.9, 0.8, -0.5), AMBULANCE_RED),
                                  ((-0.7, 1.2, 0.7, 1.55), (40, 90, 235))):
        decals.append(([tuple(tf * v) + (zd,) for v in _rect(x0, y0, x1, y1)], col))
    return _make_object(("car",), prisms, decals, scenery=False)


def _obstacle_object(o):
    kind = o["kind"]
    foot = _box_corners(o["x"], o["y"], o["angle"], o["w"], o["h"])
    H = o["height"]
    if kind == "roadblock":
        pr = {"foot": foot, "z0": 0.0, "z1": H, "top": (235, 60, 35), "side": (240, 240, 240),
              "bands": [(0.0, 0.35 * H, (70, 70, 70)), (0.35 * H, H, ((235, 40, 30), (245, 245, 245)))],
              "stripe_len": 0.75}
    elif kind == "bus":
        pr = {"foot": foot, "z0": 0.0, "z1": H, "top": (238, 238, 232), "side": (232, 178, 36),
              "bands": [(0.0, 0.6, (45, 45, 45)), (0.6, 0.55 * H, (232, 178, 36)),
                        (0.55 * H, 0.85 * H, (60, 82, 104)), (0.85 * H, H, (232, 178, 36))]}
    else:
        pr = {"foot": foot, "z0": 0.0, "z1": H, "top": (160, 140, 118), "side": (139, 119, 101)}
    return _make_object(("obstacle", o["idx"], kind), [pr])


# ==============================================================================
# Contact listener / ray cast
# ==============================================================================
class _CityContactListener(contactListener):
    """CarRacing's FrictionDetector logic (wheel.tiles <- road / sidewalk sensor tiles), minus the
    track-visit rewards, plus recording of car-vs-solid (building / obstacle) contacts."""

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        tile = obj = None
        if u1 is not None and hasattr(u1, "road_friction"):
            tile, obj = u1, u2
        if u2 is not None and hasattr(u2, "road_friction"):
            tile, obj = u2, u1
        if tile is not None:
            if obj is not None and hasattr(obj, "tiles"):
                if begin:
                    obj.tiles.add(tile)
                else:
                    obj.tiles.discard(tile)
            return
        if not begin:
            return
        for a, b in ((u1, u2), (u2, u1)):
            if a is not None and getattr(a, "is_car", False) and b is not None and getattr(b, "solid", False):
                self.env._hit_events.append(b.kind)


class _LidarCallback(rayCastCallback):
    """Closest hit against solid bodies (buildings / obstacles); sensors and the car are ignored."""

    def __init__(self):
        rayCastCallback.__init__(self)
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        u = fixture.body.userData
        if fixture.sensor or u is None or not getattr(u, "solid", False):
            return -1.0
        self.fraction = min(self.fraction, fraction)
        return fraction


# ==============================================================================
# 1. RESCUE AMBULANCE BASE ENVIRONMENT
# ==============================================================================
class RescueAmbulanceEnv(gym.Env):
    """
    Rescue Ambulance: drive the CarRacing car through a city grid to a victim at the curb.

    Constructor flags (experiment hooks):
      obstacles=True : random street obstacles (6-10 roadblocks / parked buses / debris, each only
                       partly blocking its road). False -> no street obstacles; buildings remain.
      buildings=True : buildings + sidewalks fill the blocks. False -> open grass with only the roads
                       (closest to MultiCar Racing; lidar then only sees obstacles).

    Actions (Box([-1, 0, 0], [1, 1, 1]), identical to CarRacing-v3 continuous):
      0: steer [-1..1] (-1 = left, +1 = right)   1: gas [0..1]   2: brake [0..1]

    Observation (26 floats, clipped to [-1, 1]; car frame = (right, forward)):
      0      speed |v| / V_REF
      1-2    velocity in the car frame (forward, right) / V_REF
      3      hull angular velocity / ANG_REF
      4-5    heading cos, sin (hull angle; angle 0 = facing North, +Y)
      6      front-wheel steer joint angle / 0.4 rad
      7-8    vector to the victim in the car frame (right, forward) / DIST_REF (180 m)
      9      road-graph distance to the victim / ROUTE_REF (336 m); 1 if unreachable
      10-11  unit direction to the next route waypoint (intersection or victim) in the car frame (right, forward)
      12-23  12 lidar ranges / LIDAR_RANGE (40 m) against buildings + obstacles (1 = nothing);
             ray k points 30*k degrees counter-clockwise from the car's forward axis
      24     signed offset from the nearest road centreline / ROAD_R (+ = right of it, w.r.t. the heading)
      25     on-road flag (1 if any wheel is on a road tile)

    Reward:
      + PROGRESS_K * decrease of the road-graph distance to the victim (shortest path along the road
        network, Dijkstra over intersections; segments fully blocked by obstacles are removed;
        off-road distance beyond the road edge is added). Potential-based.
      - 0.1 per step
      - 10 for each new contact with a building / obstacle, - 0.2 per step while touching one
      + 100 on arrival (hull centre within ARRIVE_R = 4 m of the victim; terminated)
      - 100 for leaving the playfield (|x| or |y| > PLAYFIELD; terminated, like CarRacing)
      Truncated after max_steps (1000) steps.

    reset(seed, options) options (all optional):
      "obstacles": list of {"kind": "roadblock"|"bus"|"debris", "x", "y", "angle"(rad, default 0),
                   "w"(opt), "h"(opt), "height"(opt)} placed exactly (overrides random sampling and the
                   obstacles flag; [] = none). w / h are full sizes along the box's local x / y axes.
      "car_start": (x, y, angle)  car hull pose (angle in Box2D / CarRacing convention, 0 = North)
      "victim":    (x, y)
    Coordinate helpers (staticmethods): intersection(i, j), heading_angle(dir), lane_point(i, j, dir, dist,
      lane), curb_point(i, j, dir, dist, side), obstacle_on_road(kind, i, j, dir, dist, lane, **overrides).

    info (reset and every step): car_pos, car_angle, speed, collision, collided_with, obstacles (placed list,
      each with its index "idx"), victim_pos, dist_to_victim (road-graph, inf if unreachable),
      dist_euclid, on_road, arrived, step.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode="rgb_array", obstacles=True, buildings=True,
                 num_obstacles=(6, 10), max_steps=MAX_STEPS):
        self.render_mode = render_mode
        self.obstacles_enabled = bool(obstacles)
        self.buildings_enabled = bool(buildings)
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps

        self.action_space = spaces.Box(np.array([-1, 0, 0], dtype=np.float32),
                                       np.array([+1, +1, +1], dtype=np.float32))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.world = None
        self.car = None
        self._listener = None
        self._hit_events = []
        self.screen = None
        self.clock = None

        self.road_rects = []      # (x0, y0, x1, y1) road + intersection tiles
        self.sidewalk_rects = []
        self.building_defs = []   # dicts: x0, y0, x1, y1, height, color
        self.obstacle_list = []   # placed obstacle dicts
        self.victim_pos = (0.0, 0.0)
        self.step_count = 0
        self.t = 0.0
        self._scene_cache = None
        self._build_segments()

    # ------------------------------------------------------------------
    # Coordinate helpers (usable without an env instance)
    # ------------------------------------------------------------------
    @staticmethod
    def intersection(i, j):
        """World (x, y) of intersection (i, j), i = column (West->East), j = row (South->North), 0..N_BLOCKS."""
        return ((i - N_BLOCKS / 2) * PITCH, (j - N_BLOCKS / 2) * PITCH)

    @staticmethod
    def heading_angle(direction):
        """Car hull angle facing compass direction 'N' | 'E' | 'S' | 'W' (CarRacing: forward = (-sin a, cos a))."""
        dx, dy = DIRS[direction]
        return math.atan2(-dx, dy)

    @staticmethod
    def lane_point(i, j, direction, dist, lane=1.0):
        """(x, y, car_angle) on the road leaving intersection (i, j) toward `direction`, `dist` metres
        from the intersection centre. lane = lateral offset in half-lane-widths to the right of the
        travel direction: +1 = centre of the right-hand (driving) lane, -1 = oncoming lane, 0 = centreline,
        +-2 = curb line. car_angle faces `direction`."""
        x0, y0 = RescueAmbulanceEnv.intersection(i, j)
        dx, dy = DIRS[direction]
        off = lane * LANE_W / 2
        return (x0 + dx * dist + dy * off, y0 + dy * dist - dx * off,
                RescueAmbulanceEnv.heading_angle(direction))

    @staticmethod
    def curb_point(i, j, direction, dist, side=1):
        """(x, y) at the curb (CURB_OFFSET from the centreline) on the right (side=+1) or left (-1) of the
        road leaving intersection (i, j) toward `direction` (default victim placement style)."""
        x, y, _ = RescueAmbulanceEnv.lane_point(i, j, direction, dist, lane=side * CURB_OFFSET / (LANE_W / 2))
        return (x, y)

    @staticmethod
    def obstacle_on_road(kind, i, j, direction, dist, lane=None, **overrides):
        """Obstacle dict (for options["obstacles"]) on the road leaving intersection (i, j) toward `direction`.
        roadblock: barrier across the lane (default right lane, lane=+1; use w=ROAD_W, lane=0 to close
        the road); bus: parked along the curb of the right lane; debris: box in the right lane.
        Any key (x, y, angle, w, h, height) can be overridden."""
        spec = dict(OBSTACLE_SPECS[kind])
        spec.update({k: v for k, v in overrides.items() if k in ("w", "h", "height")})
        if lane is None:
            lane = (ROAD_R - spec["h"] / 2 - 0.3) / (LANE_W / 2) if kind == "bus" else 1.0
        x, y, _ = RescueAmbulanceEnv.lane_point(i, j, direction, dist, lane)
        dx, dy = DIRS[direction]
        phi = math.atan2(dy, dx)
        angle = phi + math.pi / 2 if kind == "roadblock" else phi
        o = {"kind": kind, "x": x, "y": y, "angle": angle, "w": spec["w"], "h": spec["h"],
             "height": spec["height"]}
        o.update({k: v for k, v in overrides.items() if k in ("x", "y", "angle")})
        return o

    # ------------------------------------------------------------------
    # Static road graph
    # ------------------------------------------------------------------
    def _build_segments(self):
        """Road segments between neighbouring intersections (centreline A -> B, unit dir u)."""
        nodes = [(i, j) for i in range(N_BLOCKS + 1) for j in range(N_BLOCKS + 1)]
        self.node_index = {n: k for k, n in enumerate(nodes)}
        self.node_pos = np.array([self.intersection(i, j) for i, j in nodes], dtype=np.float64)
        segs = []
        for i in range(N_BLOCKS + 1):
            for j in range(N_BLOCKS):
                segs.append(((i, j), (i, j + 1)))   # vertical (North-South) road
        for j in range(N_BLOCKS + 1):
            for i in range(N_BLOCKS):
                segs.append(((i, j), (i + 1, j)))   # horizontal (East-West) road
        self.seg_nodes = np.array([(self.node_index[a], self.node_index[b]) for a, b in segs])
        self.seg_a = self.node_pos[self.seg_nodes[:, 0]]
        self.seg_u = (self.node_pos[self.seg_nodes[:, 1]] - self.seg_a) / PITCH
        self.seg_r = np.stack([self.seg_u[:, 1], -self.seg_u[:, 0]], axis=1)   # right of A->B
        self.seg_blocks = [[] for _ in segs]

    def _nearest_segment(self, x, y):
        """(segment index, param t in [0, 1] along A->B, signed lateral offset right of A->B)."""
        rel = np.array([x, y]) - self.seg_a
        along = np.einsum("ij,ij->i", rel, self.seg_u)
        t = np.clip(along / PITCH, 0.0, 1.0)
        diff = rel - self.seg_u * (t * PITCH)[:, None]
        e = np.einsum("ij,ij->i", diff, diff)
        s = int(np.argmin(e))
        return s, float(t[s]), float(rel[s] @ self.seg_r[s])

    def _compute_blocks(self, obstacles):
        """Per segment: params t of obstacle clusters leaving no lateral gap >= CAR_PASS_W."""
        blocks = [[] for _ in range(len(self.seg_nodes))]
        if not obstacles:
            return blocks
        corners = [_box_corners(o["x"], o["y"], o["angle"], o["w"], o["h"]) for o in obstacles]
        for s in range(len(self.seg_nodes)):
            A, u, r = self.seg_a[s], self.seg_u[s], self.seg_r[s]
            items = []
            for cs in corners:
                lon = [(cx - A[0]) * u[0] + (cy - A[1]) * u[1] for cx, cy in cs]
                lat = [(cx - A[0]) * r[0] + (cy - A[1]) * r[1] for cx, cy in cs]
                if max(lon) < ROAD_R or min(lon) > PITCH - ROAD_R:
                    continue   # not on this segment's road (intersections are ignored)
                if max(lat) < -ROAD_R or min(lat) > ROAD_R:
                    continue
                items.append((min(lon), max(lon), max(min(lat), -ROAD_R), min(max(lat), ROAD_R)))
            items.sort()
            k = 0
            while k < len(items):
                lo, hi = items[k][0], items[k][1]
                group = [items[k]]
                k += 1
                while k < len(items) and items[k][0] <= hi + CAR_LEN:
                    group.append(items[k])
                    hi = max(hi, items[k][1])
                    k += 1
                spans = sorted((g[2], g[3]) for g in group)
                cur, gap = -ROAD_R, 0.0
                for a, b in spans:
                    gap = max(gap, a - cur)
                    cur = max(cur, b)
                gap = max(gap, ROAD_R - cur)
                if gap < CAR_PASS_W:
                    blocks[s].append(float(np.clip(0.5 * (lo + hi) / PITCH, 0.0, 1.0)))
        return blocks

    def _compute_graph(self):
        """Dijkstra from the victim over the intersection graph (blocked segments removed)."""
        n_nodes = len(self.node_pos)
        V = n_nodes
        adj = [[] for _ in range(n_nodes + 1)]
        for s, (a, b) in enumerate(self.seg_nodes):
            if not self.seg_blocks[s]:
                adj[a].append((b, PITCH)); adj[b].append((a, PITCH))
        sv, tv, lv = self._nearest_segment(*self.victim_pos)
        self._victim_seg = (sv, tv, lv)
        a, b = self.seg_nodes[sv]
        if not any(tb < tv for tb in self.seg_blocks[sv]):
            adj[V].append((a, tv * PITCH)); adj[a].append((V, tv * PITCH))
        if not any(tb > tv for tb in self.seg_blocks[sv]):
            adj[V].append((b, (1 - tv) * PITCH)); adj[b].append((V, (1 - tv) * PITCH))
        dist = [math.inf] * (n_nodes + 1)
        nxt = [None] * (n_nodes + 1)   # next hop toward the victim
        dist[V] = 0.0
        heap = [(0.0, V)]
        while heap:
            d, k = heapq.heappop(heap)
            if d > dist[k]:
                continue
            for m, w in adj[k]:
                if d + w < dist[m]:
                    dist[m] = d + w
                    nxt[m] = k
                    heapq.heappush(heap, (d + w, m))
        self._graph_dist, self._graph_next, self._V = dist, nxt, V

    def _node_xy(self, k):
        return tuple(self.victim_pos) if k == self._V else tuple(self.node_pos[k])

    def road_distance(self, x, y):
        """(shortest road-network distance to the victim, next waypoint (x, y) or None)."""
        s, t, lat = self._nearest_segment(x, y)
        blocks = self.seg_blocks[s]
        sv, tv, lv = self._victim_seg
        a, b = self.seg_nodes[s]
        best, wp = math.inf, None
        if s == sv and not any(min(t, tv) < tb < max(t, tv) for tb in blocks):
            best, wp = abs(t - tv) * PITCH + abs(lat - lv), self._V
        if not any(tb < t for tb in blocks):
            d = t * PITCH + self._graph_dist[a]
            if d < best:
                best, wp = d, a
        if not any(tb > t for tb in blocks):
            d = (1 - t) * PITCH + self._graph_dist[b]
            if d < best:
                best, wp = d, b
        if wp is None or not math.isfinite(best):
            return math.inf, None
        if wp != self._V:
            px, py = self._node_xy(wp)
            if math.hypot(x - px, y - py) < 1.2 * ROAD_R and self._graph_next[wp] is not None:
                wp = self._graph_next[wp]
        return best + max(0.0, abs(lat) - ROAD_R), self._node_xy(wp)

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------
    def _add_tile(self, x0, y0, x1, y1, friction):
        body = self.world.CreateStaticBody(
            position=((x0 + x1) / 2, (y0 + y1) / 2),
            fixtures=fixtureDef(shape=polygonShape(box=((x1 - x0) / 2, (y1 - y0) / 2)), isSensor=True))
        body.userData = body
        body.road_friction = friction
        return body

    def _add_solid(self, x, y, angle, w, h, kind, idx):
        body = self.world.CreateStaticBody(position=(x, y), angle=angle,
                                           shapes=polygonShape(box=(w / 2, h / 2)))
        body.userData = body
        body.solid = True
        body.kind = kind
        body.idx = idx
        return body

    def _build_city(self, rng):
        xs = [self.intersection(i, 0)[0] for i in range(N_BLOCKS + 1)]
        self.road_rects, self.sidewalk_rects, self.building_defs = [], [], []
        # Road tiles: segments between intersections + intersection squares (one sensor body each)
        for i, x in enumerate(xs):
            for j, y in enumerate(xs):
                self.road_rects.append((x - ROAD_R, y - ROAD_R, x + ROAD_R, y + ROAD_R))
                if j < N_BLOCKS:
                    self.road_rects.append((x - ROAD_R, y + ROAD_R, x + ROAD_R, xs[j + 1] - ROAD_R))
                if i < N_BLOCKS:
                    self.road_rects.append((x + ROAD_R, y - ROAD_R, xs[i + 1] - ROAD_R, y + ROAD_R))
        for r in self.road_rects:
            self._add_tile(*r, ROAD_FRICTION)
        if not self.buildings_enabled:
            return
        # Sidewalk rings + 1-4 buildings per block
        for bi in range(N_BLOCKS):
            for bj in range(N_BLOCKS):
                ox0, ox1 = xs[bi] + ROAD_R, xs[bi + 1] - ROAD_R
                oy0, oy1 = xs[bj] + ROAD_R, xs[bj + 1] - ROAD_R
                S = SIDEWALK_W
                for r in ((ox0, oy0, ox1, oy0 + S), (ox0, oy1 - S, ox1, oy1),
                          (ox0, oy0 + S, ox0 + S, oy1 - S), (ox1 - S, oy0 + S, ox1, oy1 - S)):
                    self.sidewalk_rects.append(r)
                    self._add_tile(*r, SIDEWALK_FRICTION)
                ix0, ix1, iy0, iy1 = ox0 + S, ox1 - S, oy0 + S, oy1 - S

                def split(a, b):
                    if rng.random() < 0.5:
                        return [(a, b)]
                    m = a + (b - a) * float(rng.uniform(0.35, 0.65))
                    return [(a, m - ALLEY_W / 2), (m + ALLEY_W / 2, b)]
                for x0, x1 in split(ix0, ix1):
                    for y0, y1 in split(iy0, iy1):
                        col = BUILDING_PALETTE[int(rng.integers(len(BUILDING_PALETTE)))]
                        self.building_defs.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                                   "height": float(rng.uniform(*BUILDING_H)), "color": col})
        for k, b in enumerate(self.building_defs):
            self._add_solid((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, 0.0,
                            b["x1"] - b["x0"], b["y1"] - b["y0"], "building", k)

    def _segment_pose(self, s, t, forward, lane):
        """(x, y, car_angle) at param t on segment s, travelling A->B (forward) or B->A, lane as lane_point."""
        (i0, j0), (i1, j1) = [divmod(int(k), N_BLOCKS + 1) for k in self.seg_nodes[s]]
        u = self.seg_u[s]
        d = {(1, 0): "E", (0, 1): "N", (-1, 0): "W", (0, -1): "S"}[(int(round(u[0])), int(round(u[1])))]
        if forward:
            return self.lane_point(i0, j0, d, t * PITCH, lane)
        back = {"E": "W", "W": "E", "N": "S", "S": "N"}[d]
        return self.lane_point(i1, j1, back, (1 - t) * PITCH, lane)

    def _sample_start(self, rng):
        s = int(rng.integers(len(self.seg_nodes)))
        return self._segment_pose(s, float(rng.uniform(0.3, 0.7)), bool(rng.random() < 0.5), 1.0)

    def _sample_victim(self, rng, start):
        segs = list(range(len(self.seg_nodes)))
        rng.shuffle(segs)
        for s in segs:
            mid = self.seg_a[s] + self.seg_u[s] * PITCH / 2
            if abs(mid[0] - start[0]) + abs(mid[1] - start[1]) >= 2 * PITCH:
                break
        side = 1.0 if rng.random() < 0.5 else -1.0
        x, y, _ = self._segment_pose(s, float(rng.uniform(0.3, 0.7)), True, side * CURB_OFFSET / (LANE_W / 2))
        return (x, y)

    def _complete_obstacle(self, o, idx):
        kind = o["kind"]
        if kind not in OBSTACLE_SPECS:
            raise ValueError(f"unknown obstacle kind {kind!r}; expected one of {OBSTACLE_KINDS}")
        spec = OBSTACLE_SPECS[kind]
        return {"idx": idx, "kind": kind, "x": float(o["x"]), "y": float(o["y"]),
                "angle": float(o.get("angle", 0.0)),
                "w": float(o.get("w", spec["w"])), "h": float(o.get("h", spec["h"])),
                "height": float(o.get("height", spec["height"]))}

    def _sample_obstacles(self, rng, start, victim):
        """Random street obstacles on road segments; never closing a road, near the start or on the victim."""
        lo, hi = self.num_obstacles
        n_target = int(rng.integers(lo, hi + 1))
        placed, attempts = [], 0
        while len(placed) < n_target and attempts < 300:
            attempts += 1
            kind = OBSTACLE_KINDS[int(rng.choice(3, p=[0.35, 0.3, 0.35]))]
            s = int(rng.integers(len(self.seg_nodes)))
            t = float(rng.uniform(0.3, 0.7))
            fwd = bool(rng.random() < 0.5)
            spec = dict(OBSTACLE_SPECS[kind])
            if kind == "roadblock":
                x, y, a = self._segment_pose(s, t, fwd, 1.0)
                angle = a + math.pi / 2 + math.pi / 2   # car angle -> world road angle (+pi/2), then across (+pi/2)
            elif kind == "bus":
                x, y, a = self._segment_pose(s, t, fwd, (ROAD_R - spec["h"] / 2 - 0.3) / (LANE_W / 2))
                angle = a + math.pi / 2
            else:
                spec["w"] = float(rng.uniform(1.0, 2.2)); spec["h"] = float(rng.uniform(1.0, 2.2))
                spec["height"] = float(rng.uniform(0.8, 1.6))
                x, y, a = self._segment_pose(s, t, fwd, float(rng.uniform(-1.5, 1.5)))
                angle = float(rng.uniform(-math.pi, math.pi))
            cand = self._complete_obstacle({"kind": kind, "x": x, "y": y, "angle": angle, **spec}, len(placed))
            rad = 0.5 * math.hypot(cand["w"], cand["h"])
            if math.hypot(x - start[0], y - start[1]) < rad + 12.0:
                continue
            if math.hypot(x - victim[0], y - victim[1]) < rad + 6.0:
                continue
            if any(math.hypot(x - o["x"], y - o["y"]) < rad + 0.5 * math.hypot(o["w"], o["h"]) + 3.0
                   for o in placed):
                continue
            if any(self._compute_blocks(placed + [cand])):
                continue   # would close a road
            placed.append(cand)
        return placed

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = dict(options or {})

        self.car = None
        self._listener = _CityContactListener(self)
        self.world = b2World(gravity=(0, 0), contactListener=self._listener)
        self._hit_events = []
        self.step_count = 0
        self.t = 0.0
        self._scene_cache = None
        self._prev_touching = False

        self._build_city(rng)

        if options.get("car_start") is not None:
            sx, sy, sa = (float(v) for v in options["car_start"])
        else:
            sx, sy, sa = self._sample_start(rng)
        if options.get("victim") is not None:
            self.victim_pos = (float(options["victim"][0]), float(options["victim"][1]))
        else:
            self.victim_pos = self._sample_victim(rng, (sx, sy))

        if options.get("obstacles") is not None:
            self.obstacle_list = [self._complete_obstacle(o, k) for k, o in enumerate(options["obstacles"])]
        elif self.obstacles_enabled:
            self.obstacle_list = self._sample_obstacles(rng, (sx, sy), self.victim_pos)
        else:
            self.obstacle_list = []
        for o in self.obstacle_list:
            self._add_solid(o["x"], o["y"], o["angle"], o["w"], o["h"], o["kind"], o["idx"])

        # Victim marker: sensor disc (category 0x0002 so the wheels' friction sensors ignore it)
        vb = self.world.CreateStaticBody(
            position=self.victim_pos,
            fixtures=fixtureDef(shape=circleShape(radius=VICTIM_R), isSensor=True, categoryBits=0x0002))
        vb.userData = vb
        vb.kind = "victim"

        # Ambulance = CarRacing car (identical physics), only repainted
        self.car = Car(self.world, sa, sx, sy)
        self.car.hull.color = tuple(c / 255.0 for c in AMBULANCE_WHITE)
        self.car.hull.userData = self.car.hull
        self.car.hull.is_car = True
        for w in self.car.wheels:
            w.is_car = True

        # One physics tick so the wheels register the road tiles they start on (CarRacing's reset step)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self._hit_events = []

        self.seg_blocks = self._compute_blocks(self.obstacle_list)
        self._compute_graph()
        self.prev_dist, _ = self.road_distance(*self.car.hull.position)

        if self.render_mode == "human":
            self.render()
        collision, kind = self._touching()
        return self._get_obs(), self._get_info(self.prev_dist, collision, kind, False)

    # ------------------------------------------------------------------
    def _touching(self):
        """(touching any building / obstacle now, its kind)."""
        for body in [self.car.hull] + list(self.car.wheels):
            for ce in body.contacts:
                if ce.contact.touching:
                    u = ce.other.userData
                    if u is not None and getattr(u, "solid", False):
                        return True, u.kind
        return False, None

    def _on_road(self):
        return any(getattr(t, "road_friction", 0.0) >= ROAD_FRICTION for w in self.car.wheels for t in w.tiles)

    def step(self, action):
        if self.car is None:
            raise RuntimeError("RescueAmbulanceEnv: call reset() before step().")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        gas = float(np.clip(a[1], 0.0, 1.0))
        brake = float(np.clip(a[2], 0.0, 1.0))
        self.car.steer(-float(np.clip(a[0], -1.0, 1.0)))
        self.car.gas(gas)
        hull = self.car.hull
        forw = hull.GetWorldVector((0, 1))
        v_fwd = forw[0] * hull.linearVelocity[0] + forw[1] * hull.linearVelocity[1]
        if brake > 0.0 and gas == 0.0 and v_fwd < REVERSE_ENGAGE_SPEED:
            # Stopped (or already reversing) with brake held: drive the rear wheels backwards instead,
            # holding about REVERSE_MAX_SPEED * brake (light braking bleeds off wheel spin above it)
            below_target = v_fwd > -REVERSE_MAX_SPEED * brake
            self.car.brake(0.0 if below_target else REVERSE_HOLD_BRAKE)
            for w in self.car.wheels[2:4]:
                w.gas = -REVERSE_GAS if below_target else 0.0
        else:
            self.car.brake(brake)

        self._hit_events = []
        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.step_count += 1
        self.car.fuel_spent = 0.0

        x, y = self.car.hull.position
        dist, _ = self.road_distance(x, y)
        touching, touch_kind = self._touching()
        collision = touching or bool(self._hit_events)
        collided_with = self._hit_events[0] if self._hit_events else touch_kind

        reward = -STEP_COST
        if math.isfinite(dist) and math.isfinite(self.prev_dist):
            reward += PROGRESS_K * (self.prev_dist - dist)
        self.prev_dist = dist
        if self._hit_events:
            reward -= COLLISION_PENALTY
        if collision:
            reward -= CONTACT_COST

        arrived = math.hypot(x - self.victim_pos[0], y - self.victim_pos[1]) < ARRIVE_R
        terminated, truncated = False, False
        if arrived:
            reward += ARRIVE_BONUS
            terminated = True
        elif abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            reward = -OUT_PENALTY
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()
        return (self._get_obs(), float(reward), terminated, truncated,
                self._get_info(dist, collision, collided_with, arrived))

    # ------------------------------------------------------------------
    def lidar(self):
        """LIDAR_N ranges (metres) from the hull centre, ray k at 30*k deg CCW from forward."""
        p = self.car.hull.position
        a = float(self.car.hull.angle)
        out = []
        for k in range(LIDAR_N):
            th = a + 2 * math.pi * k / LIDAR_N
            d = (-math.sin(th), math.cos(th))
            cb = _LidarCallback()
            self.world.RayCast(cb, (p[0], p[1]), (p[0] + d[0] * LIDAR_RANGE, p[1] + d[1] * LIDAR_RANGE))
            out.append(cb.fraction * LIDAR_RANGE)
        return out

    def _get_obs(self):
        hull = self.car.hull
        x, y = hull.position
        a = float(hull.angle)
        fwd = (-math.sin(a), math.cos(a))
        right = (math.cos(a), math.sin(a))
        v = hull.linearVelocity
        v_f = v[0] * fwd[0] + v[1] * fwd[1]
        v_r = v[0] * right[0] + v[1] * right[1]
        dvx, dvy = self.victim_pos[0] - x, self.victim_pos[1] - y
        dist, wp = self.road_distance(x, y)
        if wp is None:
            wx, wy = dvx, dvy
        else:
            wx, wy = wp[0] - x, wp[1] - y
        wn = math.hypot(wx, wy) + 1e-6
        s, t, lat = self._nearest_segment(x, y)
        if fwd[0] * self.seg_u[s][0] + fwd[1] * self.seg_u[s][1] < 0:
            lat = -lat
        obs = [
            math.hypot(v[0], v[1]) / V_REF,
            v_f / V_REF, v_r / V_REF,
            hull.angularVelocity / ANG_REF,
            math.cos(a), math.sin(a),
            self.car.wheels[0].joint.angle / 0.4,
            (dvx * right[0] + dvy * right[1]) / DIST_REF, (dvx * fwd[0] + dvy * fwd[1]) / DIST_REF,
            dist / ROUTE_REF if math.isfinite(dist) else 1.0,
            (wx * right[0] + wy * right[1]) / wn, (wx * fwd[0] + wy * fwd[1]) / wn,
        ]
        obs += [r / LIDAR_RANGE for r in self.lidar()]
        obs += [lat / ROAD_R, 1.0 if self._on_road() else 0.0]
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _get_info(self, dist, collision, collided_with, arrived):
        x, y = self.car.hull.position
        v = self.car.hull.linearVelocity
        return {
            "car_pos": (float(x), float(y)),
            "car_angle": float(self.car.hull.angle),
            "speed": float(math.hypot(v[0], v[1])),
            "collision": bool(collision),
            "collided_with": collided_with,
            "obstacles": [dict(o) for o in self.obstacle_list],
            "victim_pos": tuple(self.victim_pos),
            "dist_to_victim": float(dist),
            "dist_euclid": float(math.hypot(x - self.victim_pos[0], y - self.victim_pos[1])),
            "on_road": bool(self._on_road()),
            "arrived": bool(arrived),
            "step": self.step_count,
        }

    # ------------------------------------------------------------------
    # Scene description shared by the top-down render and the multi-view wrapper
    # ------------------------------------------------------------------
    def static_scene(self):
        """Per-episode static geometry: {"ground": [(plan polygon, color, tag)] in draw order (z = 0),
        "objects": [prism objects]}. Tags: ("ground",), ("victim",), ("building", k), ("obstacle", idx, kind)."""
        if self._scene_cache is not None:
            return self._scene_cache
        g = ("ground",)
        ground = [(_rect(-PLAYFIELD, -PLAYFIELD, PLAYFIELD, PLAYFIELD), BG_COLOR, g)]
        n = int(round(2 * PLAYFIELD / GRASS_DIM))
        for ix in range(n):
            for iy in range(n):
                if (ix + iy) % 2 == 0:
                    x0, y0 = -PLAYFIELD + ix * GRASS_DIM, -PLAYFIELD + iy * GRASS_DIM
                    ground.append((_rect(x0, y0, x0 + GRASS_DIM, y0 + GRASS_DIM), GRASS_COLOR, g))
        for r in self.sidewalk_rects:
            ground.append((_rect(*r), SIDEWALK_COLOR, g))
        for r in self.road_rects:
            ground.append((_rect(*r), ROAD_COLOR, g))
        # Dashed white centrelines along each segment
        for s in range(len(self.seg_nodes)):
            A, u, rr = self.seg_a[s], self.seg_u[s], self.seg_r[s]
            d = ROAD_R + 1.5
            while d + 3.0 <= PITCH - ROAD_R - 1.0:
                p0 = A + u * d
                p1 = A + u * (d + 3.0)
                ground.append(([tuple(p0 - rr * 0.15), tuple(p1 - rr * 0.15), tuple(p1 + rr * 0.15),
                                tuple(p0 + rr * 0.15)], MARK_COLOR, g))
                d += 6.0
        # Victim: red disc with a white cross
        vx, vy = self.victim_pos
        ground.append(([(vx + VICTIM_R * math.cos(2 * math.pi * k / 20), vy + VICTIM_R * math.sin(2 * math.pi * k / 20))
                        for k in range(20)], VICTIM_COLOR, ("victim",)))
        for x0, y0, x1, y1 in ((-0.25, -1.0, 0.25, 1.0), (-1.0, -0.25, 1.0, 0.25)):
            ground.append((_rect(vx + x0, vy + y0, vx + x1, vy + y1), (250, 250, 250), ("victim",)))

        objects = []
        for k, b in enumerate(self.building_defs):
            objects.append(_make_object(("building", k), [{
                "foot": _rect(b["x0"], b["y0"], b["x1"], b["y1"]), "z0": 0.0, "z1": b["height"],
                "top": _mix(b["color"], (255, 255, 255), 0.25), "side": b["color"]}]))
        for o in self.obstacle_list:
            objects.append(_obstacle_object(o))
        # Victim beacon pole (red / white bands) so the victim also shows in the elevation views
        objects.append(_make_object(("victim",), [{
            "foot": _rect(vx - 0.2, vy - 0.2, vx + 0.2, vy + 0.2), "z0": 0.0, "z1": 3.0,
            "top": VICTIM_COLOR, "side": VICTIM_COLOR,
            "bands": [(0.0, 1.0, VICTIM_COLOR), (1.0, 1.6, (250, 250, 250)), (1.6, 3.0, VICTIM_COLOR)]}]))
        self._scene_cache = {"ground": ground, "objects": objects}
        return self._scene_cache

    def top_items(self, W, H, zoom=TOP_ZOOM):
        """Top-down draw list [(pts, color, tag)]: centred on the car, heading up (like CarRacing)."""
        scene = self.static_scene()
        cx, cy = self.car.hull.position
        a = float(self.car.hull.angle)
        fx, fy = -math.sin(a), math.cos(a)
        rx, ry = math.cos(a), math.sin(a)

        def proj(x, y):
            dx, dy = x - cx, y - cy
            return (W / 2 + (dx * rx + dy * ry) * zoom, H / 2 - (dx * fx + dy * fy) * zoom)

        def visible(pts):
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return max(xs) >= 0 and min(xs) <= W and max(ys) >= 0 and min(ys) <= H

        items = []
        for poly, col, tag in scene["ground"]:
            pts = [proj(x, y) for x, y in poly]
            if visible(pts):
                pts = _clip_screen(pts, W, H)
                if len(pts) >= 3:
                    items.append((pts, col, tag))
        tops = []
        for obj in scene["objects"] + [_car_object(self.car)]:
            for pr in obj["prisms"]:
                tops.append((pr["z1"], [(x, y) for x, y in pr["foot"]], pr["top"], obj["tag"]))
            for verts, col in obj["decals"]:
                tops.append((verts[0][2], [(v[0], v[1]) for v in verts], col, obj["tag"]))
        tops.sort(key=lambda it: it[0])
        for _, poly, col, tag in tops:
            pts = [proj(x, y) for x, y in poly]
            if visible(pts):
                pts = _clip_screen(pts, W, H)
                if len(pts) >= 3:
                    items.append((pts, col, tag))
        return items

    # ------------------------------------------------------------------
    def render(self):
        """Car-centred top-down view (heading up), VIDEO_H x VIDEO_W x 3 uint8 (or a window in human mode)."""
        if self.render_mode is None or self.car is None:
            return None
        W, H = 1000, 800
        surf = pygame.Surface((W, H))
        surf.fill(BG_COLOR)
        _paint(surf, self.top_items(W, H))
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((VIDEO_W, VIDEO_H))
                self.clock = pygame.time.Clock()
            pygame.event.pump()
            self.screen.blit(pygame.transform.smoothscale(surf, (VIDEO_W, VIDEO_H)), (0, 0))
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        return _surf_to_img(surf)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None


# ==============================================================================
# Painting helpers
# ==============================================================================
def _tag_matches(tag, target):
    if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
        return tag[0] == "obstacle" and tag[1] == int(target)
    if target in ("obstacle", "building", "car", "victim"):
        return tag[0] == target
    if target in OBSTACLE_SPECS:
        return tag[0] == "obstacle" and tag[2] == target
    raise ValueError(f"unknown mask target {target!r}")


def _paint(surf, items, target=None):
    """Fill polygons in order. target=None: real colours (anti-aliased edges, like the MultiCar wrapper);
    otherwise binary mask paint (target white, everything else black, no anti-aliasing)."""
    for pts, color, tag in items:
        ipts = [(int(round(p[0])), int(round(p[1]))) for p in pts]
        if len(ipts) < 3:
            continue
        if target is not None:
            color = (255, 255, 255) if _tag_matches(tag, target) else (0, 0, 0)
        try:
            if target is None:
                gfxdraw.aapolygon(surf, ipts, color)
            gfxdraw.filled_polygon(surf, ipts, color)
        except Exception:
            pass


def _surf_to_img(surf, size=(VIDEO_W, VIDEO_H)):
    img = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
    if cv2 is not None:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small = pygame.transform.smoothscale(surf, size)
    return np.transpose(pygame.surfarray.array3d(small), (1, 0, 2))


# ==============================================================================
# 2. 3D ORTHOGRAPHIC MULTI-VIEW WRAPPER
# ==============================================================================
class RescueAmbulanceOrthographicWrapper(gym.Wrapper):
    """
    Extrudes the city into 3D and returns four views (same layout and camera as the MultiCar wrapper):
      1. Top-Down : car-centred plan view, heading up (TOP_ZOOM px/m, like CarRacing's render)
      2. Rear     : orthographic elevation from behind the ego car (25 px/m), section cut REAR_CUT
                    behind the car (scenery behind the cut is removed so it never hides the car)
      3. Side     : orthographic elevation from the car's right side (25 px/m), section cut SIDE_CUT to
                    the right of the car (buildings / obstacles beyond it are removed)
      4. FPV      : perspective chase camera camera_dist (12 m) behind / camera_height (8 m) above the car,
                    fov 500, orbiting the car with cam_yaw / cam_pitch (Left/Right/Up/Down/Center).
    Coordinates: world X/Y = Box2D plan, Z = height above the ground (0).

    Occlusion: flat ground layers are drawn first; buildings, obstacles, the victim beacon and the car
    are vertical prisms ordered per view by a topological sort of their pairwise occlusion relation
    (separating-axis test of the plan footprints against the camera position / view direction), with
    back-face culling inside each prism. So buildings correctly hide what lies behind them in the Rear
    and FPV views (e.g. a roadblock around a corner is visible in Top but not in FPV).

    render_masks(target) returns binary VISIBLE masks with the same projection and draw order.
    Maintains persistent camera yaw / pitch / roll state.
    """

    REAR_CUT = 3.0   # metres behind the car centre (rear bumper at 2.4 m)
    SIDE_CUT = 1.6   # metres right of the car centre (wheels reach 1.25 m)
    NEAR = 0.5       # FPV near plane (metres)

    def __init__(self, env, step_yaw_deg=8.0, step_pitch_deg=5.0):
        super().__init__(env)
        self.WINDOW_W = 1000
        self.WINDOW_H = 800

        # Camera Parameters (as MultiCar)
        self.fov = 500.0          # Used for 3D Perspective
        self.scale_factor = 25.0  # Used for Orthographic
        self.camera_height = 8.0
        self.camera_dist = 12.0
        self.top_zoom = TOP_ZOOM

        # Incremental turn step sizes
        self.step_yaw = math.radians(step_yaw_deg)
        self.step_pitch = math.radians(step_pitch_deg)

        # Persistent camera orientation angles
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0

        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        # Reset camera angles to canonical trailing position upon reset
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    def _car_frame(self):
        base = self.env.unwrapped
        x, y = base.car.hull.position
        a = float(base.car.hull.angle)
        return float(x), float(y), (-math.sin(a), math.cos(a)), (math.cos(a), math.sin(a))

    @property
    def _cam_anchor(self):
        """Un-orbited chase camera: camera_dist behind the car along its heading, camera_height up."""
        base = self.env.unwrapped
        if getattr(base, "car", None) is None:
            return (0.0, 0.0, 0.0)
        x, y, f, _ = self._car_frame()
        return (x - self.camera_dist * f[0], y - self.camera_dist * f[1], self.camera_height)

    @property
    def cam_pos(self):
        """World X, Y, Z of the orbited FPV camera."""
        base = self.env.unwrapped
        if getattr(base, "car", None) is None:
            return (0.0, 0.0, 0.0)
        ax, ay, az = self._cam_anchor
        _, _, f, r = self._car_frame()
        od, ol, ov = orbit_camera_offset(self.cam_yaw, self.cam_pitch, self.camera_dist)
        return (ax + od * f[0] + ol * r[0], ay + od * f[1] + ol * r[1], az + ov)

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
    def _object_items(self, objects, project, side_ok, top_ok, viewer, key_of):
        """Project prism objects for one view and return their faces back-to-front."""
        projected = []
        for obj in objects:
            parts = []
            for pr in obj["prisms"]:
                pf = [(res, col) for verts, col in _prism_faces(pr, side_ok, top_ok)
                      for res in (project(verts),) if res is not None]
                if pf:
                    xs = [p[0] for res, _ in pf for p in res]
                    ys = [p[1] for res, _ in pf for p in res]
                    parts.append({"faces": pf, "foot": pr["foot"], "key": key_of(pr["foot"]),
                                  "bbox": (min(xs), min(ys), max(xs), max(ys))})
            # Composite objects (the car): order their prisms with the same occlusion sort; decals last
            order = _order_objects(parts, viewer) if len(parts) > 1 else range(len(parts))
            faces = [f for k in order for f in parts[k]["faces"]]
            for verts, col in obj["decals"]:
                if top_ok(verts[0][2]):
                    res = project(verts)
                    if res is not None:
                        faces.append((res, col))
            if not faces:
                continue
            xs = [p[0] for f in faces for p in f[0]]
            ys = [p[1] for f in faces for p in f[0]]
            projected.append({"obj": obj, "faces": faces, "foot": obj["foot"],
                              "bbox": (min(xs), min(ys), max(xs), max(ys)), "key": key_of(obj["foot"])})
        items = []
        for k in _order_objects(projected, viewer):
            p = projected[k]
            items.extend((f[0], f[1], p["obj"]["tag"]) for f in p["faces"])
        return items

    def _draw_lists(self):
        """Per-view draw lists [(pts, color, tag)] for the current state and camera (camera not modified)."""
        base = self.env.unwrapped
        W, H = self.WINDOW_W, self.WINDOW_H
        scene = base.static_scene()
        car_obj = _car_object(base.car)
        x, y, f, r = self._car_frame()
        S = self.scale_factor
        out = {"top": base.top_items(W, H, self.top_zoom)}

        # ---------------- Rear / Side orthographic section views ----------------
        def ortho(view):
            if view == "rear":
                cut = (f[0], f[1], self.REAR_CUT - (x * f[0] + y * f[1]))          # keep rel.f >= -REAR_CUT
                vdir = (-f[0], -f[1])
                axis_x, axis_d = r, f                                              # screen x, depth (far +)
            else:
                cut = (-r[0], -r[1], self.SIDE_CUT + (x * r[0] + y * r[1]))        # keep rel.r <= SIDE_CUT
                vdir = (r[0], r[1])
                axis_x, axis_d = f, (-r[0], -r[1])
            objs = []
            for obj in scene["objects"]:
                prisms = []
                for pr in obj["prisms"]:
                    foot = _clip_halfplane(pr["foot"], *cut)
                    if len(foot) >= 3:
                        prisms.append(dict(pr, foot=foot))
                if prisms:
                    objs.append(_make_object(obj["tag"], prisms, obj["decals"]))
            objs.append(car_obj)

            def project(verts):
                pts = [(((vx - x) * axis_x[0] + (vy - y) * axis_x[1]) * S + W / 2,
                        (self.camera_height - vz) * S + H / 2) for vx, vy, vz in verts]
                pts = _clip_screen(pts, W, H)
                return pts if len(pts) >= 3 else None

            def key_of(foot):
                return sum((px - x) * axis_d[0] + (py - y) * axis_d[1] for px, py in foot) / len(foot)

            return self._object_items(objs, project,
                                      lambda p, nx, ny: nx * vdir[0] + ny * vdir[1] > 1e-6,
                                      lambda z: False, ("dir", vdir[0], vdir[1]), key_of)

        out["rear"] = ortho("rear")
        out["side"] = ortho("side")

        # ---------------- FPV perspective chase camera ----------------
        ax, ay, az = self._cam_anchor
        cx_w, cy_w, cz_w = self.cam_pos
        yaw, pitch, piv, fov, near = self.cam_yaw, self.cam_pitch, self.camera_dist, self.fov, self.NEAR

        def to_cam(vx, vy, vz):
            dx, dy = vx - ax, vy - ay
            return orbit_camera_frame(dx * f[0] + dy * f[1], dx * r[0] + dy * r[1], vz - az, yaw, pitch, piv)

        def project(verts):
            cv = [to_cam(*v) for v in verts]
            if max(c[0] for c in cv) < near:
                return None
            cv = _clip_near(cv, near)
            if len(cv) < 3:
                return None
            pts = _clip_screen([(W / 2 + c[1] / c[0] * fov, H / 2 - c[2] / c[0] * fov) for c in cv], W, H)
            return pts if len(pts) >= 3 else None

        def key_of(foot):
            mx = sum(p[0] for p in foot) / len(foot)
            my = sum(p[1] for p in foot) / len(foot)
            return math.hypot(mx - cx_w, my - cy_w)

        fpv = []
        for poly, col, tag in scene["ground"]:
            pts = project([(px, py, 0.0) for px, py in poly])
            if pts is not None:
                fpv.append((pts, col, tag))
        fpv += self._object_items(scene["objects"] + [car_obj], project,
                                  lambda p, nx, ny: (cx_w - p[0]) * nx + (cy_w - p[1]) * ny > 1e-6,
                                  lambda z: cz_w > z, ("point", cx_w, cy_w), key_of)
        out["fpv"] = fpv
        out["horizon"] = H / 2 - math.tan(pitch) * fov
        return out

    def _paint_views(self, lists, target=None):
        W, H = self.WINDOW_W, self.WINDOW_H
        mask = target is not None
        surfs = []
        for name in ("top", "rear", "side", "fpv"):
            surf = pygame.Surface((W, H))
            if mask:
                surf.fill((0, 0, 0))
            elif name == "top":
                surf.fill(BG_COLOR)
            else:
                surf.fill(SKY_COLOR)
                gy = H // 2 if name != "fpv" else int(round(min(max(lists["horizon"], -1), H + 1)))
                if gy < H:
                    pygame.draw.rect(surf, BG_COLOR, (0, max(gy, 0), W, H - max(gy, 0)))
            _paint(surf, lists[name], target)
            surfs.append(surf)
        return surfs

    # ------------------------------------------------------------------
    def render(self, camera_pose="Center"):
        self.update_camera_pose(camera_pose)
        base = self.env.unwrapped
        if getattr(base, "car", None) is None:
            blank = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
            return blank, blank, blank, blank
        self.cam_x, self.cam_y, self.cam_z = self.cam_pos
        surfs = self._paint_views(self._draw_lists())
        return tuple(_surf_to_img(s) for s in surfs)

    def render_masks(self, target):
        """Binary VISIBLE masks (top, rear, side, fpv), each VIDEO_H x VIDEO_W uint8 in {0, 1}: the full
        scene painted with the same projection and draw order as render(), `target` white, rest black.
        target: "obstacle" | "building" | "car" | "victim" | an obstacle kind ("roadblock" | "bus" |
        "debris") | int obstacle index (info["obstacles"][k]["idx"]). Does not change the camera state."""
        base = self.env.unwrapped
        if getattr(base, "car", None) is None:
            blank = np.zeros((VIDEO_H, VIDEO_W), np.uint8)
            return blank, blank, blank, blank
        surfs = self._paint_views(self._draw_lists(), target)
        return tuple(np.where(_surf_to_img(s)[..., 0] >= 128, 1, 0).astype(np.uint8) for s in surfs)


# Main demo
if __name__ == "__main__":
    import os
    import sys
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    pygame.init()

    env = RescueAmbulanceEnv()
    wrapper = RescueAmbulanceOrthographicWrapper(env)
    obs, info = wrapper.reset(seed=0)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Victim at {info['victim_pos']}, road distance {info['dist_to_victim']:.1f} m, "
          f"{len(info['obstacles'])} obstacles")

    display = None if headless else pygame.display.set_mode((VIDEO_W * 2, VIDEO_H * 2))
    if display is not None:
        pygame.display.set_caption("Rescue Ambulance - Orthographic Views")
    clock = pygame.time.Clock()

    total_reward = 0.0
    for step in range(500):
        if display is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        # Simple route follower: steer toward the next waypoint (obs 10-11), moderate gas
        steer = float(np.clip(2.0 * obs[10], -1, 1))
        action = np.array([steer, 0.3 if obs[0] < 0.25 else 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapper.step(action)
        total_reward += reward

        if step % 5 == 0 and display is not None:
            views = wrapper.render()
            for k, im in enumerate(views):
                display.blit(pygame.surfarray.make_surface(im.transpose(1, 0, 2)),
                             ((k % 2) * VIDEO_W, (k // 2) * VIDEO_H))
            pygame.display.flip()
            clock.tick(30)

        if terminated or truncated:
            print(f"Episode ended at step {step}, total reward: {total_reward:.2f}, arrived: {info['arrived']}")
            obs, info = wrapper.reset()
            total_reward = 0.0

    env.close()
    pygame.quit()
