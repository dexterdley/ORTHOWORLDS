"""
Parkour Runner: held-out test env for occlusion / obstacle transfer experiments.

Same walker body, physics, 4-d joint-torque action space, 24-d observation
(hull state + joints + 10 lidar rays), reward and termination as gymnasium's
BipedalWalker-v3 (normal mode).  The only novelty is a course of hand-placeable
obstacles laid on top of the normal smooth Bipedal Walker terrain.

Obstacle kinds (all sizes in Box2D world units, TERRAIN_STEP = 14/30 ~= 0.467):
    "hurdle"  low box, 1x1 terrain steps (= hardcore STUMP)    traversable
    "crate"   medium box, 2x2 terrain steps (= big STUMP)      traversable (hard)
    "wall"    tall box, 1 step wide x 4 steps high             NOT traversable:
              there is no jump action, so the walker cannot get over it.  Walls
              are meant for occlusion scenarios (the walker stops / falls before
              it, and the wall hides what lies behind it in the 3D/front views).
    "gap"     pit, 3 steps wide x 4 steps deep (= hardcore PIT) traversable
    "crest"   smooth raised-cosine hill, 24 x 3.5 steps          traversable
              (~11.2 x 1.6 units, max slope ~25 deg).
              Natural occluder for the 3D chase camera: an obstacle placed just
              past a crest is hidden from the default 3D pose while the walker
              approaches, but visible in the Top / Side views (which span 20
              units from hull_x - 4).  E.g. course
                  [{"kind": "crest", "x": 6.6},
                   {"kind": "hurdle", "x": 6.6 + 24 * TERRAIN_STEP + 0.6}]
              puts the hurdle in the Side view but hidden in 3D at reset; it is
              revealed once the hull reaches x ~= 8.3 (part-way up the near slope,
              peak at 12.2): the chase camera trails 8.3 units behind but rises
              with the walker.  Taller crests reveal later (h=5 steps: x ~= 10.7)
              at the cost of steeper slopes; crests < ~3 steps high hide nothing.

"Traversable" means physically passable with the 4 joint torques (hardcore-sized
stumps / pits, walkable slope).  It is not a promise for any given policy: the
gymnasium scripted heuristic gets past a hurdle but fails crates, gaps and crests;
hardcore-trained policies are the reference for those.

Obstacles are static Box2D bodies in category 0x0001, so they collide with the walker
(hull contact = game over, as with the ground) and are seen by the 10 lidar rays.

With obstacles=False the env is the BipedalWalker-v3 twin: identical terrain, RNG stream
and observations for the same seed.  Renders differ on purpose: Parkour Runner uses its own
skin (teal-hoodie runner with a head, concrete ground, dusk sky) instead of the purple robot
on green grass.
"""
import math

import numpy as np
import pygame
from pygame import gfxdraw

import gymnasium
from gymnasium.envs.box2d.bipedal_walker import (
    BipedalWalker, TERRAIN_STEP, TERRAIN_LENGTH, TERRAIN_HEIGHT, TERRAIN_GRASS)

try:
    from .bipedal_orthographic import BipedalOrthographicWrapper
except ImportError:  # executed as a script from inside game_envs/
    from bipedal_orthographic import BipedalOrthographicWrapper

# Course bounds in terrain steps (walker spawns at TERRAIN_STARTPAD / 2 = step 10)
COURSE_MIN_STEP = 14
COURSE_MAX_STEP = TERRAIN_LENGTH - TERRAIN_GRASS - 4
RANDOM_MIN_STEP = 30      # random courses leave a clear run-up after the start pad
RANDOM_MARGIN = 6         # min clear steps between random obstacles

TERRAIN_COLOR = (102, 153, 76)   # BipedalWalker terrain_poly colour (replaced by GROUND_COLOR at reset)

# Parkour skin (urban runner at dusk), deliberately unlike Bipedal Walker's purple robot on green grass.
# Purely visual: physics, observations and rewards are untouched.
GROUND_COLOR = (128, 128, 136)                              # concrete
SKY_COLOR = (250, 222, 190)                                 # dusk
RUNNER_TORSO = ((0, 170, 160), (0, 110, 105))               # teal hoodie (color1, color2)
RUNNER_THIGH = [((45, 55, 80), (30, 38, 58)),               # near / far leg: dark trousers
                ((75, 85, 112), (55, 63, 88))]
RUNNER_SHIN = [((235, 235, 240), (185, 185, 195)),          # near / far leg: light running tights
               ((200, 200, 210), (160, 160, 172))]
RUNNER_HEAD = ((245, 205, 160), (205, 165, 125))            # skin tone
HEAD_LOCAL = [(4 / 30, 9 / 30), (18 / 30, 9 / 30),          # head square on the hull's top front,
              (18 / 30, 23 / 30), (4 / 30, 23 / 30)]        # hull-local units (HULL_POLY / SCALE)

OBSTACLE_KINDS = ("hurdle", "crate", "wall", "gap", "crest")
TRAVERSABLE_KINDS = ("hurdle", "crate", "gap", "crest")

# Per-kind defaults (in terrain steps), random size ranges, extrusion depth (Z, world
# units) and colour families (body / top highlight / side shade) — Mario-like bricks.
KIND_SPECS = {
    "hurdle": dict(w=1, h=1.0, w_rng=(1, 2), h_rng=(1.0, 1.0), z=(-4.0, 4.0),
                   body=(235, 140, 40), top=(255, 190, 95), side=(190, 100, 25)),
    "crate":  dict(w=2, h=2.0, w_rng=(2, 3), h_rng=(1.5, 2.0), z=(-2.5, 2.5),
                   body=(185, 122, 68), top=(210, 148, 90), side=(140, 88, 45)),
    "wall":   dict(w=1, h=4.0, w_rng=(1, 2), h_rng=(4.0, 5.0), z=(-9.9, 9.9),
                   body=(170, 60, 45), top=(205, 95, 70), side=(125, 40, 30)),
    "gap":    dict(w=3, h=4.0, w_rng=(3, 4), h_rng=(4.0, 4.0), z=(-9.9, 9.9),
                   body=(55, 40, 35), top=GROUND_COLOR, side=(40, 30, 25)),
    "crest":  dict(w=24, h=3.5, w_rng=(20, 28), h_rng=(2.5, 3.5), z=(-10.0, 10.0),
                   body=(196, 170, 112), top=(196, 170, 112), side=(196, 170, 112)),
}


class ParkourRunnerEnv(BipedalWalker):
    """
    BipedalWalker (normal mode) whose terrain carries an obstacle course.

    reset(options=...) keys (all optional):
        "course":   list of {"kind", "x", "width", "height"} dicts. "x" is the left
                    (near) edge in world units; width/height in world units default
                    to KIND_SPECS. Boxes and gaps snap to the terrain grid. For "gap",
                    "height" is the pit depth. Ignored when obstacles=False.
        "n_random": number of random obstacles (kinds from random_kinds) used
                    when no course is given.

    info keys (reset and every step):
        hull_pos, hull_angle, obstacles (placed obstacle dicts, world coords),
        next_obstacle_dx, obstacle_contact, obstacle_contact_ids, fell.
    """

    def __init__(self, render_mode=None, obstacles=True, n_random=5,
                 random_kinds=TRAVERSABLE_KINDS):
        super().__init__(render_mode=render_mode, hardcore=False)
        self.obstacles_enabled = bool(obstacles)
        self.n_random = int(n_random)
        self.random_kinds = tuple(random_kinds)
        self.obstacles = []            # placed obstacle dicts (world units)
        self.obstacle_bodies = []      # (body, obstacle_index, kind, verts) for solid boxes
        self.terrain_poly_tags = []    # parallel to terrain_poly: obstacle index or -1
        self._reset_options = {}

    # ------------------------------------------------------------------
    # Course construction
    # ------------------------------------------------------------------
    def _sample_course(self, rng, n):
        """Random course of n obstacles in evenly spaced slots with jitter (steps)."""
        course = []
        if n <= 0 or not self.random_kinds:
            return course
        slot = (COURSE_MAX_STEP - RANDOM_MIN_STEP) / n
        for k in range(n):
            kind = self.random_kinds[int(rng.integers(len(self.random_kinds)))]
            spec = KIND_SPECS[kind]
            w = int(rng.integers(spec["w_rng"][0], spec["w_rng"][1] + 1))
            w = max(1, min(w, int(slot) - RANDOM_MARGIN))
            h = float(rng.uniform(*spec["h_rng"]))
            free = max(0.0, slot - RANDOM_MARGIN - w)
            start = RANDOM_MIN_STEP + k * slot + float(rng.uniform(0.0, free))
            course.append({"kind": kind, "x": start * TERRAIN_STEP,
                           "width": w * TERRAIN_STEP, "height": h * TERRAIN_STEP})
        return course

    def _normalise_course(self, course):
        """Fill defaults, clamp to the course bounds and sort by x."""
        out = []
        for ob in course:
            kind = str(ob["kind"]).lower()
            if kind not in KIND_SPECS:
                raise ValueError(f"Unknown obstacle kind '{kind}', expected one of {OBSTACLE_KINDS}")
            spec = KIND_SPECS[kind]
            w = float(ob.get("width", spec["w"] * TERRAIN_STEP))
            h = float(ob.get("height", spec["h"] * TERRAIN_STEP))
            x = float(ob.get("x", COURSE_MIN_STEP * TERRAIN_STEP))
            lo = COURSE_MIN_STEP * TERRAIN_STEP
            hi = COURSE_MAX_STEP * TERRAIN_STEP - w
            out.append({"kind": kind, "x": min(max(x, lo), max(lo, hi)), "width": w, "height": h})
        return sorted(out, key=lambda o: o["x"])

    def _generate_terrain(self, hardcore):
        # Course RNG is a jumped copy of np_random so the main stream (terrain,
        # clouds, initial push) is consumed exactly as in plain BipedalWalker.
        course_rng = np.random.Generator(self.np_random.bit_generator.jumped())

        # 1. Plain smooth BipedalWalker terrain (creates edges + terrain_poly)
        super()._generate_terrain(False)
        self.obstacles = []
        self.obstacle_bodies = []
        self.terrain_poly_tags = [-1] * len(self.terrain_poly)
        if not self.obstacles_enabled:
            return

        opts = self._reset_options
        if opts.get("course") is not None:
            course = self._normalise_course(opts["course"])
        else:
            course = self._normalise_course(
                self._sample_course(course_rng, int(opts.get("n_random", self.n_random))))
        if not course:
            return

        # 2. Throw away the plain edges, reshape terrain_y, rebuild
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        ty = self.terrain_y
        tx = self.terrain_x

        # Crests first (smooth bumps), so boxes / gaps can sit on top of them
        for ob in course:
            if ob["kind"] != "crest":
                continue
            x0, w, h = ob["x"], ob["width"], ob["height"]
            for i in range(TERRAIN_LENGTH):
                t = (tx[i] - x0) / w
                if 0.0 < t < 1.0:
                    ty[i] += h * 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
            ob["x0"], ob["x1"] = x0, x0 + w
            span = [ty[i] for i in range(TERRAIN_LENGTH) if x0 <= tx[i] <= x0 + w]
            ob["y_base"] = min(span) if span else TERRAIN_HEIGHT
            ob["y_top"] = max(span) if span else TERRAIN_HEIGHT

        # Boxes and gaps: snap to grid and flatten a pad around the footprint
        boxes = []   # (verts [bl, br, tr, tl], obstacle dict, kind spec)
        for ob in course:
            kind = ob["kind"]
            if kind == "crest":
                continue
            spec = KIND_SPECS[kind]
            n = max(1, int(round(ob["width"] / TERRAIN_STEP)))
            i0 = int(round(ob["x"] / TERRAIN_STEP))
            i0 = min(max(i0, COURSE_MIN_STEP), COURSE_MAX_STEP - n - 2)
            x0, x1 = tx[i0], tx[i0] + n * TERRAIN_STEP
            y_flat = ty[i0]
            for i in range(max(0, i0 - 3), min(TERRAIN_LENGTH, i0 + n + 4)):
                ty[i] = y_flat
            h = ob["height"]
            ob["x0"], ob["x1"], ob["width"] = x0, x1, n * TERRAIN_STEP
            if kind == "gap":
                # Hardcore-style pit: points inside dropped, 1-step wall boxes at both lips
                for i in range(i0, i0 + n + 1):
                    ty[i] = y_flat - h
                for wx in (x0 - TERRAIN_STEP, x1):
                    verts = [(wx, y_flat - h), (wx + TERRAIN_STEP, y_flat - h),
                             (wx + TERRAIN_STEP, y_flat), (wx, y_flat)]
                    boxes.append((verts, ob, spec))
                ob["y_base"], ob["y_top"] = y_flat - h, y_flat
            else:
                sink = 0.5 * TERRAIN_STEP  # bury the base so sloped ground never shows a gap
                verts = [(x0, y_flat - sink), (x1, y_flat - sink), (x1, y_flat + h), (x0, y_flat + h)]
                boxes.append((verts, ob, spec))
                ob["y_base"], ob["y_top"] = y_flat, y_flat + h

        # Index + record obstacles in course order
        for idx, ob in enumerate(course):
            spec = KIND_SPECS[ob["kind"]]
            ob["index"] = idx
            ob["x"] = ob["x0"]
            ob["z_min"], ob["z_max"] = spec["z"]
            ob["traversable"] = ob["kind"] in TRAVERSABLE_KINDS
            for key in ("x", "x0", "x1", "width", "height", "y_base", "y_top"):
                ob[key] = float(ob[key])
        self.obstacles = course

        def owner(xa, xb):
            """Obstacle index owning a terrain segment [xa, xb] (crests / pits), else -1."""
            xm = 0.5 * (xa + xb)
            for ob in course:
                if ob["kind"] == "crest" and ob["x0"] < xm < ob["x1"]:
                    return ob["index"]
                if ob["kind"] == "gap" and ob["x0"] - TERRAIN_STEP < xm < ob["x1"] + TERRAIN_STEP:
                    return ob["index"]
            return -1

        # Rebuild edges + terrain_poly exactly like BipedalWalker, tagging owners
        self.terrain_poly = []
        self.terrain_poly_tags = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [(tx[i], ty[i]), (tx[i + 1], ty[i + 1])]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            idx = owner(tx[i], tx[i + 1])
            t.userData = {"obstacle_index": idx}  # userData survives Box2D proxy re-wrapping
            self.terrain.append(t)
            color = TERRAIN_COLOR if idx < 0 else KIND_SPECS[course[idx]["kind"]]["body"]
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
            self.terrain_poly_tags.append(idx)

        # Solid obstacle boxes: category 0x0001 so walker collides and lidar sees them
        for verts, ob, spec in boxes:
            self.fd_polygon.shape.vertices = verts
            t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
            t.color1, t.color2 = spec["body"], spec["side"]
            t.userData = {"obstacle_index": ob["index"]}
            self.terrain.append(t)
            # Keep our own vertex order: Box2D re-orders polygon vertices internally
            self.obstacle_bodies.append((t, ob["index"], ob["kind"], verts))
        self.terrain.reverse()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self._reset_options = dict(options or {})
        obs, _ = super().reset(seed=seed, options=options)
        self._apply_skin()
        return obs, self._get_info()

    def _apply_skin(self):
        """Repaint ground and walker with the Parkour palette (BipedalWalker sets its own at reset)."""
        self.terrain_poly = [(poly, GROUND_COLOR if color == TERRAIN_COLOR else color)
                             for poly, color in self.terrain_poly]
        self.hull.color1, self.hull.color2 = RUNNER_TORSO
        # self.legs = [thigh, shin] for the near leg (i = -1), then the far leg (i = +1)
        for k, leg in enumerate(self.legs):
            side, is_shin = divmod(k, 2)
            leg.color1, leg.color2 = (RUNNER_SHIN if is_shin else RUNNER_THIGH)[side]

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self._get_info())
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        hx, hy = float(self.hull.position.x), float(self.hull.position.y)

        # Obstacles currently touched by hull or legs
        touched = set()
        for body in [self.hull] + self.legs:
            for ce in body.contacts:
                if ce.contact.touching:
                    ud = ce.other.userData
                    idx = ud.get("obstacle_index", -1) if isinstance(ud, dict) else -1
                    if idx >= 0:
                        touched.add(idx)
        touched = sorted(touched)

        next_dx = math.inf
        for ob in self.obstacles:
            if ob["x1"] >= hx:
                next_dx = min(next_dx, max(0.0, ob["x0"] - hx))

        return {
            "hull_pos": (hx, hy),
            "hull_angle": float(self.hull.angle),
            "obstacles": [dict(ob) for ob in self.obstacles],
            "next_obstacle_dx": next_dx,
            # Walking on a crest is just walking on terrain, so crests are excluded here
            "obstacle_contact": any(self.obstacles[i]["kind"] != "crest" for i in touched),
            "obstacle_contact_ids": touched,
            "fell": bool(self.game_over),
        }


def make_parkour_runner(obstacles=True, render_mode="rgb_array", max_episode_steps=1600, **kwargs):
    """ParkourRunnerEnv + the same 1600-step TimeLimit as BipedalWalker-v3, in the 4-view wrapper."""
    env = ParkourRunnerEnv(render_mode=render_mode, obstacles=obstacles, **kwargs)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return ParkourRunnerOrthographicWrapper(env)


class ParkourRunnerOrthographicWrapper(BipedalOrthographicWrapper):
    """
    BipedalOrthographicWrapper with obstacle boxes / pits / crests extruded into the
    same Top, Front(ortho), Side(ortho) and 3D chase views, plus visible-mask rendering.

    Obstacles share terrain's layer so terrain (e.g. a crest) can occlude them; each
    box is depth-sorted as one object (nearest vertex) to avoid painter artefacts with
    the long terrain slabs.  The walker stays on top (layer 2), as in the base wrapper.
    """

    def render(self, camera_pose="Center"):
        # 0. Accumulate camera orientation delta for 3D FPV
        self.update_camera_pose(camera_pose)
        base_env = self.env.unwrapped
        if base_env.screen is None and getattr(base_env, "render_mode", None) == "human":
            return base_env.render()
        return self._render_views(mask_target=None)

    def render_masks(self, target="obstacle"):
        """
        Visible binary masks (uint8 0/1) in the same order / HxW as render():
        (top, front, side, fpv).  target: "obstacle" (all placed obstacles), "walker",
        an obstacle kind, an int obstacle index, or a list/tuple of those.
        Does not move the camera.
        """
        return self._render_views(mask_target=target)

    # ------------------------------------------------------------------
    def _tag_matches(self, tag, target, obstacles):
        if isinstance(target, (list, tuple, set)):
            return any(self._tag_matches(tag, t, obstacles) for t in target)
        if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
            return tag == int(target)
        target = str(target).lower()
        if target == "walker":
            return tag == "walker"
        if not isinstance(tag, int) or tag < 0:
            return False
        if target == "obstacle":
            return True
        return obstacles[tag]["kind"] == target

    def _build_polys(self, base_env):
        """3D polygons as (verts, color, layer, tag, group); group != None => one solid."""
        polys_3d = []

        def extrude(verts, z_min, z_max, c_cap, edge_colors, layer, tag, group):
            V_back = [(vx * self.SCALE, vy * self.SCALE, z_min * self.SCALE) for vx, vy in verts]
            V_front = [(vx * self.SCALE, vy * self.SCALE, z_max * self.SCALE) for vx, vy in verts]
            polys_3d.append((V_back, c_cap, layer, tag, group))
            polys_3d.append((V_front, c_cap, layer, tag, group))
            num_v = len(V_back)
            for i in range(num_v):
                idx_curr, idx_next = i, (i + 1) % num_v
                edge_poly = [V_back[idx_curr], V_back[idx_next], V_front[idx_next], V_front[idx_curr]]
                polys_3d.append((edge_poly, edge_colors(i), layer, tag, group))

        # --- 1. Extrude Terrain (identical to base; crest / pit segments tagged) ---
        TERRAIN_Z_MIN, TERRAIN_Z_MAX = -10.0, 10.0
        tags = getattr(base_env, "terrain_poly_tags", None) or [-1] * len(base_env.terrain_poly)
        for (poly, color), tag in zip(base_env.terrain_poly, tags):
            V_back = [(p[0] * self.SCALE, p[1] * self.SCALE, TERRAIN_Z_MIN * self.SCALE) for p in poly]
            V_front = [(p[0] * self.SCALE, p[1] * self.SCALE, TERRAIN_Z_MAX * self.SCALE) for p in poly]
            polys_3d.append((V_front, color, 0, tag, None))
            polys_3d.append((V_back, color, 0, tag, None))
            num_v = len(V_back)
            for i in range(num_v):
                idx_curr, idx_next = i, (i + 1) % num_v
                connect_poly = [V_back[idx_curr], V_back[idx_next], V_front[idx_next], V_front[idx_curr]]
                polys_3d.append((connect_poly, color, 0, tag, None))

        # --- 2. Extrude Obstacle Boxes (same layer as terrain so crests occlude them) ---
        # Box verts are [bottom-left, bottom-right, top-right, top-left] -> edges 0..3 =
        # bottom, far face, top, near face (static bodies, so stored world verts are exact)
        for k, (body, idx, kind, verts) in enumerate(getattr(base_env, "obstacle_bodies", [])):
            spec = KIND_SPECS[kind]
            z_min, z_max = spec["z"]
            face = {0: spec["side"], 1: spec["side"], 2: spec["top"], 3: spec["body"]}
            extrude(verts, z_min, z_max, spec["body"], face.get, 0, idx, k)

        # --- 3. Extrude Start Flag ---
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
            extrude(verts, -0.5, 0.5, c1, lambda i, c1=c1, c2=c2: c2 if i % 2 == 0 else c1, 1, "flag", None)

        # --- 4. Extrude Robot Parts ---
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
                        verts = [tuple(f.body.transform * v) for v in f.shape.vertices]
                        extrude(verts, z_min, z_max, obj.color1,
                                lambda i, o=obj: o.color2 if i % 2 == 0 else o.color1, 2, "walker", None)

        # --- 5. Runner head (visual only, rides on the hull) ---
        head = [tuple(base_env.hull.transform * v) for v in HEAD_LOCAL]
        extrude(head, -0.6, 0.6, RUNNER_HEAD[0],
                lambda i: RUNNER_HEAD[1] if i % 2 == 0 else RUNNER_HEAD[0], 2, "walker", None)
        return polys_3d

    def _render_views(self, mask_target=None):
        base_env = self.env.unwrapped
        mask_mode = mask_target is not None
        surfs = [pygame.Surface((self.VIEWPORT_W, self.VIEWPORT_H)) for _ in range(4)]
        sky_color = (0, 0, 0) if mask_mode else SKY_COLOR
        for surf in surfs:
            surf.fill(sky_color)

        if base_env.hull is None:
            dummy = np.zeros((self.VIEWPORT_H, self.VIEWPORT_W) + (() if mask_mode else (3,)), dtype=np.uint8)
            return dummy, dummy, dummy, dummy

        scroll_x = base_env.scroll * self.SCALE
        polys_3d = self._build_polys(base_env)
        obstacles = getattr(base_env, "obstacles", [])

        # --- 5. Apply Projections (same maths as BipedalOrthographicWrapper) ---
        projs = [[], [], [], []]   # top, front, side, fpv
        cy, cx = self.VIEWPORT_H / 2, self.VIEWPORT_W / 2
        cam_x, cam_y, cam_z = self.cam_pos
        if not mask_mode:
            self.cam_x, self.cam_y, self.cam_z = cam_x, cam_y, cam_z

        cos_p, sin_p = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        cos_y, sin_y = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        fwd_x, fwd_z, fwd_y = cos_y * cos_p, -sin_y * cos_p, -sin_p
        right_x, right_z, right_y = sin_y, cos_y, 0.0
        up_x, up_z, up_y = -cos_y * sin_p, sin_y * sin_p, cos_p

        for p_vertices, color, layer, tag, group in polys_3d:
            pts_t, pts_f, pts_s, pts_fpv = [], [], [], []
            depths_t, depths_f, depths_s, depths_fpv = [], [], [], []
            valid_fpv = True
            for vx, vy, vz in p_vertices:
                vx_scrolled = vx - scroll_x
                pts_t.append((vx_scrolled, vz + cy))                  # A. Top-Down
                depths_t.append(-vy)
                pts_f.append((vz + cx, self.VIEWPORT_H - vy))         # B. Front Elevation
                depths_f.append(vx_scrolled)
                pts_s.append((vx_scrolled, self.VIEWPORT_H - vy))     # C. Side Elevation
                depths_s.append(vz)
                rx, ry, rz = vx - cam_x, vy - cam_y, vz - cam_z       # D. True 3D Perspective
                dx_c = rx * right_x + ry * right_y + rz * right_z
                dy_c = rx * up_x + ry * up_y + rz * up_z
                d_c = rx * fwd_x + ry * fwd_y + rz * fwd_z
                if d_c < 10.0:  # Near clipping plane
                    valid_fpv = False
                else:
                    px = (dx_c / d_c) * self.fov + cx
                    py = self.VIEWPORT_H - ((dy_c / d_c) * self.fov + cy)
                    pts_fpv.append((px, py))
                    depths_fpv.append(d_c)
            if mask_mode:
                color = (255, 255, 255) if self._tag_matches(tag, mask_target, obstacles) else (0, 0, 0)
            for view, pts, depths, ok in [(0, pts_t, depths_t, True), (1, pts_f, depths_f, True),
                                          (2, pts_s, depths_s, True), (3, pts_fpv, depths_fpv, valid_fpv)]:
                if ok and len(pts) >= 3:
                    projs[view].append([sum(depths) / len(depths), pts, color, layer, group, min(depths)])

        # --- 6. Z-Index Sorting & Draw ---
        # Key (layer, -primary, -own depth): primary is the polygon's own mean depth, or for
        # obstacle boxes the box's nearest-vertex depth so the box sorts as one solid.
        # For ungrouped polygons this reduces to the base wrapper's (layer, -depth) order.
        out_imgs = []
        for proj_list, surf in zip(projs, surfs):
            group_min = {}
            for d, _, _, _, g, dmin in proj_list:
                if g is not None:
                    group_min[g] = min(group_min.get(g, dmin), dmin)
            proj_list.sort(key=lambda e: (e[3], -(group_min[e[4]] if e[4] is not None else e[0]), -e[0]))
            for _, pts, color, _, _, _ in proj_list:
                safe_color = (int(color[0]), int(color[1]), int(color[2]))
                ipts = [(int(round(p[0])), int(round(p[1]))) for p in pts]
                try:
                    gfxdraw.aapolygon(surf, ipts, safe_color)
                    gfxdraw.filled_polygon(surf, ipts, safe_color)
                except Exception:
                    pass
            img = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
            out_imgs.append((img[..., 0] > 127).astype(np.uint8) if mask_mode else img)
        return out_imgs[0], out_imgs[1], out_imgs[2], out_imgs[3]


if __name__ == "__main__":
    import os
    from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    # Crest occlusion scenario: a hurdle hidden just past a hill, then a pit
    course = [
        {"kind": "crest", "x": 6.6},
        {"kind": "hurdle", "x": 6.6 + 24 * TERRAIN_STEP + 0.6},
        {"kind": "gap", "x": 30.0},
        {"kind": "crate", "x": 40.0},
    ]
    wrapper = make_parkour_runner(obstacles=True)
    obs, info = wrapper.reset(seed=0, options={"course": course})
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {wrapper.action_space}")
    for ob in info["obstacles"]:
        print(f"  #{ob['index']} {ob['kind']:6s} x=[{ob['x0']:.2f}, {ob['x1']:.2f}] "
              f"y=[{ob['y_base']:.2f}, {ob['y_top']:.2f}] traversable={ob['traversable']}")

    heuristics = BipedalWalkerHeuristics()
    total_reward = 0.0
    for step in range(1600):
        action = heuristics.step_heuristic(obs)
        obs, reward, terminated, truncated, info = wrapper.step(action)
        total_reward += reward
        if step % 100 == 0:
            views = wrapper.render()
            vis = wrapper.render_masks(1)
            print(f"Step {step}: x={info['hull_pos'][0]:.2f} total={total_reward:.1f} "
                  f"next_dx={info['next_obstacle_dx']:.2f} contact={info['obstacle_contact']} "
                  f"hurdle visible px (top/front/side/3d)={[int(m.sum()) for m in vis]}")
        if terminated or truncated:
            print(f"Episode ended at step {step}: fell={info['fell']} total={total_reward:.1f}")
            break
    wrapper.close()
