"""
Rescue Ambulance (RescueAmbulanceEnv + RescueAmbulanceOrthographicWrapper), top-down, held-out.

GymBox views (rescue_ambulance_orthographic.py): every view is drawn at 1000x800 and cv2-resized
(INTER_AREA) to 600x400, an anisotropic squash. Godot renders at 750x600 (same 5:4 aspect) and the
hi-fi wrapper's resize to view_size reproduces the squash exactly.
  top   car-centred plan view, HEADING-UP (screen up = car forward), TOP_ZOOM = 16.2 px/m on the
        800 px internal height -> ortho size 49.38 m; the car sits at the frame centre
  rear  orthographic elevation from behind the car (looking along its forward axis, screen right =
        car right), 25 px/m (size 32 m), screen y = (8 - z) * 25 + 400 -> frame centre at z = 8 m;
        section cut REAR_CUT = 3 m behind the car centre: the camera's near plane IS that cut plane
  side  orthographic elevation from the car's right (looking along -right, screen right = forward),
        same scale / centre height; section cut SIDE_CUT = 1.6 m right of the car = near plane
  fpv   chase camera camera_dist 12 m behind / camera_height 8 m up, orbiting a pivot 8 m above the
        car (cam_yaw / cam_pitch), focal 500 px on 800 (vfov 77.3 deg), near clip 0.5 m
GymBox quirks fixed: the elevation views' fake horizon at the 8 m camera level (sky above, flat
green below) -> real ground, sky and a soil cross-section below the cut; the section cut is a real
near plane (cut faces are capped per view); painter-sorted prisms -> real depth.
One consistent world: every object (buildings, lamps, trees, the distant skyline, ...) is drawn in
every view whose frame contains it. The only exception is the section cut itself: whatever lies
between an elevation camera and its cut plane (in front of the car) is clipped by the near plane.
The elevation far planes follow the car so that they always reach the far side of the skyline
ring (the ortho sky card sits at 0.97 * far).

Coordinates: Box2D (x, y) at height h -> Godot (x, h, -y); hull angle 0 faces +Y (forward =
(-sin a, cos a), right = (cos a, sin a)), so the car is modelled facing Godot -Z.
"""
import math

from game_envs.rescue_ambulance_orthographic import (
    CITY_HALF, PITCH, PLAYFIELD, ROAD_R, SIDEWALK_W, VICTIM_R, RescueAmbulanceEnv,
    RescueAmbulanceOrthographicWrapper)

from .base import Exporter, P_top, cam_ortho, cam_persp, r, rv, vfov_from_focal, xf_top

WIN_H = 800              # GymBox internal frame height (px)
TOP_ZOOM = 16.2          # px/m (CarRacing ZOOM * SCALE)
ORTHO_S = 25.0           # px/m of the Rear / Side elevations
CAM_H = 8.0              # wrapper camera_height (frame centre of Rear / Side, FPV pivot height)
REAR_CUT = RescueAmbulanceOrthographicWrapper.REAR_CUT
SIDE_CUT = RescueAmbulanceOrthographicWrapper.SIDE_CUT
CUT_BACK = 20.0          # Rear / Side camera distance behind the cut plane (= its near clip)
WHEEL_R = 0.42           # visual wheel radius (Godot world.gd uses the same value)
SKYLINE_R = 352.0        # outer bound of the skyline towers (city.gd: radius <= 330 + half diagonal 21.2)
CARD = 0.97              # the ortho sky card sits at CARD * far


class RescueAmbulanceExporter(Exporter):
    name = "rescue_ambulance"
    title = "Rescue Ambulance"
    fps = 50
    view_size = (600, 400)
    render_size = (750, 600)
    heldout = True
    mask_groups = {"obstacle": ["roadblock", "bus", "debris"]}

    def __init__(self, obstacles=True, **env_kwargs):
        super().__init__(obstacles=obstacles, **env_kwargs)
        self.obstacles = bool(obstacles)
        self.last_action = (0.0, 0.0, 0.0)

    def make_env(self):
        kw = dict(self.env_kwargs)
        kw.pop("obstacles", None)
        return RescueAmbulanceOrthographicWrapper(RescueAmbulanceEnv(obstacles=self.obstacles, **kw))

    def on_reset(self, env):
        self.last_action = (0.0, 0.0, 0.0)

    def on_step(self, env, action):
        try:
            self.last_action = tuple(float(a) for a in list(action)[:3])
        except TypeError:
            self.last_action = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ camera state
    def advance_camera(self, env, command):
        env.update_camera_pose(command)
        env.cam_x, env.cam_y, env.cam_z = env.cam_pos   # render() stores the orbited position

    # ------------------------------------------------------------------ scene
    def scene(self, env):
        b = env.unwrapped
        dashes = []
        for s in range(len(b.seg_nodes)):
            A, u = b.seg_a[s], b.seg_u[s]
            d = ROAD_R + 1.5
            while d + 3.0 <= PITCH - ROAD_R - 1.0:
                c = A + u * (d + 1.5)
                dashes.append([r(c[0]), r(c[1]), int(abs(u[0]) > 0.5)])   # centre, 1 = along x
                d += 6.0
        return {
            "roads": [rv(q) for q in b.road_rects],
            "sidewalks": [rv(q) for q in b.sidewalk_rects],
            "blocks": self._blocks(b),
            "buildings": [[r(d["x0"]), r(d["y0"]), r(d["x1"]), r(d["y1"]), r(d["height"]),
                           [int(c) for c in d["color"]]] for d in b.building_defs],
            "dashes": dashes,
            "victim": rv(b.victim_pos) + [r(VICTIM_R)],
            "victim_dir": self._curb_dir(b),
            "obstacles": [{"idx": int(o["idx"]), "kind": o["kind"], "x": r(o["x"]), "y": r(o["y"]),
                           "angle": r(o["angle"]), "w": r(o["w"]), "h": r(o["h"]),
                           "height": r(o["height"])} for o in b.obstacle_list],
            "nodes": [rv(p) for p in b.node_pos],
            "road_r": ROAD_R, "sidewalk_w": SIDEWALK_W, "city_half": CITY_HALF, "playfield": PLAYFIELD,
            "pitch": PITCH, "wheel_r": WHEEL_R,
        }

    @staticmethod
    def _blocks(b):
        """Block footprints (sidewalk ring outer rects): one paved slab per block."""
        out = []
        sw = b.sidewalk_rects
        for k in range(0, len(sw), 4):
            xs = [q[0] for q in sw[k:k + 4]] + [q[2] for q in sw[k:k + 4]]
            ys = [q[1] for q in sw[k:k + 4]] + [q[3] for q in sw[k:k + 4]]
            out.append([r(min(xs)), r(min(ys)), r(max(xs)), r(max(ys))])
        return out

    @staticmethod
    def _curb_dir(b):
        """Unit vector from the victim towards the nearest road edge (where the casualty lies)."""
        s, _, lat = b._nearest_segment(*b.victim_pos)
        rr = b.seg_r[s]
        sgn = 1.0 if lat >= 0 else -1.0
        return [r(rr[0] * sgn), r(rr[1] * sgn)]

    # ------------------------------------------------------------------ state
    def _frame(self, env):
        car = env.unwrapped.car
        x, y = float(car.hull.position[0]), float(car.hull.position[1])
        a = float(car.hull.angle)
        return x, y, a, (-math.sin(a), math.cos(a)), (math.cos(a), math.sin(a))

    def state(self, env):
        b = env.unwrapped
        if b.car is None:
            return {"xf": {}}
        car = b.car
        hull = car.hull
        xf = {"car": xf_top(hull.position[0], hull.position[1], hull.angle, 0.0)}
        wheels = []
        for i, w in enumerate(car.wheels):
            xf[f"wheel{i}"] = xf_top(w.position[0], w.position[1], w.angle, WHEEL_R)
            wheels.append([r(w.phase, 3), r(w.joint.angle, 4)])
        x, y, a, f, rt = self._frame(env)
        v = hull.linearVelocity
        v_fwd = float(v[0] * f[0] + v[1] * f[1])
        steer, gas, brake = self.last_action
        return {"xf": xf, "p": {
            "wheels": wheels, "v": r(v_fwd, 3), "gas": r(max(gas, 0.0), 3), "brake": r(max(brake, 0.0), 3),
            # car frame for the per-view section caps (cut planes of the Rear / Side cameras)
            "car": [r(x), r(y), r(a, 5)], "rear_cut": REAR_CUT, "side_cut": SIDE_CUT}}

    # ------------------------------------------------------------------ cameras
    def cameras(self, env):
        b = env.unwrapped
        if b.car is None:
            x, y, a, f, rt = 0.0, 0.0, 0.0, (0.0, 1.0), (1.0, 0.0)
        else:
            x, y, a, f, rt = self._frame(env)
        top_size = WIN_H / TOP_ZOOM
        el_size = WIN_H / ORTHO_S
        fwd_g = (f[0], 0.0, -f[1])            # Godot direction of the car's forward axis
        # Rear: camera behind the car looking along +forward; its near plane is the REAR_CUT plane
        rb = REAR_CUT + CUT_BACK
        rear_pos = P_top(x - f[0] * rb, y - f[1] * rb, CAM_H)
        # Side: camera on the car's right looking along -right; near plane = SIDE_CUT plane
        sb = SIDE_CUT + CUT_BACK
        side_pos = P_top(x + rt[0] * sb, y + rt[1] * sb, CAM_H)
        # far planes: past the farthest skyline tower along each viewing axis
        rear_far = (rb + SKYLINE_R - (x * f[0] + y * f[1])) / CARD
        side_far = (sb + SKYLINE_R + (x * rt[0] + y * rt[1])) / CARD
        cx, cy, cz = env.cam_pos
        return [
            cam_ortho("top", P_top(x, y, 40.0), P_top(x, y, 0.0), fwd_g, top_size, 1.0, 46.0),
            cam_ortho("rear", rear_pos, P_top(x, y, CAM_H), (0.0, 1.0, 0.0), el_size, CUT_BACK, rear_far),
            cam_ortho("side", side_pos, P_top(x, y, CAM_H), (0.0, 1.0, 0.0), el_size, CUT_BACK, side_far),
            cam_persp("fpv", P_top(cx, cy, cz), P_top(x, y, CAM_H), (0.0, 1.0, 0.0),
                      vfov_from_focal(env.fov, WIN_H), RescueAmbulanceOrthographicWrapper.NEAR, 1500.0),
        ]

    def hud(self, env):
        b = env.unwrapped
        if b.car is None:
            return {}
        x, y = b.car.hull.position
        d = math.hypot(x - b.victim_pos[0], y - b.victim_pos[1])
        return {"text": "victim %.0f m  obstacles %d" % (d, len(b.obstacle_list))}
