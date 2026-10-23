"""
Parkour Runner (held-out): ParkourRunnerEnv (a BipedalWalker subclass) + TimeLimit(1600) +
ParkourRunnerOrthographicWrapper, built by make_parkour_runner(obstacles=...) as in ortho/utils.py.

The wrapper subclasses BipedalOrthographicWrapper: same SCALE, viewport, cam_pos, update_camera_pose
and projections, so the cameras are the Bipedal Walker ones (see bipedal_walker.py). With
obstacles=False the env is the bit-exact BipedalWalker twin (same terrain); the world renders the
same robot family in the runner palette.

Course (parkour_runner_orthographic.py): obstacles with exact Box2D extents and the KIND_SPECS z
extents (hurdle +-4, crate +-2.5, wall +-9.9, gap +-9.9, crest +-10). Crests and pits are part of
terrain_y; pit lips and boxes are static bodies (obstacle_bodies). The 3D chase camera's crest
occlusion depends on these heights, so they are sent unchanged.

Masks: GymBox render_masks(target) accepts "obstacle", "walker", a kind, an int index or a list.
The walker parts (hull + head, 4 leg links) are class "walker"; obstacle i is id "obs<i>" with its
kind as class (boxes, pit lips, and the crest / pit terrain segments GymBox tags with it).
"""
from game_envs.parkour_runner_orthographic import KIND_SPECS, OBSTACLE_KINDS, make_parkour_runner

from .base import r, rv
from .bipedal_walker import BipedalWalkerExporter


class ParkourRunnerExporter(BipedalWalkerExporter):
    name = "parkour_runner"
    title = "Parkour Runner"
    heldout = True
    mask_groups = {"obstacle": list(OBSTACLE_KINDS)}

    def __init__(self, obstacles=True, **env_kwargs):
        super().__init__(**env_kwargs)
        self.obstacles = bool(obstacles)

    def make_env(self):
        return make_parkour_runner(obstacles=self.obstacles, **self.env_kwargs)

    def scene(self, env):
        msg = super().scene(env)
        b = env.unwrapped
        obs = []
        for ob in getattr(b, "obstacles", []):
            obs.append({"i": int(ob["index"]), "kind": ob["kind"], "x0": r(ob["x0"]), "x1": r(ob["x1"]),
                        "yb": r(ob["y_base"]), "yt": r(ob["y_top"]), "h": r(ob["height"]),
                        "z": rv(KIND_SPECS[ob["kind"]]["z"])})
        boxes = []
        for _body, idx, kind, verts in getattr(b, "obstacle_bodies", []):
            boxes.append({"i": int(idx), "kind": kind, "v": [rv(v) for v in verts],
                          "z": rv(KIND_SPECS[kind]["z"])})
        tags = getattr(b, "terrain_poly_tags", None) or [-1] * (len(b.terrain_x) - 1)
        msg.update({"obstacles": obs, "boxes": boxes, "tags": [int(t) for t in tags]})
        return msg
