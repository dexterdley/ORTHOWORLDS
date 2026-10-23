"""
wrappers.py — Drop-in hi-fi versions of the GymBox orthographic envs.

HiFiOrthographicWrapper keeps the GymBox env (physics, rewards, seeds, camera state) and swaps
only the pixels: render(camera_pose) returns the same (top, rear, side, fpv) tuple of HxWx3 uint8
frames, same shapes, rendered by Godot. cam_x/cam_y/cam_z/cam_yaw/cam_pitch/cam_roll match the
low-fi wrapper's, so camera dicts logged by collect_rollouts stay comparable.

  from wrappers import make_hifi_env
  env = make_hifi_env("tank duel")            # starts Godot (see renderer.py for options)
  obs, _ = env.reset(seed=0)
  top, rear, side, fpv = env.render(camera_pose="Left")

pair=True also runs the low-fi render() each call (env.last_lowfi) for paired data; the camera
state then advances through the wrapper's own render(), exactly as in GymBox.
aov=("rgb", "mask") also renders object-ID masks (env.last_masks, env.render_masks(target)).

Registries mirror ortho/utils.py: build_hifi_registry() (training envs) and
build_hifi_heldout_registry(obstacles) (held-out obstacle-transfer envs).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exporters import ENVS, get_exporter, resolve  # noqa: E402
from exporters.base import VIEWS  # noqa: E402
from renderer import GodotRenderer, mask_ids  # noqa: E402


class HiFiOrthographicWrapper:
    def __init__(self, exporter, renderer=None, aov=("rgb",), pair=False, **renderer_kwargs):
        self.exporter = exporter
        self.env = exporter.make_env()
        self._own_renderer = renderer is None
        self.renderer = renderer if renderer is not None else GodotRenderer(**renderer_kwargs)
        self.aov = tuple(aov)
        self.pair = pair
        self.last_lowfi = None
        self.last_masks = None
        self.cam_x = self.cam_y = self.cam_z = 0.0
        self._step = 0
        self._scene = None

    # ------------------------------------------------------------------ gym API
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        if name in ("env", "exporter", "renderer"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._step = 0
        self.exporter.on_reset(self.env)
        self.sync_scene()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._step += 1
        self.exporter.on_step(self.env, action)
        return out

    def sync_scene(self):
        r = self.renderer
        if (r.env_name, tuple(getattr(r, "aov", ())), getattr(r, "mode", None)) != \
                (self.exporter.name, self.aov, "lockstep"):
            r.hello(self.exporter.name, self.exporter.size, self.aov, mode="lockstep", fps=self.exporter.fps)
        self._scene = self.exporter.scene(self.env)
        r.scene(self._scene)

    # ------------------------------------------------------------------ rendering
    def state_message(self, camera_pose="Center"):
        """Advances the camera like the GymBox render() would and builds the state message."""
        if self.pair:
            self.last_lowfi = self.env.render(camera_pose=camera_pose)
        else:
            self.exporter.advance_camera(self.env, camera_pose)
        self.cam_x, self.cam_y, self.cam_z = self.exporter.cam_log(self.env)
        msg = self.exporter.state(self.env)
        msg.setdefault("step", self._step)
        msg.setdefault("t", round(self._step / float(self.exporter.fps), 4))
        msg["cams"] = self.exporter.cameras(self.env)
        return msg

    def render(self, camera_pose="Center"):
        if self._scene is None:
            self.sync_scene()
        self.renderer.state(self.state_message(camera_pose))
        frames = self.renderer.render()
        out = tuple(self._fit(frames[(v, "rgb")]) for v in VIEWS)
        if "mask" in self.aov:
            self.last_masks = {v: self._fit(frames[(v, "mask")], nearest=True) for v in VIEWS}
        return out

    def _fit(self, img, nearest=False):
        w, h = self.exporter.view_size
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def render_masks(self, target):
        """Binary uint8 masks (one per view) of the objects matching target, from the last render.
        target: object id, class name, exporter.mask_groups alias, int index (-> id "obs<i>"), or a list."""
        if self.last_masks is None:
            raise RuntimeError("render with aov=('rgb', 'mask') first")
        targets = list(target) if isinstance(target, (list, tuple, set)) else [target]
        groups = getattr(self.exporter, "mask_groups", {})
        targets = [t for g in targets for t in (groups.get(g, [g]) if isinstance(g, str) else [g])]
        targets = [f"obs{t}" if isinstance(t, (int, np.integer)) else t for t in targets]
        objs, classes = self.renderer.mask_table.get("objects", {}), self.renderer.mask_table.get("classes", {})
        want_idx = {objs[t] for t in targets if t in objs}
        want_cls = {classes[t] for t in targets if t in classes}
        out = []
        for v in VIEWS:
            idx, cls = mask_ids(self.last_masks[v])
            m = np.isin(idx, list(want_idx)) | np.isin(cls, list(want_cls))
            out.append(m.astype(np.uint8))
        return out

    def close(self):
        try:
            self.env.close()
        finally:
            if self._own_renderer:
                self.renderer.close()


# ---------------------------------------------------------------------- factories / registries
_shared_renderer = None


def shared_renderer(**kwargs):
    """One Godot process reused by every env built from the registries (it switches worlds)."""
    global _shared_renderer
    if _shared_renderer is None or not _shared_renderer.alive():
        _shared_renderer = GodotRenderer(**kwargs)
    return _shared_renderer


def make_hifi_env(name, renderer=None, aov=("rgb",), pair=False, env_kwargs=None, **renderer_kwargs):
    exporter = get_exporter(resolve(name), **(env_kwargs or {}))
    return HiFiOrthographicWrapper(exporter, renderer=renderer, aov=aov, pair=pair, **renderer_kwargs)


def build_hifi_registry(renderer=None, aov=("rgb",), **renderer_kwargs):
    """[(display_name, factory)] for the training envs, like ortho/utils.py _build_registry."""
    out = []
    for name, (title, _, _) in ENVS.items():
        if get_exporter(name).heldout:
            continue
        out.append((title, _factory(name, {}, renderer, aov, renderer_kwargs)))
    return out


def build_hifi_heldout_registry(obstacles=True, renderer=None, aov=("rgb",), **renderer_kwargs):
    """Held-out obstacle-transfer envs, like ortho/utils.py _build_heldout_registry."""
    out = []
    for name, (title, _, _) in ENVS.items():
        if not get_exporter(name).heldout:
            continue
        out.append((title, _factory(name, {"obstacles": obstacles}, renderer, aov, renderer_kwargs)))
    return out


def _factory(name, env_kwargs, renderer, aov, renderer_kwargs):
    def make():
        r = renderer if renderer is not None else shared_renderer(**renderer_kwargs)
        return HiFiOrthographicWrapper(get_exporter(name, **env_kwargs), renderer=r, aov=aov)
    return make
