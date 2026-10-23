"""
renderer.py — Starts the Godot hi-fi renderer and talks to it (lockstep or live).

Godot binary lookup (first hit wins):
  1. the `godot` argument / HIFI_GODOT environment variable
  2. hifi/tools/bin/Godot_*_win64_console.exe   (Windows build: GPU + Forward+ from WSL)
  3. hifi/tools/bin/Godot_*_linux.x86_64
  4. `godot` on PATH
A Windows .exe is launched through WSL interop with Windows paths; it reaches the WSL server via
localhost forwarding, so the server listens on 0.0.0.0 in that case.

display:
  "window"  normal window (default for live play)
  "xvfb"    Linux build only: off-screen X server (xvfb-run), software Vulkan; no window pops up
  "auto"    "window" for a Windows build, "xvfb" for a Linux build when xvfb-run exists
"""
import atexit
import fcntl
import glob
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import weakref

import numpy as np

from bridge import PROTOCOL, BridgeError, GodotBridge, decode_frames

HIFI = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HIFI, "godot")
VIEWS = ("top", "rear", "side", "fpv")


def find_godot(godot=None):
    cands = [godot, os.environ.get("HIFI_GODOT")]
    cands += sorted(glob.glob(os.path.join(HIFI, "tools", "bin", "Godot_*_win64_console.exe")))
    cands += sorted(glob.glob(os.path.join(HIFI, "tools", "bin", "Godot_*_linux.x86_64")))
    cands.append(shutil.which("godot"))
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("No Godot 4.3+ binary found: pass --godot, set HIFI_GODOT, or run hifi/tools/get_godot.py")


def is_windows_binary(path):
    return path.lower().endswith(".exe")


def to_windows_path(path):
    return subprocess.check_output(["wslpath", "-w", path], text=True).strip()


def ensure_class_cache(godot):
    """Godot resolves class_name scripts (EnvWorld, Look, M, ...) through .godot/global_script_class_cache.cfg,
    which only the editor writes. Rebuild it (headless editor import) when any script is newer."""
    cache = os.path.join(PROJECT, ".godot", "global_script_class_cache.cfg")
    # class_name scripts live in core/ and shared/ only (env worlds are loaded by path)
    scripts = [p for d in ("core", "shared") for p in glob.glob(os.path.join(PROJECT, d, "**", "*.gd"), recursive=True)]
    newest = max((os.path.getmtime(p) for p in scripts), default=0.0)
    if os.path.exists(cache) and os.path.getmtime(cache) >= newest:
        return
    os.makedirs(os.path.join(HIFI, "test_out"), exist_ok=True)
    with open(os.path.join(HIFI, "test_out", ".import.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(cache) and os.path.getmtime(cache) >= newest:
            return
        linux = shutil.which("godot") or next(iter(sorted(glob.glob(os.path.join(HIFI, "tools", "bin", "Godot_*_linux.x86_64")))), None)
        exe = linux or godot
        proj = to_windows_path(PROJECT) if is_windows_binary(exe) else PROJECT
        print("[hifi] refreshing the Godot script class cache ...", flush=True)
        subprocess.run([exe, "--headless", "--path", proj, "--import"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=300)
        if os.path.exists(cache):
            os.utime(cache)


_LIVE = weakref.WeakSet()


def _cleanup_all():
    for r in list(_LIVE):
        r.kill()


atexit.register(_cleanup_all)


def _install_sigterm():
    """SIGTERM (e.g. `timeout`) would kill Python without running cleanup and orphan Godot."""
    if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


class GodotRenderer:
    """Owns one Godot process and its bridge. Reusable across envs (send a new hello)."""

    def __init__(self, godot=None, port=0, display="auto", renderer="forward_plus", window=(960, 640),
                 launch=True, host=None, verbose=False, timeout=120.0, log_tag="godot"):
        self.godot = find_godot(godot) if launch else None
        self.windows = bool(self.godot) and is_windows_binary(self.godot)
        if display == "auto":
            display = "window" if (self.windows or not shutil.which("xvfb-run")) else "xvfb"
        self.display = display
        self.timeout = timeout
        bind = host or ("0.0.0.0" if self.windows else "127.0.0.1")
        self.bridge = GodotBridge(bind, port)
        self.port = self.bridge.port
        self.proc = None
        self.info = {}
        self.env_name = None
        self.mask_table = {"objects": {}, "classes": {}}
        self._render_id = 0
        self.log_path = None
        self.log_tag = log_tag
        _LIVE.add(self)
        _install_sigterm()
        if launch:
            ensure_class_cache(self.godot)
            self._launch(renderer, window, verbose)
        print(f"[hifi] waiting for Godot on port {self.port} ...", flush=True)
        if not self.bridge.wait_for_client(timeout):
            self.close()
            raise BridgeError(f"Godot did not connect within {timeout:.0f}s" +
                              (f" (log: {self.log_path})" if self.log_path else ""))

    def _launch(self, renderer, window, verbose):
        proj = to_windows_path(PROJECT) if self.windows else PROJECT
        cmd = [self.godot, "--path", proj]
        if renderer == "compatibility":
            cmd += ["--rendering-method", "gl_compatibility"]
        elif not self.windows:
            cmd += ["--rendering-driver", "vulkan"]
        cmd += ["--resolution", f"{window[0]}x{window[1]}", "--", f"--port={self.port}", "--host=127.0.0.1",
                "--exit-on-disconnect"]
        if self.display == "xvfb":
            cmd = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"] + cmd
        log_dir = os.path.join(HIFI, "test_out", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{self.log_tag}_{self.port}.log")
        out = None if verbose else open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if out else None,
                                     stdin=subprocess.DEVNULL, start_new_session=True)

    # ------------------------------------------------------------------ protocol
    def hello(self, env_name, size, aov=("rgb",), mode="lockstep", fps=50):
        self.bridge.send({"type": "hello", "protocol": PROTOCOL, "env": env_name, "mode": mode,
                          "size": [int(size[0]), int(size[1])], "aov": list(aov), "fps": fps})
        self.env_name, self.size, self.aov, self.mode = env_name, tuple(size), tuple(aov), mode

    def scene(self, msg):
        msg = dict(msg, type="scene")
        self.bridge.send(msg)
        ready = self.bridge.wait_for("ready", self.timeout)
        self.info = ready
        self.mask_table = ready.get("mask", self.mask_table)
        return ready

    def state(self, msg):
        self.bridge.send(dict(msg, type="state"))

    def render(self):
        """Renders the last state; returns {(view, aov): HxWx3 uint8}."""
        self._render_id += 1
        rid = self._render_id
        self.bridge.send({"type": "render", "id": rid})
        msg = self.bridge.wait_for("frames", self.timeout, match=lambda m: m.get("id") == rid)
        if "mask" in msg:
            self.mask_table = msg["mask"]
        return decode_frames(msg)

    def probe(self, points):
        self._render_id += 1
        rid = self._render_id
        self.bridge.send({"type": "probe", "id": rid, "points": [list(map(float, p)) for p in points]})
        return self.bridge.wait_for("probe", self.timeout, match=lambda m: m.get("id") == rid)["pixels"]

    def inspect(self, max_instances=4000):
        """Debug: every rendered geometry with world AABBs, render layers and visibility range
        (EnvWorld.inspect); used by tools/check_views.py."""
        self._render_id += 1
        rid = self._render_id
        self.bridge.send({"type": "inspect", "id": rid, "max_instances": int(max_instances)})
        return self.bridge.wait_for("inspect", self.timeout, match=lambda m: m.get("id") == rid)["nodes"]

    def reveal(self, on):
        """Debug: on=True shows every geometry in all four views (ignores per-view hiding) until off."""
        self._render_id += 1
        rid = self._render_id
        self.bridge.send({"type": "reveal", "id": rid, "on": bool(on)})
        self.bridge.wait_for("revealed", self.timeout, match=lambda m: m.get("id") == rid)

    def poll(self):
        return self.bridge.poll()

    @property
    def latest_input(self):
        return self.bridge.latest_input

    def alive(self):
        return self.bridge.connected and (self.proc is None or self.proc.poll() is None)

    def close(self):
        try:
            self.bridge.close()
        finally:
            if self.proc is not None:
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.kill()
                self.proc = None
        _LIVE.discard(self)

    def kill(self):
        """Hard stop of the Godot process group (xvfb-run, Xvfb and Godot)."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def mask_ids(mask_img):
    """Decodes a mask frame: returns (instance index HxW int32, class code HxW uint8). 0 = background."""
    m = mask_img.astype(np.int32)
    return m[..., 0] + 256 * m[..., 1], mask_img[..., 2]
