"""
godot_bridge.py — Streams GymBox env state to a Godot renderer over TCP.

Protocol: newline-delimited JSON, Python is the server, Godot connects as the client.

  Python -> Godot
    {"type": "track", "seed", "width", "playfield", "center": [[x, y, beta], ...],
     "curbs": [[x1, y1, x2, y2, x3, y3, x4, y4, is_red], ...]}           once per reset
    {"type": "state", "step", "t", "reward", "ret", "lap", "cars": [car, ...]}  every step
      car = {"name", "color": [r, g, b], "x", "y", "a", "vx", "vy",
             "gas", "brake", "progress",
             "wheels": [[steer, phase, skid, on_grass], ...]}   (FL, FR, RL, RR)

  Godot -> Python
    {"type": "input", "steer", "gas", "brake"}   latest player input, sent every frame
    {"type": "reset"}                            player asked for a new episode

Coordinates are Box2D world units; Godot maps (x, y) -> Vector3(x, height, -y) and
yaw = hull angle, so no conversion happens on the Python side.
"""
import json
import socket

import numpy as np
from gymnasium.envs.box2d.car_racing import PLAYFIELD, TRACK_WIDTH

EGO_COLOR = (0.8, 0.08, 0.06)


# ---------------------------------------------------------------------- state export
def export_track(env, seed=None):
    base = env.unwrapped
    curbs = []
    for poly, color in base.road_poly:
        if isinstance(color, tuple):  # border kerbs are (255,255,255)/(255,0,0); road tiles are arrays
            curbs.append([float(c) for xy in poly for c in xy] + [1.0 if color[1] == 0 else 0.0])
    return {
        "type": "track",
        "seed": -1 if seed is None else int(seed),
        "width": float(TRACK_WIDTH),
        "playfield": float(PLAYFIELD),
        "center": [[float(t[2]), float(t[3]), float(t[1])] for t in base.track],
        "curbs": curbs,
    }


def export_car(name, car, color, progress=0.0):
    hull = car.hull
    wheels = []
    for w in car.wheels:
        skid = w.skid_start is not None or w.skid_particle is not None
        wheels.append([float(w.joint.angle), float(w.phase), int(skid), int(len(w.tiles) == 0)])
    return {
        "name": name,
        "color": list(color),
        "x": float(hull.position[0]),
        "y": float(hull.position[1]),
        "a": float(hull.angle),
        "vx": float(hull.linearVelocity[0]),
        "vy": float(hull.linearVelocity[1]),
        "gas": float(car.wheels[2].gas),
        "brake": float(car.wheels[0].brake),
        "progress": float(progress),
        "wheels": wheels,
    }


def export_state(env, step, ep_return):
    base = env.unwrapped
    progress = getattr(base, "progress", {})
    cars = [export_car("ego", base.car, EGO_COLOR, progress.get("ego", 0.0))]
    opp = getattr(base, "opponent_car", None)
    if opp is not None:
        color = tuple(float(c) for c in opp.hull.color)
        cars.append(export_car("opponent", opp, color, progress.get("opponent", 0.0)))
    return {
        "type": "state",
        "step": int(step),
        "t": float(base.t),
        "reward": float(base.reward - base.prev_reward),
        "ret": float(ep_return),
        "lap": float(base.tile_visited_count / max(len(base.track), 1)),
        "cars": cars,
    }


# ---------------------------------------------------------------------- transport
class GodotBridge:
    """Single-client TCP server. Non-blocking after the client connects."""

    def __init__(self, host="127.0.0.1", port=9876):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.conn = None
        self._buf = b""
        self.latest_input = {"steer": 0.0, "gas": 0.0, "brake": 0.0}

    @property
    def connected(self):
        return self.conn is not None

    def wait_for_client(self, timeout=None):
        """Blocks until Godot connects (or timeout seconds pass). Returns True if connected."""
        self.server.settimeout(timeout)
        try:
            conn, _ = self.server.accept()
        except socket.timeout:
            return False
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setblocking(False)
        self.conn, self._buf = conn, b""
        self.latest_input = {"steer": 0.0, "gas": 0.0, "brake": 0.0}
        return True

    def send(self, msg):
        if self.conn is None:
            return
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        try:
            self.conn.setblocking(True)  # a track message can exceed the socket buffer
            self.conn.sendall(data)
            self.conn.setblocking(False)
        except OSError as e:
            self._drop(f"send failed: {e}")

    def poll(self):
        """Reads pending messages. Returns the list of non-input messages (e.g. resets)."""
        events = []
        if self.conn is None:
            return events
        while True:
            try:
                chunk = self.conn.recv(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError as e:
                self._drop(f"receive failed: {e}")
                return events
            if not chunk:
                self._drop("Godot closed the connection (window closed or Godot crashed - check Godot's console output)")
                return events
            self._buf += chunk
        *lines, self._buf = self._buf.split(b"\n")
        for line in lines:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("type") == "input":
                self.latest_input = msg
            else:
                events.append(msg)
        return events

    def action(self):
        a = self.latest_input
        return np.array([a.get("steer", 0.0), a.get("gas", 0.0), a.get("brake", 0.0)], dtype=np.float64)

    def _drop(self, reason=None):
        if reason:
            print(f"[bridge] {reason}")
        try:
            self.conn.close()
        except OSError:
            pass
        self.conn = None

    def close(self):
        if self.conn is not None:
            self._drop()
        self.server.close()
