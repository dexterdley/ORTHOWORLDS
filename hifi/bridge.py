"""
bridge.py — TCP link between the GymBox envs (Python) and the Godot hi-fi renderer (protocol v2).

Python is the server; Godot (hifi/godot) connects as the client. Messages are newline-delimited
JSON in both directions, plus one binary payload: a "frames" header line is followed by exactly
the number of raw bytes it announces.

  Python -> Godot
    {"type": "hello", "protocol": 2, "env", "mode": "lockstep"|"live", "size": [w, h],
     "aov": ["rgb", "mask"], "fps"}                                   select env world / views
    {"type": "scene", ...}                                            once per reset (static content)
    {"type": "state", "step", "t", "xf": {id: [x, y, z, qx, qy, qz, qw]}, "dyn": [{id, k, xf, ...}],
     "p": {...}, "cams": [cam x 4], "vis": {view: [hidden ids]}, "hud": {...}}   every step
         cam = {"view", "proj": "o"|"p", "pos", "look", "up", "size"|"fov", "near", "far"}
    {"type": "render", "id"}                                          lockstep: reply "frames"
    {"type": "probe", "id", "points": [[x, y, z], ...]}               reply "probe"
    {"type": "quit"}

  Godot -> Python
    {"type": "ready", "env", "renderer", "device", "mask": {"objects": {id: idx}, "classes": {name: code}}}
    {"type": "frames", "id", "w", "h", "items": [{"view", "aov", "bytes"}]}  + raw RGB8 bytes
    {"type": "probe", "id", "pixels": {view: [[u, v, behind], ...]}}
    {"type": "input", "keys": [...], "cam": "Center"|"Left"|"Right"|"Up"|"Down"}   live mode
    {"type": "reset"} / {"type": "bye"}

All positions are Godot world coordinates (metres, Y up); the exporters do every conversion.
"""
import json
import select
import socket
import time

import numpy as np

PROTOCOL = 2


class BridgeError(RuntimeError):
    pass


class GodotBridge:
    """Single-client TCP server speaking protocol v2."""

    def __init__(self, host="127.0.0.1", port=9876):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.conn = None
        self._buf = bytearray()
        self.pending = []          # messages received while waiting for something else
        self.latest_input = {"keys": [], "cam": "Center"}

    @property
    def connected(self):
        return self.conn is not None

    def wait_for_client(self, timeout=None):
        self.server.settimeout(timeout)
        try:
            conn, _ = self.server.accept()
        except socket.timeout:
            return False
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setblocking(True)
        self.conn, self._buf = conn, bytearray()
        self.pending = []
        return True

    # ------------------------------------------------------------------ send
    def send(self, msg):
        if self.conn is None:
            raise BridgeError("Godot is not connected")
        data = (json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n").encode()
        try:
            self.conn.sendall(data)
        except OSError as e:
            self._drop()
            raise BridgeError(f"send failed: {e}") from e

    # ------------------------------------------------------------------ receive
    def _fill(self, timeout):
        """Reads whatever is available (waiting up to timeout s). Returns False on timeout."""
        r, _, _ = select.select([self.conn], [], [], timeout)
        if not r:
            return False
        try:
            chunk = self.conn.recv(1 << 20)
        except OSError as e:
            self._drop()
            raise BridgeError(f"receive failed: {e}") from e
        if not chunk:
            self._drop()
            raise BridgeError("Godot closed the connection (window closed or crashed - see Godot's output)")
        self._buf += chunk
        return True

    def _read_line(self, deadline):
        while True:
            i = self._buf.find(b"\n")
            if i >= 0:
                line = bytes(self._buf[:i])
                del self._buf[:i + 1]
                return line
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            if left == 0.0 or not self._fill(left):
                return None

    def _read_exact(self, n, deadline):
        while len(self._buf) < n:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            if left == 0.0 or not self._fill(left):
                raise BridgeError(f"timed out reading a {n}-byte payload")
        data = bytes(self._buf[:n])
        del self._buf[:n]
        return data

    def recv(self, timeout=None):
        """Returns the next message (dict), or None on timeout. Frames carry their payload in msg['_data']."""
        if self.conn is None:
            raise BridgeError("Godot is not connected")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            line = self._read_line(deadline)
            if line is None:
                return None
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("type") == "frames":
                total = sum(int(it["bytes"]) for it in msg.get("items", []))
                msg["_data"] = self._read_exact(total, None if deadline is None else deadline + 30.0)
            if msg.get("type") == "input":
                self.latest_input = msg
                continue
            return msg

    def wait_for(self, mtype, timeout=60.0, match=None):
        """Blocks until a message of type mtype arrives (other messages are queued in self.pending)."""
        for i, m in enumerate(self.pending):
            if m.get("type") == mtype and (match is None or match(m)):
                return self.pending.pop(i)
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise BridgeError(f"timed out after {timeout:.0f}s waiting for '{mtype}' from Godot")
            m = self.recv(left)
            if m is None:
                continue
            if m.get("type") == mtype and (match is None or match(m)):
                return m
            if m.get("type") == "error":
                raise BridgeError(f"Godot error: {m.get('message')}")
            self.pending.append(m)

    def poll(self):
        """Non-blocking: returns all queued and newly arrived non-input messages."""
        out, self.pending = self.pending, []
        if self.conn is None:
            return out
        while True:
            m = self.recv(0.0)
            if m is None:
                return out
            out.append(m)

    def _drop(self):
        try:
            self.conn.close()
        except OSError:
            pass
        self.conn = None

    def close(self):
        if self.conn is not None:
            try:
                self.send({"type": "quit"})
            except BridgeError:
                pass
            self._drop()
        self.server.close()


def decode_frames(msg, size_override=None):
    """Turns a 'frames' message into {(view, aov): HxWx3 uint8 array}."""
    w, h = int(msg["w"]), int(msg["h"])
    out, off, data = {}, 0, msg["_data"]
    for it in msg["items"]:
        n = int(it["bytes"])
        arr = np.frombuffer(data, dtype=np.uint8, count=n, offset=off).reshape(h, w, 3)
        out[(it["view"], it["aov"])] = arr
        off += n
    return out
