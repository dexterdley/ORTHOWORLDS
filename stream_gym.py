import socket
import json
import time
import math
import gymnasium as gym

UDP_IP = "127.0.0.1"
UDP_PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 1 Gym meter = 6 Godot units for great cinematic scale
SCALE = 6.0

env = gym.make("LunarLander-v3", render_mode=None)
obs, info = env.reset()

print(f"Streaming Gym states to Godot on {UDP_IP}:{UDP_PORT}...")

try:
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        payload = {
            "x": float(obs[0] * SCALE),
            "y": float(obs[1] * SCALE),
            "angle": float(obs[4] * (180.0 / math.pi)),
            "main_engine": bool(action == 2),
            "left_engine": bool(action == 1),
            "right_engine": bool(action == 3),
            "left_contact": bool(obs[6] > 0),
            "right_contact": bool(obs[7] > 0)
        }

        sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))

        if terminated or truncated:
            obs, info = env.reset()

        time.sleep(0.02)

except KeyboardInterrupt:
    print("Stopped.")
finally:
    env.close()
    sock.close()