"""
controls.py — Keyboard bindings for live hi-fi play (hifi/run_hifi.py).

Same bindings as ortho/play_env.py (copied here because importing play_env pulls in the training
stack). Godot sends the pressed keys by name ("a".."z", "space"); arrows orbit the 3D FPV camera.
"""
import numpy as np

# Per-env: list of (action_dim, key_positive, key_negative); a None negative = button (+1 while held)
CONTROLS = {
    "bipedal_walker": [(0, "q", "a"), (1, "w", "s"), (2, "e", "d"), (3, "r", "f")],
    "multicar_racing": [(0, "d", "a"), (1, "w", None), (2, "s", None)],
    "mario_escape": [(0, "d", "a"), (1, "space", None), (1, "w", None)],
    "drone_dogfight": [(0, "d", "a"), (1, "w", "s"), (3, "space", None)],  # dim 2 (aim) uses J/L
    "excavator": [(0, "d", "a"), (1, "w", "s"), (2, "i", "k"), (3, "l", "j")],
    "tank_duel": [(0, "w", "s"), (1, "a", "d"), (2, "j", "l"), (3, "space", None)],
    "catapult_war": [(0, "w", "s"), (1, "i", "k"), (2, "space", None)],
    "robot_sumo": [(0, "w", "s"), (1, "a", "d"), (2, "j", "l")],
    "mountain_goat": [(0, "e", "q"), (1, "s", "w"), (2, "d", "a"), (3, "k", "i"), (4, "l", "j")],
    "wrecking_ball": [(0, "d", "a"), (1, "w", "s"), (2, "i", "k")],
    "toxic_gas_escape": [(0, "w", "s"), (1, "a", "d")],
    "rescue_ambulance": [(0, "d", "a"), (1, "w", None), (2, "s", None)],
    "parkour_runner": [(0, "q", "a"), (1, "w", "s"), (2, "e", "d"), (3, "r", "f")],
    "snowball_fight": [(0, "d", "a"), (1, "w", "s"), (2, "q", "e"), (3, "space", None)],
}

HELP = {
    "bipedal_walker": "Q/A hip1, W/S knee1, E/D hip2, R/F knee2",
    "lunar_lander": "W main engine, A left engine, D right engine",
    "multicar_racing": "A/D steer, W gas, S brake",
    "mario_escape": "A/D run, Space or W jump",
    "drone_dogfight": "WASD thrust, J/L turn aim, Space fire",
    "excavator": "A/D drive, W/S boom, I/K dipper, J/L bucket (L curl, J dump)",
    "tank_duel": "W/S drive, A/D hull turn, J/L turret, Space fire",
    "catapult_war": "W/S winch tension, I/K aim up/down, Space fire",
    "robot_sumo": "W/S drive, A/D turn, J/L plow",
    "mountain_goat": "Q/E torso pitch, W/S + A/D legs A (hip, knee), I/K + J/L legs B (hip, knee)",
    "wrecking_ball": "A/D drive, W/S boom, I/K winch (I reel in)",
    "toxic_gas_escape": "W/S throttle, A/D steer",
    "rescue_ambulance": "A/D steer, W gas, S brake (hold S when stopped to reverse)",
    "parkour_runner": "Q/A hip1, W/S knee1, E/D hip2, R/F knee2",
    "snowball_fight": "W/S forward/back along the facing, A/D strafe, Q/E turn, Space throw",
}

AIM_KEYS = ("j", "l")   # Drone Dogfight: rotate the aim heading left / right
AIM_RATE = 0.02         # aim heading change per step, in units of 180 deg


def read_action(env_name, env, keys, state):
    """Action for the pressed key names (as ortho/play_env.py builds it)."""
    keys = set(keys)
    space = env.action_space
    if hasattr(space, "n"):  # Discrete (Lunar Lander): 0 idle, 1 left, 2 main, 3 right
        if "w" in keys:
            return 2
        if "a" in keys:
            return 1
        if "d" in keys:
            return 3
        return 0
    action = np.zeros(space.shape, dtype=np.float32)
    for dim, pos, neg in CONTROLS.get(env_name, []):
        if dim < action.size:
            value = float(pos in keys) - float(neg is not None and neg in keys)
            if value != 0.0:
                action[dim] = value
    if env_name == "drone_dogfight":
        state["aim"] = state.get("aim", 0.0) + AIM_RATE * (float(AIM_KEYS[0] in keys) - float(AIM_KEYS[1] in keys))
        state["aim"] = (state["aim"] + 1.0) % 2.0 - 1.0
        action[2] = state["aim"]
    return np.clip(action, space.low, space.high)
