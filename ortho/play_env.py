"""
play_env.py — Play any registered orthographic game with the keyboard.

Shows all four views (Top, Rear, Side, 3D FPV) in a 2x2 grid; the whole window is
--width x --height (default 600x400, so each view is 300x200).

Usage (from the `ortho` directory):
  python play_env.py --env catapult
  python play_env.py --env "tank duel" --fps 30
  python play_env.py --env snowball --no-obstacles   (held-out env, obstacle-free twin)

Common keys:
  Arrow keys  orbit the 3D camera (Left/Right/Up/Down); release to recentre
  R           reset the episode
  Esc         quit
Game keys are printed at start-up (see CONTROLS below).
"""
import sys
import argparse

import numpy as np
import cv2
import pygame

from utils import _build_registry, _build_heldout_registry

VIEW_NAMES = ["Top", "Rear", "Side", "3D FPV"]  # env.render() return order

# Per-game key bindings: list of (action_dim, key_positive, key_negative) for continuous games.
# Keys are pygame key names. Fire/jump style buttons use (dim, key, None) and output +1 while held.
CONTROLS = {
    "bipedal walker": [(0, "q", "a"), (1, "w", "s"), (2, "e", "d"), (3, "r", "f")],
    "multicar racing": [(0, "d", "a"), (1, "w", None), (2, "s", None)],
    "mario escape": [(0, "d", "a"), (1, "space", None), (1, "w", None)],
    "drone dogfight": [(0, "d", "a"), (1, "w", "s"), (3, "space", None)],  # dim 2 (aim) uses J/L, see AIM_KEYS
    "excavator": [(0, "d", "a"), (1, "w", "s"), (2, "i", "k"), (3, "l", "j")],
    "tank duel": [(0, "w", "s"), (1, "a", "d"), (2, "j", "l"), (3, "space", None)],
    "catapult war": [(0, "w", "s"), (1, "i", "k"), (2, "space", None)],
    "robot sumo": [(0, "w", "s"), (1, "a", "d"), (2, "j", "l")],
    "mountain goat": [(0, "e", "q"), (1, "s", "w"), (2, "d", "a"), (3, "k", "i"), (4, "l", "j")],
    "wrecking ball": [(0, "d", "a"), (1, "w", "s"), (2, "i", "k")],
    "toxic gas escape": [(0, "w", "s"), (1, "a", "d")],
    "rescue ambulance": [(0, "d", "a"), (1, "w", None), (2, "s", None)],
    "parkour runner": [(0, "q", "a"), (1, "w", "s"), (2, "e", "d"), (3, "r", "f")],
    "snowball fight": [(0, "d", "a"), (1, "w", "s"), (2, "q", "e"), (3, "space", None)],
}

HELP = {
    "bipedal walker": "Q/A hip1, W/S knee1, E/D hip2, R/F knee2",
    "lunar lander": "W main engine, A left engine, D right engine",
    "multicar racing": "A/D steer, W gas, S brake",
    "mario escape": "A/D run, Space or W jump",
    "drone dogfight": "WASD thrust, J/L turn aim, Space fire",
    "excavator": "A/D drive, W/S boom, I/K dipper, J/L bucket (L curl, J dump)",
    "tank duel": "W/S drive, A/D hull turn, J/L turret, Space fire",
    "catapult war": "W/S winch tension, I/K aim up/down, Space fire",
    "robot sumo": "W/S drive, A/D turn, J/L plow",
    "mountain goat": "Q/E torso pitch, W/S + A/D legs A (hip, knee), I/K + J/L legs B (hip, knee)",
    "wrecking ball": "A/D drive, W/S boom, I/K winch (I reel in)",
    "toxic gas escape": "W/S throttle, A/D steer",
    "rescue ambulance": "A/D steer, W gas, S brake (hold S when stopped to reverse)",
    "parkour runner": "Q/A hip1, W/S knee1, E/D hip2, R/F knee2",
    "snowball fight": "W/S forward/back along the facing, A/D strafe, Q/E turn, Space throw",
}

AIM_KEYS = ("j", "l")    # Drone Dogfight: rotate the aim heading left / right
AIM_RATE = 0.02          # aim heading change per step, in units of 180 deg


def match_env(registry, query):
    q = query.lower().replace("_", " ").replace("-", " ").strip()
    for name, factory in registry:
        if q == name.lower():
            return name, factory
    hits = [(n, f) for n, f in registry if q in n.lower()]
    if not hits:
        sys.exit(f"No env matches '{query}'. Available: {', '.join(n for n, _ in registry)}")
    return hits[0]


def key_down(keys, name):
    return name is not None and keys[pygame.key.key_code(name)]


def read_action(env_key, env, keys, state):
    """Builds an action for the current key state."""
    space = env.action_space

    if hasattr(space, "n"):  # Discrete (Lunar Lander): 0 idle, 1 left, 2 main, 3 right
        if key_down(keys, "w"):
            return 2
        if key_down(keys, "a"):
            return 1
        if key_down(keys, "d"):
            return 3
        return 0

    action = np.zeros(space.shape, dtype=np.float32)
    for dim, pos, neg in CONTROLS.get(env_key, []):
        if dim < action.size:
            # Several bindings may share a dim (e.g. Space or W to jump); keep whichever is active
            value = float(key_down(keys, pos)) - float(key_down(keys, neg))
            if value != 0.0:
                action[dim] = value

    if env_key == "drone dogfight":
        state["aim"] += AIM_RATE * (float(key_down(keys, AIM_KEYS[0])) - float(key_down(keys, AIM_KEYS[1])))
        state["aim"] = (state["aim"] + 1.0) % 2.0 - 1.0  # wrap to [-1, 1)
        action[2] = state["aim"]

    return np.clip(action, space.low, space.high)


def camera_command(keys):
    if keys[pygame.K_LEFT]:
        return "Left"
    if keys[pygame.K_RIGHT]:
        return "Right"
    if keys[pygame.K_UP]:
        return "Up"
    if keys[pygame.K_DOWN]:
        return "Down"
    return "Center"


def compose_grid(views, cell_w, cell_h, font):
    """Resizes the 4 views into a 2x2 grid surface with labels."""
    grid = np.zeros((cell_h * 2, cell_w * 2, 3), dtype=np.uint8)
    for i, img in enumerate(views[:4]):
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        r, c = divmod(i, 2)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cv2.resize(
            img[..., :3], (cell_w, cell_h), interpolation=cv2.INTER_AREA)

    surf = pygame.surfarray.make_surface(np.transpose(grid, (1, 0, 2)))
    for i, name in enumerate(VIEW_NAMES):
        r, c = divmod(i, 2)
        surf.blit(font.render(name, True, (255, 255, 255), (0, 0, 0)), (c * cell_w + 4, r * cell_h + 4))
    pygame.draw.line(surf, (0, 0, 0), (cell_w, 0), (cell_w, cell_h * 2), 2)
    pygame.draw.line(surf, (0, 0, 0), (0, cell_h), (cell_w * 2, cell_h), 2)
    return surf


def main(args):
    registry = _build_registry() + _build_heldout_registry(obstacles=not args.no_obstacles)
    name, factory = match_env(registry, args.env)
    env_key = name.lower()
    env = factory()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"{name} — R reset, Esc quit")
    # Disable text input so an active IME (e.g. Chinese input) can't swallow Space/letter keys
    pygame.key.stop_text_input()
    font = pygame.font.SysFont("arial", 12)
    clock = pygame.time.Clock()
    cell_w, cell_h = args.width // 2, args.height // 2

    print(f"[{name}] {HELP.get(env_key, 'no key bindings defined')}")
    print("Arrow keys orbit the 3D camera, R resets, Esc quits.")

    obs, _ = env.reset(seed=args.seed)
    state = {"aim": 0.0}
    ep_return, ep_steps, episode = 0.0, 0, 1
    reward = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, _ = env.reset()
                state["aim"] = 0.0
                print(f"Episode {episode} reset by player: return {ep_return:+.2f} in {ep_steps} steps")
                ep_return, ep_steps, episode = 0.0, 0, episode + 1
        if not running:
            break

        keys = pygame.key.get_pressed()
        action = read_action(env_key, env, keys, state)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_steps += 1

        views = env.render(camera_pose=camera_command(keys))
        screen.blit(compose_grid(views, cell_w, cell_h, font), (0, 0))
        hud = f"ep {episode}  step {ep_steps}  r {float(reward):+.2f}  return {ep_return:+.2f}"
        screen.blit(font.render(hud, True, (255, 255, 0), (0, 0, 0)), (4, args.height - 16))
        # Live input readout: shows exactly what the game receives (helps check key bindings)
        act_str = np.array2string(np.atleast_1d(action), precision=1, suppress_small=True)
        base = env.unwrapped
        if hasattr(base, "is_grounded"):
            act_str += f"  grounded {int(base.is_grounded)}"
        screen.blit(font.render(f"action {act_str}", True, (0, 255, 255), (0, 0, 0)), (4, args.height - 32))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode {episode} {'terminated' if terminated else 'truncated'}: "
                  f"return {ep_return:+.2f} in {ep_steps} steps")
            obs, _ = env.reset()
            state["aim"] = 0.0
            ep_return, ep_steps, episode = 0.0, 0, episode + 1

        clock.tick(args.fps)

    env.close()
    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Play an orthographic game with the keyboard (4-view display)")
    parser.add_argument("--env", type=str, default="catapult", help="Env name or substring, e.g. 'catapult', 'tank duel'")
    parser.add_argument("--width", type=int, default=600, help="Window width (each view is width/2)")
    parser.add_argument("--height", type=int, default=400, help="Window height (each view is height/2)")
    parser.add_argument("--fps", type=int, default=30, help="Game steps per second")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the first episode")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="Held-out envs only: play the obstacle-free twin")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
