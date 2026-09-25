"""
generate_game_env.py
--------------------
Generator for Box2D orthographic game environments based on game_envs_descriptions.md.
Uses FuxiAPI for generation without injecting base code.

Usage:
    cd car_racing
    python generate_game_env.py                   # Interactive selection
    python generate_game_env.py --game "Mountain Goat"  # Specific game
    python generate_game_env.py --list            # List available games
"""

import sys
import os
import re
import argparse
import asyncio
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.fuxi_api import FuxiAPI

DESCRIPTIONS_FILE = os.path.join(HERE, "game_envs", "game_envs_descriptions.md")


def parse_game_descriptions(file_path):
    """Parses game_envs_descriptions.md into structured dict of games."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Description file not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    categories = re.split(r'\n###\s+', content)
    games = []

    for cat_block in categories:
        lines = cat_block.strip().split('\n')
        category_name = lines[0].replace('#', '').strip()
        
        # Match game blocks like "1. Mountain Goat 🐐"
        game_matches = re.split(r'\n(?=\d+\.\s+)', cat_block)
        for g_block in game_matches:
            lines_g = [l.strip() for l in g_block.strip().split('\n') if l.strip()]
            if not lines_g:
                continue
            
            title_match = re.match(r'^\d+\.\s+(.*)', lines_g[0])
            if not title_match:
                continue
            
            game_name = title_match.group(1).strip()
            reward_desc = ""
            action_desc = ""

            for i, line in enumerate(lines_g):
                if "- Reward Objective:" in line:
                    reward_desc = line.split("- Reward Objective:", 1)[1].strip()
                elif "- Action Space:" in line:
                    action_desc = line.split("- Action Space:", 1)[1].strip()

            games.append({
                "category": category_name,
                "name": game_name,
                "reward": reward_desc,
                "action": action_desc,
                "raw": g_block.strip()
            })

    return games


def build_system_prompt(game_info):
    """Constructs a strict specification prompt WITHOUT injecting reference source code."""
    name = game_info["name"]
    category = game_info["category"]
    reward = game_info["reward"]
    action = game_info["action"]

    # Convert game name to valid python identifier (CamelCase & snake_case)
    clean_name = re.sub(r'[^\w\s]', '', name)
    camel_name = "".join(w.capitalize() for w in clean_name.split())
    snake_name = "_".join(w.lower() for w in clean_name.split())

    prompt = f"""You are an expert Gymnasium and Box2D game developer.
Your task is to create a complete, self-contained Python file containing a custom 2D Box2D physics environment and an orthographic multi-view 3D renderer wrapper.

GAME SPECIFICATION:
- Category: {category}
- Game Name: {name}
- Environment Class Name: {camel_name}Env
- Wrapper Class Name: {camel_name}OrthographicWrapper
- Target File Name: {snake_name}_orthographic.py
- Reward Objective: {reward}
- Action Space Description: {action}

STRICT ARCHITECTURAL REQUIREMENTS:
1. Impors & Dependencies:
   - Use `gym` or `gymnasium` (support both via try-except where appropriate).
   - Use `Box2D.b2` for physics (`world`, `polygonShape`, `circleShape`, `revoluteJointDef`, `rayCastCallback`, etc.).
     Box2D SWIG Rules:
     * If performing raycasts with `world.RayCast(cb, p1, p2)`, `cb` MUST be a subclass of `rayCastCallback` implementing `ReportFixture(self, fixture, point, normal, fraction)`. Never pass a raw lambda or function.
     * Always cast float values and pass booleans positionally: e.g. `body.ApplyTorque(float(torque), True)` or `body.ApplyForceToCenter((float(fx), float(fy)), True)`. Do NOT use `wake=True` as a keyword argument (it causes a SWIG TypeError).
     * CRITICAL SWIG FIX: When assigning to Box2D attributes like `body.angle`, `joint.motorSpeed`, etc. from Numpy arrays (e.g. from `action`), you MUST cast the value to a native python `float`. Assigning a `numpy.float32` directly to `.angle` causes `TypeError: argument 3 of type 'float32'`.
   - Use `pygame` and `pygame.gfxdraw` for rendering.
   - Use `numpy` and `math`.

2. Environment Class (`{camel_name}Env(gym.Env)`):
   - Action space (`self.action_space`): `spaces.Box(low=-1.0, high=1.0, shape=(N,), dtype=np.float32)` matching the action space description.
   - Observation space (`self.observation_space`): `spaces.Box` covering all relevant state variables (positions, velocities, angles, targets/hazards, key joint states).
   - `reset(seed=None, options=None)`: Re-creates the Box2D `world`, terrain/boundaries, main character/vehicle body, joint structures, obstacles, and goal entities. Returns `(obs, info)`.
   - `step(action)`: Clips actions, applies forces/torques/motor speeds to Box2D bodies, steps world by `1/50`s. Computes dense rewards matching the objective ({reward}) plus terminal rewards (+100 for success, -50 for failure). Returns `(obs, reward, terminated, truncated, info)`.
   - `_get_obs()`: Returns flat float32 numpy array.
   - `render()`: Renders top-down 2D view using Pygame, returns RGB uint8 array (shape: H x W x 3) when `render_mode == 'rgb_array'`.

3. Orthographic Multi-View Wrapper (`{camel_name}OrthographicWrapper(gym.Wrapper)`):
   - Wraps `{camel_name}Env`.
   - Implements `render()` which MUST return a tuple of 4 RGB uint8 numpy images (each shape: 600x800x3 or 800x600x3):
     `(im_top_down, im_rear, im_side, im_fpv)`
   - Extrudes 2D Box2D shapes (polygons, circles) into 3D bounding prisms/cylinders using Z-extrusion.
   - Projects 3D geometry onto Top-Down, Rear, Side, and FPV views.
   - CRITICAL: Camera functions MUST subtract player/character position (`v[0] - player.x`, `v[1] - player.y`) so the main character remains centered in all viewports instead of being drawn off-screen!
   - FPV CRITICAL: The FPV (First Person View) MUST make sense. Project the world from the perspective of the main character (looking forward). Shift the FPV camera position slightly forward and up so the view is clear. You MUST exclude the player's own geometry from being rendered in the FPV (e.g. using a `no_fpv` flag), and apply near-plane clipping (like Sutherland-Hodgman at Z=0.1) so large polygons spanning the camera don't explode or get dropped completely.
   - Ensure all depth/layer constants (e.g. `Z_NEAR`, `Z_FAR`, depths) are defined at module level, NOT inside local function scopes.
   - Uses painter's algorithm (Z-depth sorting) with Pygame polygon fill & anti-aliased outlines (`gfxdraw.aapolygon`).

4. Aesthetics & Visual Theme:
   - Use vibrant, thematic colors matching '{name}' (e.g. desert hues, neon sci-fi, metallic steel, mud/dirt, ice/snow).
   - Character/vehicle parts, hazards, interactive items, terrain must have clear color distinctions.

DO NOT import any external project files. The generated file must be completely self-contained in a single code block.

Return ONLY valid Python code enclosed in a ```python ... ``` block. No conversational preamble.
"""
    return prompt


def extract_python_code(text):
    """Extracts python code from markdown code blocks."""
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    match_generic = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match_generic:
        return match_generic.group(1)
    return text.strip()


async def generate_env_for_game(game_info, model_name="claude-opus-4-6"):
    name = game_info["name"]
    clean_name = re.sub(r'[^\w\s]', '', name)
    snake_name = "_".join(w.lower() for w in clean_name.split())
    out_filename = f"{snake_name}_orthographic.py"
    out_path = os.path.join(HERE, "game_envs", out_filename)

    print(f"\n🚀 Generating environment for: \033[1;96m{name}\033[0m")
    print(f"   Model: {model_name}")
    print(f"   Output: game_envs/{out_filename}")
    print("   Sending request to FuxiAPI...")

    prompt = build_system_prompt(game_info)
    api = FuxiAPI(model_name=model_name)

    code_raw = await api.get_response(prompt)
    await api.close()

    if code_raw.startswith("❌"):
        print(f"❌ Generation failed: {code_raw}")
        return False

    code = extract_python_code(code_raw)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"✅ Saved code to {out_path} ({len(code)} bytes)")
    
    # Simple syntax and import check
    print("🧪 Testing environment instantiation...")
    try:
        scope = {}
        exec(compile(code, out_filename, 'exec'), scope)
        env_cls = [v for k, v in scope.items() if k.endswith("Env") and isinstance(v, type)][0]
        wrapper_cls = [v for k, v in scope.items() if k.endswith("Wrapper") and isinstance(v, type)][0]
        
        env = wrapper_cls(env_cls())
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        imgs = env.render()
        env.close()

        print(f"✨ Success! Environment '{name}' generated and verified working.")
        print(f"   Action space: {env.action_space}")
        print(f"   Obs shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
        if isinstance(imgs, (tuple, list)) and len(imgs) == 4:
            print(f"   Render outputs 4 views: {[im.shape for im in imgs if im is not None]}")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Created file but test execution failed:\n   {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate new game environments via FuxiAPI.")
    parser.add_argument("--game", type=str, help="Name of game from game_envs_descriptions.md")
    parser.add_argument("--list", action="store_true", help="List all available game descriptions")
    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="FuxiAPI model name")

    args = parser.parse_args()

    games = parse_game_descriptions(DESCRIPTIONS_FILE)

    if args.list:
        print(f"\n📋 Available Game Descriptions ({len(games)} total):\n")
        curr_cat = ""
        for i, g in enumerate(games, 1):
            if g["category"] != curr_cat:
                curr_cat = g["category"]
                print(f"\n--- {curr_cat} ---")
            print(f"  {i:2d}. {g['name']}")
        return

    selected_game = None
    if args.game:
        for g in games:
            if g["name"].lower() == args.game.lower() or args.game.lower() in g["name"].lower():
                selected_game = g
                break
        if not selected_game:
            print(f"❌ Could not find game matching '{args.game}'. Use --list to see options.")
            return
    else:
        # Interactive selection
        print(f"\n🎮 Select a game to generate from {len(games)} descriptions:\n")
        curr_cat = ""
        for i, g in enumerate(games, 1):
            if g["category"] != curr_cat:
                curr_cat = g["category"]
                print(f"\n--- {curr_cat} ---")
            print(f"  [{i:2d}] {g['name']}")

        try:
            choice = input("\nEnter game number (or press Enter for #1 Mountain Goat): ").strip()
            idx = int(choice) - 1 if choice else 0
            selected_game = games[idx]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

    asyncio.run(generate_env_for_game(selected_game, model_name=args.model))


if __name__ == "__main__":
    main()
