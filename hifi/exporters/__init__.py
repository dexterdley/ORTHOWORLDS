"""
Per-env exporters: GymBox orthographic wrapper -> Godot hi-fi renderer messages.

ENVS maps the Godot world name (hifi/godot/envs/<name>) to (display title, module, class).
Display titles match ortho/utils.py so registries line up.
"""
import importlib
import os
import sys

_ORTHO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ortho")
if _ORTHO not in sys.path:
    sys.path.insert(0, _ORTHO)  # GymBox envs live in ortho/game_envs (imported, never modified)

ENVS = {
    # training registry (ortho/utils.py _build_registry), in its order
    "bipedal_walker": ("Bipedal Walker", "bipedal_walker", "BipedalWalkerExporter"),
    "lunar_lander": ("Lunar Lander", "lunar_lander", "LunarLanderExporter"),
    "multicar_racing": ("MultiCar Racing", "multicar_racing", "MultiCarRacingExporter"),
    "mario_escape": ("Mario Escape", "mario_escape", "MarioEscapeExporter"),
    "drone_dogfight": ("Drone Dogfight", "drone_dogfight", "DroneDogfightExporter"),
    "excavator": ("Excavator", "excavator", "ExcavatorExporter"),
    "tank_duel": ("Tank Duel", "tank_duel", "TankDuelExporter"),
    "catapult_war": ("Catapult War", "catapult_war", "CatapultWarExporter"),
    "robot_sumo": ("Robot Sumo", "robot_sumo", "RobotSumoExporter"),
    "mountain_goat": ("Mountain Goat", "mountain_goat", "MountainGoatExporter"),
    "wrecking_ball": ("Wrecking Ball", "wrecking_ball", "WreckingBallExporter"),
    "toxic_gas_escape": ("Toxic Gas Escape", "toxic_gas_escape", "ToxicGasEscapeExporter"),
    # held-out obstacle-transfer envs (ortho/utils.py _build_heldout_registry)
    "rescue_ambulance": ("Rescue Ambulance", "rescue_ambulance", "RescueAmbulanceExporter"),
    "parkour_runner": ("Parkour Runner", "parkour_runner", "ParkourRunnerExporter"),
    "snowball_fight": ("Snowball Fight", "snowball_fight", "SnowballFightExporter"),
}


def resolve(query):
    """Env name from a name, title or substring ('tank', 'Tank Duel', 'tank_duel')."""
    q = query.lower().replace("-", " ").replace("_", " ").strip()
    for name, (title, _, _) in ENVS.items():
        if q in (name.replace("_", " "), title.lower()):
            return name
    hits = [n for n, (t, _, _) in ENVS.items() if q in n.replace("_", " ") or q in t.lower()]
    if not hits:
        raise KeyError(f"no env matches '{query}'; known: {', '.join(ENVS)}")
    return hits[0]


def get_exporter(name, **env_kwargs):
    name = resolve(name)
    _, module, cls = ENVS[name]
    mod = importlib.import_module(f"exporters.{module}")
    return getattr(mod, cls)(**env_kwargs)


def available():
    """Env names whose exporter module exists."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [n for n, (_, m, _) in ENVS.items() if os.path.exists(os.path.join(here, f"{m}.py"))]
