import os
import random
import numpy as np
import torch
import torch.nn as nn
from diffusers import WanPipeline
from peft import LoraConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

def set_seed(seed):
    """Set random seed for reproducibility across random, numpy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1. ENVIRONMENT REGISTRY
# ==========================================
def _build_registry():
    """Return a list of (display_name, factory_fn) tuples.
    Each factory_fn must return a fully-wrapped env that exposes:
        env.reset()  -> (obs, info)
        env.step(a)  -> (obs, reward, term, trunc, info)
        env.render() -> (im_top_down, im_rear, im_side, im_fpv)  [H x W x 3 uint8]
        env.action_space.sample()
        env.close()
    """
    registry = []

    def _try(name, factory):
        registry.append((name, factory))
    # ------------------------------------------------------------------
    # Bipedal Walker (wraps gymnasium's BipedalWalker-v3)
    try:
        import gymnasium
        from game_envs.bipedal_orthographic import BipedalOrthographicWrapper
        _try("Bipedal Walker",
             lambda: BipedalOrthographicWrapper(
                 gymnasium.make("BipedalWalker-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register Bipedal Walker: {e}")
    # ------------------------------------------------------------------
    # Lunar Lander (wraps gymnasium's LunarLander-v3)
    try:
        import gymnasium
        from game_envs.lunar_orthographic import LunarOrthographicWrapper
        _try("Lunar Lander",
             lambda: LunarOrthographicWrapper(
                 gymnasium.make("LunarLander-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register Lunar Lander: {e}")
    # ------------------------------------------------------------------
    # MultiCar Racing (wraps gymnasium's CarRacing-v3)
    try:
        import gymnasium
        from game_envs.multicar_racing_orthographic import MultiCarOrthographicWrapper
        _try("MultiCar Racing",
             lambda: MultiCarOrthographicWrapper(
                 gymnasium.make("CarRacing-v3", render_mode="rgb_array")))
    except Exception as e:
        print(f"[WARN] Could not register MultiCar Racing: {e}")        
    # ------------------------------------------------------------------
    # Mario Escape
    try:
        from game_envs.mario_orthographic import (
            MarioEscapeEnv, MarioOrthographicWrapper)
        _try("Mario Escape",
             lambda: MarioOrthographicWrapper(MarioEscapeEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Mario Escape: {e}")

    # ------------------------------------------------------------------
    # Drone Dogfight
    try:
        from game_envs.drone_dogfight_orthographic import (
            DroneDogfightEnv, DroneDogfightOrthographicWrapper)
        _try("Drone Dogfight",
             lambda: DroneDogfightOrthographicWrapper(DroneDogfightEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Drone Dogfight: {e}")

    # ------------------------------------------------------------------
    # Excavator
    try:
        from game_envs.excavator_orthographic import (
            ExcavatorEnv, ExcavatorOrthographicWrapper)
        _try("Excavator",
             lambda: ExcavatorOrthographicWrapper(ExcavatorEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Excavator: {e}")

    # ------------------------------------------------------------------
    # Tank Duel
    try:
        from game_envs.tank_duel_orthographic import (
            TankDuelEnv, TankDuelOrthographicWrapper)
        _try("Tank Duel",
             lambda: TankDuelOrthographicWrapper(TankDuelEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Tank Duel: {e}")

    # ------------------------------------------------------------------
    # Catapult War
    try:
        from game_envs.catapult_war_orthographic import (
            CatapultWarEnv, CatapultWarOrthographicWrapper)
        _try("Catapult War",
             lambda: CatapultWarOrthographicWrapper(CatapultWarEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Catapult War: {e}")

    # ------------------------------------------------------------------
    # Robot Sumo
    try:
        from game_envs.robot_sumo_orthographic import (
            RobotSumoEnv, RobotSumoOrthographicWrapper)
        _try("Robot Sumo",
             lambda: RobotSumoOrthographicWrapper(RobotSumoEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Robot Sumo: {e}")

    # ------------------------------------------------------------------
    # Mountain Goat
    try:
        from game_envs.mountain_goat_orthographic import (
            MountainGoatEnv, MountainGoatOrthographicWrapper)
        _try("Mountain Goat",
             lambda: MountainGoatOrthographicWrapper(MountainGoatEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Mountain Goat: {e}")

    # ------------------------------------------------------------------
    # Wrecking Ball
    try:
        from game_envs.wrecking_ball_orthographic import (
            WreckingBallEnv, WreckingBallOrthographicWrapper)
        _try("Wrecking Ball",
             lambda: WreckingBallOrthographicWrapper(WreckingBallEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Wrecking Ball: {e}")

    # ------------------------------------------------------------------
    # Toxic Gas Escape
    try:
        from game_envs.toxic_gas_escape_orthographic import (
            ToxicGasEscapeEnv, ToxicGasEscapeOrthographicWrapper)
        _try("Toxic Gas Escape",
             lambda: ToxicGasEscapeOrthographicWrapper(ToxicGasEscapeEnv()))
    except Exception as e:
        print(f"[WARN] Could not register Toxic Gas Escape: {e}")
        
    return registry

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space, hidden_dim=256):
        super().__init__()
        self.is_discrete = hasattr(action_space, 'n')
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Value head (Critic)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Policy head (Actor)
        if self.is_discrete:
            self.action_dim = action_space.n
            self.actor_head = nn.Linear(hidden_dim, self.action_dim)
        else:
            self.action_dim = action_space.shape[0]
            self.actor_mean = nn.Linear(hidden_dim, self.action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, state):
        features = self.feature_net(state)
        value = self.value_head(features)
        
        if self.is_discrete:
            logits = self.actor_head(features)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = torch.tanh(self.actor_mean(features))
            std = torch.exp(self.actor_log_std.expand_as(mean))
            dist = torch.distributions.Normal(mean, std)
            
        return dist, value

    def get_value(self, state):
        features = self.feature_net(state)
        return self.value_head(features)

def load_agent(ckpt_path, state_dim, action_space, device):
    """Instantiate ActorCritic and load state dict from checkpoint."""
    ac = ActorCritic(state_dim, action_space).to(device)
    if os.path.exists(ckpt_path):
        ac.load_state_dict(torch.load(ckpt_path, map_location=device))
        print((f"  [LOADED] PPO Agent weights from {ckpt_path}"))
    else:
        print((f"  [WARN] PPO Agent Checkpoint '{ckpt_path}' not found!"))
    ac.eval()
    return ac

# ==========================================
# PER-ENVIRONMENT PROMPT COMPOSER
# ==========================================
_ACTIVE_ENV_NAME = "Bipedal Walker"


def set_active_env(name):
    global _ACTIVE_ENV_NAME
    _ACTIVE_ENV_NAME = name


ENV_PROMPT_TEMPLATES = {
    "bipedal walker": {
        "title": "Bipedal Walker 2D robot locomotion",
        "format": lambda a: (
            f"leg torques [hip1: {a[0].item():+.2f}, knee1: {a[1].item():+.2f}, "
            f"hip2: {a[2].item():+.2f}, knee2: {a[3].item():+.2f}]"
            if len(a) >= 4 else f"joint torques [{', '.join(f'{x.item():+.2f}' for x in a)}]"
        ),
        "suffix": "balancing robot movement across terrain",
    },
    "lunar lander": {
        "title": "Lunar Lander spacecraft descent",
        "format": lambda a: (
            f"main throttle {a[0].item():.2f}, steering thrusters {a[1].item():+.2f}"
            if len(a) >= 2 else (
                {0: "thrusters idle", 1: "left thruster active", 2: "main thruster active", 3: "right thruster active"}.get(
                    int(round(a[0].item())), f"thruster state {int(round(a[0].item()))}"
                ) if len(a) == 1 else "thruster control active"
            )
        ),
        "suffix": "accurate touchdown on landing pad",
    },
    "mario escape": {
        "title": "Mario Escape platformer",
        "format": lambda a: (
            f"horizontal move {a[0].item():+.2f}, {'jump impulse active' if a[1].item() > 0.5 else 'ground running'}"
            if len(a) >= 2 else f"movement input {a[0].item():+.2f}"
        ),
        "suffix": "navigating obstacles towards flagpole",
    },
    "multicar racing": {
        "title": "MultiCar circuit racing",
        "format": lambda a: (
            f"steer {a[0].item():+.2f}, throttle {a[1].item():.2f}, brake {a[2].item():.2f}"
            if len(a) >= 3 else (
                f"steer {a[0].item():+.2f}, throttle {a[1].item():.2f}"
                if len(a) == 2 else f"steer {a[0].item():+.2f}"
            )
        ),
        "suffix": "racing at high speed on a track",
    },
    "drone dogfight": {
        "title": "Drone Dogfight aerial combat",
        "format": lambda a: (
            f"thrust ({a[0].item():+.2f}, {a[1].item():+.2f}), {'firing weapon' if a[3].item() > 0.5 else 'holding fire'}"
            if len(a) >= 4 else (
                f"thrust ({a[0].item():+.2f}, {a[1].item():+.2f})"
                if len(a) >= 2 else "maneuvering thrusters"
            )
        ),
        "suffix": "aerial engagement against opponent",
    },
    "excavator": {
        "title": "Excavator heavy machinery",
        "format": lambda a: (
            f"drive {a[0].item():+.2f}, boom {a[1].item():+.2f}, dipper {a[2].item():+.2f}, bucket {a[3].item():+.2f}"
            if len(a) >= 4 else f"actuators [{', '.join(f'{x.item():+.2f}' for x in a)}]"
        ),
        "suffix": "scooping rock payload into bin",
    },
    "tank duel": {
        "title": "Tank Duel armored combat",
        "format": lambda a: (
            f"treads ({a[0].item():+.2f}, {a[1].item():+.2f})"
            if len(a) >= 2 else f"controls [{', '.join(f'{x.item():+.2f}' for x in a)}]"
        ),
        "suffix": "maneuvering and aiming turret",
    },
    "catapult war": {
        "title": "Catapult War siege artillery",
        "format": lambda a: (
            f"tension {a[0].item():.2f}, angle {a[1].item():+.2f}"
            if len(a) >= 2 else f"tension {a[0].item():.2f}"
        ),
        "suffix": "aiming and launching projectile",
    },
    "robot sumo": {
        "title": "Robot Sumo wrestling arena",
        "format": lambda a: (
            f"wheel drive ({a[0].item():+.2f}, {a[1].item():+.2f})"
            if len(a) >= 2 else f"drive {a[0].item():+.2f}"
        ),
        "suffix": "pushing opponent out of dohyo ring",
    },
    "mountain goat": {
        "title": "Mountain Goat cliff climbing",
        "format": lambda a: (
            f"body pitch {a[0].item():+.2f}, leap impulse {a[1].item():.2f}"
            if len(a) >= 2 else f"movement {a[0].item():+.2f}"
        ),
        "suffix": "ascending procedural rock cliffs",
    },
    "wrecking ball": {
        "title": "Wrecking Ball demolition crane",
        "format": lambda a: (
            f"boom rotation {a[0].item():+.2f}, hoist cable {a[1].item():+.2f}"
            if len(a) >= 2 else f"crane control {a[0].item():+.2f}"
        ),
        "suffix": "swinging demolition ball into structures",
    },
    "toxic gas escape": {
        "title": "Toxic Gas Escape platformer",
        "format": lambda a: (
            f"horizontal run {a[0].item():+.2f}, {'jump impulse active' if a[1].item() > 0.5 else 'ground running'}"
            if len(a) >= 2 else f"run {a[0].item():+.2f}"
        ),
        "suffix": "scrambling outwards to evade toxic gas",
    },
}


def compose_player_prompts(action_batch, env_name=None):
    """
    Composes text prompts for Wan2.1 text encoder tailored to each environment.
    Supports either a single env_name string or a list/tuple of env_names for mixed batches.
    Example for Bipedal Walker:
      'Bipedal Walker 2D robot locomotion: leg torques [hip1: +0.20, knee1: -0.45, hip2: +0.10, knee2: +0.80], balancing across terrain.'
    """
    # Normalize action_batch to iterable of 1D action vectors
    if isinstance(action_batch, torch.Tensor):
        if action_batch.ndim == 1:
            action_batch = action_batch.unsqueeze(0)
    elif isinstance(action_batch, np.ndarray):
        if action_batch.ndim == 1:
            action_batch = np.expand_dims(action_batch, 0)
    elif not isinstance(action_batch, (list, tuple)):
        action_batch = [action_batch]

    num_samples = len(action_batch)
    if isinstance(env_name, (list, tuple)):
        env_list = list(env_name)
    else:
        env_list = [env_name] * num_samples

    prompts = []
    for a, sample_env in zip(action_batch, env_list):
        if sample_env is None:
            sample_env = _ACTIVE_ENV_NAME or "Generic"

        env_key = sample_env.lower().replace("-", " ").replace("_", " ").strip()

        # Find matching template
        matched_template = None
        for key, tmpl in ENV_PROMPT_TEMPLATES.items():
            if key in env_key or env_key in key:
                matched_template = tmpl
                break

        if matched_template is None:
            clean_name = sample_env.replace("_", " ").replace("-", " ").title() if sample_env else "Game"
            matched_template = {
                "title": f"{clean_name} simulation",
                "format": lambda act: f"action [{', '.join(f'{x.item():+.2f}' if hasattr(x, 'item') else f'{float(x):+.2f}' for x in act)}]",
                "suffix": "navigating environment",
            }

        if isinstance(a, (int, float, np.integer, np.floating)):
            a = torch.tensor([float(a)])
        elif isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        elif not isinstance(a, torch.Tensor):
            a = torch.tensor(list(a), dtype=torch.float32)

        try:
            action_str = matched_template["format"](a)
        except Exception:
            action_str = f"action [{', '.join(f'{x.item():+.2f}' for x in a)}]"

        full_prompt = (
            f"{matched_template['title']}: {action_str}, {matched_template['suffix']}."
        )
        prompts.append(full_prompt)

    return prompts



# ==========================================
# MODEL SETUP WITH LORA
# ==========================================
def load_model(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", lora_rank=16):
    """
    Loads Wan2.1-1.3B, freezes VAE and Text Encoder, applies LoRA to DiT Transformer.
    """
    print(f"\n[INFO] Loading Wan2.1 components from: {model_id}")
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # 1. Freeze VAE and Text Encoder
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()

    text_encoder = pipe.text_encoder
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    tokenizer = pipe.tokenizer

    # 2. Setup LoRA on Wan Transformer DiT
    transformer = pipe.transformer
    transformer.requires_grad_(False)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj"],
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Move to device
    vae.to(device)
    text_encoder.to(device)
    transformer.to(device)

    return transformer, vae, text_encoder, tokenizer


def load_base_model(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"):
    """
    Loads Wan2.1-1.3B WITHOUT applying LoRA. Use this in eval scripts before
    calling PeftModel.from_pretrained() to avoid double-wrapping the transformer,
    which mangles adapter key paths into 'base_model.model.base_model.model...'.

    Example:
        transformer, vae, text_encoder, tokenizer = load_base_model(model_id)
        transformer = PeftModel.from_pretrained(transformer, ckpt_dir)
        transformer.eval()
    """
    print(f"\n[INFO] Loading Wan2.1 base components (no LoRA) from: {model_id}")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)

    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    text_encoder = pipe.text_encoder
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)

    tokenizer = pipe.tokenizer

    transformer = pipe.transformer
    transformer.requires_grad_(False)
    transformer.to(device)

    return transformer, vae, text_encoder, tokenizer
