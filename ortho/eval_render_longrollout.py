"""
eval_render_longrollout.py — Base Wan Neural Video Game Rendering
==================================================================
Transforms coarse 2D simulation rollouts (or Orthographic World Model predictions)
into high-fidelity, photorealistic 3D video game visuals using the base Wan2.1 model.

The 2D simulation / Gym physics provide geometric ground-truth (movement, collision,
camera angle, agent trajectory), while the base Wan2.1 model provides neural rendering
(PBR materials, volumetric lighting, ray tracing, next-gen textures, particle effects).

Generates:
  - results/render_longrollout/comparison_{env}.mp4  <- Side-by-side [2D Sim | Base Wan Game Render]
  - results/render_longrollout/game_render_{env}.mp4 <- Fullscreen neural video game footage
  - results/render_longrollout/filmstrip_{env}.png   <- High-res side-by-side progression strip
  - results/render_longrollout/game_render_{env}.gif <- Animated preview

Usage:
  # Render directly from Gym environment rollout (MultiCar Racing):
  python eval_render_longrollout.py --env_name "MultiCar Racing" --rollout_steps 32

  # Render with custom game art aesthetic (e.g. Cyberpunk / Unreal Engine 5):
  python eval_render_longrollout.py --env_name "MultiCar Racing" \\
      --render_prompt "Cyberpunk 2077 night race, wet neon asphalt, sports car, raytracing, Unreal Engine 5"

  # Render using an existing trained Orthographic World Model checkpoint:
  python eval_render_longrollout.py --ckpt_dir ./checkpoints/wan_ortho_12_lora_final \\
      --env_name "Bipedal Walker" --render_strength 0.65
"""

import os
import math
import random
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from utils import (
    _build_registry,
    load_base_model,
    compose_player_prompts,
    set_seed,
    load_agent,
)
from buffer import OrthoTransitionBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# CURATED AAA VIDEO GAME VISUAL PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_NEGATIVE_PROMPT = (
    "2D cartoon, flat colors, simple 2D polygons, box2d physics graphics, low poly, pixel art, "
    "flat shading, basic geometric shapes, drawing, anime, vector art, blurry, oversaturated, "
    "amateur render, low resolution, bad lighting, deformed, noisy artifacts, washed out, retro graphics, canvas painting"
)

GAME_RENDER_PROMPTS = {
    "multicar": (
        "Photorealistic next-gen AAA racing video game, Forza Motorsport and Gran Turismo 7 graphics, "
        "realistic wet asphalt race track with rubber tire marks, high-performance GT3 sports car, "
        "detailed carbon fiber aerodynamics, glossy metallic paint reflections, realistic tire smoke, "
        "volumetric sunset lighting, screen-space reflections, Unreal Engine 5.4, 8k resolution, photorealism"
    ),
    "car racing": (
        "Photorealistic next-gen AAA racing video game, Forza Motorsport and Gran Turismo 7 graphics, "
        "realistic wet asphalt race track with rubber tire marks, high-performance GT3 sports car, "
        "detailed carbon fiber aerodynamics, glossy metallic paint reflections, realistic tire smoke, "
        "volumetric sunset lighting, screen-space reflections, Unreal Engine 5.4, 8k resolution, photorealism"
    ),
    "bipedal": (
        "Next-gen sci-fi mecha adventure video game, photorealistic bipedal titanium combat robot "
        "walking across rocky rugged alien planet terrain, metallic battle wear and scratches, "
        "hydraulic pistons, dynamic dust clouds kicked up by feet, dramatic sunset ray-traced lighting, "
        "Unreal Engine 5.4, 8k resolution, cinematic masterpiece"
    ),
    "lunar": (
        "Photorealistic 3D space exploration simulator, highly detailed Apollo lunar descent module "
        "hovering over realistic cratered moon surface, detailed moon dust particles, glowing rocket thruster exhaust, "
        "Earth glowing in deep black starry space, cinematic hard shadow sunlight, 8k photorealism"
    ),
    "mario": (
        "Next-generation 3D adventure video game, rich vibrant stylized fantasy world, detailed brick and cobblestone "
        "textures, atmospheric sun god rays, lush foliage, Pixar and Unreal Engine 5 aesthetic, 4K"
    ),
    "drone": (
        "High-speed combat drone racing video game, realistic futuristic quadcopter, carbon fiber chassis, "
        "lens flare, motion blur, volumetric cloudscape, Unreal Engine 5.4, photorealistic 4K"
    ),
    "tank": (
        "Battlefield modern military combat simulator, heavily armored battle tank traversing dusty desert terrain, "
        "realistic weathered armor plates, dust kick-up, dynamic muzzle flash, Unreal Engine 5.4, photorealistic 8K"
    ),
    "catapult": (
        "Medieval siege warfare video game, realistic heavy wooden trebuchet catapult, detailed stone castle fortress, "
        "dramatic overcast lighting, realistic wood and metal textures, Unreal Engine 5.4, cinematic 4K"
    ),
    "excavator": (
        "Heavy machinery simulator video game, realistic hydraulic excavator digging wet gravel, "
        "photorealistic mud and steel bucket, industrial realism, Unreal Engine 5.4, 4K"
    ),
    "robot sumo": (
        "Cyberpunk robot arena battle game, polished chrome gladiators, neon lighting reflections on wet steel floor, "
        "spark particles, ray-traced reflections, Unreal Engine 5.4"
    ),
    "mountain goat": (
        "Photorealistic alpine wildlife adventure game, wild mountain goat scaling sheer granite cliffs, "
        "dynamic wind and snow particles, realistic rock textures, beautiful mountain vista, 4K"
    ),
    "wrecking ball": (
        "Next-gen physics demolition game, heavy cast-iron wrecking ball smashing realistic brick and concrete structure, "
        "crumbling debris physics, dynamic volumetric dust, Unreal Engine 5.4"
    ),
    "toxic gas": (
        "Photorealistic survival horror video game, abandoned industrial bunker, volumetric green toxic mist, "
        "emergency hazard lights, wet metallic surfaces, Unreal Engine 5.4"
    ),
}

DEFAULT_GAME_PROMPT = (
    "A photorealistic AAA 8k video game screenshot, Unreal Engine 5.4 render, ray-traced global illumination, "
    "PBR materials, volumetric lighting, cinematic motion blur, octane render, masterpiece, ultra-detailed textures"
)


def get_game_render_prompt(
    env_name: str,
    view: str = "fpv",
    custom_prompt: Optional[str] = None,
    camera_pose: Optional[str] = None,
) -> str:
    """Returns a rich, tailored AAA video game prompt for the given environment, camera view, and camera pose."""
    if custom_prompt and custom_prompt.strip():
        base_p = custom_prompt.strip()
    else:
        clean_key = (env_name or "").lower().replace("-", " ").replace("_", " ")
        base_p = DEFAULT_GAME_PROMPT
        for k, p in GAME_RENDER_PROMPTS.items():
            if k in clean_key:
                base_p = p
                break

    view_tags = {
        "fpv": "first-person driver cockpit view, looking through windshield at race track ahead",
        "rear": "third-person chase camera view following directly behind the vehicle",
        "top": "cinematic aerial drone broadcast camera view looking down at the race track",
        "side": "side profile dynamic tracking camera shot",
    }
    view_str = view_tags.get(str(view).lower(), "")

    cam_str = ""
    if camera_pose and str(camera_pose).strip().lower() not in ("none", "center", ""):
        cam_str = f"camera perspective tilting {camera_pose.strip().title()}"

    extra_parts = [p for p in [view_str, cam_str] if p]
    if extra_parts:
        return f"{base_p}, {', '.join(extra_parts)}."
    return f"{base_p}."


# ─────────────────────────────────────────────────────────────────────────────
# ARROW KEY HUD OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def overlay_arrow_pad(np_img: np.ndarray, pose_str: str) -> np.ndarray:
    """
    Overlays an inverted-T keyboard arrow pad in the corner with the active direction
    highlighted in yellow.
    np_img: HWC uint8 RGB image
    """
    out = np_img.copy()
    h, w, _ = out.shape

    key_size = max(10, int(min(h, w) * 0.10))
    gap = max(1, int(key_size * 0.15))
    margin = max(4, int(min(h, w) * 0.04))

    keys = {
        "Up":    (margin + key_size + gap, margin, key_size, key_size),
        "Left":  (margin, margin + key_size + gap, key_size, key_size),
        "Down":  (margin + key_size + gap, margin + key_size + gap, key_size, key_size),
        "Right": (margin + 2 * (key_size + gap), margin + key_size + gap, key_size, key_size),
    }

    pad_w = 3 * key_size + 2 * gap + 2 * gap
    pad_h = 2 * key_size + gap + 2 * gap
    overlay = out.copy()
    cv2.rectangle(overlay, (margin - gap, margin - gap), (margin - gap + pad_w, margin - gap + pad_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

    active_name = str(pose_str or "").strip().title()

    for name, (kx, ky, kw, kh) in keys.items():
        is_active = (name == active_name)
        cx, cy = kx + kw // 2, ky + kh // 2
        d = max(2, kw // 3)

        if is_active:
            key_bg = (255, 230, 0)      # Bright yellow in RGB
            arrow_color = (20, 20, 20)
            border_color = (255, 255, 120)
        else:
            key_bg = (40, 42, 48)       # Inactive dark key
            arrow_color = (180, 180, 185)
            border_color = (75, 78, 85)

        cv2.rectangle(out, (kx, ky), (kx + kw, ky + kh), key_bg, -1)
        cv2.rectangle(out, (kx, ky), (kx + kw, ky + kh), border_color, 1)

        if name == "Up":
            pts = np.array([[cx, cy - d], [cx - d, cy + d - 1], [cx + d, cy + d - 1]], np.int32)
        elif name == "Down":
            pts = np.array([[cx, cy + d], [cx - d, cy - d + 1], [cx + d, cy - d + 1]], np.int32)
        elif name == "Left":
            pts = np.array([[cx - d, cy], [cx + d - 1, cy - d], [cx + d - 1, cy + d]], np.int32)
        elif name == "Right":
            pts = np.array([[cx + d, cy], [cx - d + 1, cy - d], [cx - d + 1, cy + d]], np.int32)

        cv2.fillPoly(out, [pts], arrow_color)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# BASE WAN FLOW-MATCHING SDE NEURAL RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def _get_wan_latent_stats(vae, target_device, target_dtype):
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        mean = torch.tensor(vae.config.latents_mean, device=target_device, dtype=target_dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=target_device, dtype=target_dtype).view(1, -1, 1, 1, 1)
    else:
        mean = torch.zeros(1, 16, 1, 1, 1, device=target_device, dtype=target_dtype)
        std = torch.ones(1, 16, 1, 1, 1, device=target_device, dtype=target_dtype)
    return mean, std


def encode_frames_to_latents(vae, frames_tensor: torch.Tensor, out_dtype=None) -> torch.Tensor:
    """
    Encodes RGB frame tensors (B, T, 3, H, W) in [-1, 1] into Wan 3D VAE normalized latents.
    Returns: (B, 16, T, h, w)
    """
    b, t, c, h, w = frames_tensor.shape
    flat_frames = frames_tensor.view(b * t, c, 1, h, w).to(dtype=vae.dtype, device=vae.device)

    with torch.no_grad():
        latent_dist = vae.encode(flat_frames).latent_dist
        latents = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()
        mean, std = _get_wan_latent_stats(vae, latents.device, latents.dtype)
        latents = (latents - mean) / std

    _, c_lat, _, h_lat, w_lat = latents.shape
    latents = latents.view(b, t, c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4)
    if out_dtype is not None:
        latents = latents.to(dtype=out_dtype)
    return latents


def decode_latents_to_frames(vae, latents: torch.Tensor) -> torch.Tensor:
    """
    Decodes normalized Wan latents (B, 16, T, h, w) back to RGB image tensors (B, T, 3, H, W) in [0, 1].
    """
    b, c_lat, t, h_lat, w_lat = latents.shape
    flat_latents = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c_lat, 1, h_lat, w_lat).to(dtype=vae.dtype)

    mean, std = _get_wan_latent_stats(vae, flat_latents.device, flat_latents.dtype)
    flat_latents = flat_latents * std + mean

    with torch.no_grad():
        decoded = vae.decode(flat_latents).sample  # (B*T, 3, 1, H, W)

    decoded = decoded.squeeze(2)
    frames = decoded.view(b, t, 3, decoded.shape[-2], decoded.shape[-1])
    return torch.clamp((frames + 1.0) / 2.0, 0.0, 1.0)


@torch.no_grad()
def neural_render_frames_with_base_wan(
    base_transformer,
    vae,
    text_encoder,
    tokenizer,
    sim_frames_tensor: torch.Tensor,
    prompt: str,
    negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT,
    guidance_scale: float = 6.0,
    render_strength: float = 0.75,
    steps: int = 25,
    shift: float = 3.0,
) -> torch.Tensor:
    """
    Neural Video Game Renderer using Base Wan2.1 with Classifier-Free Guidance (CFG).
    
    sim_frames_tensor: (1, T, 3, H, W) in [0, 1] — coarse simulation frames.
    prompt: High-fidelity AAA video game prompt.
    negative_prompt: Negative prompt to strongly suppress flat 2D / cartoon artifacts.
    guidance_scale: CFG scale (>1.0 forces strong adherence to photorealistic game styling).
    render_strength: float in (0.0, 1.0].
      - 0.0: Exact VAE reconstruction of 2D simulation.
      - 0.75: Sweet spot: retains vehicle/agent motion & track geometry while generating
              photorealistic PBR materials, ray-traced reflections, and volumetric lighting.
      - 1.0: Full text-to-video creative synthesis.
    """
    base_transformer.eval()
    T = sim_frames_tensor.shape[1]

    # 1. Encode conditioning prompt with UMT5 text encoder
    text_inputs = tokenizer(
        [prompt], padding="max_length", max_length=128, truncation=True, return_tensors="pt"
    ).to(device)
    prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=base_transformer.dtype)

    # 2. Encode negative prompt for Classifier-Free Guidance (CFG)
    neg_embeds = None
    if guidance_scale > 1.0 and negative_prompt:
        neg_inputs = tokenizer(
            [negative_prompt], padding="max_length", max_length=128, truncation=True, return_tensors="pt"
        ).to(device)
        neg_embeds = text_encoder(**neg_inputs).last_hidden_state.to(dtype=base_transformer.dtype)

    # 3. Ensure spatial dimensions are strictly divisible by 16 (VAE factor 8 * DiT patch size 2)
    b, t, c, h_in, w_in = sim_frames_tensor.shape
    h_snap = max(16, int(round(h_in / 16.0)) * 16)
    w_snap = max(16, int(round(w_in / 16.0)) * 16)
    if h_in != h_snap or w_in != w_snap:
        flat = sim_frames_tensor.view(b * t, c, h_in, w_in)
        flat = F.interpolate(flat, size=(h_snap, w_snap), mode="bilinear", align_corners=False)
        sim_frames_tensor = flat.view(b, t, c, h_snap, w_snap)

    # 4. Encode simulation frames into normalized Wan latents ~ N(0, I)
    sim_norm = sim_frames_tensor * 2.0 - 1.0
    z_clean = encode_frames_to_latents(vae, sim_norm, out_dtype=torch.float32)  # (1, 16, T, h, w)

    # Ensure latent spatial dimensions are both strictly even (divisible by patch size 2)
    even_h = (z_clean.shape[-2] // 2) * 2
    even_w = (z_clean.shape[-1] // 2) * 2
    if z_clean.shape[-2] != even_h or z_clean.shape[-1] != even_w:
        z_clean = z_clean[:, :, :, :even_h, :even_w]

    noise = torch.randn_like(z_clean)

    # 5. Calibrated Flow-Matching Schedule
    # tau in [0, 1] parameterizes linear progression: tau_start = render_strength down to 0.0
    clamped_strength = float(np.clip(render_strength, 0.05, 1.0))
    tau_steps = torch.linspace(clamped_strength, 0.0, steps + 1, device=z_clean.device, dtype=torch.float32)
    sigmas = (shift * tau_steps) / (1.0 + (shift - 1.0) * tau_steps)

    # Noise level at step 0 precisely matches sigmas[0]:
    sigma_init = sigmas[0].item()
    x_tau = (1.0 - sigma_init) * z_clean + sigma_init * noise

    b, c, num_t, h, w = z_clean.shape

    # 6. Euler integration loop with Classifier-Free Guidance (CFG)
    for i in range(steps):
        s_curr = sigmas[i]
        s_next = sigmas[i + 1]
        dt = s_curr - s_next

        t_tensor = torch.full((b,), s_curr * 1000.0, device=device, dtype=base_transformer.dtype)
        in_latents = x_tau.to(dtype=base_transformer.dtype)

        if guidance_scale > 1.0 and neg_embeds is not None:
            # Memory-safe sequential evaluation to avoid OOM on video latents
            v_cond = base_transformer(
                hidden_states=in_latents,
                timestep=t_tensor,
                encoder_hidden_states=prompt_embeds,
            ).sample.float()

            v_uncond = base_transformer(
                hidden_states=in_latents,
                timestep=t_tensor,
                encoder_hidden_states=neg_embeds,
            ).sample.float()

            if v_cond.shape != x_tau.shape:
                min_h = min(x_tau.shape[-2], v_cond.shape[-2])
                min_w = min(x_tau.shape[-1], v_cond.shape[-1])
                x_tau = x_tau[:, :, :, :min_h, :min_w]
                v_cond = v_cond[:, :, :, :min_h, :min_w]
                v_uncond = v_uncond[:, :, :, :min_h, :min_w]

            # CFG extrapolation
            pred_velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_velocity = base_transformer(
                hidden_states=in_latents,
                timestep=t_tensor,
                encoder_hidden_states=prompt_embeds,
            ).sample.float()

            if pred_velocity.shape != x_tau.shape:
                min_h = min(x_tau.shape[-2], pred_velocity.shape[-2])
                min_w = min(x_tau.shape[-1], pred_velocity.shape[-1])
                x_tau = x_tau[:, :, :, :min_h, :min_w]
                pred_velocity = pred_velocity[:, :, :, :min_h, :min_w]

        x_tau = x_tau - dt * pred_velocity

    # 7. Decode latents back to photorealistic RGB video frames
    rendered_frames = decode_latents_to_frames(vae, x_tau.to(dtype=vae.dtype))  # (1, T, 3, H, W) in [0, 1]

    # If original input dimensions differed, resize back to original h_in, w_in
    if rendered_frames.shape[-2:] != (h_in, w_in):
        flat_r = rendered_frames.view(b * t, c, rendered_frames.shape[-2], rendered_frames.shape[-1])
        flat_r = F.interpolate(flat_r, size=(h_in, w_in), mode="bilinear", align_corners=False)
        rendered_frames = flat_r.view(b, t, c, h_in, w_in)

    return rendered_frames


# ─────────────────────────────────────────────────────────────────────────────
# ROLLOUT COLLECTION & SYNTHESIS
# ─────────────────────────────────────────────────────────────────────────────
def collect_env_trajectory(env, agent, n_steps=32, seed=42, target_h: Optional[int] = None, target_w: Optional[int] = None):
    """Gathers an interactive rollout with sustained camera bursts from the environment."""
    obs, _ = env.reset(seed=seed)
    curr_pose = random.choice(["Left", "Right", "Up", "Down"])
    pose_hold = random.randint(4, 8)

    top_frames, rear_frames, side_frames, fpv_frames = [], [], [], []
    pose_seq = []

    for _ in range(n_steps):
        if pose_hold <= 0:
            choices = [p for p in ["Left", "Right", "Up", "Down"] if p != curr_pose]
            curr_pose = random.choice(choices)
            pose_hold = random.randint(4, 8)
        pose_seq.append(curr_pose)
        pose_hold -= 1

        views = env.render(camera_pose=curr_pose)
        top_frames.append(views[0])
        rear_frames.append(views[1])
        side_frames.append(views[2])
        fpv_frames.append(views[3])

        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    def _to_tensor(f_list):
        arr = np.stack(f_list, axis=0)  # (T, H, W, 3)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W) in [0, 1]
        _, _, h, w = t.shape
        # Default coarse Gym simulations (e.g. 96x96) to at least 448x448 so Wan DiT has enough latent capacity
        req_h = target_h if (target_h and target_h > 0) else max(448 if h < 384 else h, 16)
        req_w = target_w if (target_w and target_w > 0) else max(448 if w < 384 else w, 16)
        req_h = max(16, int(round(req_h / 16.0)) * 16)
        req_w = max(16, int(round(req_w / 16.0)) * 16)
        if h != req_h or w != req_w:
            t = F.interpolate(t, size=(req_h, req_w), mode="bilinear", align_corners=False)
        return t

    return {
        "top": _to_tensor(top_frames),
        "rear": _to_tensor(rear_frames),
        "side": _to_tensor(side_frames),
        "fpv": _to_tensor(fpv_frames),
        "poses": pose_seq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION & VIDEO CREATION
# ─────────────────────────────────────────────────────────────────────────────
def save_render_comparison_video(
    sim_frames: torch.Tensor,
    game_frames: torch.Tensor,
    poses: List[str],
    out_dir: str,
    env_name: str,
    fps: int = 15,
    overlay_hud: bool = True,
):
    """
    Creates and saves:
      1. comparison_{env}.mp4: Side-by-side [2D Simulation | Base Wan Game Render]
      2. game_render_{env}.mp4: Fullscreen AAA game footage
      3. filmstrip_{env}.png: Multi-frame progression filmstrip
      4. game_render_{env}.gif: Animated preview GIF
    """
    os.makedirs(out_dir, exist_ok=True)
    clean_tag = "".join(c for c in env_name if c.isalnum() or c in ("_", "-")).lower()

    # Ensure sim_frames and game_frames match in spatial dimensions
    if sim_frames.shape[-2:] != game_frames.shape[-2:]:
        sim_frames = F.interpolate(sim_frames, size=game_frames.shape[-2:], mode="bilinear", align_corners=False)

    T, C, H, W = game_frames.shape
    header_h = 32
    composite_w = W * 2
    composite_h = H + header_h

    frames_bgr = []
    frames_game_only_bgr = []
    frames_rgb = []

    for t in range(T):
        s_np = (sim_frames[t].permute(1, 2, 0).cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        g_np = (game_frames[t].permute(1, 2, 0).cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Apply Arrow Keys HUD onto the FPV view
        if overlay_hud and t < len(poses):
            s_np = overlay_arrow_pad(s_np, poses[t])
            g_np = overlay_arrow_pad(g_np, poses[t])

        canvas = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
        canvas[:header_h, :] = (24, 26, 32)  # Dark sleek header

        canvas[header_h:, 0:W] = cv2.cvtColor(s_np, cv2.COLOR_RGB2BGR)
        canvas[header_h:, W:2 * W] = cv2.cvtColor(g_np, cv2.COLOR_RGB2BGR)

        # Divider line
        cv2.line(canvas, (W, 0), (W, composite_h), (90, 95, 105), 1)
        cv2.line(canvas, (0, header_h), (composite_w, header_h), (70, 75, 85), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "2D Simulation (Coarse World)", (12, 21), font, 0.40, (210, 210, 215), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Base Wan Neural Game Render", (W + 12, 21), font, 0.40, (50, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"t={t+1}/{T}", (composite_w - 68, 21), font, 0.35, (160, 160, 165), 1, cv2.LINE_AA)

        frames_bgr.append(canvas)
        frames_game_only_bgr.append(cv2.cvtColor(g_np, cv2.COLOR_RGB2BGR))
        frames_rgb.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    # 1. Save Side-by-Side MP4
    comp_video_path = os.path.join(out_dir, f"comparison_{clean_tag}.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vwriter = cv2.VideoWriter(comp_video_path, fourcc, fps, (composite_w, composite_h))
        for f in frames_bgr:
            vwriter.write(f)
        vwriter.release()
        print(f"  [SAVED] Side-by-side comparison video -> {comp_video_path}")
    except Exception as e:
        print(f"  [WARN] Failed writing comparison video: {e}")

    # 2. Save Pure Neural Game Footage MP4
    game_video_path = os.path.join(out_dir, f"game_render_{clean_tag}.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vwriter = cv2.VideoWriter(game_video_path, fourcc, fps, (W, H))
        for f in frames_game_only_bgr:
            vwriter.write(f)
        vwriter.release()
        print(f"  [SAVED] Neural video game footage -> {game_video_path}")
    except Exception as e:
        print(f"  [WARN] Failed writing pure game video: {e}")

    # 3. Save Filmstrip Image
    strip_path = os.path.join(out_dir, f"filmstrip_{clean_tag}.png")
    try:
        sample_indices = np.linspace(0, T - 1, min(6, T), dtype=int)
        strip_frames = [frames_bgr[i] for i in sample_indices]
        full_strip = np.vstack(strip_frames)
        cv2.imwrite(strip_path, full_strip)
        print(f"  [SAVED] Comparison progression strip -> {strip_path}")
    except Exception as e:
        print(f"  [WARN] Failed writing filmstrip: {e}")

    # 4. Save Preview GIF
    gif_path = os.path.join(out_dir, f"game_render_{clean_tag}.gif")
    try:
        pil_images = [Image.fromarray(f) for f in frames_rgb]
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"  [SAVED] Preview GIF -> {gif_path}")
    except Exception as e:
        print(f"  [WARN] Failed writing GIF: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Neural Video Game Rendering with Base Wan2.1")
    parser.add_argument("--env_name", type=str, default="MultiCar Racing", help="Gym environment name")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="Hugging Face model ID")
    parser.add_argument("--rollout_steps", type=int, default=32, help="Number of rollout frames to render")
    parser.add_argument("--render_strength", type=float, default=0.75, help="Neural rendering stylization strength (0.3=subtle, 0.75=photorealistic AAA game, 1.0=pure T2V)")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-Free Guidance scale (>1.0 forces strong adherence to AAA game style)")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt to suppress flat 2D / cartoon artifacts")
    parser.add_argument("--render_prompt", type=str, default=None, help="Custom AAA video game prompt override")
    parser.add_argument("--target_view", type=str, default="fpv", choices=["fpv", "top", "rear", "side"], help="Camera view to render")
    parser.add_argument("--eval_steps", type=int, default=25, help="Euler integration steps for neural render")
    parser.add_argument("--out_dir", type=str, default="results/render_longrollout", help="Directory to save videos and strips")
    parser.add_argument("--fps", type=int, default=15, help="Video framerate")
    parser.add_argument("--img_h", type=int, default=448, help="Target frame height (auto-snapped to multiple of 16)")
    parser.add_argument("--img_w", type=int, default=448, help="Target frame width (auto-snapped to multiple of 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_hud", action="store_true", help="Disable arrow key HUD overlay")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  BASE WAN2.1 NEURAL VIDEO GAME RENDERER")
    print(f"  Environment:     {args.env_name}")
    print(f"  View:            {args.target_view.upper()}")
    print(f"  Rollout Horizon: {args.rollout_steps} steps")
    print(f"  Render Strength: {args.render_strength} (SDE Flow Bridge)")
    print(f"  Guidance Scale:  {args.guidance_scale} (CFG)")
    print(f"  Euler Steps:     {args.eval_steps}")
    print(f"  Resolution:      {args.img_h}x{args.img_w}")
    print("=" * 80)

    # 1. Build environment and gather simulation trajectory
    registry = _build_registry()
    reg_dict = dict(registry)

    if args.env_name not in reg_dict:
        # Fallback search
        matching = [k for k in reg_dict.keys() if args.env_name.lower() in k.lower()]
        if matching:
            args.env_name = matching[0]
            print(f"[INFO] Resolved environment to: '{args.env_name}'")
        else:
            raise ValueError(f"Environment '{args.env_name}' not found. Available: {list(reg_dict.keys())}")

    print(f"\n[1/3] Gathering simulation trajectory from: {args.env_name}...")
    env = reg_dict[args.env_name]()
    obs, _ = env.reset(seed=args.seed)
    state_dim = int(np.prod(obs.shape if hasattr(obs, "shape") else env.observation_space.shape))
    clean_name = "".join(c for c in args.env_name if c.isalnum() or c in ("_", "-")).lower()
    agent = load_agent(f"checkpoints/ppo_{clean_name}.pt", state_dim, env.action_space, device)

    trajectory = collect_env_trajectory(
        env, agent, n_steps=args.rollout_steps, seed=args.seed,
        target_h=args.img_h, target_w=args.img_w
    )
    env.close()

    target_key = args.target_view.lower()
    sim_frames = trajectory[target_key].to(device)  # (T, 3, H, W) in [0, 1]
    pose_seq = trajectory["poses"]

    # 2. Load Base Wan2.1 Model (no fine-tuned LoRA, raw generative capacity)
    print(f"\n[2/3] Loading Base Wan2.1 Model components ({args.model_id})...")
    base_transformer, vae, text_encoder, tokenizer = load_base_model(model_id=args.model_id)

    # 3. Formulate rich AAA game prompt
    active_prompt = get_game_render_prompt(
        env_name=args.env_name,
        view=args.target_view,
        custom_prompt=args.render_prompt,
        camera_pose=pose_seq[0] if pose_seq else None,
    )
    print(f"\n[3/3] Neural Rendering with AAA Game Prompt:")
    print(f"  \"{active_prompt}\"")
    if args.guidance_scale > 1.0:
        print(f"  Negative Prompt (CFG {args.guidance_scale}):")
        print(f"  \"{args.negative_prompt}\"")

    # Neural Render via Base Wan SDE Flow Bridge with CFG
    print(f"\nRunning Flow-Matching Neural Rendering ({args.eval_steps} Euler steps with CFG {args.guidance_scale})...")
    sim_batch = sim_frames.unsqueeze(0)  # (1, T, 3, H, W)
    rendered_batch = neural_render_frames_with_base_wan(
        base_transformer=base_transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        sim_frames_tensor=sim_batch,
        prompt=active_prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        render_strength=args.render_strength,
        steps=args.eval_steps,
    )
    rendered_frames = rendered_batch.squeeze(0).cpu()  # (T, 3, H, W)

    # 4. Save High-Quality Visual Outputs
    print(f"\n[OUTPUT] Generating side-by-side comparison video and filmstrip...")
    save_render_comparison_video(
        sim_frames=sim_frames.cpu(),
        game_frames=rendered_frames,
        poses=pose_seq,
        out_dir=args.out_dir,
        env_name=args.env_name,
        fps=args.fps,
        overlay_hud=(not args.no_hud),
    )

    print("\n" + "=" * 80)
    print(f"  Neural Game Rendering Complete!")
    print(f"  Results saved to: {os.path.abspath(args.out_dir)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
