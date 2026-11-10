"""
Validation shared by the ortho training scripts: autoregressive video rollouts and their TensorBoard logging.

The arms differ only through keyword arguments:
  per_view  FPV-Single layout: each view is its own batch item (B=4, one frame each) instead of one 4-frame item
  sketch    Ortho-Project: the sampler takes the projector sketch (transformer.sketch_proj) and the camera
            features; the decoded sketch is returned and logged
  img_hw    rollout resolution (must match training)
"""

import numpy as np
import cv2
import torch

from utils import compose_player_prompts, _annotate_frame_pose, _psnr
from models import encode_views_to_latents, decode_latents_to_views, cam_feats
from data_rollout import get_registries, ReplayEnv, _skip_intro, _camera_state
from flow import predict_next_view_latents, predict_next_view_latents_sketch

VAL_ENVS = ["Parkour Runner", "Lunar Lander", "MultiCar Racing"]


@torch.no_grad()
def autoregressive_rollout(transformer, vae, text_encoder, tokenizer, env_name, env, agent=None,
                           rollout_steps=16, steps=20, camera_command="Right", seed=42,
                           img_hw=(224, 224), per_view=False, sketch=False):
    """
    Autoregressive video rollout for all 4 orthographic views (per_view: each view predicted on its own).
    Returns:
      gen_views: (vid_top, vid_rear, vid_side, vid_fpv) each (1, T, 3, H, W) in [0, 1]
      gt_views:  (gt_top,  gt_rear,  gt_side,  gt_fpv)  each (1, T, 3, H, W) in [0, 1]
      vis:       (gen_fpv_vis, gt_fpv_vis) pose-annotated FPV videos for logging
      sketch:    decoded sketch video (1, T, 3, H, W), or None without `sketch`
    """
    device = vae.device
    img_h, img_w = img_hw
    transformer.eval()

    obs, _ = env.reset(seed=seed)
    obs = _skip_intro(env, env_name, obs)
    env.action_space.seed(seed)

    # Same first render as collect_rollouts: "Center" is a no-op right after reset (yaw = pitch = 0)
    img_top, im_rear, im_side, im_fpv = env.render(camera_pose="Center")

    vt = [cv2.resize(x, (img_w, img_h), interpolation=cv2.INTER_AREA) for x in [img_top, im_side, im_rear, im_fpv]]
    vt = np.stack([np.transpose(v, (2, 0, 1)) for v in vt])
    # per_view: each view becomes its own batch item: (4, 1, 3, H, W) -> latents (4, C, 1, h, w)
    vt = np.expand_dims(vt, axis=1 if per_view else 0)

    curr_views = torch.from_numpy(vt).float().to(device) / 127.5 - 1.0
    curr_latents = encode_views_to_latents(vae, curr_views, out_dtype=transformer.dtype)
    num_views = curr_latents.shape[0]   # batch size: 4 with per_view, else 1

    # decoded_views is (4, 1, 3, H, W) with per_view, else (1, 4, 3, H, W)
    view = (lambda d, v: d[v, 0]) if per_view else (lambda d, v: d[0, v])

    # Frame storage (0: Top, 1: Side, 2: Rear, 3: FPV)
    gen_top, gen_side, gen_rear, gen_fpv = [], [], [], []
    gt_top,  gt_side,  gt_rear,  gt_fpv  = [], [], [], []
    gen_fpv_vis, gt_fpv_vis = [], []  # pose-annotated copies, for logged videos only
    gen_sketch = []  # decoded sketch (the FPV's starting point), for logging only

    print(f"Starting {rollout_steps}-step Autoregressive Rollout for [{env_name}] (camera {camera_command})...")

    for step_i in range(rollout_steps):
        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        # Render the GT target before predicting: training prompts use the camera state read right after this render
        next_top, next_rear, next_side, next_fpv = env.render(camera_pose=camera_command)
        cam_state = env.camera_state if isinstance(env, ReplayEnv) else _camera_state(env, camera_command)

        # Same float32 action vector the buffer stores (a scalar discrete action becomes length 1)
        act = np.atleast_1d(np.asarray(action, dtype=np.float32))
        prompt = compose_player_prompts(
            [act],
            env_name=env_name,
            camera_pose=cam_state,
        )

        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_embeds = text_encoder(**text_inputs).last_hidden_state.to(dtype=transformer.dtype)
        if per_view:
            prompt_embeds = prompt_embeds.expand(num_views, -1, -1)

        if sketch:
            cam = cam_feats([cam_state])
            # Same sketch the sampler starts the FPV from; decoded for logging only
            sketch_views = decode_latents_to_views(vae, transformer.sketch_proj(curr_latents[:, :, :3], cam))  # (1, 1, 3, H, W)
            gen_sketch.append(sketch_views[0, 0].cpu())
            next_latents = predict_next_view_latents_sketch(
                transformer=transformer,
                latents_t=curr_latents,
                prompt_embeds=prompt_embeds,
                cam=cam,
                steps=steps,
            )
        else:
            next_latents = predict_next_view_latents(
                transformer=transformer,
                latents_t=curr_latents,
                prompt_embeds=prompt_embeds,
                steps=steps,
            )

        decoded_views = decode_latents_to_views(vae, next_latents)  # in [0, 1]

        # Store generated frames (0: Top, 1: Side, 2: Rear, 3: FPV)
        gen_top.append(view(decoded_views, 0).cpu())
        gen_side.append(view(decoded_views, 1).cpu())
        gen_rear.append(view(decoded_views, 2).cpu())

        vn = [cv2.resize(x, (img_w, img_h), interpolation=cv2.INTER_AREA) for x in [next_top, next_side, next_rear, next_fpv]]
        vn = np.stack([np.transpose(v, (2, 0, 1)) for v in vn])

        gt_vn = torch.from_numpy(vn).float() / 255.0  # (4, 3, H, W)
        gt_top.append(gt_vn[0])
        gt_side.append(gt_vn[1])
        gt_rear.append(gt_vn[2])

        # Raw FPV frames are used for PSNR; the pose overlay only goes on the logged videos
        curr_gen_fpv = view(decoded_views, 3).cpu()
        curr_gt_fpv = gt_vn[3]

        gen_fpv.append(curr_gen_fpv)
        gt_fpv.append(curr_gt_fpv)
        gen_fpv_vis.append(_annotate_frame_pose(curr_gen_fpv, camera_command))
        gt_fpv_vis.append(_annotate_frame_pose(curr_gt_fpv, camera_command))

        # Re-encode decoded RGB frame back through VAE to stay on the clean VAE manifold
        next_views_norm = decoded_views * 2.0 - 1.0
        curr_latents = encode_views_to_latents(vae, next_views_norm, out_dtype=transformer.dtype)

        # Training windows never cross an episode reset, so stop instead of jumping to a new episode
        if terminated or truncated:
            break

    vid_gen_top  = torch.stack(gen_top,  dim=0).unsqueeze(0)
    vid_gen_side = torch.stack(gen_side, dim=0).unsqueeze(0)
    vid_gen_rear = torch.stack(gen_rear, dim=0).unsqueeze(0)
    vid_gen_fpv  = torch.stack(gen_fpv,  dim=0).unsqueeze(0)

    vid_gt_top   = torch.stack(gt_top,   dim=0).unsqueeze(0)
    vid_gt_side  = torch.stack(gt_side,  dim=0).unsqueeze(0)
    vid_gt_rear  = torch.stack(gt_rear,  dim=0).unsqueeze(0)
    vid_gt_fpv   = torch.stack(gt_fpv,   dim=0).unsqueeze(0)
    vid_gen_fpv_vis = torch.stack(gen_fpv_vis, dim=0).unsqueeze(0)
    vid_gt_fpv_vis  = torch.stack(gt_fpv_vis,  dim=0).unsqueeze(0)
    vid_gen_sketch  = torch.stack(gen_sketch,  dim=0).unsqueeze(0) if sketch else None

    transformer.train()
    env.close()
    return (vid_gen_top, vid_gen_rear, vid_gen_side, vid_gen_fpv,
            vid_gt_top,  vid_gt_rear,  vid_gt_side,  vid_gt_fpv,
            vid_gen_fpv_vis, vid_gt_fpv_vis, vid_gen_sketch)


@torch.no_grad()
def evaluate_and_log_videos(transformer, vae, text_encoder, tokenizer, writer, epoch, agents=None, rollout_steps=16, steps=20, fps=15, seed=42,
                            img_hw=(224, 224), per_view=False, sketch=False, val_envs=VAL_ENVS, log_gt=False,
                            hifi=False, replay_buffer=None):
    """
    Performs autoregressive video rollouts for each view and writes the videos to TensorBoard.
    `agents` maps env_name -> the policy used during data collection (random actions if missing).
    `log_gt` also writes the ground-truth videos (identical every epoch).
    With `replay_buffer`, each env replays a fixed cached segment instead of running the simulator (see ReplayEnv).
    """
    agents = agents or {}
    transformer.eval()
    unique_envs = list(val_envs)

    if replay_buffer is None:
        train_reg, heldout_reg = get_registries(hifi)
        reg_dict = dict(train_reg + heldout_reg)

    print(f"\n[VALIDATION] Running autoregressive video rollouts for Epoch {epoch} across envs: {', '.join(unique_envs)}...")

    for env_idx, env_name in enumerate(unique_envs):
        if replay_buffer is not None:
            if env_name not in replay_buffer.env_names:     # held-out env, or beyond --num_envs
                print(f"  [skip] {env_name} not in the cached buffer"); continue
            env = agent = ReplayEnv(replay_buffer, env_name, rollout_steps)
        else:
            env, agent = reg_dict[env_name](), agents.get(env_name, agents.get("Bipedal Walker"))

        (vid_top_down, vid_rear, vid_side, vid_fpv,
         vid_gt_top_down, vid_gt_rear, vid_gt_side, vid_gt_fpv,
         vid_fpv_vis, vid_gt_fpv_vis, vid_sketch) = autoregressive_rollout(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            env_name=env_name,
            env=env,
            agent=agent,
            rollout_steps=rollout_steps,
            steps=steps,
            seed=seed,
            img_hw=img_hw,
            per_view=per_view,
            sketch=sketch,
        )

        clean_tag = env_name.replace(" ", "_").lower()

        psnr_top_down = _psnr(vid_top_down, vid_gt_top_down)
        psnr_rear     = _psnr(vid_rear, vid_gt_rear)
        psnr_side     = _psnr(vid_side, vid_gt_side)
        psnr_fpv      = _psnr(vid_fpv, vid_gt_fpv)
        psnr_mean     = (psnr_top_down + psnr_rear + psnr_side + psnr_fpv) / 4.0

        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Top Down", psnr_top_down, epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Rear",     psnr_rear,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Side",     psnr_side,     epoch)
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/FPV",      psnr_fpv,      epoch)
        if vid_sketch is not None:
            writer.add_scalar(f"Validation/{clean_tag}/PSNR/Sketch", _psnr(vid_sketch, vid_gt_fpv), epoch)  # sketch vs GT FPV_{t+1}
        writer.add_scalar(f"Validation/{clean_tag}/PSNR/Mean",     psnr_mean,     epoch)

        print(f"  [Epoch {epoch}] [{env_name}] Autoregressive Video PSNR: {psnr_mean:.2f} dB "
              f"(Top-Down: {psnr_top_down:.2f}dB, Rear: {psnr_rear:.2f}dB, Side: {psnr_side:.2f}dB, FPV: {psnr_fpv:.2f}dB)")

        # Always write per-env tagged videos; also alias as generic Test/ for the first env
        writer.add_video(f"Test/{clean_tag}/Top-Down", vid_top_down, epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Rear",     vid_rear,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/Side",     vid_side,     epoch, fps=fps)
        writer.add_video(f"Test/{clean_tag}/FPV",      vid_fpv_vis,  epoch, fps=fps)
        if vid_sketch is not None:
            writer.add_video(f"Test/{clean_tag}/Sketch", vid_sketch, epoch, fps=fps)

        if log_gt:
            writer.add_video(f"GT/{clean_tag}/Top-Down", vid_gt_top_down, epoch, fps=fps)
            writer.add_video(f"GT/{clean_tag}/Rear",     vid_gt_rear,     epoch, fps=fps)
            writer.add_video(f"GT/{clean_tag}/Side",     vid_gt_side,     epoch, fps=fps)
            writer.add_video(f"GT/{clean_tag}/FPV",      vid_gt_fpv_vis,  epoch, fps=fps)

    transformer.train()
