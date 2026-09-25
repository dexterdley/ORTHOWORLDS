import os
import pygame
import torch
import numpy as np
import cv2
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

# Import your model and environment wrappers from your training script
from train_car_racing_view import RecurrentDiffusionWorldModel, MultiCarRacingContinuous, PerspectiveFirstPersonWrapper

# --- 1. SETUP & LOAD ---
DEVICE = torch.device("cuda")
torch.set_float32_matmul_precision('high')

print("Loading models...")
# Load PPO Agent
agent_ckpt = load_from_hub(
        repo_id="igpaub/ppo-CarRacing-v2",
        filename="ppo-CarRacing-v2.zip"
    )
agent = PPO.load(agent_ckpt, device=DEVICE)

# Load World Model
model = RecurrentDiffusionWorldModel(
        timesteps=200,
        hidden_dim=128,
        noise_aug_max=20,
        cond_drop_prob=0.1,
        base_ch=32
    ).to(DEVICE)

model.load_state_dict(torch.load("./car_racing/world_models/world_model_epoch_50.pth", map_location=DEVICE))
model.eval()

# Optionally compile for a massive speed boost on the 3070
model.unet = torch.compile(model.unet, mode="reduce-overhead")

# Setup Env for Burn-in
base_env = MultiCarRacingContinuous(difficulty=1.0)
env = PerspectiveFirstPersonWrapper(base_env)

# --- 2. PYGAME SETUP ---
pygame.init()
WINDOW_SIZE = (600, 600)  # Upscale the 96x96 output for playability
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Neural Game Engine: CarRacing")
clock = pygame.time.Clock()

def get_human_action():
    """Maps WASD to continuous steering, gas, and brake."""
    keys = pygame.key.get_pressed()
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    if keys[pygame.K_a]: action[0] = -1.0
    if keys[pygame.K_d]: action[0] =  1.0
    if keys[pygame.K_w]: action[1] =  1.0
    if keys[pygame.K_s]: action[2] =  0.8
    return action

# --- 3. BURN-IN PHASE (Gather initial 8 frames) ---
print("Running 50-step burn-in...")
obs, _ = env.reset()

# Get the static map track
track_vertices = np.array([[t[2], t[3]] for t in env.unwrapped.track])
g_map = np.zeros((1, 1, 400, 2), dtype=np.float32)
mask = np.zeros((1, 1, 400), dtype=bool)
t_len = min(len(track_vertices), 400)
g_map[0, 0, :t_len] = track_vertices[:t_len]
mask[0, 0, :t_len] = True

g_map_t = torch.from_numpy(g_map).to(DEVICE)
mask_t = torch.from_numpy(mask).to(DEVICE)

# Burn-in history trackers
hist_img, hist_pov, hist_pose, hist_act = [], [], [], []
hist_b_pose, hist_b_act = [], []

for _ in range(50):
    # Let PPO drive both cars during burn-in
    action, _ = agent.predict(obs, deterministic=True)
    b_obs = env.unwrapped.get_opponent_obs()
    b_action, _ = agent.predict(b_obs, deterministic=True)
    env.unwrapped.set_opponent_action(b_action)
    
    img, pov = env.render()
    next_obs, _, _, _, _ = env.step(action)
    
    # Store tensors scaled to [-1, 1]
    hist_img.append(torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0)
    hist_pov.append(torch.from_numpy(pov).permute(2, 0, 1).float() / 127.5 - 1.0)

    # 1. Unpack and append EGO pose
    car_x, car_y = env.unwrapped.car.hull.position
    car_angle = env.unwrapped.car.hull.angle
    hist_pose.append(torch.tensor([car_x, car_y, car_angle], dtype=torch.float32))
    
    # 2. Unpack and append OPPONENT pose
    blue_x, blue_y = env.unwrapped.opponent_car.hull.position
    blue_angle = env.unwrapped.opponent_car.hull.angle
    hist_b_pose.append(torch.tensor([blue_x, blue_y, blue_angle], dtype=torch.float32))
    
    # 3. Append actions
    hist_act.append(torch.from_numpy(action).float())
    hist_b_act.append(torch.from_numpy(b_action).float())
    
    obs = next_obs

# Stack history into sequences of shape (1, seq_len, ...)
curr_img_seq = torch.stack(hist_img).unsqueeze(0).to(DEVICE)
curr_pov_seq = torch.stack(hist_pov).unsqueeze(0).to(DEVICE)
curr_p_seq = torch.stack(hist_pose).unsqueeze(0).to(DEVICE)
curr_b_p_seq = torch.stack(hist_b_pose).unsqueeze(0).to(DEVICE)

curr_a_seq = torch.stack(hist_act).unsqueeze(0).to(DEVICE)
curr_b_a_seq = torch.stack(hist_b_act).unsqueeze(0).to(DEVICE)

print("Burn-in complete! Handing over control to human.")

# --- 4. REAL-TIME PLAY LOOP ---
running = True
with torch.no_grad():
    while running:
        # Event pump
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # 1. Get Inputs
        human_act = get_human_action()
        # Predict Blue Car action using the *hallucinated* previous top-down frame
        fake_b_obs_raw = ((curr_img_seq[0, -1].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)        
        fake_b_obs = cv2.resize(fake_b_obs_raw, (96, 96), interpolation=cv2.INTER_AREA)
        blue_act, _ = agent.predict(fake_b_obs, deterministic=True)
        
        # Convert to batch size 1, seq len 1
        curr_a = torch.from_numpy(human_act).unsqueeze(0).unsqueeze(0).to(DEVICE)
        curr_b_a = torch.from_numpy(blue_act).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        curr_a_seq = torch.cat([curr_a_seq[:, 1:], curr_a], dim=1)
        curr_b_a_seq = torch.cat([curr_b_a_seq[:, 1:], curr_b_a], dim=1)

        # 2. RUN DIFFUSION WORLD MODEL
        # We use bfloat16 mixed precision for a massive speed boost
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred_img, pred_pov, pred_red_d, pred_blue_d = model.sample(
                curr_img_seq, curr_pov_seq, curr_p_seq, curr_a_seq, 
                curr_b_p_seq, curr_b_a_seq, g_map_t, mask_t
            )

        # 3. Sliding Window Update
        new_red_pose = curr_p_seq[:, -1] + pred_red_d
        new_blue_pose = curr_b_p_seq[:, -1] + pred_blue_d
        
        curr_img_seq = torch.cat([curr_img_seq[:, 1:], pred_img.unsqueeze(1)], dim=1)
        curr_pov_seq = torch.cat([curr_pov_seq[:, 1:], pred_pov.unsqueeze(1)], dim=1)
        curr_p_seq = torch.cat([curr_p_seq[:, 1:], new_red_pose.unsqueeze(1)], dim=1)
        curr_b_p_seq = torch.cat([curr_b_p_seq[:, 1:], new_blue_pose.unsqueeze(1)], dim=1)

        # 4. Render to Pygame
        # Convert tensor back to numpy image
        frame = ((pred_img.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Pygame expects (Width, Height, Channels) so we transpose
        frame_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        frame_surface = pygame.transform.scale(frame_surface, WINDOW_SIZE)
        
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        # Cap logic to roughly 30 fps (though diffusion will likely be the bottleneck)
        clock.tick(30)

pygame.quit()