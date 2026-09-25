import os
import random
import math
import copy
import argparse
import numpy as np
import cv2
import pygame
from pygame import gfxdraw
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as ckpt_fn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Gym Box2D specific imports
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.car_dynamics import Car  # We need this to spawn a second car

# Enable TensorFloat32 (TF32) for massive speedups on Ampere/Hopper GPUs
torch.set_float32_matmul_precision('high')
np.bool8 = np.bool_

# ==========================================
# 0. ARGUMENT PARSER
# ==========================================
parser = argparse.ArgumentParser(description="Recurrent Diffusion World Model for CarRacing-v2")

# Resolution (common presets: 64x64, 96x96, 128x192, 200x300, 400x600)
parser.add_argument('--img_h', type=int, default=96, help='Image height (default: 96)')
parser.add_argument('--img_w', type=int, default=96, help='Image width (default: 96)')

# Training hyperparameters
parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU (default: 128)')
parser.add_argument('--seq_len', type=int, default=8, help='Sequence length for GRU burn-in (default: 8)')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
parser.add_argument('--steps_per_epoch', type=int, default=500, help='Training steps per epoch (default: 200)')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (default: 3e-4)')
parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum LR for cosine schedule (default: 1e-6)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

# Model hyperparameters
parser.add_argument('--diffusion_timesteps', type=int, default=200, help='Diffusion noise schedule steps (default: 200)')
parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM sampling steps at inference (default: 50)')
parser.add_argument('--hidden_dim', type=int, default=128, help='GRU hidden dim (default: 256)')
parser.add_argument('--noise_aug_max', type=int, default=20, help='Max noise augmentation level (default: 20)')
parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='CFG conditioning dropout probability (default: 0.1)')
parser.add_argument('--guidance_scale', type=float, default=1.5, help='CFG guidance scale at inference (default: 1.5)')
parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate (default: 0.999)')

# Data collection
parser.add_argument('--buffer_capacity', type=int, default=5000, help='Replay buffer capacity (default: 5000)')
parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup rollout steps (default: 3000)')
parser.add_argument('--collect_steps', type=int, default=300, help='Rollout steps per epoch (default: 300)')

# Eval
parser.add_argument('--eval_frames', type=int, default=50, help='Frames to generate during eval (default: 50)')
parser.add_argument('--eval_every', type=int, default=5, help='Run eval every N epochs (default: 5)')
parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')

### Example usage:
### CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_car_racing_view.py --img_h 96 --img_w 96 --batch_size 256
### CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_car_racing_main.py --img_h 400 --img_w 600 --batch_size 64
### CUDA_VISIBLE_DEVICES=7 python ./car_racing/train_car_racing_view.py --img_h 96 --img_w 96 --batch_size 256

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MultiCarRacingContinuous(CarRacing):
    def __init__(self, difficulty=0.0):
        super().__init__(render_mode="rgb_array", continuous=True)
        self.opponent_car = None
        self.opponent_action = np.array([0.0, 0.0, 0.0])
        self.difficulty = np.clip(difficulty, 0.0, 1.0) # 0=Easy, 1=Nightmare

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        if self.car is not None:
            init_x, init_y = self.car.hull.position
            car_angle = self.car.hull.angle
            
            # --- SOCIAL ADVERSITY ---
            # d=0: 5.0 units apart | d=1: 2.5 units apart
            offset_dist = 5.0 - (self.difficulty * 2.5) 
            
            spawn_x = init_x + offset_dist * math.cos(car_angle)
            spawn_y = init_y + offset_dist * math.sin(car_angle)
            
            self.opponent_car = Car(self.world, car_angle, spawn_x, spawn_y+10)
            self.opponent_car.hull.color = (0.0, 0.0, 0.8) 
            self.opponent_action = np.array([0.0, 0.0, 0.0])

            # Let internal gym steps will draw the blue car into the state array!
            original_draw = self.car.draw
            def patched_draw(surf, zoom, translation, angle, draw_particles=True, **kwargs):
                original_draw(surf, zoom, translation, angle, draw_particles, **kwargs)
                if self.opponent_car is not None:
                    self.opponent_car.draw(surf, zoom, translation, angle, draw_particles=False, **kwargs)
                    
            self.car.draw = patched_draw            
            obs = self._render("state_pixels")
            
        return obs, info

    def get_opponent_obs(self):
        if self.opponent_car is None:
            return np.zeros((96, 96, 3), dtype=np.uint8)
        temp_car = self.car
        self.car = self.opponent_car
        obs = self._render("state_pixels")
        self.car = temp_car
        return obs

    def set_opponent_action(self, action):
        if isinstance(action, np.ndarray):
            self.opponent_action = action.flatten()

    def step(self, action):
        # 1. Step Opponent
        if self.opponent_car is not None:
            steer = float(self.opponent_action[0])
            gas = float(self.opponent_action[1])
            brake = float(self.opponent_action[2])
            self.opponent_car.steer(-steer)
            self.opponent_car.gas(gas)
            self.opponent_car.brake(brake)
            self.opponent_car.step(1.0 / 50.0) 
            
        # 2. Step Player & Physics World
        if action is not None:
            if isinstance(action, np.ndarray):
                action = action.flatten()
            clean_action = np.array([float(action[0]), float(action[1]), float(action[2])])
            return super().step(clean_action)
            
        return super().step(action)

    def render(self):
        if self.car is None:
            return super().render()
        original_draw = self.car.draw
        def patched_draw(surf, zoom, translation, angle, draw_particles=True, **kwargs):
            original_draw(surf, zoom, translation, angle, draw_particles, **kwargs)
            if self.opponent_car is not None:
                self.opponent_car.draw(surf, zoom, translation, angle, draw_particles=False, **kwargs)
        self.car.draw = patched_draw
        img = super().render()
        self.car.draw = original_draw 
        return img

class PerspectiveFirstPersonWrapper(gym.Wrapper):
    """
    Wraps the environment to return both the original top-down view 
    and a mathematically projected 3D perspective first-person view.
    """
    def __init__(self, env):
        super().__init__(env)
        self.WINDOW_W = 1000
        self.WINDOW_H = 800
        
        # 3D Camera Parameters
        self.fov = 500.0
        self.camera_h = 10.0      # Height of camera above ground
        self.camera_z = 8.0       # Distance camera sits behind the car
        
    def render(self):
        # 1. Get the top-down view (which now includes the blue car thanks to our patch!)
        top_down_img = self.env.render()
        if top_down_img is None:
            return None, None

        base_env = self.env.unwrapped
        fpv_surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
        
        # Sky and Horizon
        fpv_surf.fill((135, 206, 235)) 
        pygame.draw.rect(fpv_surf, base_env.bg_color, 
                         (0, self.WINDOW_H // 2, self.WINDOW_W, self.WINDOW_H // 2))

        if base_env.car is None or not base_env.road_poly:
            return top_down_img, top_down_img

        car_x, car_y = base_env.car.hull.position
        car_angle = base_env.car.hull.angle
        
        rot_angle = -car_angle
        cos_a = math.cos(rot_angle)
        sin_a = math.sin(rot_angle)

        polys = []
        
        # 1. Recreate Grass patches -> LAYER 0
        PLAYFIELD = 2000 / 6.0
        GRASS_DIM = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                p = [
                    (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                    (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                    (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                    (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                ]
                polys.append((p, base_env.grass_color, 0)) # Notice the ', 0'
                
        # 2. Add Road polygons -> LAYER 1
        for p, c in base_env.road_poly:
            polys.append((p, c, 1)) # Notice the ', 1'

        # 🚨 Extract Opponent Car Polygons for 3D Rendering -> LAYER 2
        if hasattr(base_env, 'opponent_car') and base_env.opponent_car is not None:
            CAR_HEIGHT = 3.5 
            CAR_COLOR_MAIN = (20, 50, 200)
            CAR_COLOR_TOP = (15, 45, 180)
            CAR_COLOR_SIDE_1 = (10, 30, 150)
            CAR_COLOR_SIDE_2 = (15, 40, 180)

            for fixture in base_env.opponent_car.hull.fixtures:
                shape = fixture.shape
                shape_vertices = list(shape.vertices)
                
                V_bottom = []
                V_top = []
                
                for v in shape_vertices:
                    world_v = base_env.opponent_car.hull.transform * v
                    V_bottom.append((world_v[0], world_v[1], 0.0))
                
                for vb in V_bottom:
                    V_top.append((vb[0], vb[1], CAR_HEIGHT))
                
                # Add Top face
                polys.append((V_top, CAR_COLOR_TOP, 2))
                
                num_v = len(V_bottom)
                for i in range(num_v):
                    idx_curr = i
                    idx_next = (i + 1) % num_v
                    
                    side_poly = [
                        V_bottom[idx_curr],
                        V_bottom[idx_next],
                        V_top[idx_next],
                        V_top[idx_curr]
                    ]
                    
                    if i % 2 == 0:
                        polys.append((side_poly, CAR_COLOR_SIDE_1, 2))
                    else:
                        polys.append((side_poly, CAR_COLOR_SIDE_2, 2))

                polys.append((V_bottom, CAR_COLOR_SIDE_1, 2))
                
            for w in base_env.opponent_car.wheels:
                wheel_body = w if hasattr(w, 'fixtures') else w.wheel
                for fixture in wheel_body.fixtures:
                    shape = fixture.shape
                    vertices = [(wheel_body.transform * v) for v in shape.vertices]
                    polys.append((vertices, (30, 30, 30), 2)) # Dark Grey tires, Layer 2

        # 3. Apply 3D Perspective Projection
        projected_polys = []
        # Unpack the new layer variable
        for p_vertices, color, layer in polys:
            proj_pts = []
            depths = []
            valid = True
            
            for v in p_vertices:
                if len(v) == 3:
                    vx, vy, vh = v
                else:
                    vx, vy = v
                    vh = 0.0
                    
                dx = vx - car_x
                dy = vy - car_y
                
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                depth = ry + self.camera_z
                
                if depth < 2.5: 
                    valid = False
                    break
                
                zh = self.camera_h - vh
                
                px = (rx / depth) * self.fov + self.WINDOW_W / 2
                py = (zh / depth) * self.fov + self.WINDOW_H / 2
                
                proj_pts.append((px, py))
                depths.append(depth)
                
            if valid and len(proj_pts) >= 3: 
                avg_depth = sum(depths) / len(depths)
                # Store the layer alongside the projected points
                projected_polys.append((avg_depth, proj_pts, color, layer))
                
        # 4. FIX: Z-Index + Painter's Algorithm
        # Sort primarily by layer (Ascending: 0, 1, 2)
        # Sort secondarily by depth (Descending: furthest to closest)
        # Using a tuple (layer, -avg_depth) achieves exactly this in Python!
        projected_polys.sort(key=lambda x: (x[3], -x[0]))
        
        # 5. Draw
        for _, pts, color, _ in projected_polys:
            safe_color = (int(color[0]), int(color[1]), int(color[2]))
            gfxdraw.aapolygon(fpv_surf, pts, safe_color)
            gfxdraw.filled_polygon(fpv_surf, pts, safe_color)
            
        fpv_img = pygame.surfarray.array3d(fpv_surf)
        fpv_img = np.transpose(fpv_img, (1, 0, 2))
        
        target_shape = (top_down_img.shape[1], top_down_img.shape[0])
        fpv_img_resized = cv2.resize(fpv_img, target_shape, interpolation=cv2.INTER_AREA)
        
        return top_down_img, fpv_img_resized

# ==========================================
# 1. SEQUENCE REPLAY BUFFER
# ==========================================
class TransitionSequenceBuffer:
    def __init__(self, capacity, img_h, img_w, max_track_len=400):
        self.capacity = capacity
        self.img_h = img_h
        self.img_w = img_w
        self.ptr = 0
        self.size = 0
        self.max_track_len = max_track_len
        
        # --- Images ---
        self.img_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_first_person_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.img_next = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.im_first_person_next = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        
        # --- Red Car (Ego) ---
        self.pose_t = np.zeros((capacity, 3), dtype=np.float32)
        self.pose_next = np.zeros((capacity, 3), dtype=np.float32)
        self.action_t = np.zeros((capacity, 3), dtype=np.float32)
        
        # --- Blue Car (Opponent) ---
        self.blue_pose_t = np.zeros((capacity, 3), dtype=np.float32)
        self.blue_pose_next = np.zeros((capacity, 3), dtype=np.float32)
        self.blue_action_t = np.zeros((capacity, 3), dtype=np.float32)
        
        # --- Environment ---
        self.done_t = np.zeros(capacity, dtype=bool)
        self.global_map_t = np.zeros((capacity, max_track_len, 2), dtype=np.float32) # PADDING FOR THE GLOBAL MAP
        self.track_mask_t = np.zeros((capacity, max_track_len), dtype=bool) # Mask the Global Map Padding
        
    def push(self, img, img_first_person, pose, action,
             global_map, track_mask,
             blue_pose, blue_action,
             next_img, next_first_person_img, next_pose, next_blue_pose, done):
        
        # Resize if image doesn't match buffer resolution
        h, w = img.shape[:2]
        if h != self.img_h or w != self.img_w:
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            next_img = cv2.resize(next_img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            img_first_person = cv2.resize(img_first_person, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            next_first_person_img = cv2.resize(next_first_person_img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        
        # Images
        self.img_t[self.ptr] = np.transpose(img, (2, 0, 1))
        self.im_first_person_t[self.ptr] = np.transpose(img_first_person, (2, 0, 1))
        self.img_next[self.ptr] = np.transpose(next_img, (2, 0, 1))
        self.im_first_person_next[self.ptr] = np.transpose(next_first_person_img, (2, 0, 1))
        
        # Red Car
        self.pose_t[self.ptr] = pose
        self.action_t[self.ptr] = action
        self.pose_next[self.ptr] = next_pose
        
        # Blue Car
        self.blue_pose_t[self.ptr] = blue_pose
        self.blue_action_t[self.ptr] = blue_action
        self.blue_pose_next[self.ptr] = next_blue_pose

        # Environment
        self.done_t[self.ptr] = done
        self.global_map_t[self.ptr] = global_map
        self.track_mask_t[self.ptr] = track_mask
        
        # Advance pointers
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample_sequence(self, batch_size, seq_len, device):
        """Samples a contiguous sequence of length seq_len."""
        idxs = np.zeros(batch_size, dtype=np.int32)
        
        # Ensure sequences don't cross episode boundaries OR the ring buffer pointer
        for i in range(batch_size):
            valid = False
            while not valid:
                idx = np.random.randint(0, self.size - seq_len)
                
                # 1. Prevent crossing the ring buffer wrap-around point
                if self.size == self.capacity and idx < self.ptr <= idx + seq_len:
                    continue
                    
                # 2. Prevent crossing episode boundaries (dones)
                if self.done_t[idx : idx + seq_len - 1].any():
                    continue
                    
                valid = True
                idxs[i] = idx
                
        # Slice sequences
        i_seq = np.stack([self.img_t[idx : idx + seq_len] for idx in idxs])
        i_first_seq = np.stack([self.im_first_person_t[idx : idx + seq_len] for idx in idxs])
        p_seq = np.stack([self.pose_t[idx : idx + seq_len] for idx in idxs])
        a_seq = np.stack([self.action_t[idx : idx + seq_len] for idx in idxs])

        g_map_seq = np.stack([self.global_map_t[idx : idx + seq_len] for idx in idxs])
        mask_seq = np.stack([self.track_mask_t[idx : idx + seq_len] for idx in idxs])
        
        b_p_seq = np.stack([self.blue_pose_t[idx : idx + seq_len] for idx in idxs])
        b_a_seq = np.stack([self.blue_action_t[idx : idx + seq_len] for idx in idxs])
        
        i_next_target = np.stack([self.img_next[idx + seq_len - 1] for idx in idxs])
        i_first_next_target = np.stack([self.im_first_person_next[idx + seq_len - 1] for idx in idxs])
        p_next_target = np.stack([self.pose_next[idx + seq_len - 1] for idx in idxs])
        b_p_next_target = np.stack([self.blue_pose_next[idx + seq_len - 1] for idx in idxs])
        
        # Convert to Tensors 
        i_seq_t = (torch.from_numpy(i_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        i_first_seq_t = (torch.from_numpy(i_first_seq).to(device, non_blocking=True).float() / 127.5) - 1.0
        i_next_t = (torch.from_numpy(i_next_target).to(device, non_blocking=True).float() / 127.5) - 1.0
        i_first_next_target_t = (torch.from_numpy(i_first_next_target).to(device, non_blocking=True).float() / 127.5) - 1.0
        
        p_seq_t = torch.from_numpy(p_seq).to(device, non_blocking=True)
        a_seq_t = torch.from_numpy(a_seq).to(device, non_blocking=True)
        p_next_t = torch.from_numpy(p_next_target).to(device, non_blocking=True)

        g_map_seq_t = torch.from_numpy(g_map_seq).to(device, non_blocking=True)
        mask_seq_t = torch.from_numpy(mask_seq).to(device, non_blocking=True)
        
        b_p_seq_t = torch.from_numpy(b_p_seq).to(device, non_blocking=True)
        b_a_seq_t = torch.from_numpy(b_a_seq).to(device, non_blocking=True)
        b_p_next_t = torch.from_numpy(b_p_next_target).to(device, non_blocking=True)
        
        delta_p = p_next_t - p_seq_t[:, -1]
        delta_b_p = b_p_next_t - b_p_seq_t[:, -1]
        
        return (i_seq_t, i_first_seq_t, p_seq_t, a_seq_t,
                g_map_seq_t, mask_seq_t,
                b_p_seq_t, b_a_seq_t,
                i_next_t, i_first_next_target_t, delta_p, delta_b_p)

def collect_rollouts(env, agent, buffer, n_steps, n_episodes, MAX_TRACK_LEN=400):
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        obs, _ = env.reset()
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))
            
        track_vertices = np.array([[t[2], t[3]] for t in env.unwrapped.track])
    
        track_vertices = np.array([[t[2], t[3]] for t in env.unwrapped.track])
        padded_track = np.zeros((MAX_TRACK_LEN, 2), dtype=np.float32)
        track_mask = np.zeros(MAX_TRACK_LEN, dtype=bool)
        t_len = min(len(track_vertices), MAX_TRACK_LEN)
        padded_track[:t_len] = track_vertices[:t_len]
        track_mask[:t_len] = True
        
        for _ in range(n_steps):
    
            # --- BLUE CAR BRAIN ---
            obs_blue = env.unwrapped.get_opponent_obs()
            action_blue, _ = agent.predict(obs_blue, deterministic=True)
            env.unwrapped.set_opponent_action(action_blue)
            blue_action_floats = [float(a) for a in action_blue]
    
            # --- RED CAR BRAIN ---
            action, _states = agent.predict(obs, deterministic=True)
            action_floats = [float(a) for a in action]
            
            img_t, im_first_person_t = env.render()
            
            # --- RED CAR POSITION (t) ---
            car_x, car_y = env.unwrapped.car.hull.position
            car_angle = env.unwrapped.car.hull.angle
            pose_t = np.array([car_x, car_y, car_angle], dtype=np.float32)
    
            # --- BLUE CAR POSITION (t) ---
            blue_x = env.unwrapped.opponent_car.hull.position.x
            blue_y = env.unwrapped.opponent_car.hull.position.y
            blue_angle = env.unwrapped.opponent_car.hull.angle
            blue_pose_t = np.array([blue_x, blue_y, blue_angle], dtype=np.float32)
            
            # ==========================================
            # STEP ENVIRONMENT
            # ==========================================
            next_obs, reward, terminated, truncated, _ = env.step(action_floats)
            done = terminated or truncated
            
            img_next, im_first_person_next = env.render()
            
            # --- RED CAR POSITION (t+1) ---
            next_x, next_y = env.unwrapped.car.hull.position
            next_angle = env.unwrapped.car.hull.angle
            pose_next = np.array([next_x, next_y, next_angle], dtype=np.float32)
            
            # --- BLUE CAR POSITION (t+1) ---
            next_blue_x = env.unwrapped.opponent_car.hull.position.x
            next_blue_y = env.unwrapped.opponent_car.hull.position.y
            next_blue_angle = env.unwrapped.opponent_car.hull.angle
            next_blue_pose = np.array([next_blue_x, next_blue_y, next_blue_angle], dtype=np.float32)
            
            # --- PUSH TO BUFFER ---
            buffer.push(
                img_t, 
                im_first_person_t, 
                pose_t, 
                np.array(action_floats, dtype=np.float32), 
                padded_track,
                track_mask,
                blue_pose_t, 
                np.array(blue_action_floats, dtype=np.float32),
                img_next, 
                im_first_person_next, 
                pose_next, 
                next_blue_pose, 
                done
            )
            
            obs = next_obs
            
            if done:
                break
                
        env.close()

# ==========================================
# 2. COSINE NOISE SCHEDULE
# ==========================================
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in Nichol & Dhariwal 2021."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)
    
# ==========================================
# 3. DIFFUSION U-NET WITH ATTENTION & GRU WORLD MODEL
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    """Self-attention layer for spatial feature maps."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = self.norm(x).view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        return x + attn_out.permute(0, 2, 1).view(b, c, h, w)

class MapEncoder(nn.Module):
    """A lightweight PointNet to encode the padded global track vertices into a fixed-size vector."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, track_pts, mask):
        # Apply MLP to every point independently
        x = self.mlp(track_pts)  
        
        # Mask out the zero-padded points by setting them to a huge negative number 
        x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        # Global Max Pooling over the 400 points
        x_max, _ = x.max(dim=1) 
        return x_max


class ConditionalUNet(nn.Module):
    def __init__(self, cond_dim=256, base_ch=32, in_channels=6, out_channels=3):
        super().__init__()
        ch1, ch2, ch3, ch4, ch5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        self.noise_aug_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # UPDATED: Use in_channels (12)
        self.inc = ResBlock(in_channels, ch1, emb_dim=256)
        
        self.down1 = nn.MaxPool2d(2)
        self.res_down1 = ResBlock(ch1, ch2, emb_dim=256)
        self.down2 = nn.MaxPool2d(2)
        self.res_down2 = ResBlock(ch2, ch3, emb_dim=256)
        self.down3 = nn.MaxPool2d(2)
        self.res_down3 = ResBlock(ch3, ch4, emb_dim=256)
        self.attn_down3 = SelfAttention(ch4)
        self.down4 = nn.MaxPool2d(2)
        self.res_down4 = ResBlock(ch4, ch5, emb_dim=256)
        
        self.bot1 = ResBlock(ch5, ch5, emb_dim=256)
        self.bot2 = ResBlock(ch5, ch5, emb_dim=256)
        
        self.res_up1 = ResBlock(ch5 + ch4, ch4, emb_dim=256)
        self.attn_up1 = SelfAttention(ch4)
        self.res_up2 = ResBlock(ch4 + ch3, ch3, emb_dim=256)
        self.res_up3 = ResBlock(ch3 + ch2, ch2, emb_dim=256)
        self.res_up4 = ResBlock(ch2 + ch1, ch1, emb_dim=256)
        
        # UPDATED: Use out_channels (6)
        self.outc = nn.Conv2d(ch1, out_channels, kernel_size=3, padding=1)

    def _encoder_block(self, x, emb):
        x1 = self.inc(x, emb)
        x2 = self.res_down1(self.down1(x1), emb)
        x3 = self.res_down2(self.down2(x2), emb)
        x4 = self.attn_down3(self.res_down3(self.down3(x3), emb))
        x5 = self.res_down4(self.down4(x4), emb)
        return x1, x2, x3, x4, x5
    
    def _decoder_block(self, x5, x4, x3, x2, x1, emb):
        x5 = self.bot1(x5, emb)
        x5 = self.bot2(x5, emb)
        up5 = F.interpolate(x5, size=x4.shape[2:], mode='nearest')
        x = self.attn_up1(self.res_up1(torch.cat([up5, x4], dim=1), emb))
        up4 = F.interpolate(x, size=x3.shape[2:], mode='nearest')
        x = self.res_up2(torch.cat([up4, x3], dim=1), emb)
        up3 = F.interpolate(x, size=x2.shape[2:], mode='nearest')
        x = self.res_up3(torch.cat([up3, x2], dim=1), emb)
        up2 = F.interpolate(x, size=x1.shape[2:], mode='nearest')
        x = self.res_up4(torch.cat([up2, x1], dim=1), emb)
        return self.outc(x)

    def forward(self, x_noisy, t, x_cond, h_t, noise_aug_level=None):
        emb = self.time_mlp(t) + self.cond_mlp(h_t)
        
        if noise_aug_level is not None:
            emb = emb + self.noise_aug_mlp(noise_aug_level.float())
        
        x = torch.cat([x_noisy, x_cond], dim=1)
        
        if self.training:
            x1, x2, x3, x4, x5 = ckpt_fn(self._encoder_block, x, emb, use_reentrant=False)
            return ckpt_fn(self._decoder_block, x5, x4, x3, x2, x1, emb, use_reentrant=False)
        else:
            x1, x2, x3, x4, x5 = self._encoder_block(x, emb)
            return self._decoder_block(x5, x4, x3, x2, x1, emb)

class RecurrentDiffusionWorldModel(nn.Module):
    def __init__(self, timesteps=200, hidden_dim=256, noise_aug_max=20, cond_drop_prob=0.1, base_ch=32, guidance_scale=1.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_aug_max = noise_aug_max
        self.cond_drop_prob = cond_drop_prob
        self.guidance_scale = guidance_scale
        
        self.unet = ConditionalUNet(cond_dim=hidden_dim, base_ch=base_ch, in_channels=6, out_channels=3)
        
        # Keep the vision encoder at 6 channels for the RNN state
        enc_dim = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, enc_dim, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        
        map_feat_dim = 64
        self.map_encoder = MapEncoder(out_dim=map_feat_dim)
        
        gru_input_dim = enc_dim + 3 + 3 + 3 + 3 + map_feat_dim
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)
        
        self.red_pose_head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 3))
        self.blue_pose_head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 3))
        
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def forward(self, img_seq, pov_seq, pose_seq, act_seq, b_pose_seq, b_act_seq, g_map_seq, mask_seq, img_next_target=None, pov_next_target=None):
        b, seq_len, _, _, _ = img_seq.shape
        device = img_seq.device
        
        h = torch.zeros(b, self.hidden_dim, device=device)
        map_feats = self.map_encoder(g_map_seq[:, 0], mask_seq[:, 0]) 
        
        combined_img_seq = torch.cat([img_seq, pov_seq], dim=2)
        
        for t in range(seq_len):
            z_t = self.encoder(combined_img_seq[:, t])
            vec_in = torch.cat([
                z_t, pose_seq[:, t] / 100.0, act_seq[:, t], 
                b_pose_seq[:, t] / 100.0, b_act_seq[:, t], map_feats / 50.0   
            ], dim=1)
            h = self.gru(vec_in, h)
            
        pred_red_delta = self.red_pose_head(h)
        pred_blue_delta = self.blue_pose_head(h)
        
        if img_next_target is not None and pov_next_target is not None:
            t_diff = torch.randint(0, self.timesteps, (b,), device=device).long()
            
            # 1. Add noise to both images independently
            noise_img = torch.randn_like(img_next_target)
            noise_pov = torch.randn_like(pov_next_target)
            
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
            
            noisy_img = sqrt_alpha_t * img_next_target + sqrt_one_minus_alpha_t * noise_img
            noisy_pov = sqrt_alpha_t * pov_next_target + sqrt_one_minus_alpha_t * noise_pov
            
            # 2. Process conditions and noise augmentations
            cond_img = img_seq[:, -1]
            cond_pov = pov_seq[:, -1]
            noise_aug_level = torch.zeros(b, device=device, dtype=torch.long)
            h_unet = h.clone()
            
            if self.training:
                noise_aug_level = torch.randint(0, self.noise_aug_max, (b,), device=device).long()
                
                # Apply augmentation independently
                aug_n_img = torch.randn_like(cond_img)
                aug_n_pov = torch.randn_like(cond_pov)
                sq_a_aug = self.sqrt_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
                sq_one_m_aug = self.sqrt_one_minus_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
                
                cond_img = sq_a_aug * cond_img + sq_one_m_aug * aug_n_img
                cond_pov = sq_a_aug * cond_pov + sq_one_m_aug * aug_n_pov
                
                drop_mask = torch.rand(b, device=device) < self.cond_drop_prob
                if drop_mask.any():
                    h_unet[drop_mask] = 0.0
                    cond_img[drop_mask] = 0.0
                    cond_pov[drop_mask] = 0.0
            
            # 3. BATCH MULTIPLEXING: Stack along dim=0 (Batch dimension)
            # Shapes go from (B, 3, H, W) to (2B, 3, H, W)
            noisy_stacked = torch.cat([noisy_img, noisy_pov], dim=0)
            cond_stacked = torch.cat([cond_img, cond_pov], dim=0)
            
            # Duplicate the 1D/2D conditions to match the 2B batch size
            t_stacked = torch.cat([t_diff, t_diff], dim=0)
            h_stacked = torch.cat([h_unet, h_unet], dim=0)
            noise_aug_stacked = torch.cat([noise_aug_level, noise_aug_level], dim=0)
            
            # 4. Single Forward Pass
            pred_noise_stacked = self.unet(noisy_stacked, t_stacked, cond_stacked, h_stacked, noise_aug_level=noise_aug_stacked)
            
            # 5. SPLIT back into separate tensors
            pred_noise_img, pred_noise_pov = torch.split(pred_noise_stacked, b, dim=0)
            
            return pred_noise_img, pred_noise_pov, noise_img, noise_pov, pred_red_delta, pred_blue_delta
        
        return pred_red_delta, pred_blue_delta

    def auto_forward(self, img_seq, pov_seq, pose_seq, act_seq, b_pose_seq, b_act_seq, g_map_seq, mask_seq, img_next_target=None, pov_next_target=None, use_autoregressive=True):
        b, seq_len, _, h_img, w_img = img_seq.shape
        device = img_seq.device
        
        h = torch.zeros(b, self.hidden_dim, device=device)
        map_feats = self.map_encoder(g_map_seq[:, 0], mask_seq[:, 0]) 
        
        # 1. Initialize tracking variables for step 0
        curr_img = img_seq[:, 0]
        curr_pov = pov_seq[:, 0]
        curr_pose = pose_seq[:, 0]
        curr_b_pose = b_pose_seq[:, 0]
        
        # Accumulators
        loss_img_total = 0.0
        loss_pov_total = 0.0
        pred_red_deltas = []
        pred_blue_deltas = []
        
        for t in range(seq_len):
            # 2. State Encoding (using CURRENT images, which might be predicted!)
            combined_img = torch.cat([curr_img, curr_pov], dim=1) 
            z_t = self.encoder(combined_img)
            
            vec_in = torch.cat([
                z_t, curr_pose / 100.0, act_seq[:, t], 
                curr_b_pose / 100.0, b_act_seq[:, t], map_feats / 50.0   
            ], dim=1)
            
            h = self.gru(vec_in, h)
            
            pred_red_delta = self.red_pose_head(h)
            pred_blue_delta = self.blue_pose_head(h)
            
            pred_red_deltas.append(pred_red_delta)
            pred_blue_deltas.append(pred_blue_delta)
            
            if img_next_target is not None and pov_next_target is not None:
                # Target is the next frame in the sequence, or the final target if at the end
                target_img = img_seq[:, t+1] if t < seq_len - 1 else img_next_target
                target_pov = pov_seq[:, t+1] if t < seq_len - 1 else pov_next_target
                
                # --- A. NOISE THE TARGET IMAGE ---
                t_diff = torch.randint(0, self.timesteps, (b,), device=device).long()
                
                noise_img = torch.randn_like(target_img)
                noise_pov = torch.randn_like(target_pov)
                
                sqrt_alpha_t = self.sqrt_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
                
                noisy_target_img = sqrt_alpha_t * target_img + sqrt_one_minus_alpha_t * noise_img
                noisy_target_pov = sqrt_alpha_t * target_pov + sqrt_one_minus_alpha_t * noise_pov
                
                # --- B. PROCESS CONDITIONS (Autoregressive Context) ---
                cond_img = curr_img.clone()
                cond_pov = curr_pov.clone()
                h_unet = h.clone()
                noise_aug_level = torch.zeros(b, device=device, dtype=torch.long)
                
                if self.training:
                    noise_aug_level = torch.randint(0, self.noise_aug_max, (b,), device=device).long()
                    
                    aug_n_img = torch.randn_like(cond_img)
                    aug_n_pov = torch.randn_like(cond_pov)
                    sq_a_aug = self.sqrt_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
                    sq_one_m_aug = self.sqrt_one_minus_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
                    
                    cond_img = sq_a_aug * cond_img + sq_one_m_aug * aug_n_img
                    cond_pov = sq_a_aug * cond_pov + sq_one_m_aug * aug_n_pov
                    
                    drop_mask = torch.rand(b, device=device) < self.cond_drop_prob
                    if drop_mask.any():
                        h_unet[drop_mask] = 0.0
                        cond_img[drop_mask] = 0.0
                        cond_pov[drop_mask] = 0.0
                
                # --- C. BATCH MULTIPLEXING & PREDICTION ---
                noisy_stacked = torch.cat([noisy_target_img, noisy_target_pov], dim=0)
                cond_stacked = torch.cat([cond_img, cond_pov], dim=0)
                t_stacked = torch.cat([t_diff, t_diff], dim=0)
                h_stacked = torch.cat([h_unet, h_unet], dim=0)
                noise_aug_stacked = torch.cat([noise_aug_level, noise_aug_level], dim=0)
                
                pred_noise_stacked = self.unet(noisy_stacked, t_stacked, cond_stacked, h_stacked, noise_aug_level=noise_aug_stacked)
                pred_noise_img, pred_noise_pov = torch.split(pred_noise_stacked, b, dim=0)
                
                # --- D. ACCUMULATE LOSS FOR THIS STEP ---
                loss_img_total += F.mse_loss(pred_noise_img, noise_img)
                loss_pov_total += F.mse_loss(pred_noise_pov, noise_pov)
                
                # --- E. ESTIMATE x_0 FOR THE NEXT STEP ---
                # Crucial: wrapped in torch.no_grad() to prevent massive computational graphs
                with torch.no_grad():
                    # DDPM x_0 prediction formula
                    pred_x0_img = (noisy_target_img - sqrt_one_minus_alpha_t * pred_noise_img) / sqrt_alpha_t
                    pred_x0_pov = (noisy_target_pov - sqrt_one_minus_alpha_t * pred_noise_pov) / sqrt_alpha_t
                    
                    # Ensure predictions stay valid pixel ranges [-1, 1]
                    pred_x0_img = torch.clamp(pred_x0_img, -1.0, 1.0)
                    pred_x0_pov = torch.clamp(pred_x0_pov, -1.0, 1.0)
            
            # 3. AUTOREGRESSIVE UPDATE FOR STEP t+1
            if t < seq_len - 1:
                curr_pose = curr_pose + pred_red_delta
                curr_b_pose = curr_b_pose + pred_blue_delta
                if img_next_target is not None:
                    curr_img = pred_x0_img.detach()  
                    curr_pov = pred_x0_pov.detach()

        if img_next_target is not None and pov_next_target is not None:
            # Average the loss over the sequence length
            return loss_img_total / seq_len, loss_pov_total / seq_len, pred_red_deltas[-1], pred_blue_deltas[-1]
        
        return pred_red_deltas[-1], pred_blue_deltas[-1]
        
        
    @torch.no_grad()
    def sample(self, img_seq, pov_seq, pose_seq, act_seq, b_pose_seq, b_act_seq, g_map_seq, mask_seq):
        """Runs the diffusion sampling loop using batch multiplexing."""
        self.eval() 
        b, seq_len, _, h_img, w_img = img_seq.shape
        device = img_seq.device
        
        h = torch.zeros(b, self.hidden_dim, device=device)
        map_feats = self.map_encoder(g_map_seq[:, 0], mask_seq[:, 0]) 
        combined_img_seq = torch.cat([img_seq, pov_seq], dim=2)
        
        for t in range(seq_len):
            z_t = self.encoder(combined_img_seq[:, t])
            vec_in = torch.cat([
                z_t, pose_seq[:, t]/100.0, act_seq[:, t], 
                b_pose_seq[:, t]/100.0, b_act_seq[:, t], map_feats/50.0
            ], dim=1)
            h = self.gru(vec_in, h)
            
        pred_red_delta = self.red_pose_head(h)
        pred_blue_delta = self.blue_pose_head(h)
        
        # Start with separate noise tensors
        x_t_img = torch.randn((b, 3, h_img, w_img), device=device)
        x_t_pov = torch.randn((b, 3, h_img, w_img), device=device)
        
        # Stack conditions
        cond_stacked = torch.cat([img_seq[:, -1], pov_seq[:, -1]], dim=0)
        h_stacked = torch.cat([h, h], dim=0)

        h_uncond = torch.zeros_like(h_stacked)
        cond_uncond = torch.zeros_like(cond_stacked)
        
        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            t_stacked = torch.cat([t_tensor, t_tensor], dim=0)
            x_t_stacked = torch.cat([x_t_img, x_t_pov], dim=0)

            # --- CFG HERE ---
            x_in = torch.cat([x_t_stacked, x_t_stacked], dim=0)
            t_in = torch.cat([t_stacked, t_stacked], dim=0)
            cond_in = torch.cat([cond_stacked, cond_uncond], dim=0)
            h_in = torch.cat([h_stacked, h_uncond], dim=0)
            
            # pred_noise_stacked = self.unet(x_noisy=x_t_stacked, t=t_stacked, x_cond=cond_stacked, h_t=h_stacked)

            # Predict both
            pred_noise_all = self.unet(x_noisy=x_in, t=t_in, x_cond=cond_in, h_t=h_in)
            pred_cond, pred_uncond = pred_noise_all.chunk(2)
            
            # Apply CFG formula
            pred_noise_stacked = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)   
            
            # Split back for DDPM math
            pred_noise_img, pred_noise_pov = torch.split(pred_noise_stacked, b, dim=0)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            if i > 0:
                noise_img = torch.randn_like(x_t_img)
                noise_pov = torch.randn_like(x_t_pov)
            else:
                noise_img = torch.zeros_like(x_t_img)
                noise_pov = torch.zeros_like(x_t_pov)
                
            x_t_img = (1 / torch.sqrt(alpha_t)) * (x_t_img - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise_img) + torch.sqrt(beta_t) * noise_img
            x_t_pov = (1 / torch.sqrt(alpha_t)) * (x_t_pov - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise_pov) + torch.sqrt(beta_t) * noise_pov
            
        pred_img = torch.clamp(x_t_img, -1.0, 1.0)
        pred_pov = torch.clamp(x_t_pov, -1.0, 1.0)
        
        return pred_img, pred_pov, pred_red_delta, pred_blue_delta

@torch.no_grad()
def autoregressive_rollout(model, buffer, device, rollout_steps=50, seq_len=8):
    model.eval()
    
    # 1. Grab a continuous chunk of data to get the ground truth ACTIONS and MAP
    (i_full, pov_full, p_full, a_full, 
     g_map, mask_full, 
     b_p_full, b_a_full, 
     _, _, _, _) = buffer.sample_sequence(batch_size=1, seq_len=rollout_steps + seq_len, device=device)

    # 2. Extract the static global map
    track_padded = g_map[0, 0].cpu().numpy()
    track_mask = mask_full[0, 0].cpu().numpy()
    track = track_padded[track_mask]

    # 3. Initialize the Sliding Windows with the "Burn-in" sequence (first frames)
    curr_img_seq = i_full[:, :seq_len].clone()
    curr_pov_seq = pov_full[:, :seq_len].clone()
    curr_p_seq = p_full[:, :seq_len].clone()
    curr_b_p_seq = b_p_full[:, :seq_len].clone()
    
    # Storage for visualizations
    generated_td_imgs = []
    generated_pov_imgs = []
    generated_red_poses = [curr_p_seq[0, -1].cpu().numpy()]
    generated_blue_poses = [curr_b_p_seq[0, -1].cpu().numpy()]

    print(f"Starting {rollout_steps}-step Autoregressive Rollout...")

    # 4. The Autoregressive Loop
    for step in range(rollout_steps):
        curr_a_seq = a_full[:, step : step + seq_len]
        curr_b_a_seq = b_a_full[:, step : step + seq_len]
        
        # Run generation
        pred_img, pred_pov, pr_red_d, pr_blue_d = model.sample(
            curr_img_seq, curr_pov_seq, curr_p_seq, curr_a_seq, 
            curr_b_p_seq, curr_b_a_seq, g_map[:, 0:1], mask_full[:, 0:1]
        )

        # Calculate new absolute poses
        new_red_pose = curr_p_seq[:, -1] + pr_red_d
        new_blue_pose = curr_b_p_seq[:, -1] + pr_blue_d

        # Store outputs for logging (keep as tensors)
        generated_td_imgs.append(pred_img)
        generated_pov_imgs.append(pred_pov)
        generated_red_poses.append(new_red_pose[0].cpu().numpy())
        generated_blue_poses.append(new_blue_pose[0].cpu().numpy())

        # SLIDING WINDOW UPDATE
        curr_img_seq = torch.cat([curr_img_seq[:, 1:], pred_img.unsqueeze(1)], dim=1)
        curr_pov_seq = torch.cat([curr_pov_seq[:, 1:], pred_pov.unsqueeze(1)], dim=1)
        curr_p_seq = torch.cat([curr_p_seq[:, 1:], new_red_pose.unsqueeze(1)], dim=1)
        curr_b_p_seq = torch.cat([curr_b_p_seq[:, 1:], new_blue_pose.unsqueeze(1)], dim=1)

    return generated_td_imgs, generated_pov_imgs, generated_red_poses, generated_blue_poses, track

# ==========================================
# 4. SETUP & INITIALIZATION
# ==========================================
if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = args.img_h, args.img_w
    BATCH_SIZE = args.batch_size
    SEQ_LEN = args.seq_len
    EPOCHS = args.epochs
    STEPS_PER_EPOCH = args.steps_per_epoch
    COLLECT_STEPS = args.collect_steps
    LEARNING_RATE = args.lr
    DIFFUSION_TIMESTEPS = args.diffusion_timesteps
    DDIM_INFERENCE_STEPS = args.ddim_steps

    set_seed(args.seed)

    agent_ckpt = load_from_hub(
        repo_id="igpaub/ppo-CarRacing-v2",
        filename="ppo-CarRacing-v2.zip"
    )

    agent = PPO.load(agent_ckpt, device=DEVICE)
    base_env = MultiCarRacingContinuous(difficulty=1.0)
    env = PerspectiveFirstPersonWrapper(base_env)
    buffer = TransitionSequenceBuffer(capacity=args.buffer_capacity, img_h=IMG_H, img_w=IMG_W)

    WARMUP_STEPS = args.warmup_steps
    print(f"Warming up buffers: Collecting {WARMUP_STEPS} initial transitions per GPU...")
    collect_rollouts(env, agent, buffer, n_steps=WARMUP_STEPS, n_episodes=50)

    print("Warmup complete! Starting Recurrent Diffusion Training (v2: Noise Aug + DDIM + Attention + EMA)...")
    history_noise_loss, history_pose_loss = [], []
    experiment_name = "experiment_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")

    model = RecurrentDiffusionWorldModel(
        timesteps=DIFFUSION_TIMESTEPS,
        hidden_dim=args.hidden_dim,
        noise_aug_max=args.noise_aug_max,
        cond_drop_prob=args.cond_drop_prob,
        guidance_scale=args.guidance_scale,
        base_ch=32
    ).to(DEVICE)

    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Cosine LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * STEPS_PER_EPOCH, eta_min=args.lr_min
    )

    POSE_LOSS_WEIGHT = 10.0
    best_val_loss = 9999
    criterion_img = torch.nn.MSELoss() 
    criterion_pose = torch.nn.MSELoss()

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    print(f"OK: Starting Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss_total, epoch_loss_img, epoch_loss_pov, epoch_loss_pose = 0.0, 0.0, 0.0, 0.0
        
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}")
        if epoch > 0:
            collect_rollouts(env, agent, buffer, n_steps=COLLECT_STEPS, n_episodes=10)
            
        for step in pbar:
            # 1. Sample Batch
            (i_seq, i_first_seq, p_seq, a_seq, 
             g_map_seq, mask_seq, 
             b_p_seq, b_a_seq, 
             i_next, i_first_next, 
             delta_p, delta_b_p) = buffer.sample_sequence(BATCH_SIZE, SEQ_LEN, DEVICE)
            
            # 2. Forward Pass (Calculates Image loss internally across sequence)
            loss_img, loss_pov, pred_red_delta, pred_blue_delta = model.auto_forward(
                img_seq=i_seq, pov_seq=i_first_seq, pose_seq=p_seq, act_seq=a_seq, 
                b_pose_seq=b_p_seq, b_act_seq=b_a_seq, g_map_seq=g_map_seq, mask_seq=mask_seq, 
                img_next_target=i_next, pov_next_target=i_first_next
            )
            
            # 3. Calculate pose Losses (Comparing final predicted delta to target)
            loss_red = criterion_pose(pred_red_delta, delta_p)
            loss_blue = criterion_pose(pred_blue_delta, delta_b_p)
            loss_pose = loss_red + loss_blue
            
            total_loss = loss_img + loss_pov + (POSE_LOSS_WEIGHT * loss_pose)
            # 4. Backprop
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 5. Logging
            epoch_loss_total += total_loss.item()
            epoch_loss_img += loss_img.item()
            epoch_loss_pov += loss_pov.item()
            epoch_loss_pose += loss_pose.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Img": f"{loss_img.item():.4f}", "Pose": f"{loss_pose.item():.4f}"})

        avg_loss = epoch_loss_total / STEPS_PER_EPOCH
        avg_img_loss = epoch_loss_img / STEPS_PER_EPOCH
        avg_pov_loss = epoch_loss_pov / STEPS_PER_EPOCH
        
        writer.add_scalar("Training/Total loss", avg_loss, epoch)
        writer.add_scalar("Training/Image loss", avg_img_loss, epoch)
        writer.add_scalar("Training/POV loss", avg_pov_loss, epoch)
        writer.add_scalar("Training/LR", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")
        
        # 6. VISUALIZATION
        if epoch % 1 == 0:
            # Grab just the first item in the batch (index 0:1 preserves the batch dimension)
            pred_img, pred_pov, pr_red_d, pr_blue_d = model.sample(
                i_seq[0:1], i_first_seq[0:1], p_seq[0:1], a_seq[0:1], 
                b_p_seq[0:1], b_a_seq[0:1], g_map_seq[0:1], mask_seq[0:1]
            )
                
            # Map math for the visualizer
            track_padded = g_map_seq[0, 0].cpu().numpy()
            track_mask = mask_seq[0, 0].cpu().numpy()
            track = track_padded[track_mask]
                
            gt_red = p_seq[0, -1].cpu().numpy() + delta_p[0].cpu().numpy()
            gt_blue = b_p_seq[0, -1].cpu().numpy() + delta_b_p[0].cpu().numpy()
            pred_red = p_seq[0, -1].cpu().numpy() + pr_red_d[0].detach().cpu().numpy()
            pred_blue = b_p_seq[0, -1].cpu().numpy() + pr_blue_d[0].detach().cpu().numpy()
                
            val_img_loss = criterion_img(pred_img, i_next[0:1])
            val_pov_loss = criterion_img(pred_pov, i_first_next[0:1])
            
            writer.add_scalar("Validation/Image", val_img_loss, epoch)
            writer.add_scalar("Validation/Pose", val_pov_loss, epoch)

            # --- AUTOREGRESSIVE VIDEO ROLLOUT FOR TENSORBOARD ---
            gen_td, gen_pov, gen_red_p, gen_blue_p, track_map = autoregressive_rollout(
                model, buffer, DEVICE, rollout_steps=50, seq_len=SEQ_LEN
            )

            # Stacking along dim=1 turns it into (1, T, C, H, W)
            vid_td_tensor = torch.stack(gen_td, dim=1)
            vid_pov_tensor = torch.stack(gen_pov, dim=1)

            # 3. Normalize values to [0, 1] range for TensorBoard
            # Based on your previous code, it looks like your tensors are in [-1, 1]
            vid_td_tensor = (vid_td_tensor + 1.0) / 2.0
            vid_pov_tensor = (vid_pov_tensor + 1.0) / 2.0
            
            # Clamp to ensure strict compliance with [0, 1] limits
            vid_td_tensor = torch.clamp(vid_td_tensor, 0.0, 1.0)
            vid_pov_tensor = torch.clamp(vid_pov_tensor, 0.0, 1.0)

            # --- PLOT HALLUCINATED TRAJECTORY ---
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title(f"Hallucinated Trajectory | Epoch {epoch+1}")
            
            # Plot the track
            ax.plot(track_map[:, 0], track_map[:, 1], c='gray', linewidth=3, alpha=0.5, label="Track")
            
            # Convert the full lists of poses to numpy arrays
            red_history = np.array(gen_red_p)
            blue_history = np.array(gen_blue_p)
            
            # Plot the generated paths
            ax.plot(red_history[:, 0], red_history[:, 1], c='red', alpha=0.5, label='Red Path')
            ax.plot(blue_history[:, 0], blue_history[:, 1], c='blue', alpha=0.5, label='Blue Path')
            
            # Scatter the final positions
            ax.scatter(red_history[-1, 0], red_history[-1, 1], c='red', s=80)
            ax.scatter(blue_history[-1, 0], blue_history[-1, 1], c='blue', s=80)
            
            ax.set_aspect('equal', adjustable='datalim')
            ax.legend()
            
            # Write to TensorBoard
            writer.add_video("Test/Top-Down", vid_td_tensor, epoch, fps=15)
            writer.add_video("Test/POV", vid_pov_tensor, epoch, fps=15)
            writer.add_figure("Test/Trajectory", fig, epoch)
            plt.close(fig)


            if val_img_loss < best_val_loss:
                best_val_loss = val_img_loss
                torch.save(model.state_dict(), f"./world_models/best_world_model_epoch.pth")
                print(f"Saved model: {epoch+1}")

        writer.flush()

    writer.close()
    print("Training Complete!")