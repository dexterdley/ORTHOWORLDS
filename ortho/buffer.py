"""
buffer.py — OrthoTransitionBuffer

Circular replay buffer for synchronized orthographic multi-view transitions.
Stores (views_t, views_next, action, done, env_name) tuples and supports
sampling both random single-step batches and contiguous multi-step trajectories.
"""

import random
import numpy as np
import torch
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"


class OrthoTransitionBuffer:
    """Stores paired orthographic views (t and t+1) with actions across multiple environments."""

    def __init__(self, capacity=4000, img_h=128, img_w=128, max_action_len=8, env_name="Bipedal Walker"):
        self.capacity = capacity
        self.img_h = img_h
        self.img_w = img_w
        self.max_action_len = max_action_len
        self.default_env_name = env_name
        self.ptr = 0
        self.size = 0

        # 4 Views at time t (top, side, rear, fpv/3D)
        self.views_t = np.zeros((capacity, 4, 3, img_h, img_w), dtype=np.uint8)
        # 4 Views at time t+1
        self.views_next = np.zeros((capacity, 4, 3, img_h, img_w), dtype=np.uint8)
        # Actions & Dones
        self.actions = np.zeros((capacity, max_action_len), dtype=np.float32)
        self.action_lens = np.zeros(capacity, dtype=np.int32)
        self.dones = np.zeros(capacity, dtype=bool)
        # Environment name per transition
        self.env_names = [None] * capacity
        # Camera pose per transition (e.g. 'Center', 'Up', 'Down', 'Left', 'Right')
        self.camera_poses = ["Center"] * capacity

    def _resize(self, img):
        if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
            return cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        return img

    def push(self, top_t, side_t, rear_t, fpv_t,
             top_next, side_next, rear_next, fpv_next,
             action, done, env_name=None, camera_pose="Center"):

        vt = [self._resize(x) for x in [top_t, side_t, rear_t, fpv_t]]
        vn = [self._resize(x) for x in [top_next, side_next, rear_next, fpv_next]]

        # Convert HWC uint8 -> CHW uint8
        self.views_t[self.ptr] = np.stack([np.transpose(v, (2, 0, 1)) for v in vt])
        self.views_next[self.ptr] = np.stack([np.transpose(v, (2, 0, 1)) for v in vn])

        # Safely assign action vector or scalar
        act_arr = np.array(action, dtype=np.float32)
        if act_arr.ndim == 0:
            self.actions[self.ptr, :] = 0.0
            self.actions[self.ptr, 0] = float(act_arr)
            self.action_lens[self.ptr] = 1
        else:
            length = min(len(act_arr), self.max_action_len)
            self.actions[self.ptr, :] = 0.0
            self.actions[self.ptr, :length] = act_arr[:length]
            self.action_lens[self.ptr] = length

        self.dones[self.ptr] = done
        self.env_names[self.ptr] = env_name or self.default_env_name
        self.camera_poses[self.ptr] = camera_pose or "Center"

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_valid_trajectory_starts(self, seq_len, env_filter=None):
        """Finds all start indices `i` with `seq_len` contiguous steps in the same episode."""
        valid_starts = []
        limit = self.size - seq_len + 1
        for i in range(max(0, limit)):
            # Do not cross the circular buffer write pointer if buffer has wrapped
            if self.size == self.capacity and (i < self.ptr < i + seq_len):
                continue
            target_env = env_filter or self.env_names[i]
            if self.env_names[i] != target_env:
                continue
            # Ensure all steps belong to the same env and don't reset before the final step
            valid = True
            for k in range(seq_len - 1):
                if self.env_names[i + k + 1] != target_env or self.dones[i + k]:
                    valid = False
                    break
            if valid:
                valid_starts.append(i)
        return valid_starts

    def sample_trajectory_batch(self, batch_size, seq_len=4, target_device=device, env_filter=None, return_camera_poses=True):
        """
        Samples a batch of contiguous multi-step trajectories of length `seq_len`.
        Returns:
          views_t:          (B, 4, 3, H, W) initial views at step 0 in [-1, 1]
          views_next_seq:   list of `seq_len` tensors, each (B, 4, 3, H, W) in [-1, 1]
          actions_seq:      list of `seq_len` action lists (each of length B)
          env_names_seq:    list of `seq_len` env_name lists (each of length B)
          camera_poses_seq: list of `seq_len` camera pose lists (each of length B) [if return_camera_poses=True]
        """
        valid_starts = self._get_valid_trajectory_starts(seq_len, env_filter=env_filter)
        if not valid_starts:
            max_start = max(1, self.size - seq_len)
            start_idxs = np.random.randint(0, max_start, size=batch_size)
        else:
            start_idxs = np.random.choice(valid_starts, size=batch_size)

        views_t = torch.from_numpy(self.views_t[start_idxs]).float().to(target_device) / 127.5 - 1.0

        views_next_seq = []
        actions_seq = []
        env_names_seq = []
        camera_poses_seq = []

        for s in range(seq_len):
            step_idxs = np.minimum(start_idxs + s, self.size - 1)
            views_next_s = torch.from_numpy(self.views_next[step_idxs]).float().to(target_device) / 127.5 - 1.0
            actions_s = [
                torch.from_numpy(self.actions[idx, :self.action_lens[idx]]).float().to(target_device)
                for idx in step_idxs
            ]
            env_names_s = [self.env_names[idx] for idx in step_idxs]
            camera_poses_s = [self.camera_poses[idx] for idx in step_idxs]

            views_next_seq.append(views_next_s)
            actions_seq.append(actions_s)
            env_names_seq.append(env_names_s)
            camera_poses_seq.append(camera_poses_s)

        if return_camera_poses:
            return views_t, views_next_seq, actions_seq, env_names_seq, camera_poses_seq
        return views_t, views_next_seq, actions_seq, env_names_seq

    def sample_sequence(self, seq_len=16, env_filter=None, return_camera_poses=False):
        """Samples a contiguous sequence of transitions without crossing episode resets."""
        candidates = self._get_valid_trajectory_starts(seq_len, env_filter=env_filter)

        if not candidates:
            for i in range(max(0, self.size - seq_len)):
                if env_filter is None or self.env_names[i] == env_filter:
                    candidates.append(i)

        if not candidates:
            start_idx = random.randint(0, max(0, self.size - seq_len)) if self.size > seq_len else 0
        else:
            start_idx = random.choice(candidates)

        actual_len = min(seq_len, max(1, self.size - start_idx)) if self.size > 0 else 0
        idxs = list(range(start_idx, start_idx + actual_len))
        vt_seq = self.views_t[idxs]
        vn_seq = self.views_next[idxs]
        act_seq = [self.actions[i, :self.action_lens[i]] for i in idxs]
        env_seq = [self.env_names[i] for i in idxs]
        cam_seq = [self.camera_poses[i] for i in idxs]

        if return_camera_poses:
            return vt_seq, vn_seq, act_seq, env_seq, cam_seq
        return vt_seq, vn_seq, act_seq, env_seq
