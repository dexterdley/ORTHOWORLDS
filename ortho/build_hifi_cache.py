"""
build_hifi_cache.py — Render the train_single_dist.py --hifi rollout caches on a machine that can run Godot,
for upload to a training box that can't. Builds one cache per DDP rank, named exactly as train_single_dist.py
looks them up, so the remote finds them and skips collection.

The two image arrays are written straight to disk (memmap): at 12 envs x 4096 steps one rank is ~59 GB.

  python hifi/tools/get_godot.py                      # Windows build: GPU rendering from WSL
  python ortho/build_hifi_cache.py --world_size 4 --num_envs 12 --rollout_steps 4096 --out ~/buffer_cache
Run from the repo root: PPO weights are read from ./checkpoints/ppo_<env>.pt (missing ones fall back to an
untrained policy with only a [WARN] line).
"""
import argparse
import os
import pickle

import numpy as np

from data_rollout import get_registries, collect_all_envs_rollouts, IMG_H, IMG_W
from utils import set_seed
from buffer import OrthoTransitionBuffer


def main(args):
    registry, _ = get_registries(hifi=True)
    capacity = args.rollout_steps * min(args.num_envs, len(registry))
    for rank in (args.ranks if args.ranks else range(args.world_size)):
        name = f"s{args.seed}_c{args.seed + rank}_r{rank}_e{args.num_envs}_n{args.rollout_steps}_{IMG_H}px_hifi"
        cache_dir = os.path.join(os.path.expanduser(args.out), name)
        if os.path.exists(os.path.join(cache_dir, "meta.pkl")):
            print(f"[skip] {cache_dir} already built")
            continue
        os.makedirs(cache_dir, exist_ok=True)

        set_seed(args.seed + rank)   # same seeding as train_single_dist.main for this rank
        buffer = OrthoTransitionBuffer(capacity=capacity, img_h=IMG_H, img_w=IMG_W, max_action_len=8)
        for k in ("views_t", "views_next"):   # swap the (still untouched) RAM arrays for on-disk ones
            shape = getattr(buffer, k).shape
            setattr(buffer, k, np.lib.format.open_memmap(os.path.join(cache_dir, f"{k}.npy"),
                                                         mode="w+", dtype=np.uint8, shape=shape))

        collect_all_envs_rollouts(registry, buffer, rollout_steps=args.rollout_steps,
                                  num_envs=args.num_envs, seed=args.seed + rank)

        # Same files as OrthoTransitionBuffer.save; meta.pkl last, since its presence marks the cache complete
        for k in buffer._ARRAYS:
            arr = getattr(buffer, k)
            if isinstance(arr, np.memmap):
                arr.flush()
            else:
                np.save(os.path.join(cache_dir, f"{k}.npy"), arr[:buffer.size])
        meta = {"size": buffer.size, "ptr": buffer.ptr, "capacity": buffer.capacity, "img_hw": (buffer.img_h, buffer.img_w),
                "env_names": buffer.env_names[:buffer.size], "camera_states": buffer.camera_states[:buffer.size]}
        with open(os.path.join(cache_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"[saved] {cache_dir} ({buffer.size} transitions)")
        del buffer


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build train_single_dist.py --hifi rollout caches locally")
    p.add_argument("--world_size", type=int, required=True, help="nproc_per_node on the training box")
    p.add_argument("--ranks", type=int, nargs="*", help="Build only these ranks (e.g. split across sessions)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_envs", type=int, default=12)
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--out", default="./buffer_cache", help="Cache root; keep it off OneDrive / /mnt/c")
    main(p.parse_args())
