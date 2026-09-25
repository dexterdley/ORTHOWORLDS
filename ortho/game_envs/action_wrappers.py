try:
    import gym
except ImportError:
    import gymnasium as gym

import numpy as np
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


class ContinuousActions2Discrete(gym.ActionWrapper):
    """
    ActionWrapper that maps Discrete actions to Continuous Box actions.
    
    Usage:
        env = MarioEscapeEnv()
        env = ContinuousActions2Discrete(env)
        obs, reward, terminated, truncated, info = env.step(3) # 3 = JUMP
    """
    def __init__(self, env, mapping=None):
        super().__init__(env)
        
        if mapping is not None:
            self.mapping = {k: np.array(v, dtype=np.float32) for k, v in mapping.items()}
        else:
            # Default action mapping for 2D Mario / Platformer environments
            # Continuous Box shape = (2,) -> [h_force (-1..1), jump (>0.5)]
            self.mapping = {
                0: np.array([0.0, -1.0], dtype=np.float32),   # 0: NOOP
                1: np.array([-1.0, -1.0], dtype=np.float32),  # 1: LEFT
                2: np.array([1.0, -1.0], dtype=np.float32),   # 2: RIGHT
                3: np.array([0.0, 1.0], dtype=np.float32),    # 3: JUMP
                4: np.array([1.0, 1.0], dtype=np.float32),    # 4: JUMP_RIGHT
                5: np.array([-1.0, 1.0], dtype=np.float32),   # 5: JUMP_LEFT
            }
            
        self.action_space = spaces.Discrete(len(self.mapping))

    def action(self, act):
        if act is None:
            return self.mapping[0]
        if isinstance(act, (int, np.integer)):
            return self.mapping[act]
        return np.array(act, dtype=np.float32)


class DiscreteActions2Continuous(gym.ActionWrapper):
    """
    ActionWrapper that maps Continuous Box actions back to Discrete action indices.
    
    Given a continuous float array, selects the closest discrete action in mapping
    by Euclidean distance.
    """
    def __init__(self, env, mapping=None):
        super().__init__(env)
        
        if mapping is not None:
            self.mapping = {k: np.array(v, dtype=np.float32) for k, v in mapping.items()}
        else:
            self.mapping = {
                0: np.array([0.0, -1.0], dtype=np.float32),   # 0: NOOP
                1: np.array([-1.0, -1.0], dtype=np.float32),  # 1: LEFT
                2: np.array([1.0, -1.0], dtype=np.float32),   # 2: RIGHT
                3: np.array([0.0, 1.0], dtype=np.float32),    # 3: JUMP
                4: np.array([1.0, 1.0], dtype=np.float32),    # 4: JUMP_RIGHT
                5: np.array([-1.0, 1.0], dtype=np.float32),   # 5: JUMP_LEFT
            }
            
        self.action_keys = list(self.mapping.keys())
        self.action_vecs = np.array([self.mapping[k] for k in self.action_keys], dtype=np.float32)
        
        box_dim = self.action_vecs.shape[1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(box_dim,), dtype=np.float32)

    def action(self, act):
        if isinstance(act, (int, np.integer)):
            return act
        # Find nearest discrete action vector
        act_arr = np.array(act, dtype=np.float32)
        dists = np.linalg.norm(self.action_vecs - act_arr, axis=1)
        best_idx = int(np.argmin(dists))
        return self.action_keys[best_idx]
