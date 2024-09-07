from gymnasium.spaces import Discrete, Tuple, Box
import gymnasium as gym
import numpy as np

class CliffWalkingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CliffWalkingWrapper, self).__init__(env)
        self.nrow, self.ncol = self.env.shape
        self.target_position = np.array([3, 11], dtype=np.float32)
        self.observation_space = Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.nrow - 1, self.ncol - 1], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        
    def reset(self, **kwargs):
        self.num_step = 0
        state, info = self.env.reset(**kwargs)
        row, col = divmod(state, self.ncol)
        return np.array([row, col], dtype=np.float32), info

    def step(self, action):
        self.num_step += 1
        if not isinstance(action, int):
            action = action.item()
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.num_step > 1000:
            terminated = True
        
        row, col = divmod(state, self.ncol)
        current_position = np.array([row, col], dtype=np.float32)
        
        distance_to_target = np.sum(np.abs(current_position - self.target_position))
        additional_reward = 1.0 / (distance_to_target + 1)
        total_reward = reward + additional_reward
        return np.array([row, col], dtype=np.float32), total_reward, terminated, truncated, info
