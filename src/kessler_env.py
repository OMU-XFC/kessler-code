import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TestEnv(gym.Env):
    def __init__(self, map_size=(600, 600)):
        self.observation_space = spaces.Dict(
            {
                "ship_position": spaces.Box(low=0, high=600, shape=(2,)),
                "ship_speed": spaces.Box(low=0, high=240, shape=(1,)),
                "ship_heading": spaces.Box(low=-180, high=180, shape=(1,)),
            }
        )

        self.action_space = spaces.Dict(
            {
                "thrust": spaces.Box(low=-480, high=480, shape=(1,)),
                "turn_rate": spaces.Box(low=-180, high=180, shape=(1,)),
            }
        )

        def reset(self, seed=None):
            super().reset(seed=seed)
            pass

        def step(self, action):
            pass

