import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
from collections import deque
from src.center_coords import center_coords

THRUST_SCALE, TURN_SCALE = 480.0, 180.0
N_ASTEROIDS = 3 # obviously, fix this...

class KesslerEnv(gym.Env):
    def __init__(self, scenario):
        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = scenario
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])

        max_pos = max(scenario.map_size)
        self.observation_space = spaces.Dict(
            {
                "ship_position": spaces.Box(low=0, high=max_pos, shape=(2,), dtype=np.float64),
                "ship_speed": spaces.Box(low=-240, high=240, shape=(1,), dtype=np.float64),
                "ship_heading": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float64),
                "asteroid_positions": spaces.Box(low=0, high=max(max_pos, 360), shape=(N_ASTEROIDS, 2), dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller],
                                                    stop_on_no_asteroids=False)
        score, perf_list, game_state = next(self.game_generator)
        return self._get_obs(game_state), self._get_info()

    def step(self, action):
        thrust, turn_rate, fire, drop_mine = action[0] * THRUST_SCALE, action[1] * TURN_SCALE, False, False
        self.controller.action_queue.append(tuple([thrust, turn_rate, fire, drop_mine]))
        try:
            score, perf_list, game_state = next(self.game_generator)
            terminated = False
        except StopIteration as exp:
            score, perf_list, game_state = list(exp.args[0])
            terminated = True
        return self._get_obs(game_state), self._get_reward(), terminated, False, self._get_info()

    def _get_obs(self, game_state):
        # For now, we are assuming only one ship (ours)
        ship = game_state['ships'][0]

        asteroid_position = np.array([asteroid['position'] for asteroid in game_state['asteroids']])
        asteroid_position_relative = center_coords(ship['position'], ship['heading'], asteroid_position)

        obs = {
            "ship_position": np.array(ship['position']),
            "ship_speed": np.array([ship['speed']]),
            "ship_heading": np.array([ship['heading']]),
            "asteroid_positions": asteroid_position_relative
        }
        return obs

    # Just staying alive :)
    def _get_reward(self):
        return 1 / 30

    # Corner reward --- Unsure if we ever really need facilitate different reward functions or not.
    # def _get_reward(self, game_state):
    #     x, y = list(game_state['ships'][0]['position'])
    #     if (x < 50 or x > 350) and (y < 50 or y > 350):
    #         return 0.1
    #     if (x < 25 or x > 375) and (y < 25 or y > 375):
    #         return 0.3
    #     if (x < 5 or x > 395) and (y < 5 or y > 395):
    #         return 1
    #     return 0

    def _get_info(self):
        return {}


class DummyController(KesslerController):
    def __init__(self):
        super(DummyController, self).__init__()
        self.action_queue = deque()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        return self.action_queue.popleft()

    def name(self) -> str:
        return "Hello Mr"
