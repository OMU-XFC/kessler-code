import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
from collections import deque

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

class KesslerEnv(gym.Env):
    def __init__(self, map_size=(400, 400)):
        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = Scenario(num_asteroids=0, time_limit=180, map_size=map_size)
        self.game_generator = self.kessler_game.run(scenario=self.scenario, controllers=[self.controller],
                                                    run_step=True, stop_on_no_asteroids=False)

        self.observation_space = spaces.Dict(
            {
                "ship_position": spaces.Box(low=0, high=600, shape=(2,), dtype=np.float64),
                "ship_speed": spaces.Box(low=-240, high=240, shape=(1,), dtype=np.float64),
                "ship_heading": spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game_generator = self.kessler_game.run(scenario=self.scenario, controllers=[self.controller],
                                                    run_step=True, stop_on_no_asteroids=False)
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
        return self._get_obs(game_state), self._get_reward(game_state), terminated, False, self._get_info()

    def _get_obs(self, game_state):
        # For now, we are assuming only one ship (ours)
        ship = game_state['ships'][0]
        obs = {
            "ship_position": np.array(ship['position']),
            "ship_speed": np.array([ship['speed']]),
            "ship_heading": np.array([ship['heading']]),
        }
        return obs

    def _get_reward(self, game_state):
        # dist = np.linalg.norm(np.array(game_state['ships'][0]['position']))
        # return max(0, (50**2 - dist) / 50**2)
        x, y = list(game_state['ships'][0]['position'])
        if (x < 50 or x > 350) and (y < 50 or y > 350):
            return 0.1
        if (x < 25 or x > 375) and (y < 25 or y > 375):
            return 0.3
        if (x < 5 or x > 395) and (y < 5 or y > 395):
            return 1
        return 0
        # ship_position = np.array(game_state['ships'][0]['position'])
        # return -1 * np.linalg.norm(ship_position)

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
