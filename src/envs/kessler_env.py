import gymnasium as gym
from gymnasium import spaces
from kesslergame import TrainerEnvironment, KesslerController
from typing import Dict, Tuple
from collections import deque

from src.center_coods2 import center_coords2
from src.controller2023 import Controller
from src.reward.tomofuji_reward import tomofuji_reward
from src.scenario_list import *

THRUST_SCALE, TURN_SCALE = 480.0, 180.0
ASTEROID_MAX_SPEED = 180
SHIP_MAX_SPEED = 240
N_CLOSEST_ASTEROIDS = 5

class KesslerEnv(gym.Env):
    def __init__(self, scenario):
        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = scenario
        self.reward_function = tomofuji_reward
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])
        self.prev_state, self.current_state = None, None

        max_dist = np.sqrt(scenario.map_size[0] * scenario.map_size[1]) / 2
        max_rel = ASTEROID_MAX_SPEED + SHIP_MAX_SPEED
        self.observation_space = spaces.Dict(
            {
                "asteroid_dist": spaces.Box(low=0, high=max_dist, shape=(N_CLOSEST_ASTEROIDS,)),
                "asteroid_angle": spaces.Box(low=0, high=360, shape=(N_CLOSEST_ASTEROIDS,)),
                "asteroid_rel_speed": spaces.Box(low=-1 * max_rel, high=max_rel, shape=(N_CLOSEST_ASTEROIDS,)),
                "opponent_dist": spaces.Box(low=0, high=max_dist),
                "opponent_angle": spaces.Box(low=0, high=360),
                "opponent_rel_speed": spaces.Box(low=-1 * max_rel, high=max_rel),
                "nearest_mine_dist": spaces.Box(low=0, high=max_dist),
                "nearest_mine_angle": spaces.Box(low=0, high=360),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        rand = np.random.randint(0, len(Scenario_list))
        self.scenario = Scenario_list[rand]
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller, Controller()])
        score, perf_list, game_state = next(self.game_generator)
        self.prev_state, self.current_state = None, game_state

        return get_obs(game_state), self._get_info()

    def step(self, action):
        # Just always fire, for now...
        thrust, turn_rate, fire, drop_mine = action[0] * THRUST_SCALE, action[1] * TURN_SCALE, action[2]>=0.0, action[3]>=0.0
        self.controller.action_queue.append(tuple([thrust, turn_rate, fire, drop_mine]))
        try:
            score, perf_list, game_state = next(self.game_generator)
            terminated = False
        except StopIteration as exp:
            score, perf_list, game_state = list(exp.args[0])
            terminated = True
        self.update_state(game_state)
        obs = get_obs(game_state)
        return obs, self.reward_function(game_state, self.prev_state, obs), terminated, False, self._get_info()

    def update_state(self, game_state):
        self.prev_state = self.current_state
        self.current_state = game_state

    def _get_info(self):
        return {}


def get_obs(game_state):
    # For now, we are assuming only one ship (ours)
    ship = game_state['ships'][0]

    # handle asteroids
    asteroids = game_state['asteroids']
    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids])
    rho, phi, x, y = center_coords2(ship['position'], ship['heading'], asteroid_positions)
    asteroid_velocities = np.array([asteroid['velocity'] for asteroid in asteroids])
    asteroid_velocities_relative = asteroid_velocities - ship['velocity']
    asteroid_speed_relative = np.linalg.norm(asteroid_velocities_relative, axis=1)
    asteroid_info = np.stack([
        rho, phi, asteroid_speed_relative], axis=1)

    # Sort by first column (distance)
    asteroid_info = asteroid_info[
        asteroid_info[:, 0].argsort()]
    N_CLOSEST_ASTEROIDS = 5
    # Pad
    padding_len = N_CLOSEST_ASTEROIDS - asteroid_info.shape[0]
    if padding_len > 0:
        pad_shape = (padding_len, asteroid_info.shape[1])
        asteroid_info = np.concatenate([asteroid_info, np.empty(pad_shape)])

    # handle opponent ship
    if len(game_state['ships']) == 2:
        ship_oppose = game_state['ships'][1]
        opponent_position = np.array([ship_oppose['position']])
        opponent_velocity = np.array(ship_oppose['velocity'])
        opponent_velocity_relative = opponent_velocity - ship['velocity']
        opponent_speed_relative = np.linalg.norm(opponent_velocity_relative)
        rho_oppose, phi_oppose, x_oppose, y_oppose = center_coords2(ship['position'], ship['heading'],
                                                                    opponent_position)
    else:
        rho_oppose, phi_oppose, opponent_speed_relative = 1000, 180, 0.0

    if len(game_state['mines']) > 0:
        mine = game_state['mines']
        # 一番近い地雷を取得
        mine_positions = np.array([mine['position'] for mine in mine])
        rho_mine, phi_mine, x_mine, y_mine = center_coords2(ship['position'], ship['heading'], mine_positions)
        mine_info = np.stack([
            rho_mine, phi_mine
        ], axis=1)
        mine_info = mine_info[
            mine_info[:, 0].argsort()]
        nearest_mine = mine_info[0]
    else:
        nearest_mine = [1000, 180]

    obs = {
        "asteroid_dist": asteroid_info[:N_CLOSEST_ASTEROIDS, 0],
        "asteroid_angle": asteroid_info[:N_CLOSEST_ASTEROIDS, 1],
        "asteroid_rel_speed": asteroid_info[:N_CLOSEST_ASTEROIDS, 2],
        "opponent_dist": rho_oppose,
        "opponent_angle": phi_oppose,
        "opponent_rel_speed": opponent_speed_relative,
        "nearest_mine_dist": nearest_mine[0],
        "nearest_mine_angle": nearest_mine[1]
    }

    return obs


class DummyController(KesslerController):
    def __init__(self):
        super(DummyController, self).__init__()
        self.action_queue = deque()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        return self.action_queue.popleft()

    def name(self) -> str:
        return "Hello Mr"
