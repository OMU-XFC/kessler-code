import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
from collections import deque
from src.center_coords import center_coords

THRUST_SCALE, TURN_SCALE = 480.0, 180.0
SHIP_MAX_SPEED = 240

class RadarEnv(gym.Env):
    def __init__(self, scenario):
        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = scenario
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])

        self.observation_space = spaces.Dict(
            {
                "radar": spaces.Box(low=0, high=1, shape=(12,)),
                "forecast_1s": spaces.Box(low=0, high=1, shape=(12,)),
                #"radar_n": spaces.Box(low=0, high=1, shape=(12,)),
                #"forecast_1s_n": spaces.Box(low=0, high=1, shape=(12,)),
                #"forecast_3s": spaces.Box(low=0, high=1, shape=(12,)),
                "ship_speed": spaces.Box(low=0, high=SHIP_MAX_SPEED, shape=(1,)),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])
        score, perf_list, game_state = next(self.game_generator)
        return get_obs(game_state), self._get_info()

    def step(self, action):
        ### Guns on / off
        thrust, turn_rate, fire, drop_mine = action[0] * THRUST_SCALE, action[1] * TURN_SCALE, False, False
        self.controller.action_queue.append(tuple([thrust, turn_rate, fire, drop_mine]))
        try:
            score, perf_list, game_state = next(self.game_generator)
            terminated = False
        except StopIteration as exp:
            score, perf_list, game_state = list(exp.args[0])
            terminated = True
        return get_obs(game_state), self._get_reward(), terminated, False, self._get_info()

    def _get_reward(self):
        return 1/30

    def _get_info(self):
        return {}

def get_obs(game_state):
    ship = game_state['ships'][0]
    ship_position = np.array(ship['position'], dtype=np.float64)
    ship_heading = np.radians(ship['heading'])
    ship_velocity = np.array(ship['velocity'], dtype=np.float64)
    ship_speed = np.array([ship['speed']])

    asteroids = game_state['asteroids']
    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
    asteroid_velocity = np.array([asteroid['velocity'] for asteroid in asteroids], dtype=np.float64)
    asteroid_radii = np.array([asteroid['radius'] for asteroid in asteroids])
    map_size = np.array(game_state['map_size'])

    ship_future_position = ship_position + (30 * ship_velocity)
    asteroid_future_positions = asteroid_positions + (30 * asteroid_velocity)

    # ship_future_position_2 = ship_position + (90 * ship_velocity)
    # asteroid_future_positions_2 = asteroid_positions + (90 * asteroid_velocity)

    radar_info = get_radar(ship_position, ship_heading, asteroid_positions, asteroid_radii, map_size)
    forecast_info = get_radar(ship_future_position, ship_heading, asteroid_future_positions, asteroid_radii, map_size)
 #   forecast_info2 = get_radar(ship_future_position_2, ship_heading, asteroid_future_positions_2, asteroid_radii, map_size)

    obs = {
        "radar": radar_info,
        "forecast_1s": forecast_info,
        #"forecast_3s": forecast_info2,
        "ship_speed": ship_speed,
    }

    return obs

def get_radar(ship_position, ship_heading, asteroid_positions, asteroid_radii, map_size):
    centered_asteroids = center_coords(ship_position, ship_heading, asteroid_positions, map_size)
    asteroid_areas = np.pi * asteroid_radii * asteroid_radii
    rho, phi = centered_asteroids[:, 0], centered_asteroids[:, 1]

    is_near   = rho < 100
    is_medium = np.logical_and(rho < 300, rho >= 100)
    is_far    = np.logical_and(rho < 500, rho >= 300)

    is_front  = np.logical_or(phi < 0.25 * np.pi, phi >= 1.75 * np.pi)
    is_left   = np.logical_and(phi < 0.75 * np.pi, phi >= 0.25 * np.pi)
    is_behind = np.logical_and(phi < 1.25 * np.pi, phi >= 0.75 * np.pi)
    is_right  = np.logical_and(phi < 1.75 * np.pi, phi >= 1.25 * np.pi)

    inner_area  = np.pi * 100 * 100
    middle_area = np.pi * 300 * 300
    outer_area  = np.pi * 500 * 500
    # The area of one slice in the outer, middle, and inner donuts
    slice_areas = [(outer_area - middle_area) / 4, (middle_area - inner_area) / 4, inner_area / 4]

    radar_info = np.empty(shape=(12,))
    for idx, distance_mask in enumerate([is_far, is_medium, is_near]):
        slice_area = slice_areas[idx]
        for jdx, angle_mask in enumerate([is_front, is_left, is_behind, is_right]):
            mask = np.logical_and(distance_mask, angle_mask)
            total_asteroid_area = np.sum(asteroid_areas[mask])
            radar_info[idx * 4 + jdx] = total_asteroid_area / slice_area
    return radar_info


class DummyController(KesslerController):
    def __init__(self):
        super(DummyController, self).__init__()
        self.action_queue = deque()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        return self.action_queue.popleft()

    def name(self) -> str:
        return "Hello Mr"
