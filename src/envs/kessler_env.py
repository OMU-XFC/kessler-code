import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from kesslergame.scenario_list import *
from typing import Dict, Tuple
from collections import deque


THRUST_SCALE, TURN_SCALE = 480.0, 180.0

class KesslerEnv(gym.Env):
    def __init__(self, map_size=(1000, 800)):
        # Change to train with random scenarios
        rand = np.random.randint(0, len(Scenarios1))
        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = Scenarios1[rand]
        self.game_generator = self.kessler_game.run(scenario=self.scenario, controllers=[self.controller],
                                                    stop_on_no_asteroids=False)

        self.observation_space = spaces.Dict(
            {
                "ast_dist": spaces.Box(low=0, high=1280, shape=(5,)),
                "ast_angle": spaces.Box(low=-180, high=180, shape=(5,)),
                "rel_speed": spaces.Box(low=-480, high=480, shape=(5,))
            }
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # two continuous values

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        rand = np.random.randint(0, len(Scenarios1))
        print(rand)
        self.scenario = Scenarios1[rand]

        self.game_generator = self.kessler_game.run(scenario=self.scenario, controllers=[self.controller],
                                                    run_step=True, stop_on_no_asteroids=False)
        score, perf_list, game_state = next(self.game_generator)
        self.save_state(game_state)

        return self._get_obs(game_state), self._get_info()

    def step(self, action):
        thrust, turn_rate, = action[0] * THRUST_SCALE, action[1] * TURN_SCALE
        fire, drop_mine = action[2] >= 0.5, False
        self.fire_bullet = fire
        self.controller.action_queue.append(tuple([thrust, turn_rate, fire, drop_mine]))
        try:
            score, perf_list, game_state = next(self.game_generator)
            terminated = False
        except StopIteration as exp:
            score, perf_list, game_state = list(exp.args[0])
            terminated = True
        return self._get_obs(game_state), self._get_reward(game_state), terminated, False, self._get_info()

        # ひとつ前のステップにおける状態を保存する

    def save_state(self, game_state):
        self.prev_state = game_state




    def _get_obs(self, game_state):
        # For now, we are assuming only one ship (ours)
        ship = game_state['ships'][0]
        ast_list = np.array(game_state["asteroids"])
        if len(ast_list) == 0:
            return {
                "ast_dist": np.array([1280.0] * 5),
                "ast_angle": np.array([180] * 5),
                "rel_speed": np.array([0] * 5)
            }
        # Receive ship and game states from the game, and calculate the distances to the five nearest asteroids.
        # The state is a 10-dimensional vector, [x1, theta1, x2, theta2, ..., x5, theta5], where xi is the distance to the
        # ith nearest asteroid, and thetai is the angle to the ith nearest asteroid.







        # Dist from spaceship (d_x,d_y)
        dist_xylist = [np.array(ship['position']) - np.array(ast['position']) for ast in ast_list]
        dist_avoid_list = dist_xylist.copy()

        # よける部分に関しては画面端のことを考える，弾丸はすり抜けないから狙撃に関しては考えない
        for xy in dist_avoid_list:
            if xy[0] > 500:
                xy[0] -= 1000
            elif xy[0] < -500:
                xy[0] += 1000
            if xy[1] > 400:
                xy[1] -= 800
            elif xy[1] < -400:
                xy[1] += 800
        dist_avoid_list = np.array([np.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_avoid_list])
        sorted2_idx = np.argsort(dist_avoid_list)

        # sorteddict is a list of asteroids' info sorted by distance
        sorteddict = ast_list[sorted2_idx]

        # Consider the 5 nearest asteroids
        search_list = np.array(sorteddict[0:5])

        # dist_list is a list of the 5 nearest asteroids' distance
        dist_list = np.array(dist_avoid_list[sorted2_idx][0:5])
        ship_pos = np.array(ship['position'])
        angles = []
        for pos in search_list:
            angle_ast = np.degrees(np.arctan2(pos['position'][1] - ship_pos[1], pos['position'][0] - ship_pos[0]))
            # 角度を0から360度の範囲に調整
            angle_ast = (angle_ast + 360) % 360
            angle = angle_ast - ship['heading']
            angle = (angle + 360) % 360
            angles.append(angle)

        # 角度を0から1に正規化
        #normalized_angles = [angle / 360.0 for angle in angles]
        asteroids_info = np.stack((dist_list, angles), axis=1)


        # relative speed of the asteroid and the ship
        # relative speed of the asteroid and the ship
        rel_pos = np.array([pos['position'] for pos in search_list]) - np.array(ship['position'])
        rel_velocity = np.array([pos['velocity'] for pos in search_list]) - np.array(ship['velocity'])
        rel_speed = np.array([np.dot(-relpos, rel_vel)/np.linalg.norm(relpos) for relpos, rel_vel in zip(rel_pos, rel_velocity)])



        # if there is any asteroid in front of the ship, fire the bullet
        #self.fire_bullet = (angdiff_front < 5 or angdiff_front > 355) and min(dist_list1) < 400

        # if there are less than 5 asteroids, add dummy data
        num_missing = 5 - len(asteroids_info)
        if num_missing > 0:
            # make dummy data
            padding = np.array([1280.0, 180] * num_missing).reshape(-1, 2)
            dist_padding = np.array([1280.0] * num_missing)
            angle_padding = np.array([180] * num_missing)
            speed_padding = np.array([0] * num_missing)
            dist_list = np.concatenate((dist_list, dist_padding))
            angles = np.concatenate((angles, angle_padding))
            rel_speed = np.concatenate((rel_speed, speed_padding))


        obs = {
            "ast_dist": dist_list,
            "ast_angle": angles,
            "rel_speed": rel_speed,
        }

        return obs

    def _get_reward(self, game_state):
        # dist = np.linalg.norm(np.array(game_state['ships'][0]['position']))
        # return max(0, (50**2 - dist) / 50**2)
        prev_ship = self.prev_state['ships'][0]
        prev_ast_list = np.array(self.prev_state["asteroids"])
        prev_astnum = len(prev_ast_list)
        ast_list = np.array(game_state["asteroids"])
        ship = game_state['ships'][0]

        hit_ast = False
        collision = False

        # collision detection by is_respawning status
        if prev_ship['is_respawning'] == False and ship['is_respawning'] == True:
            collision = True

        #compare the numbers of asteroids in the current and previous steps
        # and detect the asteroids that have been destroyed

        if (len(ast_list) == prev_astnum + 2 or len(ast_list) == prev_astnum - 1) and not collision:
            hit_ast = True

        reward = 0.0
        if self.fire_bullet:
            reward -= 5.0
        if hit_ast:
            reward += 100.0
        if collision:
            reward -= 1000.0
        self.save_state(game_state)

        return reward


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
