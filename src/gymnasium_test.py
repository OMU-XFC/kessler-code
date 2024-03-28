import gymnasium as gym
from kesslergame.scenario_list import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from envs import KesslerEnv
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
import numpy as np

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train():
    kessler_env = Monitor(KesslerEnv())
#    kessler_env = make_vec_env(KesslerEnv, n_envs=4)
#    kessler_env = DummyVecEnv([lambda: kessler_env])
#    check_env(kessler_env, warn=True)
    model = PPO("MultiInputPolicy", kessler_env)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'        Mean reward: {mean_reward:.2f}')

    model.learn(5000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+5000   Mean reward: {mean_reward:.2f}')
    #model.save("kessler-out/5k")

    model.learn(50000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    #model.save("kessler-out/50k")

    model.learn(500000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+500000 Mean reward: {mean_reward:.2f}')
    #model.save("kessler-out/500k")

    print("Saving")
    model.save("kessler-out/accu_test_4")

def run():
    kessler_game = KesslerGame()
    scenario = ring_static_top
    controller = SuperDummyController()
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=True)




class SuperDummyController(KesslerController):
    def __init__(self):
        self.model = PPO.load("kessler-out/wall_top_wrap_1_500k")

    @property
    def name(self) -> str:
        return "Super Dummy"
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = self._get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn, fire = list(action[0])
        fire = (fire >= 0.5)
        return thrust * THRUST_SCALE, turn * TURN_SCALE, fire, False


    def _get_obs(self, game_state):
        # For now, we are assuming only one ship (ours)
        ship = game_state['ships'][0]
        ast_list = np.array(game_state["asteroids"])

        # Receive ship and game states from the game, and calculate the distances to the five nearest asteroids.
        # The state is a 10-dimensional vector, [x1, theta1, x2, theta2, ..., x5, theta5], where xi is the distance to the
        # ith nearest asteroid, and thetai is the angle to the ith nearest asteroid.
        # Dist from spaceship (d_x,d_y)
        dist_xylist = [np.array(ship['position']) - np.array(ast['position']) for ast in ast_list]
        dist_avoid_list = dist_xylist.copy()
        dist_list1 = [np.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_xylist]

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
        # normalized_angles = [angle / 360.0 for angle in angles]
        asteroids_info = np.stack((dist_list, angles), axis=1)

        angdiff_front = min(asteroids_info[:, 1], key=abs)

        # relative speed of the asteroid and the ship
        rel_pos = np.array([pos['position'] for pos in search_list]) - np.array(ship['position'])
        rel_velocity = np.array([pos['velocity'] for pos in search_list]) - np.array(ship['velocity'])
        rel_speed = np.array([np.dot(-relpos, rel_vel)/np.linalg.norm(relpos) for relpos, rel_vel in zip(rel_pos, rel_velocity)])


        # if there is any asteroid in front of the ship, fire the bullet
        #self.fire_bullet = (angdiff_front < 5 or angdiff_front > 355) and min(dist_list1) < 400
        avoidance = np.min(dist_avoid_list)

        # if there are less than 5 asteroids, add dummy data
        num_missing = 5 - len(asteroids_info)
        if num_missing > 0:
            # make dummy data
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


def train_repeat(rewards):
    kessler_env = Monitor(KesslerEnv())
    #    kessler_env = make_vec_env(KesslerEnv, n_envs=4)
    #    kessler_env = DummyVecEnv([lambda: kessler_env])
    #    check_env(kessler_env, warn=True)
    model = PPO("MultiInputPolicy", kessler_env)

    model.learn(50000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    model.save("kessler-out/50k")

if __name__ == '__main__':
   train()
   #run()

