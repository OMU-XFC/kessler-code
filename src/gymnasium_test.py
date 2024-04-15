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

from src.center_coods2 import center_coords2

THRUST_SCALE, TURN_SCALE = 480.0, 180.0
def train():
    kessler_env = Monitor(KesslerEnv(scenario=accuracy_test_1))
#   kessler_env = make_vec_env(KesslerEnv, n_envs=4)
#   kessler_env = DummyVecEnv([lambda: kessler_env])
#   check_env(kessler_env, warn=True)
    model = PPO("MultiInputPolicy", kessler_env)
    """mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'        Mean reward: {mean_reward:.2f}')

    model.learn(5000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+5000   Mean reward: {mean_reward:.2f}')
    #model.save("kessler-out/5k")
    model.learn(50000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    #model.save(f"kessler-out/50krand{i}")"""
    i = 0

    model.learn(1000000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+1000000 Mean reward: {mean_reward:.2f}')
    model.save(f"kessler-out/Scenariohalf_rand{i}")

    #print("Saving")
    #model.save(f"kessler-out/Scenarios1_rand{i}")

def run():
    kessler_game = KesslerGame()
    scenario = ring_static_left
    controller = SuperDummyController()
    score, perf_list = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=True)




class SuperDummyController(KesslerController):
    def __init__(self):
        self.model = PPO.load("kessler-out/Scenariohalf_rand1")

    @property
    def name(self) -> str:
        return "Super Dummy"
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = self._get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn, fire  = list(action[0])
        print(obs['asteroid_angle'])
        fire_bullet = (fire >= 0.0)
        if 0:
            with open("inout/Scenarios_full.txt", 'a') as f:
                f.write(str(list(obs.values())))
                f.write('\n')
                f.write(f"[{thrust}, {turn}, {fire}]\n")
        return thrust * THRUST_SCALE, turn * TURN_SCALE, fire_bullet, False

    def _get_obs(self, game_state):
        # For now, we are assuming only one ship (ours)
        ship = game_state['ships'][0]
        asteroids = game_state['asteroids']

        asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids])
        rho, phi, x, y = center_coords2(ship['position'], ship['heading'], asteroid_positions)

        asteroid_velocities = np.array([asteroid['velocity'] for asteroid in asteroids])
        asteroid_velocities_relative = asteroid_velocities - ship['velocity']
        asteroid_speed_relative = np.linalg.norm(asteroid_velocities_relative, axis=1)

        asteroid_info = np.stack([
            rho, phi, asteroid_speed_relative
        ], axis=1)

        # Sort by first column (distance)
        asteroid_info = asteroid_info[
            asteroid_info[:, 0].argsort()
        ]
        N_CLOSEST_ASTEROIDS = 5
        # Pad
        padding_len = N_CLOSEST_ASTEROIDS - asteroid_info.shape[0]
        if padding_len > 0:
            asteroid_info = np.concatenate([asteroid_info, np.array(padding_len *[[1000, 180, 0.0]])])
        obs = {
            "asteroid_dist": asteroid_info[:N_CLOSEST_ASTEROIDS, 0],
            "asteroid_angle": asteroid_info[:N_CLOSEST_ASTEROIDS, 1],
            "asteroid_rel_speed": asteroid_info[:N_CLOSEST_ASTEROIDS, 2],
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


