import os

import gymnasium as gym
from kesslergame.scenario_list import *
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
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



class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'model4_{self.n_calls}.zip')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f'Model saved to {model_path}')
        return True



THRUST_SCALE, TURN_SCALE = 480.0, 180.0
def train():
    #ここで選択するシナリオはダミー
    kessler_env = Monitor(KesslerEnv(scenario=threat_test_1))

    model = PPO("MultiInputPolicy", kessler_env)

    eval_callback = SaveModelCallback(save_freq=10000000, save_path="wcci-out", verbose=1)

    model.learn(1000000000, callback=eval_callback)

    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+1000000 Mean reward: {mean_reward:.2f}')
    model.save(f"kessler-out/Scenario_full")

    #print("Saving")
    #model.save(f"kessler-out/Scenarios1_rand{i}")

def run():
    kessler_game = KesslerGame()

    scenario = ring_closing
    controller = SuperDummyController()
    score, perf_list = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=True)

def run_all():
    kessler_game = TrainerEnvironment()
    for scenario in Scenario_full:
        controller = SuperDummyController()
        score, perf_list = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=True)


class SuperDummyController(KesslerController):
    def __init__(self):
        self.model = PPO.load("wcci-out/model1_70000000.zip")

    @property
    def name(self) -> str:
        return "Super Dummy"
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = self._get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn, fire  = list(action[0])
        fire_bullet = (fire >= 0.0)
        if 1:
            with open("inout/Scenarios_full12.txt", 'a') as f:
                f.write(str(list(obs.values())))
                f.write('\n')
                f.write(f"[{thrust}, {turn}, {fire}]\n")
        return thrust * THRUST_SCALE, turn * TURN_SCALE, fire_bullet, False

    def _get_obs(self, game_state):
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






        obs = {
            "asteroid_dist": asteroid_info[:N_CLOSEST_ASTEROIDS, 0],
            "asteroid_angle": asteroid_info[:N_CLOSEST_ASTEROIDS, 1],
            "asteroid_rel_speed": asteroid_info[:N_CLOSEST_ASTEROIDS, 2],

        }

        return obs


if __name__ == '__main__':
    #run()
    #run_all()
    #print("full12 done")
    train()


