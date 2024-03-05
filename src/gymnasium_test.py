from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from envs import KesslerEnv
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
import numpy as np
from src.center_coords import center_coords

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train(scenario):
    kessler_env = make_vec_env(KesslerEnv, env_kwargs={'scenario': scenario}, n_envs=4)
#    check_env(kessler_env, warn=True)
#    model = PPO("MultiInputPolicy", kessler_env)
    model = PPO.load("out/current", env=kessler_env)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=30)
    print(f'        Mean reward: {mean_reward:.2f}')

    for i in range(100):
        model.learn(1_000_000)
        mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=30)
        print(f'... Mean reward: {mean_reward:.2f}')
        model.save("out/current")


    # model.learn(5000)
    # mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    # print(f'+5000   Mean reward: {mean_reward:.2f}')
    # model.save("out/5k")
    #
    # model.learn(50000)
    # mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    # print(f'+50000  Mean reward: {mean_reward:.2f}')
    # model.save("out/50k")
    #
    # model.learn(500000)
    # mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    # print(f'+500000 Mean reward: {mean_reward:.2f}')
    # model.save("out/500k")
    #
    # print("Saving")
    # model.save("out/test")

def run(scenario):
    kessler_game = KesslerGame()
    controller = SuperDummyController()
    kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=False)
#    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=False)
    # print(score)


class SuperDummyController(KesslerController):
    def __init__(self):
        self.model = PPO.load("out/current")

    @property
    def name(self) -> str:
        return "Super Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = self._get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
#        print(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

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


if __name__ == '__main__':
    scenario = Scenario(time_limit=180, map_size=(600, 600), ship_states=[
        {'lives': 1, 'position': (300, 300)},
    ], asteroid_states=[
        {'position': (0, 0)},
        {'position': (0, 0)},
        {'position': (0, 0)},
        # {'position': (0, 0)},
        # {'position': (0, 0)},
        # {'position': (0, 0)},
    ])
    train(scenario)
    #run(scenario)

