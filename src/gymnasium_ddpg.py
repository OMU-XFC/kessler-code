from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from envs import KesslerEnv, get_obs
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
import numpy as np

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train(scenario):
    n_actions = 2
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    kessler_env = Monitor(KesslerEnv(scenario=scenario))
#    check_env(kessler_env, warn=True)
    model = DDPG("MultiInputPolicy", kessler_env, verbose=1,  action_noise=action_noise)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'        Mean reward: {mean_reward:.2f}')

    model.learn(total_timesteps=50000, log_interval=10)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    model.save("out/ddpg_50k")

def run(scenario):
    kessler_game = KesslerGame()
    controller = SuperDummyController()
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller])
    # print(score)


if __name__ == '__main__':
    my_scenario = Scenario(num_asteroids=5, time_limit=180, map_size=(800, 800))
    train(my_scenario)
    run(my_scenario)


class SuperDummyController(KesslerController):
    def __init__(self):
        self.model = DDPG.load("out/ddpg_50k")

    @property
    def name(self) -> str:
        return "Super Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
#        print(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False
