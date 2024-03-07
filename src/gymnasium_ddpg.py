import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from envs import KesslerEnv
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
import numpy as np

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train():
    n_actions = 2
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    kessler_env = Monitor(KesslerEnv())
#    kessler_env = make_vec_env(KesslerEnv, n_envs=4)
#    kessler_env = DummyVecEnv([lambda: kessler_env])
#    check_env(kessler_env, warn=True)
    model = DDPG("MultiInputPolicy", kessler_env, verbose=1,  action_noise=action_noise)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'        Mean reward: {mean_reward:.2f}')


    model.learn(total_timesteps=50000, log_interval=10)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    model.save("ddpgout/50k")


    print("Saving")
    model.save("ddpgout/test")

def run():
    kessler_game = KesslerGame()
    scenario = ring_static_top
    controller = SuperDummyController()
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=False)
    # print(score)


class SuperDummyController(KesslerController):
    def __init__(self):
        #self.model = PPO.load("out/500k")
        self.model = DDPG.load("ddpgout/50k")
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
        obs = {
            "ship_position": np.array(ship['position']),
            "ship_speed": np.array([ship['speed']]),
            "ship_heading": np.array([ship['heading']]),
        }
        print(obs['ship_position'])
        return obs

if __name__ == '__main__':
    train()
    run()

