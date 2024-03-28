import os
from typing import Dict, Tuple

from kesslergame import Scenario, KesslerGame, KesslerController
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.envs.radar_env import RadarEnv, get_obs

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train(scenario, exp_name):
    vec_env = make_vec_env(RadarEnv, n_envs=6, env_kwargs={'scenario': scenario})
    #model = PPO("MultiInputPolicy", vec_env)
    model = PPO.load('out/10_GUNS_ON_1S_FORECAST/136')
    eval_env = Monitor(RadarEnv(scenario=scenario))
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=30, return_episode_rewards=False)
    print(mean_reward)
    return

    os.makedirs(f'out/{exp_name}', exist_ok=True)
    with open(f'out/{exp_name}/data.txt', 'w', encoding='UTF-8') as f:
        for i in range(1000):
            model.learn(total_timesteps=200_000)
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=30)
            model.save(f'out/{exp_name}/{i}')
            print(f'{i:d} .. Mean reward: {mean_reward:.2f}')
            f.write(f'{i:d}\t{mean_reward:.2f}\n')

def run(scenario, model_name):
    kessler_game = KesslerGame()
    controller = SuperDummyController(model_name)
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller])

class SuperDummyController(KesslerController):
    def __init__(self, model_name):
        self.model = PPO.load(model_name)

    @property
    def name(self) -> str:
        return "Super Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

def main():
    n_asteroids = 10
    my_scenario = Scenario(time_limit=600, map_size=(800, 800),
                           asteroid_states=[{'position': (0, 0)}] * n_asteroids,
                           ship_states=[
                               {
                                   'position': (400, 400),
                                   'lives': 1
                               }
                           ])
    train(my_scenario, '10_GUNS_ON_1S_FORECAST')
    #run(my_scenario, 'out/10_GUNS_ON_1S_FORECAST/136')
    pass


if __name__ == '__main__':
    main()


#     marker_size = 50 * (asteroid_size ** 2)
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xlim(-1 * MAP_SIZE, MAP_SIZE)
#     ax.set_ylim(-1 * MAP_SIZE, MAP_SIZE)
#     ax.scatter(x=asteroid_xy[:, 0], y=asteroid_xy[:, 1], s=marker_size)
#
#     ax.plot([-353, 353], [-353, 353], color='k')
#     ax.plot([-353, 353], [353, -353], color='k')
#
#     my_ship = plt.Circle((0, 0), 25, color='red')
#     ax.add_patch(my_ship)
#
#     circle1 = plt.Circle((0, 0), 100, color='k', fill=False)
#     ax.add_patch(circle1)
#
#     circle2 = plt.Circle((0, 0), 300, color='k', fill=False)
#     ax.add_patch(circle2)
#
#     circle3 = plt.Circle((0, 0), 500, color='k', fill=False)
#     ax.add_patch(circle3)
#
#
#     plt.show()
#
#     pass
