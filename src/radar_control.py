import os
import numpy as np
from typing import Dict, Tuple

from kesslergame import KesslerGame, KesslerController
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.envs.radar_env import RadarEnv, get_obs
from src.navigation_scenario import *

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train(radar_zones, forecast_frames, name, cold_start=True):
    print(f"Starting {name}...")
    vec_env = make_vec_env(RadarEnv, n_envs=6, env_kwargs={'scenario': scenario_A()})
    if cold_start:
        model = PPO("MultiInputPolicy", vec_env)
        model.save('out/temp')
    os.makedirs(f'out/{name}', exist_ok=True)
    with open(f'out/{name}/data.txt', 'w', encoding='UTF-8') as f:
        training_scenario_list = [scenario_D, scenario_E, scenario_F]
        for i in range(1_000):
            for scenario in training_scenario_list:
                vec_env = make_vec_env(RadarEnv, n_envs=6, env_kwargs={
                    'scenario': scenario(),
                    'radar_zones': radar_zones,
                    'forecast_frames': forecast_frames,
                })
                model = PPO.load('out/temp', env=vec_env)
                model.learn(total_timesteps=324_000)
                model.save('out/temp')
            bench = benchmark(SuperDummyController('out/temp'))
            f.write(f'{i},')
            f.write(','.join([str(x) for x in bench.round(4).flatten()]))
            f.write('\n')

            bench_mean = np.mean(bench)
            print(bench_mean)
            model.save(f'out/{name}/bookmark_{i}')
    print("")

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
        obs = get_obs(game_state=game_state, forecast_frames=30, radar_zones=[100, 300, 500], bumper_range=50)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

def main():
    zones = [100, 300, 500]
    frames = 30
    #train(radar_zones=zones, forecast_frames=frames, name='April3')
    run(scenario_D(seed=2), 'out/April3/bookmark_44')

def run_benchmark():
    controller = SuperDummyController(model_name='out/10_GUNS_OFF_1S_FORECAST/9')
    results = benchmark(controller)
    print(results)
    print(np.mean(results))


if __name__ == '__main__':
    main()
    #run_benchmark()


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
