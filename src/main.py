from typing import Dict, Tuple

import torch
from kesslergame import KesslerController
from kesslergame.kessler_game_step import KesslerGameStep
from kesslergame.Scenario_list import *
from kesslergame.neural_rl import Actor


# See: https://github.com/ThalesGroup/kessler-game/blob/main/examples/test_controller.py
class TestController(KesslerController):
    def __init__(self):
        pass

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust = 300
        turn_rate = 180
        fire = True
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Test Controller"


def main():
    # 1sec has 30 steps
    scenario = Scenario(time_limit=60.0, num_asteroids=1)
    # remake model and load weights from file
    model = Actor(10, 2)
    model.load_state_dict(torch.load('sample_model.pth'))

    game = KesslerGameStep()

    # initialize game state, choose a scenario
    states = game.reset(scenario=accuracy_test_1)
    # reward = [bullet_hit, collision, survive, survive to the end]
    rewards = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

    # max 1800 frames
    for i in range(1800):
        action = model(states)
        states, reward_tag, done = game.run_step(action.detach().numpy())
        rewards += reward_tag

        if done:
            break

    print(f"hit={rewards[0]}, collision={rewards[1]}, survive frame = {rewards[2]}, survive to the end = {rewards[3]}")


if __name__ == '__main__':
    main()
