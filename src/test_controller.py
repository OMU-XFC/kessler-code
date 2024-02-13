import time
from typing import Dict, Tuple
from kesslergame import KesslerController, KesslerGame, Scenario, StopReason

# See: https://github.com/ThalesGroup/kessler-game/blob/main/examples/test_controller.py
class TestController(KesslerController):
    def __init__(self):
        pass

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust = 300
        turn_rate = 180
        fire = False
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Test Controller"

def main():
    game = KesslerGame()

    scenario = Scenario(num_asteroids=0, time_limit=50)
    controller = TestController()

    run_step = True
    for score, perf_list in game.run(scenario=scenario, controllers=[controller], run_step=run_step, stop_on_no_asteroids=False):
        print(score.stop_reason)
        pass
    scenario = Scenario(num_asteroids=25, time_limit=50)
    controller = TestController()

    run_step = True

    game_env = game.run(scenario=scenario, controllers=[controller], run_step=True)
    while True:
        try:
            score, perf_list, game_state = next(game_env)
            reward = get_reward(game_state)
            terminated = score.stop_reason != StopReason.not_stopped
            truncated = score.stop_reason == StopReason.time_expired
        except StopIteration:
            print("Done!")
            break


    # for score, perf_list in game.run(scenario=scenario, controllers=[controller], run_step=run_step):
    #
    #     print(score.stop_reason)
    #     pass


def get_reward(game_state):
    return 0


if __name__ == '__main__':
    main()
