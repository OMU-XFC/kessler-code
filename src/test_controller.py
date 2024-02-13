import time
from typing import Dict, Tuple
from kesslergame import KesslerController, KesslerGame, Scenario

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
    game = KesslerGame()
    scenario = Scenario(num_asteroids=0, time_limit=50)
    controller = TestController()

    run_step = True
    for score, perf_list in game.run(scenario=scenario, controllers=[controller], run_step=run_step, stop_on_no_asteroids=False):
        print(score.stop_reason)
        pass


if __name__ == '__main__':
    main()
