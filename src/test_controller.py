import time
from typing import Dict, Tuple
from kesslergame import KesslerController, KesslerGame, Scenario, StopReason
from navigation_scenario import *

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
    controller = TestController()
    b = benchmark(controller)
    print(b)


if __name__ == '__main__':
    main()
