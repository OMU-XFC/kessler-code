from typing import Dict, Tuple

import numpy as np
from kesslergame import KesslerGame, KesslerController, Scenario

from Controller.controller23 import FuzzyController
from Controller.lib import parse_game_state
from Controller.roomba_controller import RoombaController
from Controller.sniper_controller import SniperController


class CombinedController(KesslerController):
    def __init__(self):
        self.roomba = RoombaController()
        self.navi = FuzzyController()
        self.sniper = SniperController()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        state = parse_game_state(ship_state, game_state)
        nearest_asteroid_dist = np.min(state['asteroids']['polar_positions'][:, 0])

        if len(state['mines']['polar_positions']) > 0:
            nearest_mine_dist = np.min(state['mines']['polar_positions'][:, 0])
        else:
            nearest_mine_dist = 1000
        if nearest_asteroid_dist < 100 or nearest_mine_dist < 100:
            active = 'ROOMBA'
            thrust, turn, fire, mine, explanation = self.roomba.action_with_explain(ship_state, game_state)
        elif nearest_asteroid_dist < 400:
            active = 'NAVI'
            thrust, turn, fire, mine, explanation = self.navi.action_with_explain(ship_state, game_state)
        else:
            active = 'SNIPER'
            thrust, turn, fire, mine, explanation = self.sniper.action_with_explain(ship_state, game_state)
        #with open('../out/omu_explanation_log.txt', 'a', encoding='UTF-8') as f:
        #    f.write(my_explanation)
        #    f.write('\n')
        return thrust, turn, fire, mine

    @property
    def name(self) -> str:
        return "Combined Controller"


def main():
    scenario = Scenario(
        ship_states=[
            {
                'position': (400, 400)
            }
        # ], asteroid_states=[
        #     {
        #         'position': (500, 500),
        #         'angle': 0,
        #         'speed': 0,
        #         'size': 4,
        #     }
         ], map_size=(1000, 800),
         num_asteroids=2
    )

    game = KesslerGame()
    game.run(scenario=scenario, controllers=[CombinedController()])


if __name__ == '__main__':
    main()
