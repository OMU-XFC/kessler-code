from typing import Dict, Tuple

import numpy as np
from kesslergame import KesslerGame, KesslerController, Scenario

from src.lib import parse_game_state
from src.roomba_controller import RoombaController
from src.sniper_controller import SniperController


class CombinedController(KesslerController):
    def __init__(self):
        self.sniper = SniperController()
        self.roomba = RoombaController()
        self.active = 'SNIPER'

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        state = parse_game_state(ship_state, game_state)
        nearest_asteroid_dist = np.min(state['asteroids']['polar_positions'][:, 0])

        if nearest_asteroid_dist < 75:
            self.active = 'ROOMBA'
        elif nearest_asteroid_dist >= 175 and self.active == 'ROOMBA':
            self.active = 'SNIPER'
            self.sniper.activate()

        if self.active == 'SNIPER':
            return self.sniper.actions(ship_state, game_state)
        else:
            return self.roomba.actions(ship_state, game_state)


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
         num_asteroids=10
    )

    game = KesslerGame()
    game.run(scenario=scenario, controllers=[CombinedController()])


if __name__ == '__main__':
    main()
