from typing import Dict, Tuple

import numpy as np
from kesslergame import KesslerGame, KesslerController, Scenario

from src.center_coords import c2p
from src.lib import parse_game_state


class RoombaController(KesslerController):
    def __init__(self):
        pass


    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        state = parse_game_state(ship_state, game_state)
        nearest_asteroid_idx = np.argmin(state['asteroids']['polar_positions'][:, 0])
        angle = state['asteroids']['polar_positions'][nearest_asteroid_idx, 1]
        angle = np.degrees(angle)
        if angle > 180:
            angle -= 360

        if -90 <= angle <= 90:
            # The obstacle is in front. Face towards it, and hit reverse!
            return -480, angle, False, False
        else:
            # The obstacle is behind us
            if angle < 0:
                turn = angle + 180
            else:
                turn = angle - 180
            return 480, turn, False, False


    @property
    def name(self) -> str:
        return "Sniper Controller"


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
         num_asteroids=8
    )

    game = KesslerGame()
    game.run(scenario=scenario, controllers=[RoombaController()])


if __name__ == '__main__':
    main()
