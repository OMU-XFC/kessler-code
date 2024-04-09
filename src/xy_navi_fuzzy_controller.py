import numpy as np
from typing import Dict, Tuple
from kesslergame import KesslerController, KesslerGame, Scenario

'''
How can we hand-craft a fuzzy controller that navigates to a specific region of the map?

A very specific fuzzy membership function: When the ship is within that region, set the thrust to zero
    - Advanced: Put the ship into "reverse" - if the ship is going fast, use a negative thrust
Concern: The weighted voting may make it difficult to exercise fine-grained control over the ship
For example, imagine a very general rule: If X is "don't care" and Y is "don't care" then...
    As long as that rule has Thrust T, T > 0, then it will be _impossible_ for the ship to ever stop
    In fact, in an N rule system, the absolute minimum thrust will be T / N
    Assigning a numerical weight to each rule to increase or decrease its influence may help... need to look into it!
    
Test 1: Stop the ship within X ranges [100, 200] (don't care about Y range)


'''

N_ATTRIBUTES = 2
N_RULES = 3
SHIP_MAX_TURN = 180
SHIP_MAX_THRUST = 480
MAP_SIZE = 800

fuzzy_sets = np.array([

])

class XY_Navi_Test(KesslerController):
    def __init__(self):
        # 0: Don't care
        # 1: Low
        # 2: High
        self.fuzzy_rule = np.array([
            [2, 2],
            [0, 2],
            [1, 0],
        ])

        self.rule_outputs = np.array([
            [0.1 * SHIP_MAX_THRUST, 0.50 * SHIP_MAX_TURN, 1.0, 0.0],
            [SHIP_MAX_THRUST, 0.50 * SHIP_MAX_TURN, 0.25, 0.0],
            [0.1 * SHIP_MAX_THRUST, -0.9 * SHIP_MAX_TURN, 0.25, 0.0],
        ])

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # obs = np.random.random(size=(N_ATTRIBUTES,))
        obs = np.array(ship_state['position']) / MAP_SIZE
        affinity = get_affinity(obs, self.fuzzy_rule)
        weighted_votes = affinity[:, np.newaxis] * self.rule_outputs
        total_weight = np.sum(affinity)
        final_output = np.sum(weighted_votes, axis=0) / total_weight
        thrust, turn, fire, mine = final_output[0], final_output[1], final_output[2] > 0.5, final_output[3] > 0.5
        return thrust, turn, fire, mine

    @property
    def name(self) -> str:
        return "OMU Example Controller"


def get_affinity(X, rules):
    compat = np.ones(shape=(N_RULES, N_ATTRIBUTES))
    ob2 = np.broadcast_to(X, (N_RULES, N_ATTRIBUTES))

    for i in range(N_FUZZY_SETS):
        if i == 0:  # Don't care
            membership = np.ones_like(ob2)
        elif i == 1:  # Low
            membership = 1 - ob2
        else:  # High
            membership = ob2
        mask = rules == i
        compat[mask] = membership[mask]
    return np.prod(compat, axis=1)

import time

def main():
    controller = XY_Navi_Test()
    game = KesslerGame()
    gen = game.run_step(scenario=Scenario(num_asteroids=5,
                                      map_size=(MAP_SIZE, MAP_SIZE),
                                      ship_states=[{
                                          'lives': 999,
                                          'position': (MAP_SIZE / 2, MAP_SIZE / 2)
                                      }]),
                    controllers=[controller])

    for i in range(60):
        next(gen)



if __name__ == '__main__':
    main()
