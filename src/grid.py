import time
import random

import numpy as np
from kessler_game.new.Controller_neuro import Controller_neuro
from kessler_game.src.kesslergame import KesslerGame, GraphicsType, Scenario
from Scenario_list import *
from Controlelr23 import Controller23

# 隕石が2機体を囲むように，円状に並んで静止する

def timeout(input_data):
    # Function for testing the timing out by the simulation environment
    wait_time = random.uniform(0.02, 0.03)
    time.sleep(wait_time)


"""
class FuzzyController(ControllerBase):
    @property
    def name(self) -> str:
        return "Example Controller Name"

    def actions(self, ships: Tuple[SpaceShip], input_data: Dict[str, Tuple]) -> None:
        timeout(input_data)

        for ship in ships:
            ship.turn_rate = random.uniform(ship.turn_rate_range[0]/2.0, ship.turn_rate_range[1])
            ship.thrust = random.uniform(ship.thrust_range[0], ship.thrust_range[1])
            ship.fire_bullet = random.uniform(0.45, 1.0) < 0.5
"""

if __name__ == "__main__":
    x1 = np.linspace(0, 400, 1000)
    x2 = np.linspace(0, 180, 1000)
    x3 = np.linspace(0, 180, 1000)
    x4 = np.linspace(0, 180, 1000)
    x5 = np.linspace(0, 180, 1000)
    new_gene = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]
    gene = [-1.64531754e-01, 1.13815001e-04, 9.95596272e-02, 6.52069561e-01, 9.58253695e-01, 6.28739631e-01]
    controller = Controller23(gene, new_gene)
    memm = [(controller.mems(x1, x2)).tolist() for x1, x2 in zip(x1, x2)]













