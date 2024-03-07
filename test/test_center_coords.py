import unittest
import numpy as np
from numpy.testing import assert_allclose
from src.center_coords import center_coords


class TestCenterCoords(unittest.TestCase):

    # The ship angle ranges from 0 to 360, with 0' being "right" and 90' being "up"

    # The ship is facing upwards
    def test_ship_up(self):
        pass

    def test_ship_right(self):
        ship_coords = (25, 25)
        angle = 0
        asteroid_coords = np.array([
            (50, 25),
        ])
        expected_coords = np.array([
            [0, 25],
        ])


if __name__ == '__main__':
    unittest.main()
