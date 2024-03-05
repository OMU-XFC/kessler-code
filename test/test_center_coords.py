import unittest
import numpy as np
from numpy.testing import assert_allclose
from src.center_coords import center_coords


class TestCenterCoords(unittest.TestCase):

    # The ship angle ranges from 0 to 360, with 0' being "right" and 90' being "up"
    def test_zero_angle(self):
        ship_coords = (25, 25)
        angle = 0
        asteroid_coords = np.array([
            (50, 25),
            (100, 25),
        ])
        expected_coords = np.array([
            (25, 0),
            (75, 0),
        ])
        output_coords = center_coords(ship_coords, angle, asteroid_coords)
        assert_allclose(expected_coords, output_coords)

    def test_ship_pointing_left(self):
        ship_coords = (100, 100)
        ship_angle = 180
        asteroid_coords = np.array([
            (0, 100),  # The ship is pointing at it, from dist. 100
            (100, 188), # Directly above the ship, but remember the ship is pointing left...
            (150, 150),  # To the upper right of the ship
            (0, 0), # The lower left of the ship
        ])
        expected_coords = np.array([
            (100, 0),
            (88, 270),
            (np.sqrt(50 ** 2 + 50 ** 2), 225),
            (np.sqrt(100 ** 2 + 100 ** 2), 45),
        ])
        output_coords = center_coords(ship_coords, ship_angle, asteroid_coords)
        assert_allclose(expected_coords, output_coords)


if __name__ == '__main__':
    unittest.main()
