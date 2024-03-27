import unittest
import numpy as np
from numpy.testing import assert_allclose
from src.center_coords import center_coords


class TestCenterCoords(unittest.TestCase):

    # The ship angle ranges from 0 to 2pi, with 0 meaning "right" and pi/2 meaning "up"

    # Simple tests with the ship facing right
    def test_zero_angle(self):
        ship_coords = np.array([25., 25.])
        angle = 0
        map_size = np.array([500, 500])
        asteroid_coords = np.array([
            [50, 25],  # Front of the ship, 25 units away
            [100, 25], # Front of the ship, 75 units away
            [50, 50],  # Front-left of the ship, 25*1.41 units away
            [25, 100], # Left of the ship, 75 units away
            [0, 25],   # Behind the ship, 25 units away
            [0, 0],    # Behind-right of the ship, 25*1.41 units away
            [50, 0],   # Front-right of the ship, 25*1.41 units away
        ])
        expected_coords = np.array([
            [25, 0],
            [75, 0],
            [25 * np.sqrt(2), 0.25 * np.pi],
            [75, 0.5 * np.pi],
            [25, np.pi],
            [25 * np.sqrt(2), 1.25 * np.pi],
            [25 * np.sqrt(2), 1.75 * np.pi]
        ])

        output = center_coords(ship_coords, angle, asteroid_coords, map_size)
        assert_allclose(output, expected_coords, atol=1e-7)

    # Now the ship is facing "down" -- 3pi/2 radians!
    def test_adjust_angle(self):
        ship_coords = np.array([300., 300.])
        angle = 1.5 * np.pi
        map_size = np.array([5000, 5000])
        asteroid_coords = np.array([
            [300, 200],  # Front of the ship, 100 units away
            [400, 300],  # Left of the ship, 100 units away
            [300, 350],  # Behind the ship, 50 units away
            [200, 400],  # Behind-right of the ship, 141 units away
            [0,   0],      # Front-right of the ship, 1.41 * 300 units away
        ])
        expected_coords = np.array([
            [100, 0],
            [100, 0.5 * np.pi],
            [50, np.pi],
            [100 * np.sqrt(2), 1.25 * np.pi],
            [300 * np.sqrt(2), 1.75 * np.pi],
        ])

        output = center_coords(ship_coords, angle, asteroid_coords, map_size)
        assert_allclose(output, expected_coords, atol=1e-7)

    # Various tests where the shortest distance to each asteroid includes wrapping around the edges of the map
    def test_wraparound(self):
        ship_coords = np.array([400., 400.])
        angle = 0
        map_size = np.array([500, 500])
        asteroid_coords = np.array([
            [100, 400], # In front (looks like behind)
            [400, 100], # To the left (looks like to the right)
            [0,   300], # Front-right (looks like back-right)
        ])
        expected_coords = np.array([
            [200, 0],
            [200, 0.5 * np.pi],
            [100 * np.sqrt(2), 1.75 * np.pi]
        ])
        output = center_coords(ship_coords, angle, asteroid_coords, map_size)
        assert_allclose(output, expected_coords, atol=1e-7)

        ship_coords = np.array([900., 0.])
        angle = np.pi
        map_size = np.array([1000, 1000])
        asteroid_coords = np.array([
            [0,   0],   # Behind (looks like in front)
            [900, 900]  # Left (looks like right)
        ])
        expected_coords = np.array([
            [100, np.pi],
            [100, 0.5 * np.pi]
        ])
        output = center_coords(ship_coords, angle, asteroid_coords, map_size)
        assert_allclose(output, expected_coords, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
