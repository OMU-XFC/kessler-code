import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.envs.radar_env import get_radar


class TestRadar(unittest.TestCase):
    def test_radar_simple(self):
        # A single asteroid, 250 units directly in front of the ship (in the "middle" zone)
        centered_coords = np.array([[250, 0]])
        asteroid_radii = np.array([8])
        radar_zones = [100, 300, 500]

        radar = get_radar(centered_coords, asteroid_radii, radar_zones)

        # Slice area - 20,000pi
        # Asteroid area - 64pi
        expected = np.array([
            0, 0, 0, 0, # Far
            64 / 20000., 0, 0, 0, # Middle
            0, 0, 0, 0, # Near
        ])
        assert_allclose(radar, expected, atol=1e-7)
