import numpy as np

def center_coords(ship_position, ship_angle, asteroid_positions):
    centered_asteroids = asteroid_positions - ship_position
    centered_asteroids = c2polar(centered_asteroids)
    # Rotate all coordinates to be relative to the front of the ship
    centered_asteroids[:, 1] -= ship_angle
    centered_asteroids[:, 1] = np.mod(centered_asteroids[:, 1], 360)
    return centered_asteroids

def c2polar(positions):
    z = positions[:, 0] + 1j * positions[:, 1]
    rho = np.abs(z)
    phi = np.angle(z, deg=True)
    return np.stack([rho, phi], axis=1)
