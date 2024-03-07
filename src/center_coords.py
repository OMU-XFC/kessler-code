import numpy as np

def center_coords(ship_position, ship_angle, asteroid_positions):
    # TODO - rotate relative to ship heading...
    #ship_angle -= 90
    #ship_angle = np.radians(ship_angle)

    centered_asteroids = asteroid_positions - ship_position
    rho, phi = c2polar(centered_asteroids[:, 0], centered_asteroids[:, 1])

    # Rotate all coordinates to be relative to the front of the ship
    #phi -= ship_angle
    #phi = np.mod(phi, 2 * np.pi)
    x, y = p2cart(rho, phi)

    return rho, np.degrees(phi), x, y

def c2polar(x, y):
    z = x + 1j * y
    return np.abs(z), np.angle(z)

def p2cart(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)
