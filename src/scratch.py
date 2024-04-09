import numpy as np
import time
from center_coords import center_coords

def main():
    ship = np.random.rand(1,2)
    asteroids = np.random.rand(300,2)
    rho, phi = center_coords(ship_position=ship, ship_angle=0, asteroid_positions=asteroids)
    # near = rho[rho < 0.3]
    # medium = rho[rho < 0.6]
    # far = rho[rho < 1.0]

    # print(asteroids)
    # print(np.stack([rho, phi], axis=1))
#    print(c)

if __name__ == '__main__':
    start = time.time()
    for i in range(1_000):
        main()
    end = time.time()
    print(end - start)
