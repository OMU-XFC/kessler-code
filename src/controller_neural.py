import math
from typing import Tuple, Dict

import numpy as np
from .func import angle360, ast_angle
from .controller import KesslerController



class Controller_neuro(KesslerController):
# This controller is for calculate the avoidance and angle difference between the ship and the asteroids,
# which are the input of the controller used in 2023 XFC.


    def __init__(self, ):
        pass

    def actions(self, ownship: Dict, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        # 隕石と機体の位置関係のセクション

        ast_list = np.array(input_data["asteroids"])
        # (x,y)で表す，機体からの距離
        dist_xylist = [np.array(ownship['position']) - np.array(ast['position']) for ast in ast_list]
        dist_avoid_list = dist_xylist.copy()
        dist_list1 = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_xylist]

        # よける部分に関しては画面端のことを考える，弾丸はすり抜けないから狙撃に関しては考えない
        for xy in dist_avoid_list:
            if xy[0] > 500:
                xy[0] -= 1000
            elif xy[0] < -500:
                xy[0] += 1000
            if xy[1] > 400:
                xy[1] -= 800
            elif xy[1] < -400:
                xy[1] += 800
        dist_avoid_list = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_avoid_list]

        sorted2_idx = np.argsort(dist_avoid_list)
        sorteddict = ast_list[sorted2_idx]
        # Consider the 5 nearest asteroids
        search_list = np.array(sorteddict[0:5])

        search_dist = np.array([math.dist(ownship['position'], ast['position']) for ast in search_list])
        #speed_list = np.array([math.sqrt(ast['velocity'][0] ** 2 + ast['velocity'][1] ** 2) for ast in search_list])
        angle_dist = [np.array(ast['position']) - np.array(ownship['position']) for ast in search_list]
        angle_dist = [angle360(math.degrees((np.arctan2(near_ang[1], near_ang[0])))) - ownship['heading'] for near_ang
                      in angle_dist]

        asteroids_info = np.stack((search_dist, angle_dist), axis=1)
        for i, ang in enumerate(asteroids_info[:, 1]):
            if ang > 180:
                asteroids_info[i, 1] -= 360
            elif ang < -180:
                asteroids_info[i, 1] += 360
        angdiff_front = min(asteroids_info[:,1], key=abs)

        # if there is any asteroid in front of the ship, fire the bullet
        fire_bullet = abs(angdiff_front) < 10 and min(dist_list1) < 400
        avoidance = np.min(dist_avoid_list)


        if len(input_data['ships']) >= 2:
            angle_ships = ast_angle(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            dist = math.dist(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            if (dist <= avoidance):
                if angle_ships > 180:
                    angle_ships -= 360
                elif angle_ships < -180:
                    angle_ships += 360
                avoidance = dist
                angdiff = abs(angle_ships)


        num_missing = 5 - len(asteroids_info)
        if num_missing > 0:
            # make dummy data
            padding = np.array([999, 180] * num_missing).reshape(-1, 2)
            # add dummy data to the end of the array
            asteroids_info = np.vstack((asteroids_info, padding))


        #asteroids_info = [dist1, angle1, dist2, angle2, ..., dist5, angle5]
        return asteroids_info.flatten(), fire_bullet

    @property
    def name(self) -> str:
        return "OMU-Let's"
