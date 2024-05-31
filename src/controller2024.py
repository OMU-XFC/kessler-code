import math
from typing import Tuple, Dict
from kesslergame import KesslerController
import numpy as np
from kesslergame import Ship

from Controller.func import *


# 距離5分割
class NewController(KesslerController):
    """
         A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
        game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
        game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
        """

    def __init__(self):
        genes2 = [-422.56833678, -128.5440438, 390.55000786, -389.1460838, -79.66763919,
                  157.80320246, 94.08457887, 73.67425189, 129.10263867, 126.69284636,
                  108.57530471, 161.21374993, 440.06088036, -32.43618814, -269.25471835,
                  23.97717829, 179.54120818, 180, 11.01848299, 105.48515518,
                  124.83245825, 60.12989947, 84.42780692, 155.87859715, 12.61680382,
                  74.824144, 145.83796102, -5.56917356, 79.90114877, 170.21185764, ]
        gene = [-164.53175401560333, 0.11381500132402, 99.55962719106978, 652.0695605148196, 958.2536948627446,
                113.17313354331807]
        """
        Create your fuzzy logic controllers and other objects here
        """
        left1 = gene[0]
        left2 = gene[1]
        center = gene[2]
        right2 = gene[3]
        right1 = gene[4]
        center_angle = gene[5]

        def membership1(x):
            if x <= left1:
                return [1.0, 0.0, 0.0, 0.0, 0.0]
            elif x <= left2:
                return np.array([1.0 - (x - left1) / (left2 - left1), (x - left1) / (left2 - left1), 0.0, 0.0, 0.0])
            elif x <= center:
                return np.array([0.0, (center - x) / (center - left2),
                                 (x - left2) / (center - left2), 0.0, 0.0])
            elif x <= right2:
                return np.array([0.0, 0.0, (right2 - x) / (right2 - center), (x - center) / (right2 - center), 0.0])
            elif x <= right1:
                return np.array([0.0, 0.0, 0.0, (right1 - x) / (right1 - right2), (x - right2) / (right1 - right2)])
            else:
                return np.array([0.0, 0.0, 0.0, 0.0, 1.0])

        def membership2(angle):
            angle = abs(angle)
            if angle <= center_angle:
                return np.array([1.0 - angle / center_angle, angle / center_angle, 0.0])
            elif angle <= 180:
                return np.array([0.0, 2 - angle / center_angle, angle / center_angle - 1])

        self.membership1 = membership1
        self.membership2 = membership2

        def mems(x, angle):
            nonlocal genes2
            genes_out = np.array(genes2).copy()

            out0 = np.array([genes_out[0], genes_out[15]])
            out1 = np.array([genes_out[1], genes_out[16]])
            out2 = np.array([genes_out[2], genes_out[17]])
            out3 = np.array([genes_out[3], genes_out[18]])
            out4 = np.array([genes_out[4], genes_out[19]])
            out5 = np.array([genes_out[5], genes_out[20]])
            out6 = np.array([genes_out[6], genes_out[21]])
            out7 = np.array([genes_out[7], genes_out[22]])
            out8 = np.array([genes_out[8], genes_out[23]])
            out9 = np.array([genes_out[9], genes_out[24]])
            out10 = np.array([genes_out[10], genes_out[25]])
            out11 = np.array([genes_out[11], genes_out[26]])
            out12 = np.array([genes_out[12], genes_out[27]])
            out13 = np.array([genes_out[13], genes_out[28]])
            out14 = np.array([genes_out[14], genes_out[29]])

            k = membership1(x)
            p = membership2(angle)
            rule0 = k[0] * p[0]
            rule1 = k[0] * p[1]
            rule2 = k[0] * p[2]
            rule3 = k[1] * p[0]
            rule4 = k[1] * p[1]
            rule5 = k[1] * p[2]
            rule6 = k[2] * p[0]
            rule7 = k[2] * p[1]
            rule8 = k[2] * p[2]
            rule9 = k[3] * p[0]
            rule10 = k[3] * p[1]
            rule11 = k[3] * p[2]
            rule12 = k[4] * p[0]
            rule13 = k[4] * p[1]
            rule14 = k[4] * p[2]
            out = ((rule0 * out0) + (rule1 * out1) + (rule2 * out2) + (rule3 * out3) + (rule4 * out4) + (
                    rule5 * out5) + (rule6 * out6) + (
                           rule7 * out7) + (rule8 * out8) + (rule9 * out9) + (rule10 * out10) + (rule11 * out11) + (
                           rule12 * out12) + (rule13 * out13) + (rule14 * out14))
            return out

        self.mems = mems

        center_x = 500
        center_y = 400

    def actions(self, ownship: Dict, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        # 隕石と機体の位置関係のセクション
        ast_list = np.array(input_data["asteroids"])
        # (x,y)で表す，機体からの距離
        dist_xylist = [np.array(ownship['position']) - np.array(ast['position']) for ast in ast_list]
        dist_avoid_list = dist_xylist.copy()
        dist_list1 = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_xylist]

        # よける部分に関しては画面端のことを考える，弾丸はすり抜けないから狙撃に関しては考えない
        sidefromcenter = 500 - ownship['position'][0]
        below_center = 400 - ownship['position'][1]
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
        # ここから考えるのは近傍5つの隕石
        search_list = sorteddict[0:5]
        search_dist = np.array([math.dist(ownship['position'], ast['position']) for ast in search_list])
        angle_dist = [np.array(ast['position']) - np.array(ownship['position']) for ast in search_list]
        angle_dist = [angle360(math.degrees((np.arctan2(near_ang[1], near_ang[0])))) - ownship['heading'] for near_ang
                      in angle_dist]
        aalist = []
        for ang in angle_dist:
            if ang > 180:
                ang -= 360
            elif ang < -180:
                ang += 360
            aalist.append(ang)

        angdiff_front = min(aalist, key=abs)
        angdiff = aalist[0]
        fire_bullet = abs(angdiff_front) < 15 and min(dist_list1) < 400
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

        rule = self.mems(avoidance, angdiff)

        thrust = rule[0]
        turn_rate = rule[1] * np.sign(angdiff)
        if ownship["speed"] >= 0:
            self.str_move = "Moving forward"
        else:
            self.str_move = "Moving backwards"
        if thrust > ownship['thrust_range'][1]:
            thrust = ownship['thrust_range'][1]
        elif thrust < ownship['thrust_range'][0]:
            thrust = ownship['thrust_range'][0]
        if turn_rate > ownship['turn_rate_range'][1]:
            turn_rate = ownship['turn_rate_range'][1]
        elif turn_rate < ownship['turn_rate_range'][0]:
            turn_rate = ownship['turn_rate_range'][0]
        # 前後，回転，射撃のタプルをリターンする
        return thrust, turn_rate, fire_bullet, False

    @property
    def name(self) -> str:
        return "OMU-Let's"

    @property
    def explanation(self) -> str:
        return self.str_move