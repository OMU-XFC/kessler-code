from typing import Dict, Tuple

import numpy as np
from kesslergame import KesslerController

from src.center_coods2 import center_coords2
from src.rule import fitnesses, fits, fitness
from src.rulesets import *
from src.inout.aaa import *
from src.inout.clustered_XY11 import X_Clustered11, Y_Clustered11


class FuzzyController(KesslerController):
    def __init__(self):
        pass

    def weighted_average_similar_outs(self, rule_array, out_array, x):
        # 入力ベクトルをnumpy配列に変換
        x = np.array(x)
        rule_array = np.array(rule_array)
        # 一致している個数を数える
        differences = np.sum((rule_array - x) ** 2, axis=1)

        # 類似度が高いものを取り出す
        dif_index = np.argsort(differences)

        # 上位k個の類似度のインデックスを取得
        top_k_indices = dif_index
        weights = np.where(differences[top_k_indices] != 0, 1 / differences[top_k_indices], 1)
        weights[0] = 1
        # 正規化
        sum_weights = np.sum(weights)
        normalized_weights = weights / sum_weights
        # 類似度で加重平均を計算
        weighted_average_out = np.average(out_array[top_k_indices], axis=0, weights=normalized_weights)
        return weighted_average_out

    def switch(self, x):

        rule_array = np.array(X_Clustered11)
        out_array = np.array(Y_Clustered11)


        out = self.weighted_average_similar_outs(rule_array, out_array, x)
        #x = np.array(x)
        # 一致している個数を数える
        #differences = np.sum((rule_array - x) ** 2, axis=1)
        # 類似度が高いものを取り出す
        #dif_index = np.argsort(differences)[0]
        #out = out_array[dif_index]

#
        #print(dif_index[:5])
        ## 結果を出力
        #outs = []
        #for index in dif_index[:5]:
        #    outs.append(out_array[index])
        #out = np.average(outs, axis=0)
        #if len(outs) > 1:
        #    print(outs)
        print(out)
        return out




    def membership(self, x, k):
        # q1が0から始まるからk=k-1
        K = 5
        if k == 5: return 1.0
        b = 1 / (K - 1)
        a = k / (K - 1)

        return (max(0, (1 - np.abs(a - x) / b)))

    def _get_obs(self, ship_state, game_state):
        # For now, we are assuming only one ship (ours)
        ship = ship_state

        # handle asteroids
        asteroids = game_state['asteroids']
        asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids])
        print(ship['position'])
        print(ship['heading'])
        print(asteroid_positions)
        rho, phi, x, y = center_coords2(ship['position'], ship['heading'], asteroid_positions)
        asteroid_velocities = np.array([asteroid['velocity'] for asteroid in asteroids])
        asteroid_velocities_relative = asteroid_velocities - ship['velocity']

        asteroid_speed_relative = np.linalg.norm(asteroid_velocities_relative, axis=1)
        asteroid_info = np.stack([
            rho, phi, asteroid_speed_relative], axis=1)
        # Sort by first column (distance)
        asteroid_info = asteroid_info[
            asteroid_info[:, 0].argsort()]
        N_CLOSEST_ASTEROIDS = 5
        # Pad
        padding_len = N_CLOSEST_ASTEROIDS - asteroid_info.shape[0]
        if padding_len > 0:
            pad_shape = (padding_len, asteroid_info.shape[1])
            asteroid_info = np.concatenate([asteroid_info, np.empty(pad_shape)])

        # handle opponent ship
        if len(game_state['ships']) == 2:
            ship_oppose = game_state['ships'][2 - ship_state['id']]
            opponent_position = np.array([ship_oppose['position']])
            opponent_velocity = np.array(ship_oppose['velocity'])
            opponent_velocity_relative = opponent_velocity - ship['velocity']
            opponent_speed_relative = np.linalg.norm(opponent_velocity_relative)
            rho_oppose, phi_oppose, x_oppose, y_oppose = center_coords2(ship['position'], ship['heading'], opponent_position)
        else:
            rho_oppose, phi_oppose, opponent_speed_relative = 1000, 180, 0.0

        if len(game_state['mines']) > 0:
            mine = game_state['mines']
            # 一番近い地雷を取得
            mine_positions = np.array([mine['position'] for mine in mine])
            rho_mine, phi_mine, x_mine, y_mine = center_coords2(ship['position'], ship['heading'], mine_positions)
            mine_info = np.stack([
                rho_mine, phi_mine
            ], axis=1)
            mine_info = mine_info[
                mine_info[:, 0].argsort()]
            nearest_mine = mine_info[0]
        else:
            nearest_mine = [1000, 180]





        obs = {
            "asteroid_dist": asteroid_info[:N_CLOSEST_ASTEROIDS, 0],
            "asteroid_angle": asteroid_info[:N_CLOSEST_ASTEROIDS, 1],
            "asteroid_rel_speed": asteroid_info[:N_CLOSEST_ASTEROIDS, 2],
            #   "opponent_dist": rho_oppose,
            #   "opponent_angle": phi_oppose,
            #   "opponent_rel_speed": opponent_speed_relative,
            #   "nearest_mine_dist": nearest_mine[0],
            #   "nearest_mine_angle": nearest_mine[1]
        }

        return obs

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = self._get_obs(ship_state, game_state)
        print(obs)
        X = np.concatenate(list(obs.values()))
        row_max = np.array([ 855.79438612, 1000.,         1000.,         1000.,         1000.,
  359.99939828,  359.999461  ,  359.99979122,  359.998528  ,  359.99953199,
  320.03165503,  321.31196369,  322.37405693,  321.79295743,  321.64484877])

        row_min = np.array([5.12234548e+00, 1.27582342e+01, 1.40325813e+01, 1.49979920e+01,
 1.54089230e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 3.65472862e-05, 1.49174801e-03, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,])

        #X = (X - row_min) / (row_max - row_min)
        #fitt = fitness(X, 5)
        #maxes = [np.argmax(sublist) for sublist in fitt]
        aa = self.switch(X)
        thrust, turn, fire = aa
        fire_bullet = (fire >= 0.0)
        #ファジィコントローラの出力をファイルに書き込む
        #with open("inout/out_full22_fuzzcon.txt", 'a') as f:
        #   f.write(str(list(obs.values())))
        #   f.write('\n')
        #   f.write(f"[{thrust}, {turn}, {fire}]")
        THRUST_SCALE = 480
        TURN_SCALE = 180
        fire_bullet = fire >= 0.0
        print(thrust*THRUST_SCALE, turn*TURN_SCALE, fire_bullet, False)
        return thrust * THRUST_SCALE, turn * TURN_SCALE, fire_bullet, False

    @property
    def name(self) -> str:
        return "Test Controller"
