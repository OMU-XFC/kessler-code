import re
import time
import random


from rule import fitnesses, fits, fitness
import numpy as np
from kesslergame import KesslerGame, GraphicsType, Scenario
from kesslergame import TrainerEnvironment

from src.fuzzy_controller import FuzzyController
from inout.clustered_XY import X_Clustered, Y_Clustered
from inout.Scenarios_full11_cluster import L
from src.scenario_list import Scenario_full


# 隕石が2機体を囲むように，円状に並んで静止する

def timeout(input_data):
    # Function for testing the timing out by the simulation environment
    wait_time = random.uniform(0.02, 0.03)
    time.sleep(wait_time)

def weighted_average_similar_outs(rule_array, out_array, x):
    # 入力ベクトルをnumpy配列に変換
    x = np.array(x)

    differences = np.sum((rule_array - x) ** 2, axis=1)
    # 類似度が高いものを取り出す
    dif_index = np.argsort(differences)

    # 上位k個の類似度のインデックスを取得
    top_k_indices = dif_index[:1]
    weights = np.where(differences[top_k_indices] != 0, 1 / differences[top_k_indices], 0)
    weights[0] = 1

    # 正規化
    sum_weights = np.sum(weights)
    normalized_weights = weights / sum_weights
    # 類似度で加重平均を計算
    weighted_average_out = np.average(out_array[top_k_indices], axis=0, weights=normalized_weights)

    return weighted_average_out

def switch_direct(x):
    rule_array = np.array(X_Clustered)
    out_array = np.array(Y_Clustered)
    x = np.array(x)
    #differences = np.sum((rule_array - x) ** 2, axis=1)
    #top = np.argsort(differences)[0]
    #out = out_array[top]
    out = weighted_average_similar_outs(rule_array, out_array, x)

    return out
def view(x):
    rule_array = rules22
    out_array = np.array(out22)



"""
class FuzzyController(ControllerBase):
    @property
    def name(self) -> str:
        return "Example Controller Name"

    def actions(self, ships: Tuple[SpaceShip], input_data: Dict[str, Tuple]) -> None:
        timeout(input_data)

        for ship in ships:
            ship.turn_rate = random.uniform(ship.turn_rate_range[0]/2.0, ship.turn_rate_range[1])
            ship.thrust = random.uniform(ship.thrust_range[0], ship.thrust_range[1])
            ship.fire_bullet = random.uniform(0.45, 1.0) < 0.5
"""
def run_scenarios():
    controllers = [FuzzyController(), FuzzyController()]
    for scene in Scenario_full:
        kessler_game = TrainerEnvironment()
        score, perf_data = kessler_game.run(scenario=scene, controllers=controllers)
def actions_from_obs(obs):
        X = obs
        row_max = np.array([855.79438612, 1000., 1000., 1000., 1000.,
                            359.99939828, 359.999461, 359.99979122, 359.998528, 359.99953199,
                            320.03165503, 321.31196369, 322.37405693, 321.79295743, 321.64484877])

        row_min = np.array([5.12234548e+00, 1.27582342e+01, 1.40325813e+01, 1.49979920e+01,
                            1.54089230e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            3.65472862e-05, 1.49174801e-03, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ])
        X = (X - row_min) / (row_max - row_min)
        fitt = fitness(X, 5)
        maxes = [np.argmax(sublist) for sublist in fitt]
        aa = switch(maxes)
        thrust, turn, fire = aa
        fire_bullet = (fire >= 0.0)
        #with open("inout/out_full22_fuzzcon.txt", 'a') as f:
        #   f.write(str(obs))
        #   f.write('\n')
        #   f.write(f"[{thrust}, {turn}, {fire}]")
        THRUST_SCALE = 480
        TURN_SCALE = 180
        fire_bullet = fire >= 0.0
        return [thrust, turn, fire]




def direct_actions_from_obs(obs):
    X = obs
    row_max = np.array([855.79438612, 1000., 1000., 1000., 1000.,
                        359.99939828, 359.999461, 359.99979122, 359.998528, 359.99953199,
                        320.03165503, 321.31196369, 322.37405693, 321.79295743, 321.64484877])

    row_min = np.array([5.12234548e+00, 1.27582342e+01, 1.40325813e+01, 1.49979920e+01,
                        1.54089230e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        3.65472862e-05, 1.49174801e-03, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ])
    #X = (X - row_min) / (row_max - row_min)
    #fitt = fitness(X, 5)
    #maxes = [np.argmax(sublist) for sublist in fitt]
    aa = switch_direct(X)
    thrust, turn, fire = aa
    fire_bullet = (fire >= 0.0)
    # with open("inout/out_full22_fuzzcon.txt", 'a') as f:
    #   f.write(str(obs))
    #   f.write('\n')
    #   f.write(f"[{thrust}, {turn}, {fire}]")
    THRUST_SCALE = 480
    TURN_SCALE = 180
    fire_bullet = fire >= 0.0
    return [thrust, turn, fire]


if __name__ == "__main__":
    #NNによる制御の入出力
    file_path = 'inout/Scenarios_full11.txt'

    # ファイルを読み込んでテキストを取得
    with open(file_path, 'r') as file:
        # テキスト全体を1つの文字列として読み込む
        text = file.read()

    # 正規表現パターン
    pattern = re.compile(r"\[(.*?)\]", re.DOTALL)

    # パターンにマッチする部分を検索
    matches = re.findall(pattern, text)

    # inputのデータを格納するリスト
    X = []

    # outputのデータを格納するリスト
    Y = []

    # マッチした結果を処理
    for i, match in enumerate(matches):
        values = match.split(',')
        if len(values) == 15:  # 要素数が17の場合はXに格納
            X.append([float(val.strip()) for val in values])
        elif len(values) == 3:  # 要素数が2の場合はYに格納
            Y.append([float(val.strip()) for val in values])

    ## XとYをNumpy配列に変換
    #X = np.array(X)
    #Y = np.array(Y)
    #print(f"{file_path} loaded")
    #L = []
    ##NNの入力からファジィ制御したものの出力
    file_path = 'inout/Scenarios_full11_cluster.py'
    #i = 0
    #for x in X:
    #    print(i)
    #    line = direct_actions_from_obs(x)
    #    L.append(line)
    #    i += 1
    #ファジィ？による出力

    L = np.array(L)
    Y = np.array(Y)
    with open(file_path, 'a') as f:
        #f.write(f"X={X}\n")
        #f.write(f"Y={Y}\n")
        #f.write(f"L={L}\n")
        f.write(f"L-Y={list(L-Y)}\n")






