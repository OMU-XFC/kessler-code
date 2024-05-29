import itertools
import re
import numpy as np


def membership(x, k):
    # q1が0から始まるからk=k-1
    K = 5
    if k == K: return 1.0
    b = 1 / (K - 1)
    a = k / (K - 1)

    return (max(0, (1 - np.abs(a - x) / b)))


def fits(x, K):
    r = [membership(x, k) for k in range(K)]
    return r


def fitness(x_array, K):
    rr = [fits(x, K) for x in x_array]
    return rr


def fitnesses(X):
    K = 5
    rrr = [fitness(x_array, K) for x_array in X]
    return rrr


def load_data():
    # テキストファイルのパス
    file_path = 'inout/Scenarios_full12.txt'
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

    # XとYをNumpy配列に変換
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def mincut(X, X_max):
    for x, x_max in zip(X, X_max):
        while (np.prod(x) < 0.3):
            # argminを取得し，その変数を削除
            argmin = np.argmin(x)
            x[argmin] = 1.0
            x_max[argmin] = 5
    return X, X_max


def select_seven(maxes, max_values, Y):
    maxes_keep, max_values_keep, y_keep = [], [], []
    for max, values, y in zip(maxes, max_values, Y):
        if np.sum(np.array(max) == 5) < 7:
            maxes_keep.append(max)
            max_values_keep.append(values)
            y_keep.append(y)
    return maxes_keep, max_values_keep, y_keep
if __name__ == '__main__':

    X, Y = load_data()

    # less_zero = np.array(np.sum(X == 0.0, axis=0))
    # less_zero = less_zero < len(X) / 2
    ## If more than half of zero, the column is deleted.
    # X = X[:, less_zero]
    ## 各変数が取り得る値のリスト
    possible_values = [0, 1, 2, 3, 4, 5]
    # 15変数の組み合わせを生成
    variable_combinations = itertools.product(possible_values, repeat=15)
    row_max = np.max(X, axis=0)
    row_min = np.min(X, axis=0)
    print(row_max)
    print(row_min)
    X = (X - row_min) / (row_max - row_min)
    # Normalize X between 0 and 1 in each column
    fitt = fitnesses(X)
    max_values = [[np.max(sublist) for sublist in subs] for subs in fitt]
    maxes = [[np.argmax(sublist) for sublist in subs] for subs in fitt]
    # もっとも適合度の低い変数を削除していく

    maxes = np.array(maxes)

    # max_values, maxes = mincut(max_values, maxes)
    # maxes, max_values, Y = select_seven(maxes, max_values, Y)
    maxes = np.array(maxes)

    # max_valuesの各列において，各数値のカウントをする
    for i in range(6):
        print(f"{i}の数{np.sum(maxes == i, axis=0)}")
    maxes = maxes.tolist()

    xy_dict = {}
    for i, x_row in enumerate(maxes):
        x_row_str = str(x_row)
        if x_row_str in xy_dict:
            xy_dict[x_row_str].append(Y[i])
        else:
            xy_dict[x_row_str] = [Y[i]]

    # Xの各行が辞書のキーとして存在する場合、対応するyの値を取得する
    result = {}
    for x_row_str, y_values in xy_dict.items():
        result[x_row_str] = y_values

    # ユニークなXの要素数と各Xに対応するyの要素数を出力
    unique_X_count = len(xy_dict)
    y_counts = {x_row_str: len(y_values) for x_row_str, y_values in xy_dict.items()}
    print("ユニークなXの要素数:", unique_X_count)

    # 各Xに対応するYの平均を格納する辞書
    averages_dict = {}

    # xy_dictの各キーと値に対して処理を行う
    for x_row_str, y_values in xy_dict.items():
        # Yの平均を計算する
        y_average = np.mean(y_values, axis=0)
        # 平均を辞書に格納する
        averages_dict[x_row_str] = y_average
    # xy_dictから，キーのみを集めたリストを作る
    keys = list(xy_dict.keys())
    values = list(xy_dict.values())

    # 結果を1列に並べて出力
    print("\n結果:")
    # for x_row_str in xy_dict.keys():
    #    count = y_counts.get(x_row_str, 0)
    #    average = averages_dict.get(x_row_str, np.zeros_like(y_values[0]))
    #    print(f"elif x == {x_row_str}: out={list(average)}")
    rule_array = [eval(key) for key in keys]
    out_array = [list(average) for average in averages_dict.values()]
    print(rule_array)
    print(out_array)
