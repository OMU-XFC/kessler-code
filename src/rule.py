import itertools
import re
import numpy as np

def membership(x, k):
    #q1が0から始まるからk=k-1
    K = 3
    if k==3: return 1.0
    b = 1/(K-1)
    a = k/(K-1)
    
    return (max(0, (1-np.abs(a-x)/b)))

def fits(x, K):
    r = [membership(x, k) for k in range(K)]
    return r

def fitness(x_array, K):
    rr = [fits(x, K) for x in x_array]
    return rr

def fitnesses(X, K):
    rrr = [fitness(x_array, K) for x_array in X]
    return rrr

# テキストファイルのパス
file_path = 'inout/inout1.txt'

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
    if len(values) == 24:  # 要素数が24の場合はXに格納
        X.append([float(val.strip()) for val in values])
    elif len(values) == 2:  # 要素数が2の場合はYに格納
        Y.append([float(val.strip()) for val in values])

# XとYをNumpy配列に変換
X = np.array(X)
Y = np.array(Y)


less_zero = np.array(np.sum(X==0.0, axis=0))
less_zero = less_zero<len(X)/2

# If more than half of zero, the column is deleted.
print(X.shape)
X = X[:,less_zero]


# 各変数が取り得る値のリスト
possible_values = [0, 1, 2, 3]

# 24変数の組み合わせを生成
variable_combinations = itertools.product(possible_values, repeat=24)
row_max = np.max(X, axis=0)
X = X / row_max
#Normalize X between 0 and 1 in each column


fitt = fitnesses(X, 3)
maxes = [[np.argmax(sublist) for sublist in subs] for subs in fitt]
print(maxes)

def Rule_average(Y, maxes):
    pp = [[Y[maxes[:,i]==j] for j in range(3)] for i in range(16)]
    return pp
    
# pp = Rule_average(Y, maxes)
# for i in range(16):
#     for j in range(3):
#         print(f"{i}, {j}")
#         print(np.average(pp[i][j], axis=0))

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

unique_X_count = len(xy_dict)  # ユニークなXの要素数
y_counts = {x_row_str: len(y_values) for x_row_str, y_values in xy_dict.items()}  # 各Xに対応するyの要素数を辞書に格納

print("ユニークなXの要素数:", unique_X_count)
print("各Xに対応するyの要素数:")
for x_row_str, count in y_counts.items():
    print(x_row_str, ":", count)