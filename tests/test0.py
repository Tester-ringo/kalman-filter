#簡易的なカルマンフィルタのフィルタリング例
#カルマンフィルタの信頼区間を求めたいと思ったが、よくわからなかった。

from file24_kf.kf import core
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif' #日本語フォントの設定らしい。（ネットのコピペ）
rcParams['font.sans-serif'] = [
    'Hiragino Maru Gothic Pro', 
    'Yu Gothic', 
    'Meirio', 
    'Takao', 
    'IPAexGothic', 
    'IPAPGothic', 
    'VL PGothic', 
    'Noto Sans CJK JP'
]

# 想定システム
# 等速直線運動をする何かをフィルタリングする事とする
# 
# x_{k} = Ax_{k|k-1} + Bv_{k}  *離散化状態方程式での表現
# y_{k} = C^Tx_{k} + w_{k}
#
# x = [位置, 速度]  *想定している状態変数
#
# A = [[1, 0.01], 
#      [0, 1   ]]  *推移行列
# B = [[1, 0], 
#      [0, 1]]  *名前がわからない。モデルのノイズをこねくり回す行列
# C = [1, 0]  *観測ベクトル
#
# Ver[v] = 1e-4  *システムノイズの分散。今は100%の精度のモデリングを想定したため、小さい値とした
# Ver[w] = 1e+4  *観測ノイズの分散。今回は分散が10,000の白色ガウス雑音とした。
#
# 初期値
#
# p = [[10000, 0    ],
#      [0    , 10000]])  *共分散行列の初期値
# x = [0, 0]  *状態変数の初期値
#

#カルマンフィルタの初期設定
skf = core.KalmanFilter_SingleObservation()
skf.transition_matrix_or_function = np.array([[1, 0.01],[0, 1]]) #A
skf.noise_model_matrix = np.array([1, 1]) #B
skf.observation_matrix_or_function = np.array([1,0]) #C
skf.state_vector = np.array([0, 0]) #x
skf.prediction_covariance_matrix = np.array([[10000, 0],[0, 10000]]) #P
skf.system_noise_covariance_matrix_or_number = 1e-4 #Ver[v]
skf.observation_covariance_matrix_or_number = 1e+4 #Ver[w]

#シミュレーション

n = 1000 #データ数

y_true = [] #記録用リスト
y_observation = []
y_filtered = []

temp_x = -150 #一時変数。位置と速度と1離散時間
temp_dx = 30
temp_dt = 0.01

for i in range(n): #逐次とした
    temp_x += temp_dt*temp_dx
    temp_x_observation = temp_x+random.gauss(0, 100) #シミュレータ
    
    skf.update(temp_x_observation) #カルマンフィルタでのフィルタリング

    y_true.append(temp_x) #記録
    y_observation.append(temp_x_observation)
    y_filtered.append(skf.prediction_observation)

plt.plot(np.arange(n), y_observation, label="観測値", c="#D4D4D4") #プロット
plt.plot(np.arange(n), y_true, label="真値", c="#06FF00")
plt.plot(np.arange(n), y_filtered, label="推定値", c="#009EF5")
plt.legend()
plt.show()
