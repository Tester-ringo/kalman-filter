<img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python"></img>

# Kalman-Filter

---

## 概要
pythonでKalmanFilterを簡易的に扱うためのモジュールです

- 実装フィルタ一覧
  - 線形カルマンフィルタ(Kalman Filter)
  - 拡張カルマンフィルタ(Extended Kalman Filter)
  - 無香料カルマンフィルタ(Unscented Kalman Filter)

- 想定環境

  - version
    - python3系
  - サードパーティ製の依存ライブラリ
    - numpy

## 使い方

- [KalmanFilter](../kalman-filter/documents/KalmanFilter.md)
- [ExtendedKalmanFilter](../kalman-filter/documents/ExtendedKalmanFilter.md)
- [UnscentedKalmanFilter](../kalman-filter/documents/UnscentedKalmanFilter.md)

実装フィルタクラス
- KalmanFilter_SingleObservation
    - 観測がスカラーの線形カルマンフィルタ
- KalmanFilter_MultipleObservation
    - 観測がベクトルの線形カルマンフィルタ
- ExtendedKalmanFilter_SingleObservation
    - 観測がスカラーの拡張カルマンフィルタ
- ExtendedKalmanFilter_MultipleObservation
    - 観測がベクトルの拡張カルマンフィルタ
- UnscentedKalmanFilter_SingleObservation
    - 観測がスカラーの無香料カルマンフィルタ
- UnscentedKalmanFilter_MultipleObservation
    - 観測がベクトルの無香料カルマンフィルタ




```python
import kf
import numpy as np


# 初期値を与える

kf_instance = kf.KalmanFilter_SingleObservation()
# 遷移行列を指定
kf_instance.transition_matrix = np.array([[1, 1], [0, 1]])
# 観測ベクトルを指定
kf_instance.observation_matrix = np.array([1, 0])
# ノイズモデル行列を指定
kf_instance.noise_model_matrix = np.array([0.5, 1])
# システムノイズの分散を指定
kf_instance.system_noise_covariance_number = 1e-6
# 観測ノイズの分散を指定v
kf_instance.observation_covariance_number = 3*3
# 推定状態ベクトルの初期値を指定
kf_instance.prediction_state_vector = np.array([0, 0])
# 推定状態ベクトルの分散の初期値を指定
kf_instance.prediction_covariance_matrix = np.array([[100, 0], [0, 100]])


# 推定のループ

while ...:
    # 観測値
    observed_value = ...
    # システムへの制御入力値 (なくても良い)
    input_vector = ...
    # updateメソッドを呼び出し、推定。
    kf_instance.update(observed_value, input_vector)
    # 最新の推定値
    kf_instance.x
```


