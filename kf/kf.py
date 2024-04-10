
"""
カルマンフィルタのちょっとしたパッケージを作りたくなった。
"""

import math
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, TypeVar
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, TypeVar, Annotated
from dataclasses import dataclass
from collections import namedtuple
from kf.dtypes import *

from kf.common.kf import update_single as kf_update_single

__all__ = [
    "LowPassFilter",
    "KalmanFilter_SingleObservation",
]

class Filter_Base(metaclass=ABCMeta): #ほとんど意味をなしていないベースクラス
    def __init__(self) -> None:
        self._x: SCALAR|np.ndarray|None #状態変数の記録用
    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """状態更新"""
        pass

class LowPassFilter(Filter_Base):
    """簡易ローパスフィルタ。差分系"""
    _x: SCALAR
    alpha: SCALAR
    def __init__(self, alpha: SCALAR) -> None:
        """初期化
        """
        super().__init__() #意味のないsuper
        self.alpha = alpha
        self._x = None # type: ignore # 初期値のNoneの無視
    def update(self, y: SCALAR) -> SCALAR:
        """状態更新。

        parameters
        ----------
            y: SCALAR
                ノイズが乗った観測値

        returns
        -------
            SCALAR
                フィルタリング結果"""
        if self._x == None:
            self._x = y
        self._x = (1-self.alpha)*self._x + self.alpha*y
        return self._x
    @property
    def state(self) -> SCALAR:
        """状態変数

        returns
        -------
            SCALAR
                前回のフィルタリング値"""
        return self._x
    
class KFs_Base(Filter_Base):
    _a_or_f: ARRAY_N_N|FUNC_VECTOR_N_TO_N
    _c_or_h: ARRAY_M_N|ARRAY_1_N|FUNC_VECTOR_N_TO_M
    _b:      ARRAY_N_V
    _q:      ARRAY_V_V|SCALAR
    _r:      ARRAY_M_M|SCALAR
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    _g:      ARRAY_N_M
    def __init__(self, 
        transition_matrix_or_function:            ARRAY_N_N|FUNC_VECTOR_N_TO_N|None         =None, #a
        observation_matrix_or_function:           ARRAY_M_N|ARRAY_N|FUNC_VECTOR_N_TO_M|None =None, #c
        noise_model_matrix:                       ARRAY_N_V|None                            =None, #b
        system_noise_covariance_matrix_or_number: ARRAY_V_V|SCALAR|None                     =None, #q
        observation_covariance_matrix_or_number:  ARRAY_M_M|SCALAR|None                     =None, #r
        initial_state:                            ARRAY_N|None                              =None, #init_x
        initial_prediction_covariance_matrix:     ARRAY_N_N|None                            =None, #init_p
    ) -> None: #この書き方は果たして本当に良いのだろうか
        """カルマンフィルタ系のフィルタの基底クラス"""
        self._a_or_f= transition_matrix_or_function # type: ignore # 初期値Noneの無視
        if isinstance(observation_matrix_or_function, np.ndarray) and len(observation_matrix_or_function.shape) == 1:
            self._c_or_h = observation_matrix_or_function.reshape(1, len(observation_matrix_or_function))
        else:
            self._c_or_h = observation_matrix_or_function # type: ignore # 初期値Noneの無視
        self._b = noise_model_matrix # type: ignore # 初期値Noneの無視
        self._q = system_noise_covariance_matrix_or_number # type: ignore # 初期値Noneの無視
        self._r = observation_covariance_matrix_or_number # type: ignore # 初期値Noneの無視
        if initial_state == None:
            self._x = None # type: ignore # 初期値Noneの無視
        else:
            self._x = initial_state.reshape(len(initial_state), 1) # type: ignore # 一つ前で例外を省いているため無視
        self._p = initial_prediction_covariance_matrix # type: ignore # 初期値Noneの無視
        self._g = None # type: ignore # 初期値Noneの無視
    @property
    def transition_matrix_or_function(self) -> ARRAY_N_N|FUNC_VECTOR_N_TO_N:
        return self._a_or_f
    @transition_matrix_or_function.setter
    def transition_matrix_or_function(self, value: ARRAY_N_N|FUNC_VECTOR_N_TO_N) -> None:
        self._a_or_f = value

    @property
    def noise_model_matrix(self) -> ARRAY_N_V:
        return self._b
    @noise_model_matrix.setter
    def noise_model_matrix(self, value: ARRAY_N_V|ARRAY_N) -> None:
        if len(value.shape) == 1:
            self._b = value.reshape(len(value), 1)
        else:
            self._b = value

    @property
    def observation_matrix_or_function(self) -> ARRAY_M_N|ARRAY_N|FUNC_VECTOR_N_TO_M:
        if isinstance(self._c_or_h, np.ndarray) and self._c_or_h.shape[0] == 1:
            return self._c_or_h.reshape(len(self._c_or_h))
        else:
            return self._c_or_h
    @observation_matrix_or_function.setter
    def observation_matrix_or_function(self, value: ARRAY_M_N|ARRAY_N|FUNC_VECTOR_N_TO_M) -> None:
        if isinstance(value, np.ndarray) and len(value.shape) == 1:
            self._c_or_h = value.reshape(1, len(value))
        elif callable(value):
            self._c_or_h = value
        else:
            ValueError("呼び出し可能でも配列でもありません")

    @property
    def system_noise_covariance_matrix_or_number(self) -> ARRAY_V_V|SCALAR:
        return self._q
    @system_noise_covariance_matrix_or_number.setter
    def system_noise_covariance_matrix_or_number(self, value: ARRAY_V_V|SCALAR) -> None:
        self._q = value
    
    @property
    def observation_covariance_matrix_or_number(self) -> ARRAY_M_M|SCALAR:
        return self._r
    @observation_covariance_matrix_or_number.setter
    def observation_covariance_matrix_or_number(self, value: ARRAY_M_M|SCALAR) -> None:
        self._r = value
    
    @property
    def state_vector(self) -> ARRAY_N:
        return self._x.reshape(len(self._x))
    @state_vector.setter
    def state_vector(self, value: ARRAY_N) -> None:
        self._x = value.reshape(len(value), 1)

    @property
    def prediction_covariance_matrix(self) -> ARRAY_N_N:
        return self._p
    @prediction_covariance_matrix.setter
    def prediction_covariance_matrix(self, value: ARRAY_N_N) -> None:
        self._p = value
    
    @property
    def kalman_gain(self) -> ARRAY_N_M:
        return self._g
    
    @property
    def prediction_observation(self) -> SCALAR|ARRAY_N:
        if callable(self._c_or_h):
            return self._c_or_h(self._x).reshape(len(self._r.shape[0]))
        else:
            result = self._c_or_h@self._x
            if len(result) == 1:
                result = float(result)
            else:
                result = result.reshape(len(self._r.shape[0]))
            return result

class KalmanFilter_SingleObservation(KFs_Base):
    _a_or_f: ARRAY_N_N
    _c_or_h: ARRAY_N
    _b:      ARRAY_N_1
    _q:      SCALAR
    _r:      SCALAR
    _x:      ARRAY_N
    _p:      ARRAY_M_N
    def __init__(self, 
        transition_matrix:                    ARRAY_N_N|None =None, #a
        observation_matrix:                   ARRAY_N|None   =None, #c
        noise_model_matrix:                   ARRAY_N_1|None =None, #b
        system_noise_covariance_number:       SCALAR|None    =None, #q
        observation_covariance_number:        SCALAR|None    =None, #r
        initial_state:                        ARRAY_N|None   =None, #init_x
        initial_prediction_covariance_matrix: ARRAY_M_N|None =None, #init_p
    ) -> None:
        """線形定常カルマンフィルタ

        観測値は1つのスカラ

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : np.ndarray
                推移行列 
                size : n, n
            observation_matrix : np.ndarray
                観測行列
                size : n
            noise_model_matrix : np.ndarray
                ノイズモデル
                size : n
            system_noise_covariance_number : float
                ノイズモデルのノイズの分散
            observation_covariance_number : float
                観測値の分散
            init_x : np.ndarray
                状態変数の初期値
                size : n
            init_p : np.ndarray
                共分散行列の
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            = transition_matrix,
            observation_matrix_or_function           = observation_matrix,
            noise_model_matrix                       = noise_model_matrix,
            system_noise_covariance_matrix_or_number = system_noise_covariance_number,
            observation_covariance_matrix_or_number  = observation_covariance_number,
            initial_state                            = initial_state,
            initial_prediction_covariance_matrix     = initial_prediction_covariance_matrix,
        )
    def update(self, observation: SCALAR) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        if any(isinstance(attribute, type(None)) for attribute in 
               (self._x, self._p, self._a_or_f, self._c_or_h, self._b, self._q, self._r)):
            raise AttributeError("NoneTypeの属性または属性の配列の大きさが違います")
        result = kf_update_single(self._x, self._p, self._a_or_f, self._c_or_h, 
                                  self._b, self._q, self._r, observation)
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector


