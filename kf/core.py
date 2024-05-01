
"""
カルマンフィルタのちょっとしたパッケージを作りたくなった。
"""

import math
import numpy as np
import numpy.typing as npt
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, TypeVar, Annotated
from dataclasses import dataclass
from collections import namedtuple
from kf.dtypes import *

from kf.common.kf import update_single as kf_update_single
from kf.common.kf import update_multiple as kf_update_multiple

__all__ = [
    "LowPassFilter",
    "KalmanFilter_SingleObservation",
]


class Filter_Base(metaclass=ABCMeta): 
    #ほとんど意味をなしていないベースクラス
    def __init__(self) -> None:
        self._x: SCALAR|np.ndarray|None #状態変数の記録用(意味ない)
    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """状態更新"""
        pass


class LowPassFilter(Filter_Base):
    """簡易ローパスフィルタ。差分系"""
    _x: SCALAR #用途はおかしいが、制作の際の補完のために
    alpha: SCALAR
    def __init__(self, alpha: SCALAR) -> None:
        """初期化"""
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
    _a_or_f: ARRAY_N_N|FUNC_VECTOR_N_I_TO_N
    _c_or_h: ARRAY_M_N|ARRAY_1_N|FUNC_VECTOR_N_TO_M
    _bu:     ARRAY_N_I
    _b:      ARRAY_N_V
    _q:      ARRAY_V_V|SCALAR
    _r:      ARRAY_M_M|SCALAR
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    _g:      ARRAY_N_M
    def __init__(
            self, 
            transition_matrix_or_function:            
                ARRAY_N_N|FUNC_VECTOR_N_I_TO_N|None       =None, #a
            observation_matrix_or_function:           
                ARRAY_M_N|ARRAY_N|FUNC_VECTOR_N_TO_M|None =None, #c
            noise_model_matrix:                       
                ARRAY_N_V|None                            =None, #b
            input_matrix:                             
                ARRAY_N_I|ARRAY_N|None                    =None, #bu
            system_noise_covariance_matrix_or_number: 
                ARRAY_V_V|SCALAR|None                     =None, #q
            observation_covariance_matrix_or_number:  
                ARRAY_M_M|SCALAR|None                     =None, #r
            initial_state:                            
                ARRAY_N|None                              =None, #init_x
            initial_prediction_covariance_matrix:     
                ARRAY_N_N|None                            =None, #init_p
            ) -> None: #この書き方は果たして本当に良いのだろうか
        """カルマンフィルタ系のフィルタの基底クラス"""
        self._a_or_f= transition_matrix_or_function # type: ignore
        if (isinstance(observation_matrix_or_function, np.ndarray) and
                len(observation_matrix_or_function.shape) == 1):
            self._c_or_h = observation_matrix_or_function.reshape(1, -1)
        else:
            self._c_or_h = observation_matrix_or_function # type: ignore
        self._b = noise_model_matrix # type: ignore
        if (isinstance(input_matrix, np.ndarray) and 
                len(input_matrix.shape) == 1):
            self._bu = input_matrix.reshape(-1, 1)
        else:
            self._bu = input_matrix
        self._q = system_noise_covariance_matrix_or_number # type: ignore
        self._r = observation_covariance_matrix_or_number # type: ignore
        if initial_state == None:
            self._x = None # type: ignore
        else:
            self._x = initial_state.reshape(-1, 1) # type: ignore
        self._p = initial_prediction_covariance_matrix # type: ignore
        self._g = None # type: ignore
    @property
    def transition_matrix_or_function(
            self
            ) -> ARRAY_N_N|FUNC_VECTOR_N_I_TO_N|None:
        return self._a_or_f
    @transition_matrix_or_function.setter
    def transition_matrix_or_function(
            self, 
            value: ARRAY_N_N|FUNC_VECTOR_N_I_TO_N
            ) -> None:
        self._a_or_f = value

    @property
    def noise_model_matrix(self) -> ARRAY_N_V|None:
        return self._b
    @noise_model_matrix.setter
    def noise_model_matrix(self, value: ARRAY_N_V|ARRAY_N) -> None:
        if len(value.shape) == 1:
            self._b = value.reshape(-1, 1)
        else:
            self._b = value

    @property
    def observation_matrix_or_function(
            self
            ) -> ARRAY_M_N|ARRAY_N|FUNC_VECTOR_N_TO_M|None:
        if isinstance(self._c_or_h, np.ndarray) and self._c_or_h.shape[0] == 1:
            return self._c_or_h.reshape(-1)
        else:
            return self._c_or_h
    @observation_matrix_or_function.setter
    def observation_matrix_or_function(
            self, 
            value: ARRAY_M_N|ARRAY_N|FUNC_VECTOR_N_TO_M
            ) -> None:
        if isinstance(value, np.ndarray) and len(value.shape) == 1:
            self._c_or_h = value.reshape(1, -1)
        elif callable(value):
            self._c_or_h = value
        else:
            ValueError("呼び出し可能でも配列でもありません")
    
    @property
    def input_matrix(self) -> ARRAY_N_I|ARRAY_N|None:
        if isinstance(self._c_or_h, np.ndarray) and self._c_or_h.shape[1] == 1:
            return self._c_or_h.reshape(-1)
        else:
            return self._c_or_h
    @input_matrix.setter
    def input_matrix(self, value: ARRAY_N_I|ARRAY_N) -> None:
        if isinstance(value, np.ndarray) and len(value.shape) == 1:
            self._bu = value.reshape(-1, 1)
        else:
            self._bu = value

    @property
    def system_noise_covariance_matrix_or_number(
            self
            ) -> ARRAY_V_V|SCALAR|None:
        return self._q
    @system_noise_covariance_matrix_or_number.setter
    def system_noise_covariance_matrix_or_number(
            self, 
            value: ARRAY_V_V|SCALAR) -> None:
        self._q = value
    
    @property
    def observation_covariance_matrix_or_number(
            self
            ) -> ARRAY_M_M|SCALAR|None:
        return self._r
    @observation_covariance_matrix_or_number.setter
    def observation_covariance_matrix_or_number(
            self, 
            value: ARRAY_M_M|SCALAR
            ) -> None:
        self._r = value
    
    @property
    def state_vector(self) -> ARRAY_N|None:
        return self._x.reshape(-1)
    @state_vector.setter
    def state_vector(self, value: ARRAY_N) -> None:
        self._x = value.reshape(-1, 1)

    @property
    def prediction_covariance_matrix(self) -> ARRAY_N_N|None:
        return self._p
    @prediction_covariance_matrix.setter
    def prediction_covariance_matrix(self, value: ARRAY_N_N) -> None:
        self._p = value
    
    @property
    def kalman_gain(self) -> ARRAY_N_M|None:
        return self._g
    
    @property
    def prediction_observation(self) -> SCALAR|ARRAY_N:
        if callable(self._c_or_h):
            return self._c_or_h(self._x).reshape(-1)
        else:
            result = self._c_or_h@self._x
            if len(result) == 1:
                result = float(result)
            else:
                result = result.reshape(-1)
            return result
    
    def _is_none(self) -> None:
        if any(isinstance(attribute, type(None)) 
               for attribute in (self._x, self._p, self._a_or_f, 
                                 self._c_or_h, self._b, self._q, self._r)):
            raise AttributeError(
                "NoneTypeの属性または属性の配列の大きさが違います"
                )


class KalmanFilter_SingleObservation(KFs_Base):
    _a_or_f: ARRAY_N_N
    _c_or_h: ARRAY_1_N
    _b:      ARRAY_N_1
    _bu:     ARRAY_N_1|ARRAY_N_I|None
    _q:      SCALAR
    _r:      SCALAR
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_matrix:                    
                ARRAY_N_N|None         =None, #a
            observation_matrix:                   
                ARRAY_N|None           =None, #c
            noise_model_matrix:                   
                ARRAY_N_1|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_number:       
                SCALAR|None            =None, #q
            observation_covariance_number:        
                SCALAR|None            =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値は1つのスカラ

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n
            system_noise_covariance_number : float
                ノイズモデルのノイズの分散
            observation_covariance_number : float
                観測値の分散
            init_x : ndarray
                状態変数の初期値
                size : n
            init_p : ndarray
                共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_number,
            observation_covariance_matrix_or_number  
                = observation_covariance_number,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: SCALAR, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = 0#np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        result = kf_update_single(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, observation, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector


class KalmanFilter_Multiple(KFs_Base):
    _a_or_f: ARRAY_N_N
    _c_or_h: ARRAY_M_N
    _b:      ARRAY_N_V
    _bu:     ARRAY_N_1|ARRAY_N_I|None
    _q:      ARRAY_V_V
    _r:      ARRAY_M_M
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_matrix:                    
                ARRAY_N_N|None         =None, #a
            observation_matrix:                   
                ARRAY_M_N|None         =None, #c
            noise_model_matrix:                   
                ARRAY_N_V|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_matrix:       
                ARRAY_V_V|None         =None, #q
            observation_covariance_matrix:        
                ARRAY_M_M|None         =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値はmつ。m次元ベクトルを受け取る

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : m, n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n, v
            system_noise_covariance_number : ndarray
                ノイズモデルのノイズの共分散行列
                size : v, v
            observation_covariance_number : ndarray
                観測値の共分散行列
                size : m, m
            init_x : ndarray
                状態ベクトルの初期値
                size : n
            init_p : ndarray
                状態推定値の共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_matrix,
            observation_covariance_matrix_or_number  
                = observation_covariance_matrix,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: ARRAY_M, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        y = observation.reshape(-1, 1)
        result = kf_update_multiple(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, y, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector
    
########################################################################################

class ExtendedKalmanFilter_SingleObservation(KFs_Base):
    _a_or_f: FUNC_VECTOR_N_I_TO_N
    _a_or_f_jacobian: FUNC_VECTOR_N_TO_N_N|None
    _c_or_h: FUNC_VECTOR_N_TO_SCALAR
    _c_or_h_jacobian: FUNC_VECTOR_N_TO_1_N|None
    _b:      ARRAY_N_1
    _bu:     ARRAY_N_1|ARRAY_N_I|None
    _q:      SCALAR
    _r:      SCALAR
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_matrix:                    
                ARRAY_N_N|None         =None, #a
            observation_matrix:                   
                ARRAY_N|None           =None, #c
            noise_model_matrix:                   
                ARRAY_N_1|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_number:       
                SCALAR|None            =None, #q
            observation_covariance_number:        
                SCALAR|None            =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値は1つのスカラ

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n
            system_noise_covariance_number : float
                ノイズモデルのノイズの分散
            observation_covariance_number : float
                観測値の分散
            init_x : ndarray
                状態変数の初期値
                size : n
            init_p : ndarray
                共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_number,
            observation_covariance_matrix_or_number  
                = observation_covariance_number,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: SCALAR, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = 0#np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        result = kf_update_single(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, observation, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector


class ExtendedKalmanFilter_Multiple(KFs_Base):
    _a_or_f: FUNC_VECTOR_N_I_TO_N
    _a_or_f_jacobian: FUNC_VECTOR_N_TO_N_N|None
    _c_or_h: FUNC_VECTOR_N_TO_M
    _c_or_h_jacobian: FUNC_VECTOR_N_TO_M_N|None
    _b:      ARRAY_N_V
    _bu:     ARRAY_N_1|ARRAY_N_I|None
    _q:      ARRAY_V_V
    _r:      ARRAY_M_M
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_function:                    
                ARRAY_N_N|None         =None, #a
            observation_function:                   
                ARRAY_M_N|None         =None, #c
            noise_model_matrix:                   
                ARRAY_N_V|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_matrix:       
                ARRAY_V_V|None         =None, #q
            observation_covariance_matrix:        
                ARRAY_M_M|None         =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値はmつ。m次元ベクトルを受け取る

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : m, n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n, v
            system_noise_covariance_number : ndarray
                ノイズモデルのノイズの共分散行列
                size : v, v
            observation_covariance_number : ndarray
                観測値の共分散行列
                size : m, m
            init_x : ndarray
                状態ベクトルの初期値
                size : n
            init_p : ndarray
                状態推定値の共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_matrix,
            observation_covariance_matrix_or_number  
                = observation_covariance_matrix,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: ARRAY_M, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        y = observation.reshape(-1, 1)
        result = kf_update_multiple(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, y, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector


class UnscentedKalmanFilter_SingleObservation(KFs_Base):
    _a_or_f: FUNC_VECTOR_N_I_TO_N
    _c_or_h: FUNC_VECTOR_N_TO_SCALAR
    _r:      SCALAR
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_matrix:                    
                ARRAY_N_N|None         =None, #a
            observation_matrix:                   
                ARRAY_N|None           =None, #c
            noise_model_matrix:                   
                ARRAY_N_1|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_number:       
                SCALAR|None            =None, #q
            observation_covariance_number:        
                SCALAR|None            =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値は1つのスカラ

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n
            system_noise_covariance_number : float
                ノイズモデルのノイズの分散
            observation_covariance_number : float
                観測値の分散
            init_x : ndarray
                状態変数の初期値
                size : n
            init_p : ndarray
                共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_number,
            observation_covariance_matrix_or_number  
                = observation_covariance_number,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: SCALAR, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = 0#np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        result = kf_update_single(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, observation, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector


class UnscentedKalmanFilter_Multiple(KFs_Base):
    _a_or_f: FUNC_VECTOR_N_I_TO_N
    _c_or_h: FUNC_VECTOR_N_TO_M
    _r:      ARRAY_M_M
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_matrix:                    
                ARRAY_N_N|None         =None, #a
            observation_matrix:                   
                ARRAY_M_N|None         =None, #c
            noise_model_matrix:                   
                ARRAY_N_V|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_matrix:       
                ARRAY_V_V|None         =None, #q
            observation_covariance_matrix:        
                ARRAY_M_M|None         =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値はmつ。m次元ベクトルを受け取る

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : m, n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n, v
            system_noise_covariance_number : ndarray
                ノイズモデルのノイズの共分散行列
                size : v, v
            observation_covariance_number : ndarray
                観測値の共分散行列
                size : m, m
            init_x : ndarray
                状態ベクトルの初期値
                size : n
            init_p : ndarray
                状態推定値の共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_matrix,
            observation_covariance_matrix_or_number  
                = observation_covariance_matrix,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: ARRAY_M, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        y = observation.reshape(-1, 1)
        result = kf_update_multiple(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, y, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector

####################################################################################################################

class AKalmanFilter_SingleObservation(KFs_Base):
    _a_or_f: ARRAY_N_N
    _c_or_h: ARRAY_1_N
    _b:      ARRAY_N_1
    _bu:     ARRAY_N_1|ARRAY_N_I|None
    _q:      SCALAR
    _r:      SCALAR
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_matrix:                    
                ARRAY_N_N|None         =None, #a
            observation_matrix:                   
                ARRAY_N|None           =None, #c
            noise_model_matrix:                   
                ARRAY_N_1|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_number:       
                SCALAR|None            =None, #q
            observation_covariance_number:        
                SCALAR|None            =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値は1つのスカラ

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n
            system_noise_covariance_number : float
                ノイズモデルのノイズの分散
            observation_covariance_number : float
                観測値の分散
            init_x : ndarray
                状態変数の初期値
                size : n
            init_p : ndarray
                共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_number,
            observation_covariance_matrix_or_number  
                = observation_covariance_number,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: SCALAR, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = 0#np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        result = kf_update_single(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, observation, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector


class AKalmanFilter_Multiple(KFs_Base):
    _a_or_f: ARRAY_N_N
    _c_or_h: ARRAY_M_N
    _b:      ARRAY_N_V
    _bu:     ARRAY_N_1|ARRAY_N_I|None
    _q:      ARRAY_V_V
    _r:      ARRAY_M_M
    _x:      ARRAY_N_1
    _p:      ARRAY_N_N
    def __init__(
            self, 
            transition_matrix:                    
                ARRAY_N_N|None         =None, #a
            observation_matrix:                   
                ARRAY_M_N|None         =None, #c
            noise_model_matrix:                   
                ARRAY_N_V|None         =None, #b
            input_matrix:                         
                ARRAY_N|ARRAY_N_I|None =None, #bu
            system_noise_covariance_matrix:       
                ARRAY_V_V|None         =None, #q
            observation_covariance_matrix:        
                ARRAY_M_M|None         =None, #r
            initial_state:                        
                ARRAY_N|None           =None, #init_x
            initial_prediction_covariance_matrix: 
                ARRAY_N_N|None         =None, #init_p
            ) -> None:
        """線形定常カルマンフィルタ

        観測値はmつ。m次元ベクトルを受け取る

        状態変数をn次元ベクトルとする

        モデルのノイズベクトルをv次元ベクトルとする(ややこしい)
        
        parameters
        ----------
            transition_matrix : ndarray
                推移行列 
                size : n, n
            observation_matrix : ndarray
                観測行列
                size : m, n
            noise_model_matrix : ndarray
                ノイズモデル
                size : n, v
            system_noise_covariance_number : ndarray
                ノイズモデルのノイズの共分散行列
                size : v, v
            observation_covariance_number : ndarray
                観測値の共分散行列
                size : m, m
            init_x : ndarray
                状態ベクトルの初期値
                size : n
            init_p : ndarray
                状態推定値の共分散行列の初期値
                size : n, n
        """
        super().__init__(
            transition_matrix_or_function            
                = transition_matrix,
            observation_matrix_or_function           
                = observation_matrix,
            input_matrix                             
                = input_matrix,
            noise_model_matrix                       
                = noise_model_matrix,
            system_noise_covariance_matrix_or_number 
                = system_noise_covariance_matrix,
            observation_covariance_matrix_or_number  
                = observation_covariance_matrix,
            initial_state                            
                = initial_state,
            initial_prediction_covariance_matrix     
                = initial_prediction_covariance_matrix,
            )
    def update(
            self, 
            observation: ARRAY_M, 
            input_value: ARRAY_I|SCALAR|None = None
            ) -> ARRAY_N:
        """更新。

        returns
        -------
            ARRAY_N
                今回の推定結果"""
        self._is_none()
        if input_value == None:
            u = np.zeros([len(self._x), 1])
        else:
            if isinstance(self._bu, type(None)):
                raise AttributeError("input_matrixがNoneです")
            if isinstance(input_value, SCALAR):
                u = self._bu*input_value
            else:
                u = self._bu@input_value
        y = observation.reshape(-1, 1)
        result = kf_update_multiple(
            self._x, self._p, self._a_or_f, self._c_or_h, 
            self._b, self._q, self._r, y, u
            )
        self._p = result.p
        self._x = result.x
        self._g = result.g
        return self.state_vector
