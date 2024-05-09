
#
# ディスクリプタによる実装
#

"""カルマンフィルタ

手軽にカルマンフィルタを扱うことが目的
"""

import math
import numpy as np
import numpy.typing as npt
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, TypeVar, Annotated, Self
from dataclasses import dataclass
from collections import namedtuple, abc
from kf.dtypes import *

from kf.common.kf_algorithm import update_single as kf_update_single
from kf.common.kf_algorithm import update_multiple as kf_update_multiple
from kf.common.ekf_algorithm import update_single as ekf_update_single
from kf.common.ekf_algorithm import update_multiple as ekf_update_multiple
from kf.common.ukf_algorithm import update_single as ukf_update_single
from kf.common.ukf_algorithm import update_multiple as ukf_update_multiple
from kf.common.ekf_algorithm import jacobian


__all__ = [
    "KalmanFilter_SingleObservation",
    "KalmanFilter_MultipleObservation",
    "ExtendedKalmanFilter_SingleObservation",
    "ExtendedKalmanFilter_MultipleObservation",
    "UnscentedKalmanFilter_SingleObservation",
    "UnscentedKalmanFilter_MultipleObservation",
]

class FlaggedAttributeDescriptor(object):
    def __set_name__(self, owner, name: str) -> None:
        self._name = name
        if self._name == self._flag_name:
            raise ValueError("flag_nameに自身を指定しないでください")
    def __init__(self, flag_name: str, 
                 storage_attribute_name:str|None=None) -> None:
        self._flag_name = flag_name
        self._storage_attribute_name = storage_attribute_name
    def __set__(self, instance, value: Any) -> None:
        if self._storage_attribute_name == self._name or\
                self._storage_attribute_name == None:
            instance.__dict__[self._name] = value
        else:
            setattr(instance, self._storage_attribute_name, value)
        setattr(instance, self._flag_name, True)
    def __get__(self, instance, objtype=None) -> Any:
        if self._storage_attribute_name == self._name or\
                self._storage_attribute_name == None:
            if not self._name in instance.__dict__:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
            else:
                result = instance.__dict__[self._name]
        else:
            try:
                result = getattr(instance, self._storage_attribute_name)
            except:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
        return result


class ArrayReshapeSwapDescriptor(object):
    #名前がひどいですが、気にしないでください
    def __set_name__(self, owner, name: str) -> None:
        self._name = name
    def __init__(self, input_size: tuple[int, ...], 
                 output_size: tuple[int, ...], 
                 storage_attribute_name: str|None=None) -> None:
        self._input_size = input_size
        self._output_size = output_size
        self._storage_attribute_name = storage_attribute_name
    def __set__(self, instance:object, array_like_object:np.ndarray) -> None:
        if not isinstance(array_like_object, (np.ndarray, abc.Sequence)):
            raise ValueError("値が配列オブジェクトではありません")
        try:
            reshaped_array_like_object \
                = np.reshape(array_like_object, self._input_size)
        except:
            raise ValueError("無効なarray sizeです。"\
                             "またはarray sizeが一致していません")
        if self._storage_attribute_name == self._name or\
                self._storage_attribute_name == None:
            instance.__dict__[self._name] = reshaped_array_like_object
        else:
            setattr(instance, self._storage_attribute_name, 
                    reshaped_array_like_object)
    def __get__(self, instance: object, objtype=None) -> np.ndarray:
        if self._storage_attribute_name == self._name or\
                self._storage_attribute_name == None:
            if not self._name in instance.__dict__:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
            else:
                array_like_object = instance.__dict__[self._name]
        else:
            try:
                array_like_object = getattr(
                    instance, self._storage_attribute_name)
            except:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
        if not isinstance(array_like_object, (np.ndarray, abc.Sequence)):
            raise ValueError("値が配列オブジェクトではありません")
        try:
            reshaped_array_like_object \
                = np.reshape(array_like_object, self._output_size)
        except:
            raise ValueError("無効なarray sizeです。"\
                             "またはarray sizeが一致していません")
        return reshaped_array_like_object
    

class TypeCheckSwapDescriptor(object):
    #名前がひどいですが、気にしないでください
    def __set_name__(self, owner, name: str) -> None:
        self._name = name
    def __init__(self, types: tuple|type, 
                 storage_attribute_name: str|None=None) -> None:
        self._types = types
        self._storage_attribute_name = storage_attribute_name
    def __set__(self, instance:object, value: Any) -> None:
        if not isinstance(value, self._types):
            raise TypeError(
                f"{instance.__class__.__name__}のインスタンス{instance}の"\
                f"{self._name}アトリビュートに不正な型の値が代入されました"\
                f"正しくは{self._types}です。")
        if self._storage_attribute_name == self._name or\
                self._storage_attribute_name == None:
            instance.__dict__[self._name] = value
        else:
            setattr(instance, self._storage_attribute_name, value)
    def __get__(self, instance, objtype=None) -> Any:
        if self._storage_attribute_name == self._name or\
                self._storage_attribute_name == None:
            if not self._name in instance.__dict__:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
            else:
                result = instance.__dict__[self._name]
        else:
            try:
                result = getattr(instance, self._storage_attribute_name)
            except:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
        return result


class ReadOnlyDescriptor(object):
    def __set_name__(self, owner, name: str) -> None:
        self._name = name
    def __init__(self, reading_attribute_name: str|None=None) -> None:
        self._reading_attribute_name = reading_attribute_name
    def __set__(self, *args) -> None:
        raise AttributeError(f"アトリビュート [{self._name}] は代入出来ません")
    def __get__(self, instance, *args) -> ...:
        if self._reading_attribute_name == self._name or\
                self._reading_attribute_name == None:
            if not self._name in instance.__dict__:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
            else:
                result = instance.__dict__[self._name]
        else:
            try:
                result = getattr(instance, self._reading_attribute_name)
            except:
                raise AttributeError(
                    f"アトリビュート [{self._name}] は中身がありません")
        return result


class KalmanFilter_SingleObservation(object): 
    #アノテーションを漬ける
    #docstringを付ける
    is_updated = False
    f = FlaggedAttributeDescriptor(flag_name="is_updated")
    h = FlaggedAttributeDescriptor(flag_name="is_updated")
    b = FlaggedAttributeDescriptor(flag_name="is_updated")
    q = FlaggedAttributeDescriptor(flag_name="is_updated")
    r = FlaggedAttributeDescriptor(flag_name="is_updated")
    x = FlaggedAttributeDescriptor(flag_name="is_updated")
    p = FlaggedAttributeDescriptor(flag_name="is_updated")
    g : ...
    _g = ArrayReshapeSwapDescriptor(
        input_size=(), output_size=(-1,), storage_attribute_name="g")
    transition_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="f")
    observation_matrix = ArrayReshapeSwapDescriptor(
        input_size=(1, -1), output_size=(-1,), storage_attribute_name="h")
    noise_model_matrix = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="b")
    system_noise_covariance_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="q")
    observation_covariance_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="r")
    prediction_state_vector = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="x")
    prediction_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="p")
    kalman_gain = ReadOnlyDescriptor(reading_attribute_name="_g")
    _precheck_data: dict = {
        "f": "transition_matrix",
        "h": "observation_matrix",
        "b": "noise_model_matrix",
        "q": "system_noise_covariance_number",
        "r": "observation_covariance_number",
        "x": "prediction_state_vector",
        "p": "prediction_covariance_matrix",}
    def _validate_var_error(self) -> None:
        #あまり良いとは言えない実装だけれど許してください。
        #必要なアトリビュート。もといパラメータがセットされているかを確認
        for internal_var, interface_var in self._precheck_data.items():
            if not hasattr(self, internal_var):
                raise AttributeError(
                    f"必要なデータ:{interface_var}が存在しません。"\
                    "値を代入してください")
        #array sizeが正しい否かを確認
        n = len(self.x) #状態変数の要素数
        m = 1
        v = 1
        if not self.f.shape == (n, n):
            raise #大したものはかけないが、後々エラー文を書く。
        if not self.h.shape == (m, n):
            raise
        if not self.b.shape == (n, v): #ノイズ源を一つしか想定していない問題
            raise
        if not self.x.shape == (n, 1):
            raise
        if not self.p.shape == (n, n):
            raise
        self.is_updated = False
    def update(self, observed_value: SCALAR, 
               input_vector: ARRAY_I|None = None) -> None:
        if self.is_updated:
            self._validate_var_error()
        n = len(self.x)
        if input_vector == None:
            input_vector = np.zeros((n, 1))
        elif isinstance(input_vector, np.ndarray):
            input_vector = np.reshape(input_vector, (-1, 1))
        else:
            #数値、もしくはよくわからないものが入ってきた場合。
            #警告やエラーを出したいけれど、この部分は後々実装することとする。
            input_vector = input_vector
        result = kf_update_single(
            x = self.x, 
            p = self.p, 
            a = self.f, 
            c = self.h, 
            b = self.b, 
            q = self.q, 
            r = self.r, 
            y = observed_value, 
            u = input_vector)  
        #↑ここどうしよう。array型のアノテーションの方法を変えたほうが良いのか
        self.p = result.p
        self.x = result.x
        self.g = result.g
        self.is_updated = False


class KalmanFilter_MultipleObservation(object): 
    is_updated = False
    f = FlaggedAttributeDescriptor(flag_name="is_updated")
    h = FlaggedAttributeDescriptor(flag_name="is_updated")
    b = FlaggedAttributeDescriptor(flag_name="is_updated")
    q = FlaggedAttributeDescriptor(flag_name="is_updated")
    r = FlaggedAttributeDescriptor(flag_name="is_updated")
    x = FlaggedAttributeDescriptor(flag_name="is_updated")
    p = FlaggedAttributeDescriptor(flag_name="is_updated")
    g : ...
    transition_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="f")
    observation_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="h")
    noise_model_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="b")
    system_noise_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="q")
    observation_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="r")
    prediction_state_vector = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="x")
    prediction_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="p")
    kalman_gain = ReadOnlyDescriptor(reading_attribute_name="g")
    _precheck_data: dict = {
        "f": "transition_matrix",
        "h": "observation_matrix",
        "b": "noise_model_matrix",
        "q": "system_noise_covariance_matrix",
        "r": "observation_covariance_matrix",
        "x": "prediction_state_vector",
        "p": "prediction_covariance_matrix",}
    def _validate_var_error(self) -> None:
        #あまり良いとは言えない実装だけれど許してください。
        #必要なアトリビュート。もといパラメータがセットされているかを確認
        for internal_var, interface_var in self._precheck_data.items():
            if not hasattr(self, internal_var):
                raise AttributeError(
                    f"必要なデータ:{interface_var}が存在しません。"\
                    "値を代入してください")
        #array sizeが正しい否かを確認
        n = len(self.x) #状態変数の要素数
        m, _ = self.h.shape #観測ベクトルの次元
        _, v = self.b.shape #ノイズ源の数
        if not self.f.shape == (n, n):
            raise #大したものはかけないが、後々エラー文を書く。
        if not self.h.shape == (m, n):
            raise
        if not self.b.shape == (n, v): #ノイズ源を一つしか想定していない問題
            raise
        if not self.q.shape == (v, v):
            raise
        if not self.r.shape == (m, m):
            raise
        if not self.x.shape == (n, 1):
            raise
        if not self.p.shape == (n, n):
            raise
        self.is_updated = False
    def update(self, observed_vector: ARRAY_M, 
               input_vector: ARRAY_I|None = None) -> None:
        if self.is_updated:
            self._validate_var_error()
        observed_vector_reshaped = np.reshape(observed_vector, (-1, 1))
        n = len(self.x)
        if input_vector == None:
            input_vector = np.zeros((n, 1))
        elif isinstance(input_vector, np.ndarray):
            input_vector = np.reshape(input_vector, (-1, 1))
        else:
            #数値、もしくはよくわからないものが入ってきた場合。
            #警告やエラーを出したいけれど、この部分は後々実装することとする。
            input_vector = input_vector
        result = kf_update_multiple(
            x = self.x, 
            p = self.p, 
            a = self.f, 
            c = self.h, 
            b = self.b, 
            q = self.q, 
            r = self.r, 
            y = observed_vector_reshaped, 
            u = input_vector)  
        self.p = result.p
        self.x = result.x
        self.g = result.g
        self.is_updated = False


class ExtendedKalmanFilter_SingleObservation(object): 
    is_updated = False
    f  = FlaggedAttributeDescriptor(flag_name="is_updated")
    df = FlaggedAttributeDescriptor(flag_name="is_updated")
    h  = FlaggedAttributeDescriptor(flag_name="is_updated")
    dh = FlaggedAttributeDescriptor(flag_name="is_updated")
    b  = FlaggedAttributeDescriptor(flag_name="is_updated")
    q  = FlaggedAttributeDescriptor(flag_name="is_updated")
    r  = FlaggedAttributeDescriptor(flag_name="is_updated")
    x  = FlaggedAttributeDescriptor(flag_name="is_updated")
    p  = FlaggedAttributeDescriptor(flag_name="is_updated")
    dx = 1e-4
    g : ...
    _g = ArrayReshapeSwapDescriptor(
        input_size=(), output_size=(-1,), storage_attribute_name="g")
    transition_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="f")
    transition_jacobian = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="df")
    observation_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="h")
    observation_jacobian = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="dh")
    noise_model_matrix = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="b")
    system_noise_covariance_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="q")
    observation_covariance_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="r")
    prediction_state_vector = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="x")
    prediction_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="p")
    kalman_gain = ReadOnlyDescriptor(reading_attribute_name="_g")
    _precheck_data: dict = {
        "f": "transition_function",
        "h": "observation_function",
        "b": "noise_model_matrix",
        "q": "system_noise_covariance_number",
        "r": "observation_covariance_number",
        "x": "prediction_state_vector",
        "p": "prediction_covariance_matrix",}
    def _validate_var_error(self) -> None:
        #あまり良いとは言えない実装だけれど許してください。
        #必要なアトリビュート。もといパラメータがセットされているかを確認
        for internal_var, interface_var in self._precheck_data.items():
            if not hasattr(self, internal_var):
                raise AttributeError(
                    f"必要なデータ:{interface_var}が存在しません。"\
                    "値を代入してください")
        #array sizeが正しい否かを確認
        n = len(self.x) #状態変数の要素数
        m = 1
        v = 1
        if not self.b.shape == (n, v):
            raise
        if not self.x.shape == (n, 1):
            raise
        if not self.p.shape == (n, n):
            raise
        self.is_updated = False
    def update(self, observed_value: SCALAR, 
               input_vector: Any = None) -> None:
        if self.is_updated:
            self._validate_var_error()
        if not (hasattr(self, "df") and isinstance(self.df, Callable)):
            _f = lambda x: self.f(x, input_vector)
            f_jacobian = jacobian(f=_f, x=self.x.reshape(-1), dx=self.dx,)
        else:
            f_jacobian = self.df(self.x.reshape(-1))
        if not (hasattr(self, "dh") and isinstance(self.dh, Callable)):
            h_jacobian = jacobian(f=self.h, x=self.x.reshape(-1), dx=self.dx)
        else:
            h_jacobian = self.dh(self.x.reshape(-1))
        result = ekf_update_single(
            x = self.x,
            p = self.p,
            f = self.f,
            df = f_jacobian,
            h = self.h,
            dh = h_jacobian,
            b = self.b,
            q = self.q,
            r = self.r,
            y = observed_value,
            u = input_vector,)  
        self.p = result.p
        self.x = result.x
        self.g = result.g
        self.is_updated = False


class ExtendedKalmanFilter_MultipleObservation(object): 
    is_updated = False
    f  = FlaggedAttributeDescriptor(flag_name="is_updated")
    df = FlaggedAttributeDescriptor(flag_name="is_updated")
    h  = FlaggedAttributeDescriptor(flag_name="is_updated")
    dh = FlaggedAttributeDescriptor(flag_name="is_updated")
    b  = FlaggedAttributeDescriptor(flag_name="is_updated")
    q  = FlaggedAttributeDescriptor(flag_name="is_updated")
    r  = FlaggedAttributeDescriptor(flag_name="is_updated")
    x  = FlaggedAttributeDescriptor(flag_name="is_updated")
    p  = FlaggedAttributeDescriptor(flag_name="is_updated")
    dx = 1e-4
    g : ...
    transition_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="f")
    transition_jacobian = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="df")
    observation_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="h")
    observation_jacobian = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="dh")
    noise_model_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="b")
    system_noise_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="q")
    observation_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="r")
    prediction_state_vector = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="x")
    prediction_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="p")
    kalman_gain = ReadOnlyDescriptor(reading_attribute_name="g")
    _precheck_data: dict = {
        "f": "transition_function",
        "h": "observation_function",
        "b": "noise_model_matrix",
        "q": "system_noise_covariance_matrix",
        "r": "observation_covariance_matrix",
        "x": "prediction_state_vector",
        "p": "prediction_covariance_matrix",}
    def _validate_var_error(self) -> None:
        #あまり良いとは言えない実装だけれど許してください。
        #必要なアトリビュート。もといパラメータがセットされているかを確認
        for internal_var, interface_var in self._precheck_data.items():
            if not hasattr(self, internal_var):
                raise AttributeError(
                    f"必要なデータ:{interface_var}が存在しません。"\
                    "値を代入してください")
        #array sizeが正しい否かを確認
        n = len(self.x) #状態変数の要素数
        m = len(self.h(np.ones(n)))
        _, v = self.b.shape
        if not self.b.shape == (n, v):
            raise
        if not self.q.shape == (v, v):
            raise
        if not self.r.shape == (m, m):
            raise
        if not self.x.shape == (n, 1):
            raise
        if not self.p.shape == (n, n):
            raise
        self.is_updated = False
    def update(self, observed_vector: ARRAY_M, 
               input_vector: Any = None) -> None:
        if self.is_updated:
            self._validate_var_error()
        observed_vector_reshaped = np.reshape(observed_vector, (-1, 1))
        if not (hasattr(self, "df") and isinstance(self.df, Callable)):
            _f = lambda x: self.f(x, input_vector)
            f_jacobian = jacobian(f=_f, x=self.x.reshape(-1), dx=self.dx,)
        else:
            f_jacobian = self.df(self.x.reshape(-1))
        if not (hasattr(self, "dh") and isinstance(self.dh, Callable)):
            h_jacobian = jacobian(f=self.h, x=self.x.reshape(-1), dx=self.dx)
        else:
            h_jacobian = self.dh(self.x.reshape(-1))
        result = ekf_update_multiple(
            x = self.x,
            p = self.p,
            f = self.f,
            df = f_jacobian,
            h = self.h,
            dh = h_jacobian,
            b = self.b,
            q = self.q,
            r = self.r,
            y = observed_vector_reshaped,
            u = input_vector,)  
        self.p = result.p
        self.x = result.x
        self.g = result.g
        self.is_updated = False


class UnscentedKalmanFilter_SingleObservation(object): 
    is_updated = False
    f  = FlaggedAttributeDescriptor(flag_name="is_updated")
    h  = FlaggedAttributeDescriptor(flag_name="is_updated")
    b  = FlaggedAttributeDescriptor(flag_name="is_updated")
    q  = FlaggedAttributeDescriptor(flag_name="is_updated")
    r  = FlaggedAttributeDescriptor(flag_name="is_updated")
    k  = FlaggedAttributeDescriptor(flag_name="is_updated")
    x  = FlaggedAttributeDescriptor(flag_name="is_updated")
    p  = FlaggedAttributeDescriptor(flag_name="is_updated")
    g : ...
    _g = ArrayReshapeSwapDescriptor(
        input_size=(), output_size=(-1,), storage_attribute_name="g")
    transition_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="f")
    observation_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="h")
    noise_model_matrix = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="b")
    system_noise_covariance_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="q")
    observation_covariance_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="r")
    scaling_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="k")
    prediction_state_vector = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="x")
    prediction_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="p")
    kalman_gain = ReadOnlyDescriptor(reading_attribute_name="_g")
    _precheck_data: dict = {
        "f": "transition_function",
        "h": "observation_function",
        "b": "noise_model_matrix",
        "q": "system_noise_covariance_number",
        "r": "observation_covariance_number",
        "k": "scaling_number",
        "x": "prediction_state_vector",
        "p": "prediction_covariance_matrix",}
    def _validate_var_error(self) -> None:
        #あまり良いとは言えない実装だけれど許してください。
        #必要なアトリビュート。もといパラメータがセットされているかを確認
        for internal_var, interface_var in self._precheck_data.items():
            if not hasattr(self, internal_var):
                raise AttributeError(
                    f"必要なデータ:{interface_var}が存在しません。"\
                    "値を代入してください")
        #array sizeが正しい否かを確認
        n = len(self.x) #状態変数の要素数
        m = 1
        v = 1
        #推移関数と観測関数は今は考えない
        if not self.b.shape == (n, v):
            raise
        if not self.x.shape == (n, 1):
            raise
        if not self.p.shape == (n, n):
            raise
        self.is_updated = False
    def __init__(self) -> None:
        self.k = 0
    def update(self, observed_value: SCALAR, 
               input_vector: Any = None) -> None:
        if self.is_updated:
            self._validate_var_error()
        result = ukf_update_single(
            x = self.x,
            p = self.p,
            f = self.f,
            h = self.h,
            b = self.b,
            q = self.q,
            r = self.r,
            k = self.k,
            y = observed_value,
            u = input_vector)  
        self.p = result.p
        self.x = result.x
        self.g = result.g
        self.is_updated = False


class UnscentedKalmanFilter_MultipleObservation(object): 
    is_updated = False
    f  = FlaggedAttributeDescriptor(flag_name="is_updated")
    h  = FlaggedAttributeDescriptor(flag_name="is_updated")
    b  = FlaggedAttributeDescriptor(flag_name="is_updated")
    q  = FlaggedAttributeDescriptor(flag_name="is_updated")
    r  = FlaggedAttributeDescriptor(flag_name="is_updated")
    k  = FlaggedAttributeDescriptor(flag_name="is_updated")
    x  = FlaggedAttributeDescriptor(flag_name="is_updated")
    p  = FlaggedAttributeDescriptor(flag_name="is_updated")
    g : ...
    transition_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="f")
    observation_function = TypeCheckSwapDescriptor(
        types=(Callable,), storage_attribute_name="h")
    noise_model_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="b")
    system_noise_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="q")
    observation_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="r")
    scaling_number = TypeCheckSwapDescriptor(
        types=(int, float, complex), storage_attribute_name="k")
    prediction_state_vector = ArrayReshapeSwapDescriptor(
        input_size=(-1, 1), output_size=(-1,), storage_attribute_name="x")
    prediction_covariance_matrix = TypeCheckSwapDescriptor(
        types=np.ndarray, storage_attribute_name="p")
    kalman_gain = ReadOnlyDescriptor(reading_attribute_name="g")
    _precheck_data: dict = {
        "f": "transition_function",
        "h": "observation_function",
        "b": "noise_model_matrix",
        "q": "system_noise_covariance_matrix",
        "r": "observation_covariance_matrix",
        "k": "scaling_number",
        "x": "prediction_state_vector",
        "p": "prediction_covariance_matrix",}
    def _validate_var_error(self) -> None:
        #あまり良いとは言えない実装だけれど許してください。
        #必要なアトリビュート。もといパラメータがセットされているかを確認
        for internal_var, interface_var in self._precheck_data.items():
            if not hasattr(self, internal_var):
                raise AttributeError(
                    f"必要なデータ:{interface_var}が存在しません。"\
                    "値を代入してください")
        #array sizeが正しい否かを確認
        n = len(self.x) #状態変数の要素数
        m = len(self.h(np.ones(n)))
        _, v = self.b.shape
        if not self.b.shape == (n, v):
            raise
        if not self.q.shape == (v, v):
            raise
        if not self.r.shape == (m, m):
            raise
        if not self.x.shape == (n, 1):
            raise
        if not self.p.shape == (n, n):
            raise
        self.is_updated = False
    def __init__(self) -> None:
        self.k = 0
    def update(self, observed_vector: ARRAY_M, 
               input_vector: Any = None) -> None:
        if self.is_updated:
            self._validate_var_error()
        observed_vector_reshaped = np.reshape(observed_vector, (-1, 1))
        result = ukf_update_multiple(
            x = self.x,
            p = self.p,
            f = self.f,
            h = self.h,
            b = self.b,
            q = self.q,
            r = self.r,
            k = self.k,
            y = observed_vector_reshaped,
            u = input_vector)  
        self.p = result.p
        self.x = result.x
        self.g = result.g
        self.is_updated = False


