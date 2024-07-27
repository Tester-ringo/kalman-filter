
"""
EKFの裏処理
"""

import math
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, TypeVar, Annotated, NamedTuple
from dataclasses import dataclass
from kf.dtypes import *

__all__ = [
    "update_single", 
    "update_multiple",
    "jacobian",
    ]


class Result_Single(NamedTuple):
    x: ARRAY_N_1
    p: ARRAY_N_N
    g: ARRAY_N_1


class Result_Multiple(NamedTuple):
    x: ARRAY_N_1
    p: ARRAY_N_N
    g: ARRAY_N_M


def jacobian(f:Callable, x:np.ndarray, dx:float=1e-4) -> np.ndarray:
    n = len(x)
    dx_offset = np.eye(n)*dx
    g = lambda x: np.asarray(list(map(f, x)))
    result = (g(x+dx_offset)-g(x-dx_offset))/2/dx
    return result.T


def update_single(
        x: ARRAY_N_1, 
        p: ARRAY_N_N, 
        f: FUNC_VECTOR_N_I_TO_N, 
        df: ARRAY_N_N,
        h: FUNC_VECTOR_N_TO_M, 
        dh: ARRAY_M_N,
        b: ARRAY_N_1,
        q: SCALAR, 
        r: SCALAR, 
        y: SCALAR,
        u: Any,
        ) -> Result_Single:
    """更新"""
    #予測
    x_    = f(x.reshape(-1), u).reshape(-1, 1)
    p_    = df@p@df.T+q*b@b.T

    #カルマンゲイン
    g     = (p_@dh.T)/(dh@p_@dh.T+r)

    #修正（フィルタリング）
    x_new = x_+g@(y-dh@x_)
    p_new = (np.eye(len(x))-g@dh)@p_

    return Result_Single(x_new, p_new, g)


def update_multiple(
        x: ARRAY_N_1, 
        p: ARRAY_N_N, 
        f: FUNC_VECTOR_N_I_TO_N, 
        df: ARRAY_N_N,
        h: FUNC_VECTOR_N_TO_M, 
        dh: ARRAY_M_N,
        b: ARRAY_N_V,
        q: ARRAY_V_V, 
        r: ARRAY_M_M, 
        y: ARRAY_M_1,
        u: Any,
        ) -> Result_Multiple:
    """更新"""
    #予測
    x_    = f(x.reshape(-1), u).reshape(-1, 1)
    p_    = df@p@df.T+b@q@b.T
    
    #カルマンゲインを計算
    g     = np.linalg.solve((dh@p_@dh.T+r).T, (p_@dh.T).T).T
    
    #修正ステップ（フィルタリング）
    x_new = x_+g@(y-dh@x_)
    p_new = (np.eye(len(x))-g@dh)@p_
    
    return Result_Multiple(x_new, p_new, g)
