
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
    ]


class Result_Single(NamedTuple):
    x: ARRAY_N_1
    p: ARRAY_N_N
    g: ARRAY_N_1


class Result_Multiple(NamedTuple):
    x: ARRAY_N_1
    p: ARRAY_N_N
    g: ARRAY_N_M


def update_single(
        x: ARRAY_N_1, 
        p: ARRAY_N_N, 
        f: ARRAY_N_N, 
        h: ARRAY_1_N, 
        b: ARRAY_N_1,
        q: SCALAR, 
        r: SCALAR, 
        y: SCALAR,
        u: ARRAY_N_1,
        ) -> Result_Single:
    """更新"""
    #予測
    x_    = f(x, u)
    p_    = a@p@a.T+q*b@b.T

    #カルマンゲイン
    g     = (p_@c.T)/(c@p_@c.T+r)

    #修正（フィルタリング）
    x_new = x_+g@(y-c@x_)
    p_new = (np.eye(len(x))-g@c)@p_

    return Result_Single(x_new, p_new, g)


def update_multiple(
        x: ARRAY_N_1, 
        p: ARRAY_N_N, 
        f: ARRAY_N_N, 
        h: ARRAY_M_N, 
        b: ARRAY_N_V,
        q: ARRAY_V_V, 
        r: ARRAY_M_M, 
        y: ARRAY_M_1,
        u: ARRAY_N_1,
        ) -> Result_Multiple:
    """更新"""
    #予測
    x_    = f(x, u)
    p_    = a@p@a.T+b@q@b.T
    
    #カルマンゲイン
    g     = p_@c.T@np.linalg.solve(c@p_@c.T+r, np.eye(r.shape[0]))
    
    #修正（フィルタリング）
    x_new = x_+g@(y-c@x_)
    p_new = (np.eye(len(x))-g@c)@p_
    
    return Result_Multiple(x_new, p_new, g)
