
"""
KFの裏処理
"""

import math
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, TypeVar, Annotated, NamedTuple
from dataclasses import dataclass
from kf.dtypes import *

__all__ = ["update_single"]

class Result(NamedTuple):
    x: ARRAY_N_1
    p: ARRAY_N_N
    g: ARRAY_N_1

def update_single(x: ARRAY_N_1, 
                  p: ARRAY_N_N, 
                  a: ARRAY_N_N, 
                  c: ARRAY_1_N, 
                  b: ARRAY_N_1,
                  q: SCALAR, 
                  r: SCALAR, 
                  y: SCALAR) -> Result:
    """更新"""
    #予測
    x_ = a@x
    p_ = a@p@a.T+q*b@b.T

    #カルマンゲイン
    g     = (p_@c.T)/(c@p_@c.T+r)

    #修正（フィルタリング）
    x_new = x_+g@(y-c@x_)
    p_new = (np.eye(len(x))-g@c)@p_

    return Result(x_new, p_new, g)