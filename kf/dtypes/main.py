
"""
型置き場。何かが違う気がする。
"""

import numpy as np
import numpy.typing as npt
from typing import Any, Callable, TypeVar, Annotated, Union
from dataclasses import dataclass

N = TypeVar("N") #状態ベクトルの要素数
M = TypeVar("M") #観測ベクトルの要素数
V = TypeVar("V") #モデルのノイズベクトルの要素数

@dataclass(frozen=True)
class MatrixSize:
    row : int
    column : int

@dataclass(frozen=True)
class VectorSize:
    row : int

DType = TypeVar("DType", bound=np.number)

SCALAR = Union[int, float]
ARRAY_N = Annotated[npt.NDArray[DType], VectorSize(N)]
ARRAY_M = Annotated[npt.NDArray[DType], VectorSize(M)]
ARRAY_N_1 = Annotated[npt.NDArray[DType], MatrixSize(N, 1)]
ARRAY_1_N = Annotated[npt.NDArray[DType], MatrixSize(1, N)]
ARRAY_N_N = Annotated[npt.NDArray[DType], MatrixSize(N, N)]
ARRAY_M_1 = Annotated[npt.NDArray[DType], MatrixSize(M, 1)]
ARRAY_M_N = Annotated[npt.NDArray[DType], MatrixSize(M, 1)]
ARRAY_N_M = Annotated[npt.NDArray[DType], MatrixSize(N, M)]
ARRAY_N_V = Annotated[npt.NDArray[DType], MatrixSize(V, 1)]
ARRAY_V_V = Annotated[npt.NDArray[DType], MatrixSize(V, V)]
ARRAY_M_M = Annotated[npt.NDArray[DType], MatrixSize(M, M)]
FUNC_VECTOR_N_TO_N = Callable[[ARRAY_N_1], ARRAY_N_1] #in:ARRAY_N_1からout:ARRAY_N_1の関数アノテーション
FUNC_VECTOR_N_TO_M = Callable[[ARRAY_N_1], ARRAY_M_1] #in:ARRAY_N_1からout:ARRAY_M_1の関数アノテーション

__all__ = [
    "SCALAR",
    "ARRAY_N",
    "ARRAY_M",
    "ARRAY_N_1",
    "ARRAY_1_N",
    "ARRAY_N_N",
    "ARRAY_M_1",
    "ARRAY_M_N",
    "ARRAY_N_M",
    "ARRAY_N_V",
    "ARRAY_V_V",
    "ARRAY_M_M",
    "FUNC_VECTOR_N_TO_N",
    "FUNC_VECTOR_N_TO_M",
]

