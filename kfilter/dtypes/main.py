
"""
型置き場。何かが違う気がする。
"""

import numpy as np
import numpy.typing as npt
from typing import Any, Callable, TypeVar, Annotated, Union
from dataclasses import dataclass

N = TypeVar("N") #状態ベクトルの次元数
M = TypeVar("M") #観測ベクトルの次元数
V = TypeVar("V") #モデルのノイズベクトルの次元数
I = TypeVar("I") #入力ベクトルの次元数


@dataclass(frozen=True)
class MatrixSize:
    row : int
    column : int


@dataclass(frozen=True)
class VectorSize:
    row : int


DType = TypeVar("DType", bound=np.number)

SCALAR = Union[int, float] #数値
ARRAY_N = Annotated[npt.NDArray[DType], VectorSize(N)] #n次元ベクトル
ARRAY_M = Annotated[npt.NDArray[DType], VectorSize(M)]
ARRAY_I = Annotated[npt.NDArray[DType], VectorSize(I)]
ARRAY_N_1 = Annotated[npt.NDArray[DType], MatrixSize(N, 1)] #(n, 1)行列
ARRAY_1_N = Annotated[npt.NDArray[DType], MatrixSize(1, N)]
ARRAY_N_N = Annotated[npt.NDArray[DType], MatrixSize(N, N)]
ARRAY_N_I = Annotated[npt.NDArray[DType], MatrixSize(N, I)]
ARRAY_I_1 = Annotated[npt.NDArray[DType], MatrixSize(I, 1)]
ARRAY_M_1 = Annotated[npt.NDArray[DType], MatrixSize(M, 1)]
ARRAY_M_N = Annotated[npt.NDArray[DType], MatrixSize(M, 1)]
ARRAY_N_M = Annotated[npt.NDArray[DType], MatrixSize(N, M)]
ARRAY_N_V = Annotated[npt.NDArray[DType], MatrixSize(V, 1)]
ARRAY_V_V = Annotated[npt.NDArray[DType], MatrixSize(V, V)]
ARRAY_M_M = Annotated[npt.NDArray[DType], MatrixSize(M, M)]
FUNC_VECTOR_N_I_TO_N = Callable[[ARRAY_N, ARRAY_I], ARRAY_N] #推移関数
FUNC_VECTOR_N_TO_M = Callable[[ARRAY_N], ARRAY_M] #観測関数
FUNC_VECTOR_N_TO_N_N = Callable[[ARRAY_N], ARRAY_N_N] #推移ヤコビアン
FUNC_VECTOR_N_TO_M_N = Callable[[ARRAY_N], ARRAY_M_N] #観測ヤコビアン
FUNC_VECTOR_N_TO_SCALAR = Callable[[ARRAY_N], SCALAR] #観測値一つ
FUNC_VECTOR_N_TO_1_N = Callable[[ARRAY_N], ARRAY_1_N] #ただの関数

__all__ = [
    "SCALAR",
    "ARRAY_N",
    "ARRAY_M",
    "ARRAY_I",
    "ARRAY_N_1",
    "ARRAY_1_N",
    "ARRAY_N_N",
    "ARRAY_N_I",
    "ARRAY_I_1",
    "ARRAY_M_1",
    "ARRAY_M_N",
    "ARRAY_N_M",
    "ARRAY_N_V",
    "ARRAY_V_V",
    "ARRAY_M_M",
    "FUNC_VECTOR_N_I_TO_N",
    "FUNC_VECTOR_N_TO_M",
    "FUNC_VECTOR_N_TO_N_N",
    "FUNC_VECTOR_N_TO_M_N",
    "FUNC_VECTOR_N_TO_SCALAR",
    "FUNC_VECTOR_N_TO_1_N",
    ]

