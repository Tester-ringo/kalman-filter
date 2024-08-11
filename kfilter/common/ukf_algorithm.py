
"""
UKFの裏処理
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, Annotated, NamedTuple

import numpy as np
import numpy.typing as npt

from kfilter.dtypes import *

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
        f: FUNC_VECTOR_N_I_TO_N, 
        h: FUNC_VECTOR_N_TO_SCALAR, 
        b: ARRAY_N_1,
        q: SCALAR, 
        r: SCALAR, 
        k: SCALAR, 
        y: SCALAR,
        u: Any,
        ) -> Result_Single:
    """UKFの状態更新"""
    #許してください
    #予測。2n+1個のシグマポイントを用いて非線形性をある程度表現
    n:int                   = len(x) #nを状態ベクトルの変数量とする。
    m, _                    = (1, 0) #mを観測ベクトルの変数量とする。
    translate_function      = lambda vector: f(vector, u) #入力を内包した関数にする
    sigma_offset            = math.sqrt(n + k) * np.linalg.cholesky(p)
    sigma_point             = np.concatenate([x, x + sigma_offset, x - sigma_offset], axis=1) #シグマポイントベクトルが並んでいる状態 size:(n, 2n+1)
    weight                  = np.concatenate([[k/(n+k)], np.full(2*n, 0.5/(n+k))]).reshape(1, -1) #それぞれのシグマポイントの重みづけ size:(1, 2n+1)
    sigma_point_transition  = np.asarray(list(map(translate_function, sigma_point.T))).T #推移関数で推移させる (もう少し良い実装は無いのだろうか) size:(n, 2n+1)
    x_                      = np.add.reduce(sigma_point_transition*weight, axis=1).reshape(-1, 1) #推移後の状態変数の予測値 size:(n, 1)
    p_                      = weight*(sigma_point_transition-x_)@(sigma_point_transition-x_).T + q*b@b.T #事前誤差共分散行列を計算 size:(n, n)
    sigma_offset_again      = math.sqrt(n + k) * np.linalg.cholesky(p_)
    sigma_point_again       = np.concatenate([x_, x_ + sigma_offset_again, x_ - sigma_offset_again], axis=1) #シグマポイントを再計算 size:(n, 2n+1)
    sigma_point_observation = np.asarray(list(map(h, sigma_point_again.T))).T #新しいシグマポイントの観測値を計算 size:(m, 2n+1)
    y_                      = np.add.reduce(sigma_point_observation*weight, axis=1).reshape(-1, 1) #推定観測値を計算 size:(m, 1)
    p_yy                    = weight*(sigma_point_observation-y_)@(sigma_point_observation-y_).T #観測値の誤差共分散行列の計算 size:(m, m)
    p_xy                    = weight*(sigma_point_again-x_)@(sigma_point_observation-y_).T #シグマポイントと観測値の共分散行列の計算 size:(n, m)

    #カルマンゲイン
    g                       = p_xy@np.linalg.solve(p_yy+r, np.eye(m)) #カルマンゲインの計算！！

    #フィルタリングステップ
    x_new                   = x_ + g@(y - y_) #修正
    p_new                   = p_ - g@p_xy.T #共分散行列の更新
    
    return Result_Single(x=x_new, p=p_new, g=g)


def update_multiple(
        x: ARRAY_N_1, 
        p: ARRAY_N_N, 
        f: FUNC_VECTOR_N_I_TO_N, 
        h: FUNC_VECTOR_N_TO_M, 
        b: ARRAY_N_V,
        q: ARRAY_V_V, 
        r: ARRAY_M_M, 
        k: SCALAR, 
        y: ARRAY_M_1,
        u: Any,
        ) -> Result_Multiple:
    """UKFの状態更新"""
    #予測。2n+1個のシグマポイントを用いて非線形性をある程度表現
    n:int                   = len(x) #nを状態ベクトルの変数量とする。
    m, _                    = r.shape #mを観測ベクトルの変数量とする。
    translate_function      = lambda vector: f(vector, u) #入力を内包した関数にする
    sigma_offset            = math.sqrt(n + k) * np.linalg.cholesky(p)
    sigma_point             = np.concatenate([x, x + sigma_offset, x - sigma_offset], axis=1) #シグマポイントベクトルが並んでいる状態 size:(n, 2n+1)
    weight                  = np.concatenate([[k/(n+k)], np.full(2*n, 0.5/(n+k))]).reshape(1, -1) #それぞれのシグマポイントの重みづけ size:(1, 2n+1)
    sigma_point_transition  = np.asarray(list(map(translate_function, sigma_point.T))).T #推移関数で推移させる (もう少し良い実装は無いのだろうか) size:(n, 2n+1)
    x_                      = np.add.reduce(sigma_point_transition*weight, axis=1).reshape(-1, 1) #推移後の状態変数の予測値 size:(n, 1)
    p_                      = weight*(sigma_point_transition-x_)@(sigma_point_transition-x_).T + b@q@b.T #事前誤差共分散行列を計算 size:(n, n)
    sigma_offset_again      = math.sqrt(n + k) * np.linalg.cholesky(p_)
    sigma_point_again       = np.concatenate([x_, x_ + sigma_offset_again, x_ - sigma_offset_again], axis=1) #シグマポイントを再計算 size:(n, 2n+1)
    sigma_point_observation = np.asarray(list(map(h, sigma_point_again.T))).T #新しいシグマポイントの観測値を計算 size:(m, 2n+1)
    y_                      = np.add.reduce(sigma_point_observation*weight, axis=1).reshape(-1, 1) #推定観測値を計算 size:(m, 1)
    p_yy                    = weight*(sigma_point_observation-y_)@(sigma_point_observation-y_).T #観測値の誤差共分散行列の計算 size:(m, m)
    p_xy                    = weight*(sigma_point_again-x_)@(sigma_point_observation-y_).T #シグマポイントと観測値の共分散行列の計算 size:(n, m)

    #カルマンゲイン
    g                       = np.linalg.solve((p_yy+r).T, p_xy.T).T #カルマンゲインの計算！！

    #フィルタリングステップ
    x_new                   = x_ + g@(y - y_) #修正
    p_new                   = p_ - g@p_xy.T #共分散行列の更新
    
    return Result_Multiple(x=x_new, p=p_new, g=g)
