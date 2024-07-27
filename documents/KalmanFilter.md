# KalmanFilter

## 実装フィルタ詳細
- スカラ観測の線形カルマンフィルタ
- ベクトル観測の線形カルマンフィルタ

## 使い方詳細
```python

import kfilter as kf

...

```

## 想定システム

$$
\begin{align}
x_{k} &= Ax_{k-1} + Bv_{k} \\
y_{k} &= Cx_{k} + w_{k}\\
\end{align}
$$
ただし各種パラメータを以下とする．
$$
\begin{align}
k &= 1, 2, 3, \dots\\
A &\in \mathbb{R}^{n\times n} \\
B &\in \mathbb{R}^{n\times r} \\
C &\in \mathbb{R}^{m\times n} \\
x_k &\in \mathbb{R}^n\\
y_k &\in \mathbb{R}^m\\
v_k &\in \mathbb{R}^r\\
w_k &\in \mathbb{R}^m\\
\end{align}
$$

上記は離散状態方程式である．$x_k$は時刻$k$での状態変数，$y_k$は時刻$k$での観測値である．また$A$，$B$はそれぞれ遷移行列とシステム雑音行列であり，$C$は観測行列である．$v_k$，$w_k$はシステム雑音と観測雑音と呼ばれ，平均は共に$0$とする．

## 推定アルゴリズム

$$
\begin{align}
\hat x^-_{k} &= A\hat x_{k-1} \\
P^-_{k} &= AP_{k-1}A^\mathsf{T} + B\Sigma_{\mathrm v} B^\mathsf{T} \\
G_k &= P^-_{k}C^\mathsf{T} (CP^-_{k}C^\mathsf{T}+\Sigma_{\mathrm w})^{-1} \\
\hat x_{k} &= \hat x^-_{k} + G_k(y_k-C\hat x^-_{k}) \\
P_k &= (I-G_kC)P^-_k
\end{align}
$$

$\hat x_{k}$は状態変数$x_k$の最適推定値であり$\hat x_{k}=\mathrm E[x_k]$が成り立つ．また$\hat x_k$は事後推定値と呼ばれ，事前推定値は$\hat x^-_k$と記することとする．$P_k$は状態推定誤差の分散であり，同じく$P^-_k$はそれの事前推定値である．$G_k$はカルマンげインと呼ばれる．詳細以下の通りである．

$$
\begin{align}
\hat x_{k} &= \mathrm E[x_k] \\
P_{k} &= \mathrm E[(x_k-\hat x_k)(x_k-\hat x_k)^\mathsf{T}] \\
\Sigma_{\mathrm{v}} &= \mathrm E[v_kv_k^\mathsf{T}] \\
\Sigma_{\mathrm{w}} &= \mathrm E[w_kw_k^\mathsf{T}] \\
\end{align}
$$


## 推定アルゴリズムの概念的説明

後々書く

