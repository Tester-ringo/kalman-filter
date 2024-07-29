# ExtendedKalmanFilter

```python

import kfilter as kf

...

```

## 想定システム

```math
\begin{align}
x_{k} &= f(x_{k-1}) + Bv_{k} \\
y_{k} &= h(x_{k}) + w_{k}\\
\end{align}
```
ただし各種パラメータを以下とする
```math
\begin{align}
k &= 1, 2, 3, \dots\\
f &: \mathbb{R}^n \longrightarrow \mathbb{R}^n \\
h &: \mathbb{R}^n \longrightarrow \mathbb{R}^n \\
B &\in \mathbb{R}^{n\times r} \\
x_k &\in \mathbb{R}^n\\
y_k &\in \mathbb{R}^m\\
v_k &\in \mathbb{R}^r\\
w_k &\in \mathbb{R}^r\\
\end{align}
```

上記は離散状態方程式である．$`x_k`$は時刻$`k`$での状態変数，$`y_k`$は時刻$`k`$での観測値である．また$`f`$，$`h`$はそれぞれ遷移関数と観測関数であり，$`x_k`$周りで微分可能とする．$`B`$はシステム雑音行列である．$`v_k`$，$`w_k`$はシステム雑音と観測雑音と呼ばれ，平均は共に$`0`$とする．

## 推定アルゴリズム

```math
\begin{align}
\hat x^-_{k} &= f(\hat x_{k-1}) \\
A_{k-1} &= \left.\frac{\partial f(x)}{\partial x}\right|_{x= \hat x_{k-1}}\\
C^-_k &= \left.\frac{\partial h(x)}{\partial x}\right|_{x= \hat x^-_k}\\
P^-_{k} &= A_{k-1}P^-_{k-1}A_{k-1}^\mathsf{T} + B\Sigma_{\mathrm v} B^\mathsf{T} \\
G_k &= P^-_{k}C^-_k{}^\mathsf{T} (C^-_kP^-_{k}C^-_k{}^\mathsf{T}+\Sigma_{\mathrm w})^{-1} \\
\hat x_{k} &= \hat x^-_{k} + G_k(y_k-h(\hat x^-_{k})) \\
P_k &= (I-G_kC^-_k)P^-_k
\end{align}
```

説明はまた後で

## 推定アルゴリズムの概念的説明

後々書く

