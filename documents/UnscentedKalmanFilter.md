# UnscentedKalmanFilter

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
w_k &\in \mathbb{R}^m\\
\mathrm E[v_k] &= 0 \\
\mathrm E[w_k] &= 0 \\
\end{align}
```

上記は離散状態方程式である．$`x_k`$は時刻$`k`$での状態変数，$`y_k`$は時刻$`k`$での観測値である．また$`f`$，$`h`$はそれぞれ遷移関数と観測関数である．$`B`$はシステム雑音行列である．$`v_k`$，$`w_k`$はシステム雑音と観測雑音と呼ばれ，平均は共に$`0`$とする．

## 推定アルゴリズム

```math
\begin{align}
\mathrm x_0,{}_{k-1} &= \hat x_{k-1} \\
\mathrm x_i,{}_{k-1} &= \hat x_{k-1} + \sqrt{n+\kappa}(\sqrt{P_{k-1}})_i,\;\;i=1,2,\dots,n\\
\mathrm x_{n+i},{}_{k-1} &= \hat x_{k-1} - \sqrt{n+\kappa}(\sqrt{P_{k-1}})_i,\;\;i=1,2,\dots,n\\
\mathrm{w}_0 &= \frac{\kappa}{n+\kappa} \\
\mathrm{w}_i &= \frac{\kappa}{2(n+\kappa)},\;\;i=1,2,\dots,2n \\
\mathrm x^-_i,{}_{k} &= f(\mathrm x_i,{}_{k-1}),\;\;i=1,2,\dots,2n\\
\hat x^-_k &= \sum^{2n}_{i=0}\mathrm{w}_i\mathrm x^-_i,{}_{k} \\
P^-_k &= B\Sigma_{\mathrm v} B^\mathsf{T} + \sum^{2n}_{i=0}\mathrm{w}_i(\mathrm x^-_i,{}_{k} - \hat x^-_k)(\mathrm x^-_i,{}_{k} - \hat x^-_k)^\mathsf{T} \\
\mathrm x_0,{}_{k} &= \hat x^-_{k} \\
\mathrm x_i,{}_{k} &= \hat x^-_{k} + \sqrt{n+\kappa}(\sqrt{P^-_{k}})_i,\;\;i=1,2,\dots,n\\
\mathrm x_{n+i},{}_{k} &= \hat x^-_{k} - \sqrt{n+\kappa}(\sqrt{P^-_{k}})_i,\;\;i=1,2,\dots,n\\
\mathrm y_i,{}_{k} &= h(\mathrm x_i,{}_{k}),\;\;i=1,2,\dots,n\\
\hat y^-_k &= \sum^{2n}_{i=0}\mathrm{w}_i\mathrm y_i,{}_{k} \\
\mathrm P^-_{yy},{}_k &= \sum^{2n}_{i=0}\mathrm{w}_i(\mathrm y_i,{}_{k} - \hat y^-_k)(\mathrm y_i,{}_{k} - \hat y^-_k)^\mathsf{T} \\
\mathrm P^-_{xy},{}_k &= \sum^{2n}_{i=0}\mathrm{w}_i(\mathrm x_i,{}_{k} - \hat x^-_k)(\mathrm y_i,{}_{k} - \hat y^-_k)^\mathsf{T} \\
G_k &= \mathrm P^-_{xy},{}_k(\mathrm P^-_{yy},{}_k + \Sigma_{\mathrm w})^{-1} \\
\hat x_k &= \hat x^-_k + G_k(y_k-\hat y^-_k) \\
P_k &= P^-_k-G_k(\mathrm P^-_{xy},{}_k)^\mathsf{T}
\end{align}
```

$`(\sqrt{A})_i`$は$`A`$の平方根行列の$`i`$番目の列を表すこととする．ここでは平方根行列をこれスキー分解を用いて計算することにする．

この説明を書くのは骨が折れるため今度


## 推定アルゴリズムの概念的説明

後々書く

