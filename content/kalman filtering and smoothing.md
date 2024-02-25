---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: kalman filtering and smoothing
date created: Wednesday, January 31st 2024, 6:52:35 pm
date modified: Sunday, February 25th 2024, 7:58:29 pm
share: true
---

When the dynamics model and the observation model of a [[./state space model|state space model]] are both **Gaussian**, that is we have a [[./linear gaussian ssm|linear gaussian ssm]], we can perform inference efficiently using **Kalman filtering** methods.

We perform **filtering** to predict one step and obtain $p(\mathbf{z}_{t}| \mathbf{y}_{1:t})$. Ignoring the optional exogenous inputs, $\mathbf{u}_t$, the algorithm consists of two steps:

- **Time update** step

$$
\begin{align}
p(\mathbf{z}_t | \mathbf{y}_{1:{t-1}}) &= \mathcal{N}(\mathbf{z}_{t-1} | \mathbf{\mu}_{t|t-1}, \mathbf{\Sigma}_{t|t-1}) \\
\mathbf{\mu}_{t|t-1} &= \mathbf{F}_{t}\mathbf{\mu}_{t-1|t-1} \\
\mathbf{\Sigma}_{t|t-1} &= \mathbf{F}_{t}\mathbf{\Sigma}_{t-1|t-1}\mathbf{F}_{t}^{T} + \mathbf{Q}_{t}
\end{align}
$$

- **Measurement** step, getting the expected observation $\mathbf{\hat{y}}$

$$
\begin{align}
p(\mathbf{z}_{t}| \mathbf{y}_{1:t}) &= \mathcal{N}(\mathbf{z}_{t} | \mathbf{\mu}_{t|t}, \mathbf{\Sigma}_{t|t}) \\
\mathbf{\hat{y}} &= \mathbf{H}_{t} \mathbf{\mu}_{t|t-1} \\
\mathbf{S}_{t} &= \mathbf{H}_{t}\mathbf{\Sigma}_{t|t-1}\mathbf{H}_{t}^{T} + \mathbf{R}_{t} \\
\mathbf{K}_{t} &= \mathbf{\Sigma}_{t|t-1} \mathbf{H}_{t}^{T} \mathbf{S}_{t}^{-1} \\
\mathbf{\mu}_{t|t} &= \mathbf{\mu}_{t|t-1} + \mathbf{K}_{t} (\mathbf{y} - \mathbf{\hat{y}}) \\
\mathbf{\Sigma}_{t|t} &= \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_{t} \mathbf{H}_{t} \mathbf{\Sigma}_{t|t-1} \\
&= \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_{t} \mathbf{S}_{t} \mathbf{K}_{t}^T
\end{align}
$$

The algorithm scales as $O(N)$. The full derivation relies on the conjugate properties of Gaussians and is in both the [[./ssm resources|Kevin Murphy and Durbin and Koopman books]].

When all the data have arrived, we can perform offline **smoothing** and obtain $p(\mathbf{z}_{t}| \mathbf{y}_{1:T})$. The algorithm requires two passes through the data, a forward-filter and backward-smoother, and so scales $O(N^2)$.

$$
\begin{align}
p(\mathbf{z}_t | \mathbf{y}_{1:T}) &= \mathcal{N}(\mathbf{z}_{t} | \mathbf{\mu}_{t|T}, \mathbf{\Sigma}_{t|T}) \\
\mathbf{\mu}_{t|t+1} &= \mathbf{F}_{t} \mathbf{\mu}_{t|t} \\
\mathbf{\Sigma}_{t|t+1} &= \mathbf{F}_{t} \mathbf{\Sigma}_{t|t} \mathbf{F}_{t}^{T} + \mathbf{Q}_{t+1} \\
\mathbf{G}_{t} &= \mathbf{\Sigma}_{t|t} \mathbf{F}_{t}^{T} \mathbf{\Sigma}_{t|t+1}^{-1} \\
\mathbf{\mu}_{t|T} &= \mathbf{\mu}_{t|t} + \mathbf{G}_{t} (\mathbf{\mu}_{t+1|T} - \mathbf{\mu}_{t+1|t}) \\
\mathbf{\Sigma}_{t|T} &= \mathbf{\Sigma}_{t|t} + \mathbf{G}_{t} (\mathbf{\Sigma}_{t+1|T} - \mathbf{\Sigma}_{t+1|t}) \mathbf{G}_{t}^T
\end{align}
$$
