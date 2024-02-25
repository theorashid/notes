---
tags:
  - ssm
  - ml
  - bayesian
  - gaussian-process
folder: ssm
title: state space gaussian process
date created: Sunday, February 4th 2024, 4:27:00 pm
date modified: Sunday, February 25th 2024, 7:58:55 pm
share: true
---

Gaussian processes (GPs) scale as $O(N^3)$, where $N$ is the number of data points, which is problematic for large datasets.

By parametrising the GP as a **stochastic differential equation**, we can reformulate the GP regression problem into a [[./linear gaussian ssm|linear gaussian ssm]], where it can be solved using [[./kalman filtering and smoothing|kalman filtering and smoothing]] with linear time complexity ($O(N)$).

*See the [Temporal Gaussian Process Regression in Logarithmic Time](https://arxiv.org/pdf/2102.09964.pdf) and [Kalman filtering and smoothing solutions to temporal Gaussian process regression models](https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf)papers, as well as Adrien Corenflos' [implementation](https://github.com/EEA-sensors/parallel-gps/blob/main/pssgp/kernels/matern/common.py), the `BayesNewton` [implementation](https://github.com/AaltoML/BayesNewton/blob/main/bayesnewton/kernels.py#L238)and the `GPy` [implementation](https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/sde_matern.py). Perhaps could be implemented in [[./ssm in dynamax|dynamax]] by wrapping the filter step within a larger log-likelihood to optimise the lengthscale and variance, but [no success so far](https://github.com/theorashid/dynamax/blob/ssgp/docs/notebooks/linear_gaussian_ssm/ssgp.ipynb).*

## Mátern-5/2 kernel

This is the case $p=2$, so $\nu = p + 1/2= 5/2$, a typical choice for spatial and temporal problems.

We need to train lengthscale ($\ell$) and variance ($\alpha$) of kernel during fitting, as well as the likelihood noise $\sigma$.

$$

\begin{align*}

\underbrace{\begin{pmatrix} z^1_t \\ z^2_t \\ z^3_t \end{pmatrix}}_{z_t}

=

\underbrace{

\begin{pmatrix}

0 & 1 & 0 \\

0 & 0 & 1 \\

-\lambda^3 & -3 \lambda^2 & -3 \lambda

\end{pmatrix}

}_{F}

\underbrace{\begin{pmatrix} z^1_{t-1} \\ z^2_{t-1} \\ z^3_{t-1} \end{pmatrix}}_{z_{t-1}} + q_t

\end{align*}

$$

where

$$

\begin{align*}

\lambda = \sqrt{5} / \ell \\

q_t =

\begin{pmatrix}

0 \\

0 \\

1

\end{pmatrix} w(t)

\end{align*}

$$

and $w(t)$ is some white noise process.

The observation model is

$$

\begin{align*}

y_t

&=

\underbrace{

\begin{pmatrix}

1 & 0 & 0

\end{pmatrix}

}_{H}

\;

\underbrace{\begin{pmatrix} z^1_t \\ z^2_t \\ z^3_t \end{pmatrix}}_{z_t}

+ r_t

\end{align*}

$$

where $r_t \sim \mathcal{N}(0, \sigma^2)$.

This is a [[./linear gaussian ssm|linear gaussian ssm]].

## Matérn-1/2 kernel

Matérn-1/2 is a less common covariance function, but it results in even simpler matrices:

$$

\begin{align*}

\underbrace{\begin{pmatrix} z_t \end{pmatrix}}_{z_t}

&=

\underbrace{

\begin{pmatrix}

-\lambda

\end{pmatrix}

}_{F}

\underbrace{\begin{pmatrix} z_{t-1} \end{pmatrix}}_{z_{t-1}} + q_t\\

\lambda &= \sqrt{5} / \ell \\

q_t &= w(t)

\end{align*}

$$

with the observation model

$$

\begin{align*}

y_t

&=

\underbrace{

\begin{pmatrix}

1

\end{pmatrix}

}_{H}

\;

\underbrace{\begin{pmatrix} z_t \end{pmatrix}}_{z_t}

+ r_t

\end{align*}

$$

where $r_t \sim \mathcal{N}(0, \sigma^2)$.
