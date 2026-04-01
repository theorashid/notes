---
tags:
  - ssm
  - ml
  - bayesian
  - gaussian-process
folder: ssm
title: state space gaussian process
date created: Sunday, February 4th 2024, 4:27:00 pm
date modified: Tuesday, March 31st 2026, 6:21:51 pm
share: true
---

Gaussian processes (GPs) scale as $O(N^3)$, where $N$ is the number of data points, which is problematic for large datasets.

By parametrising the GP as a [[./stochastic calculus|stochastic differential equation]], we can reformulate the GP regression problem into a [[./linear gaussian ssm|linear gaussian ssm]], where it can be solved using [[./kalman filtering and smoothing|kalman filtering and smoothing]] with linear time complexity ($O(N)$).

The GP kernel defines a continuous-time linear SDE (a special case of the [[./continuous-discrete state space model|continuous-discrete state space model]]). The SDE is then [[./continuous-discrete state space model#Exact discretisation (linear case)|exactly discretised]] via matrix exponentials to obtain the discrete-time transition matrices $\mathbf{F}$ and noise covariance $\mathbf{Q}$. See also [this paper](https://arxiv.org/abs/2505.18187v1) on discretisation of continuous-time linear systems.

*See the [Temporal Gaussian Process Regression in Logarithmic Time](https://arxiv.org/pdf/2102.09964.pdf) and [Kalman filtering and smoothing solutions to temporal Gaussian process regression models](https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf)papers, as well as Adrien Corenflos' [implementation](https://github.com/EEA-sensors/parallel-gps/blob/main/pssgp/kernels/matern/common.py), the `BayesNewton` [implementation](https://github.com/AaltoML/BayesNewton/blob/main/bayesnewton/kernels.py#L238), the `GPy` [implementation](https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/sde_matern.py) and the [RxInfer.jl example](https://examples.rxinfer.com/categories/advanced_examples/gp_regression_by_ssm/). Perhaps could be implemented in [[./ssm in dynamax|dynamax]] by wrapping the filter step within a larger log-likelihood to optimise the lengthscale and variance, but [no success so far](https://github.com/theorashid/dynamax/blob/ssgp/docs/notebooks/linear_gaussian_ssm/ssgp.ipynb).*

## Mátern kernels

Each Mátern kernel ($\nu = p + 1/2$) corresponds to a linear SDE (a [[./continuous-discrete state space model|continuous-discrete state space model]]) with state dimension $p + 1$. We need to train lengthscale ($\ell$), variance ($\alpha$), and likelihood noise ($\sigma$).

### Mátern-5/2 ($\nu = 5/2$, $p = 2$)

A typical choice for spatial and temporal problems. The continuous-time SDE is:

$$d\mathbf{z}(t) = \underbrace{\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ -\lambda^3 & -3\lambda^2 & -3\lambda \end{pmatrix}}_{\mathbf{A}} \mathbf{z}(t)\, dt + \underbrace{\begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}}_{\mathbf{L}}\, dW(t), \qquad \lambda = \sqrt{5} / \ell$$

The Brownian motion is scalar ($d_w = 1$), entering only the third state dimension. The observation model picks out the first component of the state:

$$y_t = \underbrace{\begin{pmatrix} 1 & 0 & 0 \end{pmatrix}}_{\mathbf{H}} \mathbf{z}(t) + r_t, \qquad r_t \sim \mathcal{N}(0, \sigma^2)$$

### Mátern-1/2 ($\nu = 1/2$, $p = 0$)

A scalar SDE (state dimension 1):

$$dz(t) = \underbrace{-\lambda}_{A}\, z(t)\, dt + \underbrace{1}_{L}\, dW(t), \qquad \lambda = 1 / \ell$$

with observation $y_t = z(t) + r_t$, $r_t \sim \mathcal{N}(0, \sigma^2)$.

### discretisation to LGSSM

Each SDE is [[./continuous-discrete state space model#Exact discretisation (linear case)|exactly discretised]] to a [[./linear gaussian ssm|linear gaussian ssm]] with step size $\Delta t = t_{k+1} - t_k$:

$$\mathbf{z}_{t+1} = \mathbf{F}\, \mathbf{z}_t + \mathbf{q}_t, \qquad \mathbf{q}_t \sim \mathcal{N}(0, \mathbf{Q})$$

where:

$$\mathbf{F} = \exp(\mathbf{A}\,\Delta t), \qquad \mathbf{Q} = \int_0^{\Delta t} \exp(\mathbf{A}\,s)\, \mathbf{L}\mathbf{L}^T\, \exp(\mathbf{A}\,s)^T\, ds$$

For Mátern-1/2 this simplifies to $F = e^{-\lambda \Delta t}$ and $Q = \frac{1}{2\lambda}(1 - e^{-2\lambda \Delta t})$. For higher-order Mátern, $\mathbf{F}$ and $\mathbf{Q}$ are dense matrices computed numerically. Inference then uses [[./kalman filtering and smoothing|kalman filtering and smoothing]].
