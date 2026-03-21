---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: nonlinear gaussian ssm
date created: Sunday, February 4th 2024, 4:22:44 pm
date modified: Monday, March 16th 2026, 1:36:26 pm
share: true
---

Moving from [[./linear gaussian ssm|linear gaussian ssm]], we relax the assumptions of linear dynamics ($\mathbf{F}_t \mathbf{z}_t \rightarrow \mathbf{f}(\mathbf{z}_t)$) and linear observation models ($\mathbf{H}_{t}\mathbf{z}_t \rightarrow \mathbf{h}(\mathbf{z}_t)$).

$$
\begin{align}
\mathbf{z}_{t+1} &= \mathbf{f}(\mathbf{z}_{t}, \mathbf{u}_{t}) + \mathbf{q}_t, \qquad \mathbf{q}_t \sim \mathcal{N}(0, \mathbf{Q}_{t}) \\
\mathbf{y}_t &= \mathbf{h}(\mathbf{z}_{t}, \mathbf{u}_{t}) + \mathbf{r}_t, \qquad  \mathbf{r}_t \sim \mathcal{N}(0, \mathbf{R}_t)
\end{align}
$$

This model can be used to [track an object performing nonlinear motion](https://probml.github.io/dynamax/notebooks/nonlinear_gaussian_ssm/ekf_ukf_spiral.html) or in [[./online learning using ssm|online learning using ssm]] for [neural networks](https://probml.github.io/dynamax/notebooks/nonlinear_gaussian_ssm/ekf_mlp.html).

## inference

Since the dynamics $\mathbf{f}$ and/or observation $\mathbf{h}$ are nonlinear, we can no longer use the exact [[./kalman filtering and smoothing|Kalman filter]] directly. Instead, we **linearise** the nonlinear functions to recover a local linear-Gaussian approximation, then apply the standard Kalman update.

### Extended Kalman filter (EKF)

First-order Taylor expansion around the current state estimate $\hat{\mathbf{z}}_t$:

$$
\mathbf{f}(\mathbf{z}_t) \approx \mathbf{f}(\hat{\mathbf{z}}_t) + \mathbf{J}_f(\hat{\mathbf{z}}_t)(\mathbf{z}_t - \hat{\mathbf{z}}_t)
$$

giving the local linear parameters $\mathbf{F}_t = \mathbf{J}_f(\hat{\mathbf{z}}_t)$ (the Jacobian) and $\mathbf{c}_t = \mathbf{f}(\hat{\mathbf{z}}_t) - \mathbf{F}_t \hat{\mathbf{z}}_t$. The same applies to the observation function $\mathbf{h}$ to obtain $\mathbf{H}_t$ and $\mathbf{d}_t$.

- **Taylor linearisation** ([`cuthbert.gaussian.taylor`](https://state-space-models.github.io/cuthbert/cuthbert_api/gaussian/taylor/)): provide log densities $\log p(\mathbf{z}_t | \mathbf{z}_{t-1})$ and $\log p(\mathbf{y}_t | \mathbf{z}_t)$. cuthbert [[./automatic differentiation|auto-differentiates]] (`jax.hessian` + `jax.jacobian`) to extract the local linear-Gaussian approximation.

```python
from cuthbert.gaussian.taylor import build_filter

filter_obj = build_filter(
    get_init_log_density=...,       # returns log p(x_0) + linearisation point
    get_dynamics_log_density=...,   # returns log p(x_t | x_{t-1}) + linearisation points
    get_observation_func=...,       # returns log p(y_t | x_t) + linearisation point
)
```

- **Moments linearisation** ([`cuthbert.gaussian.moments`](https://state-space-models.github.io/cuthbert/cuthbert_api/gaussian/moments/)): provide conditional mean and Cholesky covariance functions. cuthbert linearises via `jax.jacfwd`.

```python
from cuthbert.gaussian.moments import build_filter

filter_obj = build_filter(
    get_init_params=...,            # returns (m0, chol_P0)
    get_dynamics_params=...,        # returns (mean_and_chol_cov_func, linearisation_point)
    get_observation_params=...,     # returns (mean_and_chol_cov_func, linearisation_point, y)
)
```

### Unscented Kalman filter (UKF)

Instead of linearising analytically, the UKF passes a set of deterministically chosen **sigma points** through the nonlinear functions $\mathbf{f}$ and $\mathbf{h}$, then computes the mean and covariance of the transformed points. This captures second-order effects that the EKF misses and does not require computing Jacobians.

### Particle filter (SMC)

For strongly nonlinear models where Gaussian approximations are poor, sequential Monte Carlo (SMC) / particle filtering can be used. This makes no Gaussian assumption on the posterior but scales poorly with state dimension. See [[./state space model#Inference and parameter estimation|inference methods]].
