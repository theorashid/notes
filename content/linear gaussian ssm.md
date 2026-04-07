---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: linear gaussian ssm
date created: Sunday, February 4th 2024, 4:21:59 pm
date modified: Tuesday, April 7th 2026, 1:38:48 pm
share: true
---

When the dynamics model and the observation model of a [[./state space model|state space model]] are **both Gaussian**, we have a [[linear gaussian ssm|linear gaussian ssm]],

The latent **dynamics model** is written as

$$
\mathbf{z}_{t+1} = \mathbf{F}_t \mathbf{z}_t + \mathbf{q}_t, \qquad \mathbf{q}_t \sim \mathcal{N}(0, \mathbf{Q}_t)
$$

with the **observation model**

$$
\mathbf{y}_t = \mathbf{H}_t \mathbf{z}_t + \mathbf{u}_t + \mathbf{r}_t, \qquad  \mathbf{r}_t \sim \mathcal{N}(0, \mathbf{R}_t)
$$

| variable | variable description | shape |
| ---- | ---- | ---- |
| $\mathbf{z}_t$ | state vector | `(N_states, 1)` |
| $\mathbf{y}_t$ | observation vector at time $t$ | `(N_obs, 1)` |
| $\mathbf{F}_t$ | dynamics (transition) matrix | `(N_states, N_states)` |
| $\mathbf{Q}_t$ | covariance matrix of dynamics (system) noise | `(N_states, N_states)` |
| $\mathbf{H}_t$ | emission (observation) matrix | `(N_obs, N_states)` |
| $\mathbf{R}_t$ | covariance function for emission (observation) noise | `(N_obs, N_obs)` |

$\mathbf{H}_t$ *extracts* the relevant parts of the state vector $\mathbf{z}_t$. $\mathbf{u}_t$ are exogenous inputs.

Inference can be performed efficiently using [[./kalman filtering and smoothing|kalman filtering and smoothing]]. These models have applications in [object tracking](https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/kf_tracking.html) and [[./structural time series models|structural time series models]].

## jax implementation (cuthbert)

[cuthbert](https://github.com/state-space-models/cuthbert) provides an exact Kalman filter and smoother in square-root form via [`cuthbert.gaussian.kalman`](https://state-space-models.github.io/cuthbert/cuthbert_api/gaussian/kalman/).

```python
from cuthbert import filter, smoother
from cuthbert.gaussian import kalman

filter_obj = kalman.build_filter(
    get_init_params=lambda inputs: (m0, chol_P0),          # p(x_0)
    get_dynamics_params=lambda inputs: (F, c, chol_Q),     # p(x_t | x_{t-1})
    get_observation_params=lambda inputs: (H, d, chol_R, y),  # p(y_t | x_t)
)

states = filter(filter_obj, model_inputs, parallel=True)
```

The filter supports temporal parallelisation via [`jax.lax.associative_scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html) ([[./kalman filtering and smoothing#Parallel (associative) Kalman filter|associative Kalman filter]], [cuthbert parallelization example](https://state-space-models.github.io/cuthbert/examples/temporal_parallelization_kalman/)).

*See also [[./ssm in dynamax|ssm in dynamax]] and [[./ssm resources|ssm resources]]*.

## canonical Bayesian system identification

When the system matrices are **time-invariant** ($\mathbf{F}_t = \mathbf{F}$, $\mathbf{H}_t = \mathbf{H}$, etc.), the LGSSM is a **linear time-invariant (LTI)** system – the standard object studied in control theory and system identification.

[Bryutkin et al. (2025)](https://arxiv.org/abs/2507.11535) show that Bayesian inference over the standard parameterisation $\{\mathbf{F}, \mathbf{B}, \mathbf{H}, \mathbf{D}, \mathbf{Q}, \mathbf{R}\}$ suffers from **non-identifiability**: for any invertible $\mathbf{T}$, the transformed system $(\mathbf{T}^{-1}\mathbf{F}\mathbf{T},\; \mathbf{T}^{-1}\mathbf{B},\; \mathbf{H}\mathbf{T},\; \mathbf{D})$ produces an identical likelihood. The Kalman filter marginalises out the latent states, and all observation-space quantities are invariant:

- Eigenvalues: $\det(\mathbf{T}^{-1}\mathbf{F}\mathbf{T} - \lambda \mathbf{I}) = \det(\mathbf{T}^{-1}(\mathbf{F} - \lambda \mathbf{I})\mathbf{T}) = \det(\mathbf{F} - \lambda \mathbf{I})$ via the multiplicative property of the determinant
- Transfer function: $(\mathbf{H}\mathbf{T})(z\mathbf{I} - \mathbf{T}^{-1}\mathbf{F}\mathbf{T})^{-1}(\mathbf{T}^{-1}\mathbf{B}) = \mathbf{H}(z\mathbf{I} - \mathbf{F})^{-1}\mathbf{B}$ as $\mathbf{T}$ and $\mathbf{T}^{-1}$ cancel in pairs
- Likelihood: the Kalman innovation mean $(\mathbf{H}\mathbf{T})(\mathbf{T}^{-1}\mathbf{z}_t) = \mathbf{H}\mathbf{z}_t$ and covariance $(\mathbf{H}\mathbf{T})(\mathbf{T}^{-1}\mathbf{P}_t\mathbf{T}^{-\top})(\mathbf{H}\mathbf{T})^\top + \mathbf{R} = \mathbf{H}\mathbf{P}_t\mathbf{H}^\top + \mathbf{R}$ are unchanged

There are infinitely many such equivalent systems (one per invertible $\mathbf{T}$), so putting priors on matrix entries creates a posterior with infinitely many equivalent modes that are hard to sample with MCMC.

### canonical form ($d_x = 2$, SISO)

The **controller canonical form** fixes a unique representative. For $d_x = 2$ with eigenvalues $\lambda_1, \lambda_2$, the characteristic polynomial is $\lambda^2 + a_1 \lambda + a_0 = (\lambda - \lambda_1)(\lambda - \lambda_2)$, giving $a_1 = -(\lambda_1 + \lambda_2)$ and $a_0 = \lambda_1 \lambda_2$ (Vieta's formulas). The canonical system is:

$$
\mathbf{F}_c = \begin{pmatrix} 0 & 1 \\ -a_0 & -a_1 \end{pmatrix}, \quad \mathbf{B}_c = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad \mathbf{H}_c = \begin{pmatrix} b_0 & b_1 \end{pmatrix}, \quad D_c = d_0
$$

The free parameters are just $\{a_0, a_1, b_0, b_1, d_0\}$ -- five scalars instead of the $4 + 2 + 2 + 1 = 9$ entries of $\{\mathbf{F}, \mathbf{B}, \mathbf{H}, D\}$. To enforce stability, set a prior on eigenvalues with $|\lambda_i| < 1$ (e.g., $\lambda_i \sim \text{Uniform}(-1, 1)$ for real eigenvalues) and push forward through Vieta's to get the prior on $(a_0, a_1)$.

Note, the paper focuses on SISO systems. MIMO canonical forms (needed for e.g. [[./structural time series models|structural time series models]]) are more complex and depend on the system's Kronecker indices.

### key results

**Canonical forms** resolve the non-identifiability by providing a unique, minimal representative per equivalence class. Three key results:

1. Posteriors over any **similarity-invariant** quantity (eigenvalues, transfer functions, predictive distributions) are identical whether computed from the canonical or standard parameterisation – the canonical space just has fewer parameters ($2d_x + 1$ for SISO vs $\mathcal{O}(d_x^2)$).
2. Canonical coefficients relate to eigenvalues via Vieta's formulas (invertible, with Vandermonde Jacobian), enabling **tractable eigenvalue priors** (e.g., $|\lambda_i| < 1$ for stability) – intractable with the full $\mathbf{F}$ matrix.
3. The canonical posterior satisfies **Bernstein-von Mises**: it converges to $\mathcal{N}(\hat{\theta}_c,\; \mathcal{I}(\theta_c^0)^{-1} / T)$ as $T \to \infty$. The standard parameterisation cannot, because its Fisher information is **singular** along similarity-transform directions.
