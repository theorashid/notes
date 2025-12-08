---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: linear gaussian ssm
date created: Sunday, February 4th 2024, 4:21:59 pm
date modified: Thursday, December 4th 2025, 5:55:58 pm
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

*See [[./ssm resources|ssm resources]]*.
