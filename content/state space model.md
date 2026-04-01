---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: state space model
date created: Sunday, February 11th 2024, 8:33:45 pm
date modified: Tuesday, March 31st 2026, 5:56:31 pm
share: true
---

A state space model is a partially observed **Markov model**:

- $\mathbf{z}_t$ are the latent **states**
- $\mathbf{y}_t$ are the **observations**
- $\mathbf{u}_t$ are exogenous inputs

![ssm-pgm](./notes/learning/assets/ssm-pgm.png)

Using the Markov property, the above probabilistic graphical model forms the joint distribution

$$p(\mathbf{y}_{1:T}, \mathbf{z}_{1:T} | \mathbf{u}_{1:T}) = [p(\mathbf{z}_1| \mathbf{u}_{1)}\prod_i^T\underbrace{p(\mathbf{z}_t| \mathbf{z}_{t-1}, \mathbf{u}_t)}_{\text{dynamics model}}][\prod_i^T\underbrace{p(\mathbf{y}_t| \mathbf{z}_t, \mathbf{u}_t)}_{\text{observation model}}]$$

| Dynamics model | Observation model | Model |
|---|---|---|
| Categorical (discrete states) | Any | [[./hidden markov model|hidden markov model]] |
| Linear Gaussian | Linear Gaussian | [[./linear gaussian ssm|linear gaussian ssm]] |
| Nonlinear, Gaussian noise | Nonlinear, Gaussian noise | [[./nonlinear gaussian ssm|nonlinear gaussian ssm]] |
| Gaussian (linear or nonlinear) | Non-Gaussian | [[./generalised gaussian ssm|generalised gaussian ssm]] |
| Continuous-time SDE | Discrete-time observations | [[./continuous-discrete state space model|continuous-discrete state space model]] |

*See [ssm resources](./ssm%2520resources.md#)*.

## Inference and parameter estimation

Inference has two parts: an **inner loop** (state estimation given fixed $\theta$) and an **outer loop** (learning $\theta$ using the inner loop's log marginal likelihood $\log p(\mathbf{y}_{1:T} | \theta)$).

### State estimation (inner loop)

- **Filtering**: $p(\mathbf{z}_t | \mathbf{y}_{1:t})$ -- online, forward pass
- **Smoothing**: $p(\mathbf{z}_t | \mathbf{y}_{1:T})$ -- offline, forward-backward

|                           | **HMM**  | **LGSSM** | **Nonlinear Gaussian** | **Generalised** |
| ------------------------- | -------- | --------- | ---------------------- | --------------- |
| **Discrete filter**       | Exact    | --        | --                     | --              |
| **Kalman filter**         | --       | Exact     | --                     | --              |
| **EKF / UKF**             | --       | Overkill  | Approximate            | Poor*           |
| **Particle filter (SMC)** | Overkill | Overkill  | Yes                    | Yes             |

*Local Gaussian approximation of non-Gaussian likelihoods (e.g. [[./generalised gaussian ssm|CMGF]]) may be poor for multimodal or heavy-tailed distributions.

### Parameter estimation (outer loop)

- **MLE / MAP**: differentiate $\log p(\mathbf{y}_{1:T} | \theta)$ w.r.t. $\theta$, optimise with gradient ascent (e.g. [optax](https://github.com/google-deepmind/optax))
- **MCMC**: sample $p(\theta | \mathbf{y}_{1:T})$ using the inner loop likelihood. For non-Gaussian models, a particle filter gives a noisy but unbiased likelihood estimate that still targets the correct posterior ([Particle MCMC](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)). Use with e.g. [blackjax](https://github.com/blackjax-devs/blackjax) or [numpyro](https://github.com/pyro-ppl/numpyro)
- **EM**: E-step runs filter + smoother for expected sufficient statistics; M-step updates $\theta$

Model and inference are **decoupled** -- [dynamax](https://probml.github.io/dynamax/) bundles them together; [cuthbert](https://github.com/state-space-models/cuthbert) keeps them separate.
