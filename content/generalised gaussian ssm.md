---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: generalised gaussian ssm
date created: Monday, March 16th 2026, 12:55:54 pm
date modified: Monday, March 16th 2026, 1:41:41 pm
share: true
---

Moving from [[./nonlinear gaussian ssm|nonlinear gaussian ssm]], we relax the assumption that the noise is Gaussian. The dynamics may remain Gaussian, but the **observation model** has a non-Gaussian likelihood:

$$
\begin{align}
\mathbf{z}_{t+1} &= \mathbf{f}(\mathbf{z}_{t}) + \mathbf{q}_t, \qquad \mathbf{q}_t \sim \mathcal{N}(0, \mathbf{Q}_{t}) \\
\mathbf{y}_t &\sim p(\mathbf{y}_t | \mathbf{z}_t)
\end{align}
$$

where $p(\mathbf{y}_t | \mathbf{z}_t)$ can be any distribution, e.g.:

- **Poisson**: $y_t \sim \text{Poisson}(\exp(\mathbf{h}(\mathbf{z}_t)))$ for count data
- **Bernoulli**: $y_t \sim \text{Bernoulli}(\sigma(\mathbf{h}(\mathbf{z}_t)))$ for classification
- **Student-t**: for heavy-tailed noise

This is used in [[./online learning using ssm|online learning using ssm]] for [neural network classification](https://probml.github.io/dynamax/notebooks/generalized_gaussian_ssm/cmgf_mlp_classification_demo.html).

## Inference

Exact Kalman filtering is not possible -- the non-Gaussian likelihood means the posterior is no longer Gaussian after the update step. See [[./state space model#Inference and parameter estimation|inference methods]].

- [**Conditional moments Gaussian Filter (CMGF)**](https://probml.github.io/dynamax/notebooks/generalized_gaussian_ssm/cmgf_mlp_classification_demo.html): compute $\mathbb{E}[\mathbf{y}_t | \mathbf{z}_t]$ and $\text{Cov}[\mathbf{y}_t | \mathbf{z}_t]$ and feed into the standard Kalman update. A form of assumed density filtering – projects the true posterior back onto the Gaussian family at each step. In cuthbert: [`cuthbert.gaussian.moments`](https://state-space-models.github.io/cuthbert/cuthbert_api/gaussian/moments/) or [`cuthbert.gaussian.taylor`](https://state-space-models.github.io/cuthbert/cuthbert_api/gaussian/taylor/).
- **Particle filter (SMC)**: no distributional assumptions, just sample from dynamics and evaluate observation likelihood. Asymptotically exact but scales poorly with state dimension. In cuthbert: [`cuthbert.smc.particle_filter`](https://state-space-models.github.io/cuthbert/cuthbert_api/smc/particle_filter/).
