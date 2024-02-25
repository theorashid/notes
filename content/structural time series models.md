---
tags:
  - ssm
  - bayesian
  - ml
folder: ssm
title: structural time series models
date created: Sunday, February 11th 2024, 9:07:02 pm
date modified: Sunday, February 25th 2024, 7:59:02 pm
share: true
---

The [structural time series model](https://github.com/probml/sts-jax) is a [[./linear gaussian ssm|linear gaussian ssm]] with a specific structure:

- $\mathbf{z}_t$ is a composition of states of all latent components, $\mathbf{z}_t = [c_{1, t}, c_{2, t}, ...]$, where $c_{i,t}$ is the state of latent component $c_i$ at time step $t$.
- Observation noise is the same for all $\mathbf{y}_t$, so $\mathbf{R}_t \rightarrow \sigma_t$.

$$
\begin{align}
\mathbf{z}_{t+1} &= \mathbf{F}_t \mathbf{z}_t + \mathbf{q}_t, \qquad \mathbf{q}_t \sim \mathcal{N}(0, \mathbf{Q}_{t})\\
\mathbf{y}_t &= \mathbf{H}_t \mathbf{z}_t + \mathbf{u}_t + \mathbf{\epsilon}_t, \qquad  \mathbf{\epsilon}_t \sim \mathcal{N}(0, \sigma^2_{t})
\end{align}
$$

## example for hierarchical or grouped time series

Consider an example where we are forecasting across [multiple time series](https://otexts.com/fpp2/hierarchical.html).

We want a shared (global) autoregressive effect and seasonalities, and then each site to have its own autoregressive effect and seasonalities.

In this case, we can make up the latent components such as

$$
\mathbf{z}_{t} = \begin{pmatrix}
\text{global AR effect}_{t} \\
\text{global seasonal components}_{t} ... \\
\text{local AR effect site }1_{t} \\
\text{local seasonal components, site 1}_{t} ... \\
\text{local AR effect site }2_{t} \\
\text{local seasonal components, site 2}_{t} \\
\vdots \\
\text{local AR effect site }N_{t} \\
\text{local seasonal components, site N}_{t} \\
\end{pmatrix}
$$with length $(1 + N_{\text{seasonal}}) + (1 + N_{\text{seasonal}}) \times N_{\text{sites}} = (1 + N_{\text{seasonal}})(1 + N_{\text{sites}})$. We can then use $\mathbf{H}_{t}$ to pick out the relevant components. For example, consider three sites ($N_\text{obs} = 3$) with three seasonal components (dummies here, but could be Fourier terms). Picking a timestep where the second seasonality component is active (transpose for readability):
$$

\mathbf{H}^T_t = \begin{pmatrix}

1 & 1 & 1 \\

1 & 1 & 1 \\

0 & 0 & 0 \\

1 & 0 & 0 \\ 

1 & 0 & 0 \\

0 & 0 & 0 \\

0 & 1 & 0 \\

0 & 1 & 0 \\

0 & 0 & 0 \\

0 & 0 & 1 \\

0 & 0 & 1 \\

0 & 0 & 0 \\

\end{pmatrix}

$$
- First row is all 1 as all three sites have the global autoregressive term
- Second row is all 1 for the second component of global seasonality (first term is reference for identifiability)
- R3: all 0 as third component of seasonality is not active on this timestep
- R4: AR term for site 1 only
- R5: seasonality component 2 for site 1 only
- R6: seasonality component 3 is not active on this timestep
- R7: AR term for site 2 only
- R8: seasonality component 2 for site 2 only
- R9: seasonality component 3 is not active on this timestep
- R10: AR term for site 3 only
- R11: seasonality component 3 for site 2 only
- R12: seasonality component 3 is not active on this timestep

Further modelling assumptions can be made:
- $\mathbf{Q}_t \rightarrow \sigma_q$. Diagonal and static dynamics covariance, so no covariance in the autoregressive effects.
- $\mathbf{F}_{t} = \mathbf{F} = \text{diag}(1, ..., 1, \rho_1, ..., \rho_N)$. Static diagonal dynamics matrix (vector), with perfect correlation for the global effects and measurable correlation for the local effects, and no cross-correlation.

Site-specific covariates, such as holidays, can be implemented as $\mathbf{u}_t = \sum_{i} \mathbf{X}_t \odot \mathbf{\beta}$, where both the design matrix (of zeros and ones) and the covariates have shape $N_{\text{sites}} \times N_{\text{covariates}}$.
