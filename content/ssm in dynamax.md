---
tags:
  - ssm
  - ml
  - bayesian
  - jax
  - python
folder: ssm
title: ssm in dynamax
date created: Sunday, February 4th 2024, 4:25:08 pm
date modified: Sunday, February 25th 2024, 7:58:46 pm
share: true
---

Minimal example for a [[./linear gaussian ssm|linear gaussian ssm]]. See the [dynamax docs](https://probml.github.io/dynamax/index.html) for more examples.

```python
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_smoother, lgssm_filter

latent_dim = ... # N_states
observation_dim = ... # N_obs

y = ... # shape (N_obs, 1)

lgssm = LinearGaussianSSM(latent_dim, observation_dim)
params, _ = lgssm.initialize(
	jax.random.PRNGKey(0)
    initial_mean=initial_mean, # of the state, (N_states, 1)
    initial_covariance= initial_covariance, # (N_states, N_states)
    dynamics_weights=F, # (N_states, N_states)
    dynamics_covariance=Q, # (N_states, N_states)
    emission_weights=H, # (N_obs, N_states)
    emission_covariance=R, # (N_obs, N_obs)
)

# filtering
lgssm_filtered_posterior = lgssm.filter(params, y)

# smoothing
lgssm_smoothed_posterior = lgssm.smoother(params, y)
```
