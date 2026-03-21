---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: hidden markov model
date created: Monday, March 16th 2026, 1:01:23 pm
date modified: Monday, March 16th 2026, 1:17:14 pm
share: true
---

When the dynamics model of a [[./state space model|state space model]] is categorical, so that the latent states are discrete, we have a **hidden Markov model** (HMM) ([dynamax example](https://probml.github.io/dynamax/notebooks/hmm/casino_hmm_inference.html)).

[cuthbert](https://github.com/state-space-models/cuthbert) provides exact discrete filtering and smoothing via [`cuthbert.discrete`](https://state-space-models.github.io/cuthbert/cuthbert_api/discrete/):

```python
from cuthbert import filter
from cuthbert.discrete.filter import build_filter

filter_obj = build_filter(
    get_init_dist=...,       # returns p(x_0 = i), shape (K,)
    get_trans_matrix=...,    # returns A_ij = p(x_t = j | x_{t-1} = i), shape (K, K)
    get_obs_lls=...,         # returns log p(y_t | x_t = i), shape (K,)
)

states = filter(filter_obj, model_inputs, parallel=True)
```

The discrete filter is associative and supports `jax.lax.associative_scan` for parallel-in-time inference.
