---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: kalman filtering and smoothing
date created: Wednesday, January 31st 2024, 6:52:35 pm
date modified: Monday, March 16th 2026, 1:30:10 pm
share: true
---

When the dynamics model and the observation model of a [[./state space model|state space model]] are both **Gaussian**, that is we have a [[./linear gaussian ssm|linear gaussian ssm]], we can perform inference efficiently using **Kalman filtering** methods.

We perform **filtering** to predict one step and obtain $p(\mathbf{z}_{t}| \mathbf{y}_{1:t})$. Ignoring the optional exogenous inputs, $\mathbf{u}_t$, the algorithm consists of two steps:

- **Time update** step

$$
\begin{align}
p(\mathbf{z}_t | \mathbf{y}_{1:{t-1}}) &= \mathcal{N}(\mathbf{z}_{t-1} | \mathbf{\mu}_{t|t-1}, \mathbf{\Sigma}_{t|t-1}) \\
\mathbf{\mu}_{t|t-1} &= \mathbf{F}_{t}\mathbf{\mu}_{t-1|t-1} \\
\mathbf{\Sigma}_{t|t-1} &= \mathbf{F}_{t}\mathbf{\Sigma}_{t-1|t-1}\mathbf{F}_{t}^{T} + \mathbf{Q}_{t}
\end{align}
$$

- **Measurement** step, getting the expected observation $\mathbf{\hat{y}}$

$$
\begin{align}
p(\mathbf{z}_{t}| \mathbf{y}_{1:t}) &= \mathcal{N}(\mathbf{z}_{t} | \mathbf{\mu}_{t|t}, \mathbf{\Sigma}_{t|t}) \\
\mathbf{\hat{y}} &= \mathbf{H}_{t} \mathbf{\mu}_{t|t-1} \\
\mathbf{S}_{t} &= \mathbf{H}_{t}\mathbf{\Sigma}_{t|t-1}\mathbf{H}_{t}^{T} + \mathbf{R}_{t} \\
\mathbf{K}_{t} &= \mathbf{\Sigma}_{t|t-1} \mathbf{H}_{t}^{T} \mathbf{S}_{t}^{-1} \\
\mathbf{\mu}_{t|t} &= \mathbf{\mu}_{t|t-1} + \mathbf{K}_{t} (\mathbf{y} - \mathbf{\hat{y}}) \\
\mathbf{\Sigma}_{t|t} &= \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_{t} \mathbf{H}_{t} \mathbf{\Sigma}_{t|t-1} \\
&= \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_{t} \mathbf{S}_{t} \mathbf{K}_{t}^T
\end{align}
$$

The update for the $\mathbf{\Sigma}_{t|t}$ term can also be written in [[kalman filtering and smoothing#Joseph form equivalence|Joseph form]], which is more numerically stable because it guarantees the term remains positive-symmetric even in the presence of floating-point errors.

The algorithm scales as $O(N)$. The full derivation relies on the conjugate properties of Gaussians and is in both the [[./ssm resources|Kevin Murphy and Durbin and Koopman books]].

When all the data have arrived, we can perform offline **smoothing** and obtain $p(\mathbf{z}_{t}| \mathbf{y}_{1:T})$. The algorithm requires two passes through the data, a forward-filter and backward-smoother, and so scales $O(N^2)$.

$$
\begin{align}
p(\mathbf{z}_t | \mathbf{y}_{1:T}) &= \mathcal{N}(\mathbf{z}_{t} | \mathbf{\mu}_{t|T}, \mathbf{\Sigma}_{t|T}) \\
\mathbf{\mu}_{t|t+1} &= \mathbf{F}_{t} \mathbf{\mu}_{t|t} \\
\mathbf{\Sigma}_{t|t+1} &= \mathbf{F}_{t} \mathbf{\Sigma}_{t|t} \mathbf{F}_{t}^{T} + \mathbf{Q}_{t+1} \\
\mathbf{G}_{t} &= \mathbf{\Sigma}_{t|t} \mathbf{F}_{t}^{T} \mathbf{\Sigma}_{t|t+1}^{-1} \\
\mathbf{\mu}_{t|T} &= \mathbf{\mu}_{t|t} + \mathbf{G}_{t} (\mathbf{\mu}_{t+1|T} - \mathbf{\mu}_{t+1|t}) \\
\mathbf{\Sigma}_{t|T} &= \mathbf{\Sigma}_{t|t} + \mathbf{G}_{t} (\mathbf{\Sigma}_{t+1|T} - \mathbf{\Sigma}_{t+1|t}) \mathbf{G}_{t}^T
\end{align}
$$

## parallel (associative) Kalman filter

The standard Kalman filter is sequential – each step depends on the previous filtered mean and covariance, giving $O(T)$ serial complexity. The [**parallel Kalman filter**](https://doi.org/10.1137/23M156121X) reformulates the predict + update step as an **affine map** on the filtered mean, enabling a parallel prefix scan in $O(\log T)$.

### Reformulation as an affine map

Substituting the time update (predict) into the measurement update, the filtered mean at time $t$ is:

$$
\begin{align}
\boldsymbol{\mu}_{t|t} &= \boldsymbol{\mu}_{t|t-1} + \mathbf{K}_t (\mathbf{y}_t - \mathbf{H}_t \boldsymbol{\mu}_{t|t-1} - \mathbf{d}_t) \\
&= (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)(\mathbf{F}_t \boldsymbol{\mu}_{t-1|t-1} + \mathbf{c}_t) + \mathbf{K}_t (\mathbf{y}_t - \mathbf{d}_t) \\
&= (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{F}_t \boldsymbol{\mu}_{t-1|t-1} + (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{c}_t + \mathbf{K}_t (\mathbf{y}_t - \mathbf{d}_t)
\end{align}
$$

where $\mathbf{c}_t$ is a dynamics shift and $\mathbf{d}_t$ is an observation shift from the general [linear gaussian ssm](./linear%2520gaussian%2520ssm.md#) formulation. This has the form of an **affine map** on the previous filtered mean:

$$
\boldsymbol{\mu}_{t|t} = \mathbf{A}_t \boldsymbol{\mu}_{t-1|t-1} + \mathbf{b}_t
$$

where:

$$
\begin{align}
\mathbf{A}_t &= (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{F}_t \\
\mathbf{b}_t &= (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{c}_t + \mathbf{K}_t (\mathbf{y}_t - \mathbf{d}_t)
\end{align}
$$

The covariance update $\boldsymbol{\Sigma}_{t|t}$ does not depend on the mean – it only depends on the model parameters $(\mathbf{F}_t, \mathbf{Q}_t, \mathbf{H}_t, \mathbf{R}_t)$ and can be computed independently.

### associative scan

The composition of two affine maps is itself an affine map:

$$
f_j \circ f_i(\boldsymbol{\mu}) = \mathbf{A}_j(\mathbf{A}_i \boldsymbol{\mu} + \mathbf{b}_i) + \mathbf{b}_j = (\mathbf{A}_j \mathbf{A}_i) \boldsymbol{\mu} + (\mathbf{A}_j \mathbf{b}_i + \mathbf{b}_j)
$$

This composition is **associative** (it is function composition), meaning:

$$
(f_k \circ f_j) \circ f_i = f_k \circ (f_j \circ f_i)
$$

Both sides produce the same $(\mathbf{A}, \mathbf{b})$.

To compute all $T$ filtered states, we need the prefix compositions:

$$
f_1, \quad f_2 \circ f_1, \quad f_3 \circ f_2 \circ f_1, \quad \ldots, \quad f_T \circ \cdots \circ f_1
$$

The **associative scan** (parallel prefix scan) computes all of these in $O(\log T)$ parallel steps by composing pairs in a binary tree:

$$
\begin{align}
\text{Level 0:} \quad & e_0, \quad e_1, \quad e_2, \quad e_3 \\
\text{Level 1:} \quad & e_0, \quad e_{0:1}, \quad e_2, \quad e_{2:3} \\
\text{Level 2:} \quad & e_0, \quad e_{0:1}, \quad e_{0:2}, \quad e_{0:3}
\end{align}
$$

where $e_{i:j}$ denotes the composed affine map from step $i$ to $j$. At each level, independent compositions run in parallel. Associativity guarantees that any grouping produces the correct result, e.g. $e_{0:3} = e_{0:1} \circ e_{2:3}$.

[cuthbert](https://state-space-models.github.io/cuthbert/examples/temporal_parallelization_kalman/) implements this using [`jax.lax.associative_scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html) with `parallel=True` flag:

```python
from cuthbert import filter
from cuthbert.gaussian import kalman

filter_obj = kalman.build_filter(get_init_params, get_dynamics_params, get_observation_params)
states = filter(filter_obj, model_inputs, parallel=True)
```

`cuthbert` has three functions that map to the maths above:

- `init_prepare`: creates the initial element with $\mathbf{A}_0 = \mathbf{0}$, $\mathbf{b}_0 = \boldsymbol{\mu}_0$
- `filter_prepare`: encodes one time step's parameters $(\mathbf{F}_t, \mathbf{Q}_t, \mathbf{H}_t, \mathbf{R}_t, \mathbf{y}_t)$ into the affine scan element $(\mathbf{A}_t, \mathbf{b}_t, \ldots)$ via `associative_params_single`
- `filter_combine`: composes two scan elements via the associative `filtering_operator`

In parallel mode, all elements are prepared independently via `jax.vmap(filter_prepare)`, then composed via `jax.lax.associative_scan(filter_combine, ...)`. In sequential mode, the same `filter_prepare` and `filter_combine` are called inside a `jax.lax.scan` loop.

### square-root form

In practice, the scan element carries additional terms beyond $(\mathbf{A}, \mathbf{b})$ to track the covariance in **Cholesky (square-root) form** and accumulate the log marginal likelihood. Working with Cholesky factors rather than full covariance matrices provides better numerical stability, particularly for long time series. The composition operator is more complex but remains associative.

## Joseph form equivalence

The **Joseph form** is defined as:

$$ \mathbf{\Sigma}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{\Sigma}_{t|t-1} (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)^T + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^T $$

$$ \mathbf{\Sigma}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{\Sigma}_{t|t-1} (\mathbf{I} - \mathbf{H}_t^T \mathbf{K}_t^T) + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^T $$

$$ \mathbf{\Sigma}_{t|t} = \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_t \mathbf{H}_t \mathbf{\Sigma}_{t|t-1} - \mathbf{\Sigma}_{t|t-1} \mathbf{H}_t^T \mathbf{K}_t^T + \mathbf{K}_t \mathbf{H}_t \mathbf{\Sigma}_{t|t-1} \mathbf{H}_t^T \mathbf{K}_t^T + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^T $$

$$ \mathbf{\Sigma}_{t|t} = \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_t \mathbf{H}_t \mathbf{\Sigma}_{t|t-1} - \mathbf{\Sigma}_{t|t-1} \mathbf{H}_t^T \mathbf{K}_t^T + \mathbf{K}_t (\mathbf{H}_t \mathbf{\Sigma}_{t|t-1} \mathbf{H}_t^T + \mathbf{R}_t) \mathbf{K}_t^T $$

Using the definition for the innovation covariance $\mathbf{S}_{t} = \mathbf{H}_{t}\mathbf{\Sigma}_{t|t-1}\mathbf{H}_{t}^{T} + \mathbf{R}_{t}$:

$$ \mathbf{\Sigma}_{t|t} = \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_t \mathbf{H}_t \mathbf{\Sigma}_{t|t-1} - \mathbf{\Sigma}_{t|t-1} \mathbf{H}_t^T \mathbf{K}_t^T + \mathbf{K}_t \mathbf{S}_t \mathbf{K}_t^T $$

Using the definition for the Kalman Gain $\mathbf{K}_{t} = \mathbf{\Sigma}_{t|t-1} \mathbf{H}_{t}^{T} \mathbf{S}_{t}^{-1}$, we have $\mathbf{K}_t \mathbf{S}_t = \mathbf{\Sigma}_{t|t-1} \mathbf{H}_t^T$, and so

$$ \mathbf{\Sigma}_{t|t} = \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_t \mathbf{H}_t \mathbf{\Sigma}_{t|t-1} - (\mathbf{K}_t \mathbf{S}_t) \mathbf{K}_t^T + \mathbf{K}_t \mathbf{S}_t \mathbf{K}_t^T $$

$$ \mathbf{\Sigma}_{t|t} = \mathbf{\Sigma}_{t|t-1} - \mathbf{K}_t \mathbf{H}_t \mathbf{\Sigma}_{t|t-1} $$
