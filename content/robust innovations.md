---
tags:
  - ssm
folder: learning
share: true
title: robust innovations
date created: Wednesday, October 7th 2026, 6:23:22 pm
date modified: Wednesday, October 7th 2026, 6:41:27 pm
---

A **robust innovation** limits how much an unusual observation changes the state of a [[./state space model|state space model]].

In [[./kalman filtering and smoothing|Kalman filtering]], the time update is

$$
\boldsymbol{\mu}_{t|t-1}
=
\mathbf{F}_t\boldsymbol{\mu}_{t-1|t-1},
$$

with expected observation

$$
\hat{\mathbf{y}}_t
=
\mathbf{H}_t\boldsymbol{\mu}_{t|t-1}.
$$

The measurement update is

$$
\boldsymbol{\mu}_{t|t}
=
\boldsymbol{\mu}_{t|t-1}
+
\mathbf{K}_t(\mathbf{y}_t-\hat{\mathbf{y}}_t).
$$

The innovation $\mathbf{y}_t-\hat{\mathbf{y}}_t$ is new information from the observation. An outlier creates a large state update which is then propagated into future predictions.

A robust update uses

$$
\boldsymbol{\mu}_{t|t}
=
\boldsymbol{\mu}_{t|t-1}
+
\mathbf{K}_t\boldsymbol{\psi}
(\mathbf{y}_t-\hat{\mathbf{y}}_t).
$$

Robustness changes the measurement update. Persistence remains in $\mathbf{F}_t$; for a scalar AR(1) state, $F_t=\phi$.

## raw innovations

The standard [[./kalman filtering and smoothing|kalman filter]] uses

$$
\psi(a)=a,
$$

where $a$ denotes one scalar innovation. Its influence is unbounded:

$$
\lim_{|a|\rightarrow\infty}|\psi(a)|=\infty.
$$

**This learns genuine changes quickly**, but one bad observation can substantially change the state.

## bounded innovations

[Huber introduced bounded M-estimation in 1964](https://doi.org/10.1214/aoms/1177703732):

$$
\psi_H(a)=
\begin{cases}
a, & |a|\leq c,\\
c\operatorname{sign}(a), & |a|>c.
\end{cases}
$$

It is linear near zero, bounded and monotone.

**This limits outlier influence while continuing to learn structural changes.** However, the clipping threshold $c$ must be chosen, and even an arbitrarily extreme observation moves the state by $c$.

A smooth alternative is

$$
\psi_{\tanh}(a)
=
c\tanh\left(\frac{a}{c}\right).
$$

### robust exponential smoothing

[Gelper, Fried and Croux (2010) applied bounded innovations to exponential and Holt–Winters smoothing](https://doi.org/10.1002/for.1125)).

Construct a cleaned observation:

$$
y_t^*
=
\hat y_{t|t-1}
+
\hat\sigma_t
\psi_H\left(
\frac{y_t-\hat y_{t|t-1}}{\hat\sigma_t}
\right).
$$

Then apply ordinary exponential smoothing (can instead be applied for level, trend and seasonality updates):

$$
\hat y_{t+1|t}
=
\hat y_{t|t-1}
+
\alpha(y_t^*-\hat y_{t|t-1}).
$$

Substitution gives

$$
\hat y_{t+1|t}
=
\hat y_{t|t-1}
+
\alpha\hat\sigma_t
\psi_H\left(
\frac{y_t-\hat y_{t|t-1}}{\hat\sigma_t}
\right).
$$

This is the Huber innovation applied to an exponential-smoothing state.

**This prevents an observation outlier from becoming a persistent forecast change**, but may delay adaptation to a genuine level shift.

## score-based robust filtering

[Masreliez (1975) introduced score-based non-Gaussian filtering](https://doi.org/10.1109/TAC.1975.1100882)).

Before observing $\mathbf{y}_t$, the filter has the predictive density

$$
p(\mathbf{y}_t\mid\mathbf{y}_{1:t-1}).
$$

The score with respect to its predicted location is

$$
\boldsymbol{\nabla}_t
=
\nabla_{\hat{\mathbf{y}}_t}
\log p(\mathbf{y}_t\mid\mathbf{y}_{1:t-1}).
$$

It points in the direction that would make $\mathbf{y}_t$ more likely, with magnitude measuring the strength of that evidence.

For a Gaussian predictive density,

$$
p(\mathbf{y}_t\mid\mathbf{y}_{1:t-1})
=
\mathcal{N}
(\mathbf{y}_t\mid\hat{\mathbf{y}}_t,\mathbf{S}_t),
$$

the score is

$$
\boldsymbol{\nabla}_t
=
\mathbf{S}_t^{-1}
(\mathbf{y}_t-\hat{\mathbf{y}}_t).
$$

Scaling by $\mathbf{S}_t$ recovers the raw innovation:

$$
\mathbf{S}_t\boldsymbol{\nabla}_t
=
\mathbf{y}_t-\hat{\mathbf{y}}_t.
$$

For a scalar innovation $a=y-\hat y$,

$$
\frac{\partial\log p(y\mid\hat y)}{\partial\hat y}
=
-\frac{\partial\log p(a)}{\partial a},
$$

because $\partial a/\partial\hat y=-1$.

**This derives the update from the predictive distribution.** Robustness depends on the chosen density. Masreliez filtering is also an approximation to the generally intractable non-Gaussian filter.

## GAS and DCS

[Creal, Koopman and Lucas (2013) introduced Generalised Autoregressive Score models](https://doi.org/10.1002/jae.1279)). [Harvey (2013) developed the equivalent Dynamic Conditional Score framework](https://doi.org/10.1017/CBO9781139540933)). GAS and DCS are two names for essentially the same score-driven model class.

Let $\boldsymbol{\theta}_t$ be a time-varying parameter of the predictive density:

$$
\mathbf{y}_t
\sim
p(\mathbf{y}_t\mid\boldsymbol{\theta}_t,\mathbf{y}_{1:t-1}).
$$

Calculate the score

$$
\boldsymbol{\nabla}_t
=
\nabla_{\boldsymbol{\theta}_t}
\log p(\mathbf{y}_t\mid\boldsymbol{\theta}_t,\mathbf{y}_{1:t-1}),
$$

then optionally scale it:

$$
\mathbf{s}_t
=
\mathbf{W}_t\boldsymbol{\nabla}_t.
$$

The parameter recursion is

$$
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\omega}
+
\mathbf{A}\mathbf{s}_t
+
\mathbf{B}\boldsymbol{\theta}_t.
$$

Here $\mathbf{B}$ controls persistence. In a scalar model, $B=\phi$ and $A=\kappa$:

$$
\theta_{t+1}
=
\omega(1-\phi)
+
\kappa s_t
+
\phi\theta_t.
$$

When $\boldsymbol{\theta}_t$ is the predicted location, a Gaussian score gives a raw innovation. A heavy-tailed score gives a robust innovation.

## Student-t score innovations

[Harvey and Luati (2014) used the Student-t score for robust filtering]([https://doi.org/10.1080/01621459.2014.887011](https://doi.org/10.1080/01621459.2014.887011)).

For one standardised innovation,

$$
a=\frac{y-\hat y}{\sigma},
$$

the Student-t log density is

$$
\log p(a)
=
C-\frac{\nu+1}{2}
\log\left(1+\frac{a^2}{\nu}\right).
$$

Its rescaled location score is

$$
\psi_t(a)
=
\frac{a}{1+a^2/\nu}.
$$

For small innovations,

$$
\psi_t(a)\approx a.
$$

Its derivative is

$$
\psi_t'(a)
=
\frac{1-a^2/\nu}
{(1+a^2/\nu)^2},
$$

so the update is largest at

$$
|a|=\sqrt{\nu}.
$$

After this point, larger innovations produce smaller updates:

$$
\lim_{|a|\rightarrow\infty}\psi_t(a)=0.
$$

For a GAS/DCS model, this score enters through $\mathbf{s}_t$.

The score is **redescending, strongly rejecting isolated outliers**. It can also reject genuine structural breaks: when the prediction is very wrong, its correction approaches zero.
