---
tags:
  - bayesian
folder: learning
share: true
title: marginalisation of variables and rao-blackwell
date created: Friday, January 9th 2026, 5:25:54 pm
date modified: Friday, January 9th 2026, 5:35:39 pm
---

*Adapted from [pymc example](https://www.pymc.io/projects/examples/en/latest/howto/marginalizing-models.html).*

Marginalising out variables from a model results in **lower variance estimates of parameters** in the model through Rao-Blackwell's theorem.

Samplers approximate the expectation $\mathbb{E}_{p(x, z)}[f(x, z)]$ for some function $f$ with respect to a distribution $p(x, z)$. By [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation) we know that

$$ \mathbb{E}_{p(x, z)}[f(x, z)] =  \mathbb{E}_{p(z)}\left[\mathbb{E}_{p(x \mid z)}\left[f(x, z)\right]\right] $$

Letting $g(z) = \mathbb{E}_{p(x \mid z)}\left[f(x, z)\right]$, we know by [law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance) that

$$ \mathbb{V}_{p(x, z)}[f(x, z)] = \mathbb{V}_{p(z)}[g(z)] + \mathbb{E}_{p(z)}\left[\mathbb{V}_{p(x \mid z)}\left[f(x, z)\right]\right] $$

Because the expectation is over a variance it must always be positive, and thus we know

$$ \mathbb{V}_{p(x, z)}[f(x, z)] \geq \mathbb{V}_{p(z)}[g(z)] $$

Marginalising variables ($x$) in your model lets you use $g$ instead of $f$. This lower variance manifests most directly in lower Monte-Carlo standard error, and indirectly in a generally higher effective sample size.
