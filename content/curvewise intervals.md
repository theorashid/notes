---
tags:
  - bayesian
  - visualisation
folder: learning
share: true
title: curvewise intervals
date created: Tuesday, October 6th 2026, 6:12:39 pm
date modified: Wednesday, October 7th 2026, 6:23:15 pm
---

A **pointwise interval** contains a proportion of sampled values at each $x$. A **curvewise interval** contains a proportion of complete sampled curves across every $x$.

Suppose there are $N$ sampled curves $y_i(x)$.

## pointwise percentile interval

An 80% pointwise interval uses the 10th and 90th percentiles:

$$L(x) = Q_{0.1}\{y_1(x),\ldots,y_N(x)\}$$$$U(x) = Q_{0.9}\{y_1(x),\ldots,y_N(x)\}$$

Different curves can determine the boundaries at each $x$. The interval gives 80% coverage for a **fixed** $x$, but does not generally contain 80% of complete curves.

## curvewise interval

[`ggdist::curve_interval()`](https://mjskay.github.io/ggdist/reference/curve_interval.html) assigns each complete curve a depth $D_i$. The deepest curve stays most central relative to the other curves across $x$. The central curve is the deepest sampled curve:

$$i^* = \underset{i}{\operatorname{argmax}}\ D_i$$

The central line is therefore **one actual sampled curve**.

The band boundaries generally are not: although each boundary value comes from a selected curve, the curve supplying that value can change with $x$. For a depth cutoff $c$, select

$$S_c = \{i : D_i \geq c\}$$

and construct an envelope from the selected curves:

$$L_c(x) = \min_{i \in S_c} y_i(x),\qquad U_c(x) = \max_{i \in S_c} y_i(x)$$

The cutoff is chosen so that approximately 80% of complete curves remain inside the envelope for every $x$:$$L_c(x) \leq y_i(x) \leq U_c(x)\qquad \text{for all }x$$

As the requested width decreases, less central curves are removed. With a unique deepest curve, the smallest envelope collapses to$$L(x) = U(x) = y_{i^*}(x)$$

Use pointwise intervals for uncertainty at a particular $x$. Use curvewise intervals for uncertainty about the complete shape, peak or timing of a curve.
