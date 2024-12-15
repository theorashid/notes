---
tags:
  - bayesian
folder: learning
share: true
title: probability integral transform
date created: Saturday, December 14th 2024, 1:22:52 pm
date modified: Saturday, December 14th 2024, 4:03:49 pm
---

Transform from any distribution to uniform and back:

- apply **inverse CDF** for **uniform to distribution**
	- e.g. `norm.ppf(x)` stretches the outer regions of the $\text{Uniform}(0,1)$ to yield a normal. This works for this for arbitrary (univariate) probability distributions.
- apply **CDF** (`.cdf()`) for arbitrary **distribution to uniform**

Used with [[./copula|copula]]s.
