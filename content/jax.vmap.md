---
tags:
  - jax
  - ml
  - python
  - swe
folder: learning
title: jax.vmap
date created: Sunday, February 4th 2024, 4:30:33 pm
date modified: Sunday, February 25th 2024, 12:18:51 pm
share: "true"
---

[`jax.vmap`](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#auto-vectorization-with-vmap) is a functional transform for vectorised mapping. The operation `vmap(func)(batched_array)` is the equivalent of the pure-python loop

```python
def func(x):
	...
	return result

np.stack([func(array) for array in batched_array])
```

By default, it vectorises over the leading (batch) dimension (`in_axes=0`). This argument can be used to change the dimension of vectorisation. For example, a summing function `sum_vector` can act over rows of a matrix, `rowsum_func = vmap(sum_vector, in_axes=0)`, or the columns, `colsum_func = vmap(sum_vector, in_axes=1)`.

Further examples including [converting rows of a matrix into a stack of probability vectors](https://ericmjl.github.io/dl-workshop/02-jax-idioms/01-loopless-loops.html) and [parallelising MCMC inference when sampling multiple chains](https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html#using-jax-vmap).

The developer documentation [explains how `vmap` is implemented](https://jax.readthedocs.io/en/latest/autodidax.html).
