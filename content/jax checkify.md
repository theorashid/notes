---
tags:
  - jax
  - ml
  - python
folder: learning
share: true
title: jax checkify
date created: Friday, December 5th 2025, 5:43:11 pm
date modified: Friday, December 5th 2025, 6:09:35 pm
---

[`jax.experimental.checkify`](https://docs.jax.dev/en/latest/debugging/checkify_guide.html) allows [[./jax.jit|jit]]-able runtime error checking.

[GPJax use checks](https://github.com/thomaspinder/GPJax/blob/7a7ba096f09b580c14783b91a0e3617b5c3b3b50/gpjax/parameters.py#L90-L95) to make sure the lengthscale parameters are non-negative.

```python
@checkify.checkify
def _check_is_non_negative(value):
    checkify.check(
        jnp.all(value >= 0), "value needs to be non-negative, got {value}", value=value
    )
```

This means at runtime, when this particular part of the function jit-compiled, we need to also wrap with `jax.experimental.checkify` so that the assertion is passed and [compilation is not broken](https://docs.jaxgaussianprocesses.com/sharp_bits/#jit-compilation).

```python
jit_compute_gram = jax.jit(jax.experimental.checkify(compute_gram))
error, value = jit_compute_gram(1.0)
```
