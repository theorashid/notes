---
tags:
  - jax
  - python
  - ml
folder: learning
share: true
title: jax.jit
date created: Friday, December 5th 2025, 5:40:38 pm
date modified: Friday, December 5th 2025, 5:55:38 pm
---

## tracer

`JitTracer`s are abstract stand-ins for array objects. A tracer is an abstract representation of *every possible* value with a given its essential attributes (shape, `dtype` etc), while a numpy array is a *concrete* member of that abstract class.

## jaxpr

The [`jaxpr`](https://docs.jax.dev/en/latest/jaxpr.html) of a `jax.jit` compiled function is a sequence of primitive operations defining the functional program `jax.marke_jaxpr(func)`.

## compiling

`jit` compiling a function creates `JitTracer`. The aim when designing a jax program is to minimise how often we need to recompile a function. Once a function is compiled, it will be fast for a tracer that fits, but if we need to recompile to create more-abstract tracer, the program becomes slow.

Avoid calling `jax.jit` within inner loops, because the program will spend more time compiling.

jax transformations are designed to understand the side-effect-free (functionally pure) code. i.e. do not affect any global [[./jax and state|state]].

Use `partial` to define static arguments, so the `jaxpr` recompiles every new value of the static input, but gets a less-abstract tracer. Only use static arguments if they rarely change, otherwise they would be recompiled for each static argument.

```python
from functools import partial

@partial(jax.jit, static_argnames=["n"])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))
```

During control flow operations, jax will trace through all branches and loops of the program. To avoid this static unrolling use the [control flow primitive](https://docs.jax.dev/en/latest/control-flow.html#structured-control-flow-primitives)s `jax.lax.cond`, `jax.lax.scan`, `jax.lax.while_loop`, `jax.lax.fori_loop` etc
