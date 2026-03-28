---
tags:
  - automatic-differentiation
  - jax
  - ml
  - neural-network
folder: learning
share: true
title: automatic differentiation
date created: Saturday, August 3rd 2024, 1:45:17 pm
date modified: Friday, March 27th 2026, 6:45:22 pm
---

[Automatic differentiation](https://implicit-layers-tutorial.org/implicit_functions/) (autodiff, [jax autodiff cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)) is built on two transformations: **Jacobian-vector products** (JVPs) and **vector-Jacobian products** (VJPs).

We can see this as the [last layer before the gradient of the scalar-valued loss](https://ejenner.com/post/implicit-layers/) is

$$
\partial L = \partial_{z} L \cdot \partial z(\theta),
$$

which is a vector-Jacobian product.

These also have the property that for composition of functions, we can compose their JVP/VJPs.

## jacobian-vector product

$$(x,v) \mapsto (f(x),\partial f(x) v)$$

- Right hand side is **first two terms of Taylor series** of $f(x + v)$, thus explaining what happens with a nudge from vector $v$.
- Evaluates Jacobian at each column
- `jax.jvp`

## vector-jacobian product

$$(x,w) \mapsto (f(x), w^{T}\partial f(x))$$

- Evaluates Jacobian at each row
- `jax.vjp`
