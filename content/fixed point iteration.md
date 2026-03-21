---
tags:
  - optimisation
  - automatic-differentiation
  - jax
  - python
folder: learning
share: true
title: fixed point iteration
date created: Tuesday, July 30th 2024, 11:43:45 am
date modified: Monday, March 16th 2026, 1:36:00 pm
---

## fixed point in 1-D

Rewrite $f(x) = 0$ into the form $x= g(x)$ and then label left hand side as $x_{n+1}$ and right hand side as $x_n$.

e.g. $x^{2}- x - 1 = 0$ becomes $x_{n+1} = 1 + \frac{1}{x_n}$.

- converges when $|g'(x)| < 1$
- generally try and reduce the degree of the polynomial
- [[./newton's method|newton's method]] is a special case of fixed point iteration where $g(x) = x - \frac{f(x)}{f'(x)}$

Oscar Veliz video [1](https://www.youtube.com/watch?v=OLqdJMjzib8) and [2](https://www.youtube.com/watch?v=FyCviw2ZA2o).

## generalised fixed point iteration

Naive [forward iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration), where we iterate until $z_{n+1}$ stays sufficiently close to $z_n$:

$$
z_{n+1} = f(\theta, z_n)
$$

```python
def fwd_solver(f, z_init):
	z_prev, z = z_init, f(z_init)
	while jnp.linalg.norm(z_prev - z) > 1e-5:
		z_prev, z = z, f(z)
	return z
```

### gradients

If we use a fixed-point iteration in a model where we need the gradients, such as an [[./explicit and implicit layers|implicit layer]], naively using [[./automatic differentiation|automatic differentiation]] would require the gradients at each intermediate step of the solver.

Instead, we can use the [[./implicit function theorem|implicit function theorem]] to define the backward pass *only at the solution point*.

For the [fixed point iterator](https://implicit-layers-tutorial.org/implicit_functions/), the Jacobian becomes (as we now have $z^{∗}(\theta) = f(\theta, z^{∗}(\theta))$ instead of $0$)

$$ \partial z^{∗}(\theta_{0}) = [I - \partial_{1} f(\theta_{0}, z_{0})]^{-1} \partial_{0} f(\theta_{0}, z_{0}). $$

#### Jacobian-vector product

$$ \partial z^{∗}(\theta_{0}) v = [I - \partial_{1} f(\theta_{0}, z_{0})]^{-1} \partial_{0} f(\theta_{0}, z_{0}) v$$

We want to [avoid the matrix inversion](https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/) by using the [adjoint method](https://www.jgaeb.com/2021/09/13/implicit-autodiff.html).

1. Calculate $u = \partial_{0} f(\theta_{0}, z_{0}) v$ using `jax.jvp`.
2. The final solution is now $w = [I - \partial_{1} f(\theta_{0}, z_{0})]^{-1} u$. So $w = u + \partial_{1} f(\theta_{0}, z_{0}) w$ and we can use a fixed point iterator (or other solver) to solve the linear system for $w$.

#### vector-Jacobian product

$$ w^{T}\partial z^{∗}(\theta_{0}) = w^T[I - \partial_{1} f(\theta_{0}, z_{0})]^{-1} \partial_{0} f(\theta_{0}, z_{0})$$

1. Calculate the [adjoint](https://docs.kidger.site/optimistix/api/adjoints/) $u^{T}= w^T[I - \partial_{1} f(\theta_{0}, z_{0})]^{-1}$ as $u^{T}= w^{T} + u^{T} \partial_{1} f(\theta_{0}, z_{0})$.
2. Compute $u^{T} \partial_{0} f(\theta_{0}, z_{0})$ using `jax.vjp`.

The [jax code for the VJP](https://implicit-layers-tutorial.org/implicit_functions/) (also [jax docs](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations)):

```python
from functools import partial

@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point_layer(solver, f, params, x):
	z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
	return z_star

def fixed_point_layer_fwd(solver, f, params, x):
	z_star = fixed_point_layer(solver, f, params, x)
	return z_star, (params, x, z_star)

def fixed_point_layer_bwd(solver, f, res, z_star_bar):
	params, x, z_star = res
	_, vjp_a = jax.vjp(lambda params, x: f(params, x, z_star), params, x)
	_, vjp_z = jax.vjp(lambda z: f(params, x, z), z_star)
	return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar, z_init=jnp.zeros_like(z_star)))

fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)
```
