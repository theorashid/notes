---
tags:
  - optimisation
  - python
  - jax
folder: learning
share: true
title: newton's method
date created: Monday, July 29th 2024, 4:43:38 pm
date modified: Saturday, August 3rd 2024, 3:07:18 pm
---

Newton's method for **finding roots**. Oscar Veliz [video in 1D case](https://www.youtube.com/watch?v=E24zUEKqgwQ), [generalised](https://www.youtube.com/watch?v=p0SBubUfwiI) and [global](https://www.youtube.com/watch?v=BGZfHxzZ-7c).

## newton's method in 1-D

$$
x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}
$$

- starting point needs to be near root
- not guaranteed to converge
- multiply second term by factor $\alpha$ to get damped Newton's method

```python
from typing import Callable, Tuple

def newton(
	func: Callable[float],
	df: Callable[float],
	x0: float, # initial estimate
	max_iterations: int = 100,
	tolerance: float = 1e-12
) -> Tuple[float, int]:
    curr_x: float = x0
    iteration: int = 0

    while iteration < max_iterations:
        delta_x: float = func(curr_x) / df(curr_x)

		# done enough iterations
        if abs(delta_x) < tolerance:
            break

		# x_ = x - f(x)/f'(x)
        curr_x = curr_x - delta_x
        iteration += 1

    return curr_x, iteration # value, number of iterations
```

## generalised newton's method

$$
z_{n+1} = z_{n} - \partial{f}(z_{n})^{-1} f(z_{n})
$$

where $f$ is now a vector-valued function and $\partial{f}$ is the Jacobian (matrix of all first order partial derivatives, see [here](https://implicit-layers-tutorial.org/implicit_functions/) for notation). Or avoid the inverse and solve the system of equations

$$
\partial{f}(z_{n})(z_{n+1} - z_{n})=  - f(z_{n})
$$

```python
def fwd_solver(f, z_init):
	z_prev, z = z_init, f(z_init)
	while jnp.linalg.norm(z_prev - z) > 1e-5:
		z_prev, z = z, f(z)
	return z

def newton_solver(f, z_init):
	f_root = lambda z: f(z) - z
	g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
	return fwd_solver(g, z_init)
```

Colouring domain by which roots a point converges to gives a [Newton fractal](https://www.youtube.com/watch?v=MWD2A0Vg2V0).
