---
tags:
  - physics
  - vector-calculus
folder: learning
share: true
title: surfaces and volumes of revolution
date created: Wednesday, November 11th 2026, 5:11:20 pm
date modified: Friday, November 13th 2026, 3:19:25 pm
---

Rotate a curve $\rho = \rho(z)$ in the $(\rho, z)$ half-plane about the $z$-axis.

![[./notes/learning/assets/volume-revolution.png|volume-revolution.png]]

## volume (disk method)

In [[./polar coordinates|cylindrical coordinates]] $dV = \rho\,d\rho\,d\phi\,dz$. Integrating $\phi \in [0, 2\pi]$ and $\rho \in [0, \rho(z)]$:

$$V = \int_{z_1}^{z_2} \int_0^{2\pi} \int_0^{\rho(z)} \rho\,d\rho\,d\phi\,dz = \pi \int_{z_1}^{z_2} [\rho(z)]^2\,dz$$

Each slice at height $z$ is a disk of radius $\rho(z)$, area $\pi\rho^2$, thickness $dz$.

```python
V = jnp.pi * jnp.trapezoid(jax.vmap(rho_fn)(z_grid)**2, dx=dz)
```

## surface area

Arc length along the curve: $ds = \sqrt{(dz)^2 + (d\rho)^2} = \sqrt{1 + (d\rho/dz)^2}\,dz$.

Each segment $ds$ traces a ribbon of circumference $2\pi\rho(z)$:

$$S = \int 2\pi\rho(z)\,ds = 2\pi \int_{z_1}^{z_2} \rho(z)\,\sqrt{1 + \left(\frac{d\rho}{dz}\right)^2}\,dz$$

Parametric form $(\rho(t), z(t))$: $S = 2\pi \int \rho(t)\sqrt{\dot{\rho}^2 + \dot{z}^2}\,dt$.

```python
drho = jax.vmap(jax.grad(rho_fn))(z_grid)
S = 2*jnp.pi * jnp.trapezoid(rho_vals * jnp.sqrt(1 + drho**2), dx=dz)
```

## example: cone

Line $\rho(z) = \frac{R}{h}z$ from $z=0$ to $h$:

- $V = \pi \int_0^h (\frac{R}{h}z)^2 dz = \frac{1}{3}\pi R^2 h$
- $S = 2\pi \int_0^h \frac{R}{h}z\sqrt{1 + R^2/h^2}\,dz = \pi R\sqrt{R^2 + h^2} = \pi R\ell$
