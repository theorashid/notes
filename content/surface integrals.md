---
tags:
  - physics
  - vector-calculus
folder: learning
share: true
title: surface integrals
date created: Wednesday, November 11th 2026, 5:10:44 pm
date modified: Friday, November 13th 2026, 3:33:04 pm
---

## parametric surface

Surface $S$ parameterised by $(u, v)$:

$$\mathbf{r}(u, v) = x(u, v)\,\hat{\mathbf{i}} + y(u, v)\,\hat{\mathbf{j}} + z(u, v)\,\hat{\mathbf{k}}$$

Surface element:

$$dS = \left| \frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v} \right| du\,dv, \qquad \hat{\mathbf{n}} = \frac{\frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v}}{\left| \frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v} \right|}$$

Scalar surface integral:

$$\iint_S f\,dS = \iint_D f(\mathbf{r}(u,v)) \left| \frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v} \right| du\,dv$$

Flux (oriented — flipping $u \leftrightarrow v$ reverses $\hat{\mathbf{n}}$):

$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_D \mathbf{F} \cdot \left( \frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v} \right) du\,dv$$

```python
# dS = |∂r/∂u × ∂r/∂v| du dv — tangents via jacfwd
dr_du, dr_dv = jax.jacfwd(r, argnums=(0,1))(U, V)
dS = jnp.linalg.norm(jnp.cross(dr_du, dr_dv), axis=-1)
```

## surface as $z = z(x, y)$

Parameterise with $(x, y)$: $\mathbf{r} = x\hat{\mathbf{i}} + y\hat{\mathbf{j}} + z(x,y)\hat{\mathbf{k}}$. Tangent cross product:

$$\frac{\partial \mathbf{r}}{\partial x} \times \frac{\partial \mathbf{r}}{\partial y} = -\frac{\partial z}{\partial x}\hat{\mathbf{i}} - \frac{\partial z}{\partial y}\hat{\mathbf{j}} + \hat{\mathbf{k}}$$

$$dS = \sqrt{1 + \left(\tfrac{\partial z}{\partial x}\right)^2 + \left(\tfrac{\partial z}{\partial y}\right)^2}\;dx\,dy$$

The square root is the area scaling factor — how much the slanted surface stretches relative to the $xy$-plane.

```python
dS = jnp.sqrt(1 + jax.grad(z_fn)(X)**2 + jax.grad(z_fn, argnums=1)(Y)**2)
```

## level surface

$f(x, y, z) = \text{const}$

Normal from gradient: $\hat{\mathbf{n}} = \nabla f / |\nabla f|$. Projected onto $xy$-plane:

$$dS = \frac{|\nabla f|}{|\nabla f \cdot \hat{\mathbf{k}}|}\,dx\,dy = \frac{|\nabla f|}{|\partial f/\partial z|}\,dx\,dy$$
