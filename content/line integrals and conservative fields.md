---
tags:
  - physics
  - vector-calculus
folder: learning
share: true
title: line integrals and conservative fields
date created: Wednesday, November 11th 2026, 5:11:03 pm
date modified: Friday, November 13th 2026, 3:29:37 pm
---

## line integral

For a vector field $\mathbf{F}$ along a curve $C$ parameterised by $\mathbf{r}(t)$, $t \in [a, b]$:

$$W = \int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \frac{d\mathbf{r}}{dt}\,dt$$

In components: $\int_C (F_x\,dx + F_y\,dy + F_z\,dz)$. Reversing orientation of $C$ flips the sign.

![[./notes/learning/assets/line-intergral.png|line-intergral.png]]

```python
# ∫ F(r(t)) · (dr/dt) dt
dr_dt = jax.jacfwd(r)(t_grid)                        # tangent
integrand = jnp.sum(F(*r(t_grid)) * dr_dt, axis=-1)  # F·dr/dt
```

## conservative fields

$\mathbf{F}$ is **conservative** if the line integral depends only on endpoints, not the path:

$$\oint_C \mathbf{F} \cdot d\mathbf{r} = 0 \quad \text{for every closed path } C$$

There exists a scalar **potential** $\phi$ such that $\mathbf{F} = \nabla\phi$, and:

$$\int_A^B \mathbf{F} \cdot d\mathbf{r} = \phi(B) - \phi(A)$$

## exact differentials

$\mathbf{F} \cdot d\mathbf{r} = F_x\,dx + F_y\,dy + F_z\,dz$ is an **exact differential** $d\phi$ iff the mixed partials match:

$$\frac{\partial F_x}{\partial y} = \frac{\partial F_y}{\partial x},\quad \frac{\partial F_y}{\partial z} = \frac{\partial F_z}{\partial y},\quad \frac{\partial F_z}{\partial x} = \frac{\partial F_x}{\partial z}$$

These are exactly the conditions $\nabla \times \mathbf{F} = \mathbf{0}$.

## curl condition

$\mathbf{F}$ conservative $\Longleftrightarrow \nabla \times \mathbf{F} = \mathbf{0}$ (in a simply-connected domain). Follows from [[./divergence and stokes theorems|Stokes' theorem]]: $\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \hat{\mathbf{n}}\,dS$.

In Cartesian:

$$\nabla \times \mathbf{F} = \begin{vmatrix} \hat{\mathbf{i}} & \hat{\mathbf{j}} & \hat{\mathbf{k}} \\ \partial_x & \partial_y & \partial_z \\ F_x & F_y & F_z \end{vmatrix}$$

```python
# check conservative: curl from Jacobian
J = jax.jacfwd(F)(*x)
curl = jnp.array([J[2,1]-J[1,2], J[0,2]-J[2,0], J[1,0]-J[0,1]])
```

## finding the potential

Integrate component-by-component along a convenient path (e.g. coordinate axes):

$$\phi(x, y, z) = \int_{x_0}^x F_x(t, y_0, z_0)\,dt + \int_{y_0}^y F_y(x, t, z_0)\,dt + \int_{z_0}^z F_z(x, y, t)\,dt$$
