---
tags:
  - physics
  - vector-calculus
folder: learning
share: true
title: divergence and stokes theorems
date created: Wednesday, November 11th 2026, 5:11:49 pm
date modified: Friday, November 13th 2026, 3:28:05 pm
---

Generalise the Fundamental Theorem of Calculus: the integral of a derivative over a region equals the value of the function on the boundary.

---

## divergence theorem (Gauss)

$$\iint_{\partial V} \mathbf{F} \cdot d\mathbf{S} = \iiint_V \nabla \cdot \mathbf{F}\,dV$$

Think of $\mathbf{F}$ as fluid velocity. $\nabla \cdot \mathbf{F}$ is the net outflow per unit volume at a point (sources $\nabla \cdot \mathbf{F} > 0$ create fluid, sinks ($\nabla \cdot \mathbf{F} < 0$) destroy it).

The total flux crossing the closed boundary equals the sum of all sources minus sinks inside. If water flows out of a lake (the surface integral) and there's a non-zero net flow, there must be a spring or leak inside.

## Stokes' theorem

$$\oint_{\partial S} \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \hat{\mathbf{n}}\,dS$$

The circulation of $\mathbf{F}$ around a closed loop equals the flux of curl through any surface bounded by that loop.

$\nabla \times \mathbf{F}$ is the local rotation (vorticity) — direction along the axis of spin, magnitude its strength. Sum the spin of paddlewheels (spinning at a rate proportional to the component of curl normal to it) across the surface = the net push felt along the rim. **Any two surfaces sharing the same boundary give the same flux of curl.**

![[./notes/learning/assets/stokes.png|stokes.png]]

## Green's theorem (Stokes in 2D)

Specialise Stokes to a flat surface in the xy-plane: $\hat{\mathbf{n}} = \hat{\mathbf{k}}, dS = dx\,dy, \mathbf{F} = (P(x,y), Q(x,y), 0)$.

$$(\nabla \times \mathbf{F}) \cdot \hat{\mathbf{k}} = \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}$$

$$\oint_C (P\,dx + Q\,dy) = \iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dx\,dy$$

The 2D curl $\partial Q/\partial x - \partial P/\partial y$ is a scalar — circulation per unit area in the plane. Green's is Stokes with the 3D surface squashed flat. The curl vector collapses to a single scalar component normal to the plane.

![[./notes/learning/assets/greens.png|greens.png]]

## Jacobian decomposition

The Jacobian $J_{ij} = \partial F_i / \partial x_j$ contains all vector calculus:

$$J = \underbrace{\tfrac{1}{3}(\nabla \cdot \mathbf{F})\mathbf{I}}_{\text{trace (div)}} \;+\; \underbrace{\tfrac{1}{2}(J - J^T)}_{\text{antisym (curl)}} \;+\; \underbrace{\tfrac{1}{2}(J + J^T) - \tfrac{1}{3}(\nabla \cdot \mathbf{F})\mathbf{I}}_{\text{sym. traceless (shear)}}$$

```python
J = jax.jacfwd(F)(x)                                 # 3×3 Jacobian
div = jnp.trace(J)                                   # ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z
curl = jnp.array([J[2,1]-J[1,2], J[0,2]-J[2,0], J[1,0]-J[0,1]])
curl_2d = J[1,0] - J[0,1]                            # ∂Q/∂x - ∂P/∂y
```
