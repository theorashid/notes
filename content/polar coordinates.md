---
tags:
  - physics
  - vector-calculus
folder: learning
share: true
title: polar coordinates
date created: Wednesday, November 11th 2026, 5:12:29 pm
date modified: Friday, November 13th 2026, 3:45:13 pm
---

Curvilinear coordinate systems for axial and radial symmetry.

---

## cylindrical polar coordinates $(\rho, \phi, z)$

![[./notes/learning/assets/cyclindrical-polar.png|cyclindrical-polar.png]]

### transformation

$$x = \rho\cos\phi, \quad y = \rho\sin\phi, \quad z = z, \qquad \rho \geq 0,\; \phi \in [0, 2\pi)$$

Basis vectors (position-dependent — $\hat{\boldsymbol{\rho}}$, $\hat{\boldsymbol{\phi}}$ rotate with $\phi$):

$$\hat{\boldsymbol{\rho}} = \cos\phi\,\hat{\mathbf{i}} + \sin\phi\,\hat{\mathbf{j}}, \qquad \hat{\boldsymbol{\phi}} = -\sin\phi\,\hat{\mathbf{i}} + \cos\phi\,\hat{\mathbf{j}}, \qquad \hat{\mathbf{z}} = \hat{\mathbf{k}}$$

$$\frac{\partial\hat{\boldsymbol{\rho}}}{\partial\phi} = \hat{\boldsymbol{\phi}}, \qquad \frac{\partial\hat{\boldsymbol{\phi}}}{\partial\phi} = -\hat{\boldsymbol{\rho}}$$

### line element & scale factors

$$d\mathbf{r} = d\rho\,\hat{\boldsymbol{\rho}} + \rho\,d\phi\,\hat{\boldsymbol{\phi}} + dz\,\hat{\mathbf{z}}, \qquad h_\rho = 1,\; h_\phi = \rho,\; h_z = 1$$

### Jacobian & volume element

$$J = \frac{\partial(x, y, z)}{\partial(\rho, \phi, z)} = \begin{vmatrix} \cos\phi & -\rho\sin\phi & 0 \\ \sin\phi & \rho\cos\phi & 0 \\ 0 & 0 & 1 \end{vmatrix} = \rho(\cos^2\phi + \sin^2\phi) = \rho$$

$$dV = \rho\,d\rho\,d\phi\,dz$$

Equivalently $J = h_\rho h_\phi h_z = 1 \cdot \rho \cdot 1 = \rho$.

```python
J = jnp.linalg.det(jax.jacfwd(cylindrical_to_cartesian)((ρ, φ, z)))  # = ρ
```

### operators

From $df = \nabla f \cdot d\mathbf{r} = (\nabla f)_\rho d\rho + (\nabla f)_\phi(\rho\,d\phi) + (\nabla f)_z dz$, matching with chain rule $df = \partial_\rho f\,d\rho + \partial_\phi f\,d\phi + \partial_z f\,dz$:

$$\nabla f = \frac{\partial f}{\partial\rho}\hat{\boldsymbol{\rho}} + \frac{1}{\rho}\frac{\partial f}{\partial\phi}\hat{\boldsymbol{\phi}} + \frac{\partial f}{\partial z}\hat{\mathbf{z}}$$

General curvilinear divergence $\nabla \cdot \mathbf{A} = \frac{1}{h_\rho h_\phi h_z}\big[\partial_\rho(h_\phi h_z A_\rho) + \partial_\phi(h_\rho h_z A_\phi) + \partial_z(h_\rho h_\phi A_z)\big]$:

$$\nabla \cdot \mathbf{A} = \frac{1}{\rho}\frac{\partial}{\partial\rho}(\rho A_\rho) + \frac{1}{\rho}\frac{\partial A_\phi}{\partial\phi} + \frac{\partial A_z}{\partial z}$$

Curl from the curvilinear determinant $\frac{1}{h_\rho h_\phi h_z}\begin{vmatrix} h_\rho\hat{\boldsymbol{\rho}} & h_\phi\hat{\boldsymbol{\phi}} & h_z\hat{\mathbf{z}} \\ \partial_\rho & \partial_\phi & \partial_z \\ h_\rho A_\rho & h_\phi A_\phi & h_z A_z \end{vmatrix}$:

$$\nabla \times \mathbf{A} = \left(\frac{1}{\rho}\frac{\partial A_z}{\partial\phi} - \frac{\partial A_\phi}{\partial z}\right)\hat{\boldsymbol{\rho}} + \left(\frac{\partial A_\rho}{\partial z} - \frac{\partial A_z}{\partial\rho}\right)\hat{\boldsymbol{\phi}} + \frac{1}{\rho}\left(\frac{\partial}{\partial\rho}(\rho A_\phi) - \frac{\partial A_\rho}{\partial\phi}\right)\hat{\mathbf{z}}$$

Laplacian $\nabla^2 f = \nabla \cdot (\nabla f)$:

$$\nabla^2 f = \frac{1}{\rho}\frac{\partial}{\partial\rho}\left(\rho\frac{\partial f}{\partial\rho}\right) + \frac{1}{\rho^2}\frac{\partial^2 f}{\partial\phi^2} + \frac{\partial^2 f}{\partial z^2}$$

---

## spherical polar coordinates $(r, \theta, \phi)$

![[./notes/learning/assets/spherical-polar.png|spherical-polar.png]]

### transformation

$$x = r\sin\theta\cos\phi, \quad y = r\sin\theta\sin\phi, \quad z = r\cos\theta, \qquad r \geq 0,\; \theta \in [0, \pi],\; \phi \in [0, 2\pi)$$

Basis vectors:

$$\hat{\mathbf{r}} = \sin\theta\cos\phi\,\hat{\mathbf{i}} + \sin\theta\sin\phi\,\hat{\mathbf{j}} + \cos\theta\,\hat{\mathbf{k}}$$

$$\hat{\boldsymbol{\theta}} = \cos\theta\cos\phi\,\hat{\mathbf{i}} + \cos\theta\sin\phi\,\hat{\mathbf{j}} - \sin\theta\,\hat{\mathbf{k}}$$

$$\hat{\boldsymbol{\phi}} = -\sin\phi\,\hat{\mathbf{i}} + \cos\phi\,\hat{\mathbf{j}}$$

### line element & scale factors

$$d\mathbf{r} = dr\,\hat{\mathbf{r}} + r\,d\theta\,\hat{\boldsymbol{\theta}} + r\sin\theta\,d\phi\,\hat{\boldsymbol{\phi}}, \qquad h_r = 1,\; h_\theta = r,\; h_\phi = r\sin\theta$$

### Jacobian & volume element

$$J = \begin{vmatrix} \sin\theta\cos\phi & r\cos\theta\cos\phi & -r\sin\theta\sin\phi \\ \sin\theta\sin\phi & r\cos\theta\sin\phi & r\sin\theta\cos\phi \\ \cos\theta & -r\sin\theta & 0 \end{vmatrix} = r^2\sin\theta$$

Equivalently $J = h_r h_\theta h_\phi = 1 \cdot r \cdot r\sin\theta = r^2\sin\theta$.

$$dV = r^2\sin\theta\,dr\,d\theta\,d\phi$$

```python
J = jnp.linalg.det(jax.jacfwd(spherical_to_cartesian)((r, θ, φ)))  # = r² sin θ
```

### operators

From $df = \nabla f \cdot d\mathbf{r} = (\nabla f)_r dr + (\nabla f)_\theta(r\,d\theta) + (\nabla f)_\phi(r\sin\theta\,d\phi)$, matching with chain rule $df = \partial_r f\,dr + \partial_\theta f\,d\theta + \partial_\phi f\,d\phi$:

$$\nabla f = \frac{\partial f}{\partial r}\hat{\mathbf{r}} + \frac{1}{r}\frac{\partial f}{\partial\theta}\hat{\boldsymbol{\theta}} + \frac{1}{r\sin\theta}\frac{\partial f}{\partial\phi}\hat{\boldsymbol{\phi}}$$

General curvilinear divergence $\nabla \cdot \mathbf{A} = \frac{1}{h_r h_\theta h_\phi}\big[\partial_r(h_\theta h_\phi A_r) + \partial_\theta(h_r h_\phi A_\theta) + \partial_\phi(h_r h_\theta A_\phi)\big]$:

$$\nabla \cdot \mathbf{A} = \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 A_r) + \frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}(\sin\theta\,A_\theta) + \frac{1}{r\sin\theta}\frac{\partial A_\phi}{\partial\phi}$$

Curl from the curvilinear determinant $\frac{1}{h_r h_\theta h_\phi}\begin{vmatrix} h_r\hat{\mathbf{r}} & h_\theta\hat{\boldsymbol{\theta}} & h_\phi\hat{\boldsymbol{\phi}} \\ \partial_r & \partial_\theta & \partial_\phi \\ h_r A_r & h_\theta A_\theta & h_\phi A_\phi \end{vmatrix}$:

$$\begin{gathered}
\nabla \times \mathbf{A} = \frac{1}{r\sin\theta}\left(\frac{\partial}{\partial\theta}(\sin\theta\,A_\phi) - \frac{\partial A_\theta}{\partial\phi}\right)\hat{\mathbf{r}} \\
\quad + \frac{1}{r}\left(\frac{1}{\sin\theta}\frac{\partial A_r}{\partial\phi} - \frac{\partial}{\partial r}(r A_\phi)\right)\hat{\boldsymbol{\theta}} \\
\quad + \frac{1}{r}\left(\frac{\partial}{\partial r}(r A_\theta) - \frac{\partial A_r}{\partial\theta}\right)\hat{\boldsymbol{\phi}}
\end{gathered}$$

Laplacian $\nabla^2 f = \nabla \cdot (\nabla f)$:

$$\nabla^2 f = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial f}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial f}{\partial\theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 f}{\partial\phi^2}$$

The radial part can also be written $\frac{1}{r}\frac{\partial^2}{\partial r^2}(rf)$, useful for Laplace's equation.

```python
# gradient in curvilinear via composite function
df_dcoords = jax.grad(lambda c: f_cartesian(*cylindrical_to_cartesian(*c)))((ρ, φ, z))
grad_physical = df_dcoords / jnp.array([1., ρ, 1.])  # divide by h_i

# vector Laplacian in Cartesian (avoid curvilinear operators):
lap_A = lambda x: jnp.array([jnp.trace(jax.hessian(lambda x: A(x)[i])(x))
                              for i in range(3)])
```
