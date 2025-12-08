---
tags:
  - optimisation
  - automatic-differentiation
  - jax
folder: learning
share: true
title: implicit function theorem
date created: Saturday, August 3rd 2024, 11:46:19 am
date modified: Tuesday, November 25th 2025, 5:24:42 pm
---

$f(\theta, z)=0$ defines a system of nonlinear [[./explicit and implicit layers|equations]] on $z$, parameterised by $\theta$.

Take a *solution mapping* function $z^*$ (e.g. solving fixed point equation with root $z^{∗}(\theta)$), which satisfies

$$f(\theta, z^{∗}(\theta))=0.$$

Differentiate both sides with respect to $\theta$ and evaluate at the point $(\theta_0,z_0)$.

$$\partial_{0} f(\theta_{0}, z_{0}) + \partial_{1} f(\theta_{0}, z_{0}) \partial z^{∗}(\theta_{0}) = 0$$

And then rearrange for the **Jacobian**

$$ \partial z^{∗}(\theta_{0}) = - [\partial_{1} f(\theta_{0}, z_{0})]^{-1} \partial_{0} f(\theta_{0}, z_{0}). $$

- The Jacobian of the solution mapping can be expressed **just in terms of Jacobians** of $f$ **at the solution point** (regardless of how we get to the solution).
- We can skip propagating derivatives through the solver using [[./automatic differentiation|automatic differentiation]] systems:
	- In PyTorch, use `torch.no_grad()` when running the solver and add the backward pass with `z.register_hook(lambda grad : torch.solve(grad[:,:,None], J.transpose(1,2))[0][:,:,0])`.
	- In jax, use [`jax.custom_jvp` and `jax.custom_vjp`](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).
