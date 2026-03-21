---
tags:
  - neural-network
  - ml
  - automatic-differentiation
folder: learning
share: true
title: explicit and implicit layers
date created: Saturday, August 3rd 2024, 11:48:41 am
date modified: Monday, March 16th 2026, 1:36:03 pm
---

## explicit

- Inputs $\rightarrow$ outputs
- Forward pass in a neural network classically means applying the function $f$ at each layer

$$z = f(x; \theta)$$

## [implicit](https://implicit-layers-tutorial.org/)

- Specify the *conditions that we want the layer’s output to satisfy*. For example, a layer with

$$ g(x, z; \theta) = 0$$

- The output of the layer is the **solution** to this equation for $z$.
	- relatively few or even just one implicit layer are often [enough](https://ejenner.com/post/implicit-layers/)
	- Calculate [[./automatic differentiation|gradients]] using [[./implicit function theorem|implicit function theorem]] rather than backpropagating through each step of the solve (memory intensive)

### examples

- [[./fixed point iteration|Fixed point]] layer
	- $g(x, z) = \tanh(wz + x)$
	- Solving for $z$ is the equivalent of repeatedly applying a standard dense layer with activation to the same input.
- (neural) [ODE](https://docs.kidger.site/diffrax/examples/neural_ode/) / SDE / [equilibrium models](https://implicit-layers-tutorial.org/)
