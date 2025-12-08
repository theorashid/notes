---
tags:
  - neural-network
  - ml
  - jax
  - python
folder: learning
title: linear model as a neural network
date created: Sunday, February 4th 2024, 1:32:42 pm
date modified: Sunday, December 7th 2025, 6:52:55 pm
share: true
---

*Adapted from Ravin Kumar's [GenAI Guidebook](https://ravinkumar.com/GenAiGuidebook/model_basics/SimpleLinRegFlax.html), but updated to use [[./neural networks in jax|equinox]] instead of flax.*

## linear regression in numpy

Linear regression (1-D case)

$$
y = \mathbf{\beta} \cdot \mathbf{x} + c
$$

where $\mathbf{x}$ and $\mathbf{\beta}$ have shape `(N_features,)`.

Vectorising to operate on the entire $\mathbf{y}$ at once, in (jax-)numpy we write:

- `y = (x @ coefficients.T) + bias` 
- `y = jnp.einsum('ij,kj->ki', coefficients, x) + bias` using using [[./einsum|einsum]] notation

## linear regression in flax

In neural network notation, we call the regression coefficients the "weights", $\mathbf{w}$, and the intercept the bias, $b$, giving us

$$
F(x) = \mathbf{w}^T \mathbf{x} + b
$$

This is a **dense neural network with one layer** and no activation function.

```python
import equinox as eqx

# equally eqx.nn.linear
class LinearLayer(eqx.Module):
	weight: jax.Array
	bias: jax.Array
	def __init__(self, in_size, out_size, key):
		wkey, bkey = jax.random.split(key)
		self.weight = jax.random.normal(wkey, (out_size, in_size))
		self.bias = jax.random.normal(bkey, (out_size,))
		
	def __call__(self, x):
		return self.weight @ x + self.bias

class LinearRegression(eqx.Module):
	def __init__(self, in_size: int, out_size: int, key: jax.random.PRNGKey):
		self.linear = LinearLayer(in_size, out_size, key=key)

	def __call__(self, x: jax.Array):
		y_pred = self.linear(x)
		return y_pred
```

Use `optax` in a training loop to tune `params` via gradient descent.

```python
optimizer = optax.adam(learning_rate=0.001)

# it only makes sense to train the arrays in our model,
# so filter out everything else using `eqx.filter`
params = optim.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(key, params, optimizer, x, y):
	def loss_fn(params):
		y_pred = jax.vmap(model)(x) # vectorise the model over a batch of data
		return optax.l2_loss(y_pred, y).sum()

	loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
	updates, params = optimizer.update(grads, params)
	model = eqx.apply_updates(model, updates)
  return model
```

Linear regression cannot deal with nonlinear data. We can introduce nonlinearity by adding a second layer and an activation function.

Then just scale up and train for longer.
