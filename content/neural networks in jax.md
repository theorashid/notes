---
tags:
  - neural-network
  - jax
  - ml
  - python
folder: learning
share: true
title: neural networks in jax
date created: Thursday, August 29th 2024, 3:02:10 pm
date modified: Sunday, December 7th 2025, 7:36:41 pm
---

Implement [[./linear model as a neural network|linear model as a neural network]] in different jax-based neural network libraries: [[neural networks in jax#equinox|equinox]], [[neural networks in jax#nnx|nnx]], ~~flax~~ (deprecated in favour of the nnx api where we can call `model(x)` – no need for a separate `apply` method), ~~penzai~~ (stale project, but had some useful contributions such as [[./treescope|treescope]] and [[./penzai named array axis|named array axis]]).

## nnx

```python
from flax import nnx

class LinearRegression(nnx.Module):
	def __init__(self, in_size: int, out_size: int, *, rngs: nnx.Rngs):
		self.linear = nnx.Linear(in_size, out_size, rngs=rngs)

	def __call__(self, x: jax.Array):
		y_pred = self.linear(x)
		return y_pred
```

```python
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

@nnx.jit  # automatic state management
def train_step(model, optimizer, x, y):
	def loss_fn(model: MLP):
		y_pred = model(x)
		return optax.l2_loss(y_pred, y).sum()
		
	loss, grads = nnx.value_and_grad(loss_fn)(model)
	optimizer.update(grads)  # inplace updates

	return loss
```

## equinox

`eqx.Module` is defined for a **single input vector**.

- mini-batching must be done via an external `vmap` call
- can apply `model` directly like a function

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
