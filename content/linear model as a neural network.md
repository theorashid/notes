---
tags:
  - neural-network
  - ml
  - jax
  - python
folder: learning
title: linear model as a neural network
date created: Sunday, February 4th 2024, 1:32:42 pm
date modified: Thursday, August 29th 2024, 7:31:37 pm
share: true
---

*Adapted from Ravin Kumar's [GenAI Guidebook](https://ravinkumar.com/GenAiGuidebook/model_basics/SimpleLinRegFlax.html).*

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
import flax.linen as nn

class LinearRegression(nn.Module):
    def setup(self):
        self.dense = nn.Dense(features=1)

	# or use `@nn.compact` to do this inline
    # Define the forward pass
    def __call__(self, x):
        y_pred = self.dense(x)
        return y_pred

model = LinearRegression()
y_pred = model.apply(params, x)
```

Use `optax` in a training loop to tune `params` via gradient descent.

```python
import optax
from flax.training import train_state  # dataclass to keep train state

key = jax.random.PRNGKey(0)
params = model.init(key, x_obs)

@jax.jit
def flax_l2_loss(params, x, y_true):
	y_pred = model.apply(params, x)
	total_loss = optax.l2_loss(y_pred, y_true).sum()
	return total_loss

optimizer = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model, params=params, tx=optimizer)

_loss = []
for epoch in range(10000):
	# Calculate the gradient
	loss, grads = jax.value_and_grad(flax_l2_loss)(state.params, x_obs, y_obs_noisy)
	_loss.append(loss)
	# Update the model parameters
	state = state.apply_gradients(grads=grads)
```

Linear regression cannot deal with nonlinear data. We can introduce nonlinearity by adding a second layer and an activation function.

Then just scale up and train for longer.
