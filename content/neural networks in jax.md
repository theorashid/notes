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
date modified: Sunday, September 1st 2024, 12:54:34 pm
---

Implement [[./linear model as a neural network|linear model as a neural network]] in different jax-based neural network libraries: [[neural networks in jax#flax|flax]], [[neural networks in jax#equinox|equinox]], [[neural networks in jax#nnx|nnx]], [[neural networks in jax#penzai|penzai]].

Visualise using [[./treescope|treescope]].

## flax

- [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) of parameters for the model 
	- parameters are not stored with the models themselves
	- cannot directly call `model(x)`
		- `__call__` function wrapped up in the `apply` function
		- need to call `model.init(key, x)`

```python
import flax.linen as nn

class LinearRegression(nn.Module):
	features: int
	
	@nn.compact
	def __call__(self, x):
		y_pred = nn.Dense(self.features)(x)
		return y_pred
```

```python
optimizer = optax.adam(learning_rate=0.001)

# better to use flax.training.train_state
@jax.jit
def train_step(key, params, optimizer, x, y):
	def loss_fn(params):
		y_pred = model.apply(params, x)
		return optax.l2_loss(y_pred, y).sum()

	loss, grads = jax.value_and_grad(loss_fn)(params)
	updates, params = optimizer.update(grads, params)
	params = optax.apply_updates(params, updates)
  return params
```

## nnx

- express **models using regular python objects**, which are modelled as *pygraphs* (instead of pytrees)
- module itself [**holds the state** directly](https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html)
	- dynamic state is usually stored in `nnx.Param`
	- access using `graphdef, params, rngs = nnx.split(model, nnx.Param, nnx.RngState)` to get the `params`
	- parameters were already initialised during model instantiation so [no need to call](https://flax.readthedocs.io/en/latest/nnx/haiku_linen_vs_nnx.html#) `model.init()`
	- can call `model(x)`, no need for a separate `apply` method
- using `nnx.split` and `nnx.merge` to operate on module (from [gpjax](https://docs.jaxgaussianprocesses.com/_examples/backend/#transforming-multiple-parameters))
	- use this to [separate different types of parameters to do different optimisations](https://docs.jaxgaussianprocesses.com/_examples/backend/#nnx-modules)

```python
graphdef, state = nnx.split(posterior) # state behaves like a pytree
updated_state = jax.tree_utils.tree_map(lambda x: x + 1, state) # increment each parameter's value by 1
updated_posterior = nnx.merge(graphdef, updated_state) # reconstruct the posterior distribution using the updated state
```

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

- [Equinox](https://docs.kidger.site/equinox/all-of-equinox/) **represent models as pytrees** using `eqx.Module`
	- then use any jax transformation (such as `jit`, `grad` [[./jax.vmap|jax.vmap]]) or other jax code
	- class-like behaviour can be obtained by [avoiding extra abstractions](https://sscardapane.notion.site/JAX-and-Equinox-An-Introduction-6898d847d9bc4a15817637e07185236b) and simply exploiting pytrees
- supports arbitrary python objects for its leaves (e.g. activation function)
	- need to use *filtering* (`@eqx.filter_jit`, `@eqx.filter_grad` etc) to operate on:
		- `params`, the leaves that are arrays; and not
		- `static`, everything else that cannot be operated on
- `eqx.Module` is defined for a **single input vector**
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

- `eqx.tree_at mutates` a particular leaf or leaves of a pytree
- `equinox.tree_serialise_leaves` to save leaves of a pytree to a file

## penzai

*API is in flux but some useful concepts such as named arrays and [[./treescope|treescope]] for visualising arrays of a pytree. Currently best used for inspecting trained neural networks rather than building and training.*

- [Penzai models](https://penzai.readthedocs.io/en/stable/notebooks/how_to_think_in_penzai.html) are registered as [**jax pytree** nodes](https://jax.readthedocs.io/en/latest/pytrees.html) (similar to Equinox)
	- so any Penzai model can be traversed using `jax.tree_util` or other jax transformation
	- forward pass of a model, call it like a function
- Penzai model objects are immutable
	- modifications to Penzai models always occur by making a *modified copy* of the model
- Each Penzai model tree has two types of leaves:
	- jax arrays and scalars, which are immutable and often represent hyperparameters, and
	- variable objects, which can be modified:
	    - `Parameter`s, usually modified by optimisers (not by the model),
	    - `StateVariable`s, usually updated as the model runs.

```python
# each layer has the same signature
# exactly one positional argument – the input from the previous layer
class Layer(pz.Struct, abc.ABC):
	@abc.abstractmethod
	def __call__(self, argument: Any, /, **side_inputs) -> Any:
		...

# for example
@struct.pytree_dataclass
class Attention(Layer):
	input_to_query: Layer
	input_to_key: Layer
	input_to_value: Layer
	query_key_to_attn: Layer
	attn_value_to_output: Layer

	def __call__(self, x: NamedArray, **side_inputs) -> NamedArray:
		query = self.input_to_query(x, **side_inputs)
		key = self.input_to_key(x, **side_inputs)
		value = self.input_to_value(x, **side_inputs)
		attn = self.query_key_to_attn((query, key), **side_inputs)
		output = self.attn_value_to_output((attn, value), **side_inputs)
		return output
```

```python
@pz.pytree_dataclass
class LinearRegression(pz.nn.Sequential):
	sublayers: list[pz.nn.Layer]

	@classmethod
	def from_config(
		cls,
		name: str,
		init_base_rng: jax.Array | None,
		feature_sizes: Sequence[int],
		features_axis: str = "features",
	) -> "LinearRegression":
	sublayers = []
	for i in range(len(feature_sizes) - 1):
		sublayers.append(Linear.from_config(
			name=f"{name}/block_{i}/linear",
			init_base_rng=init_base_rng,
			in_features=feature_sizes[i],
			out_features=feature_sizes[i + 1],
			features_axis=features_axis,
		))
		sublayers.append(Bias.from_config(
			name=f"{name}/block_{i}/bias",
			init_base_rng=init_base_rng,
			features=feature_sizes[i + 1],
			features_axis=features_axis,
		))
	return cls(sublayers)
```

[example training loop](https://penzai.readthedocs.io/en/stable/notebooks/how_to_think_in_penzai.html#putting-it-all-together-a-basic-penzai-neural-network)

### named axes

- `pz.nx.NamedArray` class [wraps an ordinary array](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html), and assigns each axis to either a **position** *or* a **name** (but not both)
- two shapes:
	- `.positional_shape` attribute is the sequence of dimension sizes for the positional axes in the array
	- `.named_shape` attribute is a dictionary mapping axis names to their dimension sizes

```python
pz.nx.wrap(jnp.ones([3, 4, 5])).tag("foo", "bar", "baz")
my_named_array.untag("bar", "foo", "baz").unwrap()
```

- operations on NamedArrays **always act only on the positional axes**
	- **untag**, move those axes to the `.positional_shape`
	- **operate**, calling `pz.nx.nmap(some_positional_op)(...args...)`
	- **retag**, `.tag` to move the resulting axes from the `.positional_shape` back to the `.named_shape`

```python
def named_dot(x: pz.nx.NamedArray, features_axis: str) -> pz.nx.NamedArray:
	pos_x = x.untag(features_axis)
	pos_kernel = kernel.value.untag("out_features", "in_features")
	pos_y = pz.nx.nmap(jnp.dot)(pos_kernel, pos_x)
	return pos_y.tag(features_axis)
```
