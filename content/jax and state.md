---
tags:
  - jax
  - ml
  - python
folder: learning
share: true
title: jax and state
date created: Friday, December 5th 2025, 5:48:28 pm
date modified: Sunday, December 7th 2025, 7:48:41 pm
---

jax `Array`s are immutable. Immutability is useful for optimised [[./jax.jit|compilation]] and [[./jax.vmap|parallelisation]].

You [cannot update a global state](https://docs.jax.dev/en/latest/stateful-computations.html) within a function. Model parameters are a state.

Instead, we pass the state into the function as an argument, and emit a state.

```diff
- def stateful_method(...) -> Output # updates State object
+ def statless_method(state: State) -> (Output, State)
```

For example, with model parameters:

```python
class Params(NamedTuple):
  weight: jnp.ndarray
  bias: jnp.ndarray

@jax.jit
def update(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> Params:
  grad = jax.grad(loss)(params, x, y)
  new_params = jax.tree.map(
      lambda param, g: param - g * LEARNING_RATE, params, grad)

  return new_params
```

This is also why we must [[./splitting keys in jax|pass the random state]] into a function.

## examples

### blackjax

[blackjax](https://github.com/blackjax-devs/blackjax) holds the `State` as a `NamedTuple` of parameters.

blackjax `SamplingAlogrithm` are `NamedTuple` [holding two functions](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/base.py), both of which return `NamedTuple`:

- `InitFn`, which is `__call__` with the signature `(position: Position, rng_key: Optional[PRNGKey]) -> State`
- `UpdateFn` which is `__call__` with the signature `(rng_key: PRNGKey, state: State) -> tuple[State, Info]`, returning a new state and some information (e.g. whether a step is accepted) about the transition

With an [algorithm](https://github.com/blackjax-devs/blackjax/blob/main/blackjax/mcmc/hmc.py), these `init` and `step_fn` methods are defined, and these functions are wrapped to match the signature of each `Protocol` above.

```python
def as_top_level_api(
    logdensity_fn: Callable,
    *,
    ...
) -> SamplingAlgorithm:
    kernel = build_kernel(...)

	# for algo.init
    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

	# for algo.step
    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            ...
        )

    return SamplingAlgorithm(init_fn, step_fn)
```

### dynamax

Dynamax also uses `NamedTuple` to hold the parameters of the state space model.

The model [`initialize`](https://github.com/probml/dynamax/blob/main/dynamax/linear_gaussian_ssm/models.py#L92) method outputs: 

- `Params`, a `NamedTuple` of the distribution parameters
- `ParameterProperties`, the corresponding [parameter metadata](https://github.com/probml/dynamax/blob/main/dynamax/parameters.py#L29) for each of the distribution parameters, which is a class registered as a [[./jax and pytree|pytree]] node. This contains a flag for whether the parameter is trainable and the bijection mapping for constraining.

Then, when fitting either with [gradient descent](https://github.com/probml/dynamax/blob/main/dynamax/ssm.py#L458) or with [sampling](https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/lgssm_hmc.html#infer-the-model-parameters-using-hmc), the parameters are first unconstrained (`to_unconstrained`) to the real number line.

### nnx

But neural nets have many layers, and a state becomes difficult to pipe around. [[./neural networks in jax|nnx]] module extends jax to [hold a state](<[**holds the state** directly](https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html)>).

- `State` (usually stored in `nnx.Param`) is a mapping from strings to `Variable`s or nested `State`s.
- `GraphDef` contains all the static information needed to reconstruct a `Module` graph, it is analogous to [jax’s `PyTreeDef`](https://jax.readthedocs.io/en/latest/pytrees.html#internal-pytree-handling).

A nnx `Module` can be decomposed into `State`  and `GraphDef` using the `nnx.split` function, which can be used to [separate different types of parameters to do different optimisations](https://docs.jaxgaussianprocesses.com/_examples/backend/#nnx-modules).

```python
graphdef, state = nnx.split(posterior) # state behaves like a pytree
updated_state = jax.tree_utils.tree_map(lambda x: x + 1, state) # increment each parameter's value by 1
updated_posterior = nnx.merge(graphdef, updated_state) # reconstruct the posterior distribution using the updated state
```

[GPJax](https://docs.jaxgaussianprocesses.com/_examples/backend/#parameters) uses `Parameter(nnx.Variable)` which are tagged with `ParameterTag` (`str`) which is linked to the corresponding bijection to transform to an unconstrained space. The pattern within the `transform` function is

```python
def transform(
    params: nnx.State,
    params_bijection: tp.Dict[str, npt.Transform],
    inverse: bool = False,
) -> nnx.State:
    def _inner(param):
	    # Each Parameter has a tag
	    # Each tag has an associated numpyro bijector
	    # DEFAULT_BIJECTION[param.tag]
        bijector = params_bijection.get(param.tag, npt.IdentityTransform())
        if inverse:
            transformed_value = bijector.inv(param.value)
        else:
            transformed_value = bijector(param.value)

		# update the State
        param = param.replace(transformed_value)
        return param

	# split the State to get the Parameter variables
    gp_params, *other_params = params.split(Parameter, ...)

    # transform each parameter in the state using the associated bijector
    transformed_gp_params: nnx.State = jtu.tree_map(
        lambda x: _inner(x) if isinstance(x, Parameter) else x,
        gp_params,
        is_leaf=lambda x: isinstance(x, Parameter),
    )
    
    # update the state with the transformed variables
    return nnx.State.merge(transformed_gp_params, *other_params)

# single parameter
# realise the _state_ of our model
_, _params = nnx.split(meanf, Parameter)
tranformed_params = transform(_params, DEFAULT_BIJECTION, inverse=True)

# multiple parameters
# `State` contains information on the parameters' state
# `GraphDef` contains the information required to reconstruct a PyGraph from a give `State`.
graphdef, state = nnx.split(posterior)

# unconstrain
transformed_state = transform(state, DEFAULT_BIJECTION, inverse=True)

# constrain
retransformed_state = transform(transformed_state, DEFAULT_BIJECTION, inverse=False)

# only extract PositiveReal part of State
graphdef, positive_reals, other_params = nnx.split(posterior, PositiveReal, ...)
```

### equinox

For [stateful layers](https://docs.kidger.site/equinox/examples/stateful/), equinox requires threading the additional `state` object in and out of every call. An updated state object is returned.

```python
def __call__(self, x, state):
	x, state = self.norm1(x, state)
	x, state = self.spectral_linear(x, state)
	x = jax.nn.relu(x)
	x, state = self.norm2(x, state)
	return x, state
```

`eqx.nn.make_with_state(Model)(key)` returns:

- `model`, a [[./jax and pytree|pytree]] holding all the initial parameters (just like any other model), and
- `state`, a pytree holding the initial state.
