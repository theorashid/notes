---
tags:
  - jax
  - python
  - neural-network
  - ml
folder: learning
share: true
title: treescope
date created: Saturday, August 31st 2024, 2:37:32 pm
date modified: Sunday, December 7th 2025, 6:46:34 pm
---

[Treescope](https://treescope.readthedocs.io/en/stable/) is an interactive HTML **pretty-printer** and ND-array/tensor visualiser. Originally designed for penzai [[./neural networks in jax|neural networks in jax]].

- show **shapes** and the distribution of their values `NDArray`
- colour-coding parts of neural network models to emphasise shared **structures**

render:

- **numpy/jax/pytorch** `NDArray`/`tensor` or penzai `NamedArray`
- neural network models
	- [[./neural networks in jax#penzai|penzai]] / [[./neural networks in jax#equinox|equinox]] pytree dataclasses
	- [[./neural networks in jax#nnx|nnx]] `nnx.display(model)`
	- pytorch dynamic Python objects
- dicts, lists, tuples, and sets
- dataclasses and namedtuples
- functions
- builtins and literals
- arbitrary pytree types

visualise some arrays

```python
with treescope.active_autovisualizer.set_scoped(
	treescope.ArrayAutovisualizer()
):
	treescope.display(arrays) # inspect nested object containing NDArrays
	treescope.render_array(arr) # NDArray visualizer
```

set as default **ipython** pretty-printer

```python
treescope.basic_interactive_setup()
```

render to **html**

```python
with treescope.active_autovisualizer.set_scoped(
	treescope.ArrayAutovisualizer()
):
	contents = treescope.render_to_html(some_arrays)

with open("/tmp/treescope_output.html", "w") as f:
	f.write(contents)
```
