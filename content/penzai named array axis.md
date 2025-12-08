---
tags:
  - jax
  - ml
  - python
folder: learning
share: true
title: penzai named array axis
date created: Sunday, December 7th 2025, 6:47:55 pm
date modified: Sunday, December 7th 2025, 6:48:58 pm
---

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
