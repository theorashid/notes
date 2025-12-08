---
tags:
  - jax
  - python
folder: learning
share: true
title: jax and pytree
date created: Friday, December 5th 2025, 6:27:37 pm
date modified: Sunday, December 7th 2025, 7:45:34 pm
---

```python
import jax
import jax.numpy as jnp

pytree = [
  [1, 'x', object()],
  {'a': 1},
  {
    'b': 2,
    'c': (3, 4),
    'd': None,
  },
  jnp.array([1, 2, 3]),
]

leaves, treedef = jax.tree.flatten(pytree)
print(f"leaves = {leaves}")
print(f"treedef = {treedef}")
```

```sh
leaves = [1, 'x', <object object at 0x1196ab980>, 1, 2, 3, 4, Array([1, 2, 3], dtype=int32)]
treedef = PyTreeDef([[*, *, *], {'a': *}, {'b': *, 'c': (*, *), 'd': None}, *])
```

- `None` is not a leaf because its defined as a pytree with no children.
- Any object whose type is *not* in the pytree container registry will be treated as a leaf node in the tree.
	- This can be changed by registering [custom pytree node](https://docs.jax.dev/en/latest/custom_pytrees.html).
	- Dataclasses are not automatically pytrees but they can be using `jax.tree_util.register_dataclass`
- Arrays, which are leaves, in jax are as [[./jax and state|immutable]].
- `NamedTuple` is often used to store params.

`jax.tree.map(f: Callable[..., Any], tree: Any)` maps a function over a tree's leaves.

## equinox

The [equinox](https://docs.kidger.site/equinox/all-of-equinox/) [module](https://docs.kidger.site/equinox/api/module/module/) makes the class **both a dataclass and a pytree**. [tinygp](https://github.com/dfm/tinygp/commit/1e798b7c6129aae411b3ea1a6515feb052751ec6) uses `equinox.Module` to simplify dataclass so we don't need to use `jax.tree_util.register_pytree_node`.

- class-like behaviour can be obtained by [avoiding extra abstractions](https://sscardapane.notion.site/JAX-and-Equinox-An-Introduction-6898d847d9bc4a15817637e07185236b) and simply exploiting pytrees

Equinox adds extra tree methods:

- `eqx.tree_at mutates` a particular leaf or leaves of a pytree
- `equinox.tree_serialise_leaves` to save leaves of a pytree to a file

Equinox supports arbitrary python objects for its leaves (e.g. activation function). Need to use *filtering* (`@eqx.filter_jit`, `@eqx.filter_grad` etc) to operate on:

- `params`, the leaves that are arrays; and not
- `static`, everything else that cannot be operated on

`eqx.is_array` is a boolean *filter function* specifying whether each leaf should go into `params` or into `static` by returning `True` for jax and numpy arrays, and `False` for everything else.
