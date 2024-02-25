---
tags:
  - ml
  - python
  - swe
  - jax
folder: learning
title: einshape
date created: Sunday, February 4th 2024, 1:50:24 pm
date modified: Sunday, February 25th 2024, 12:16:52 pm
share: "true"
---

The [einshape](https://github.com/google-deepmind/einshape) library uses similar notation to [[./einsum|einsum]], but focused on unifying `reshape`, `squeeze`, `expand_dims`, and `transpose` operations (although `einsum` can be used for transposing).

| array | einshape |
| ---- | ---- |
| `expand_dims(x, axis=1)` three times | `einshape('n->n111', x)` |
| `squeeze(x, axis=[1,3,4])` | `einshape('a1b11->ab', x)` |
| `transpose(x, perm=[0,3,1,2])` | `einshape('nhwc->nchw', x)` |
| reshape combining leading two dimensions (without knowing rank) | `einshape('mn...->(mn)...', x)` |
| reshape splitting the leading dimension into two | `einshape('(mn)hwc->mnhw', x, n=batch_size)` |
| batch flatten | `einshape('n...->n(...)', x)` |
| inserts a trailing dimension and tiles along it | `einshape("ij->ijk", x, k=3)` |
