---
tags:
  - ml
  - python
  - swe
  - jax
folder: learning
title: einsum
date created: Wednesday, January 31st 2024, 4:14:17 pm
date modified: Sunday, February 25th 2024, 12:01:24 pm
share: "true"
---

[Einstein summation convention on the operands](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html). Implemented in most tensor libraries. Saves time transposing and with broadcasting.

| operation | array | einsum |
| ---- | ---- | ---- |
| $\mathbf{a}$ (view) | `a` | `einsum('i', A)` |
| $\mathbf{a} \cdot \mathbf{b}$ (1-D) | `dot(a, b)`, `inner(a, b)` | `einsum('i,i', a, b)` |
| $\mathbf{a} \odot \mathbf{b}$ (1-D) | `multiply(a, b)` `a * b` | `einsum('i,i->i', a, b)` |
| $\mathbf{a} \cdot \mathbf{b}^T=\mathbf{a}\otimes\mathbf{b}$ | `outer(a, b)` | `einsum('i,j->ij', a, b)` |
| $\mathbf{A}^T$ | `A.T` | `einsum('ji', A)` |
| $A_{ii}$ | `diag(A)` | `einsum('ii->i', A)` |
| $\text{tr}{\mathbf{A}} = \sum_{i}A_{ii}$ | `trace(A)` | `einsum('ii', A)` |
| $\sum_{ij}{A_{ij}}$ | `sum(A)` | `einsum('ij->', A)` |
| $\sum_{i}{A_{ij}}$ | `sum(A, axis=0)` | `einsum('ij->j', A)` |
| $\mathbf{A} \mathbf{B}$ | `matmul(A, B)`, `A @ B` | `einsum('ij,jk->ik', A, B)` |
| $\mathbf{A} \mathbf{B}^T$ | `matmul(A, B.T)`, `A @ B.T` | `einsum('ij,kj->ik', A, B)` |
| batched-matrix-batched-matrix |  | `einsum('bij,bjk->bik', A, B)` |
| each value of $\mathbf{A}$ multiplied by $\mathbf{B}$ | `A[:, :, None, None] * B` | `einsum('ij,kl->ijkl', A, B)` |
| $\sum_{kl}{A_{ik} B_{jkl} C_{il}}$ |  | `einsum('ik,jkl,il->ij', [A, B, C])` |

Omitting the arrow `'->'` will take the labels that appeared once and arrange them in alphabetical order. For example, `'ij,jk->ik'` is equivalent to `'ij,jk'`.

`einsum` allows the ellipses syntax `'...'` for axes we’re not particularly interested in, like batch dimensions. For example, `einsum('...ij,ji...->...', A, B)` would multiply just the last two axes of `A` with the first two of `B`.

Transposes such as `einsum('ijk...->kji...', A)` are the same as `swapaxes(A, 0, 2)`.

Further resources from [Tim Rocktäschel](https://rockt.github.io/2018/04/30/einsum), [ajcr](https://ajcr.net/Basic-guide-to-einsum/) and [tensorchiefs](https://tensorchiefs.github.io/dlday2018/tutorial/einsum.html) . For a visual understanding of what is going on under the hood, see [Olexa Bilaniuk's post](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/).

Similar notation has been used for shapes in [[./einshape|einshape]].
