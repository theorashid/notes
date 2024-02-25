---
tags:
  - ml
folder: learning
title: kernel trick
date created: Wednesday, February 14th 2024, 8:00:01 pm
date modified: Sunday, February 25th 2024, 7:58:06 pm
share: true
---

**Support vector machines** rely on taking the dot product between data points, $\mathbf{x_{i}}\cdot \mathbf{x_j}$. We can increase the complexity by transforming with a nonlinear mapping as $\phi(\mathbf{x_{i}})\cdot \phi(\mathbf{x_j})$. Nonlinear mapping can introduce higher (or even infinite) dimensional terms.

Instead, we define a **similarity function** (kernel) which implicitly defines the nonlinear feature map, $k(x_{i}, x_{j}) = \phi(\mathbf{x_{i}})\cdot \phi(\mathbf{x_j})$.

This is just a **function on the original coordinates** rather than a dot product on the set of transformed coordinates.

As an example, choose a space with two features and a polynomial order 2 feature space.

```python
# polynomial transformation gives more features than just (x[0], x[1])
phiX = [(x[0], x[1], x[0], x[1], x[0]**2, x[1]**2) for x in X]

# linear is SVC().fit(x, y)
# this is slow
SVC().fit(phiX, y)

# instead, use a kernel function 
# k(x, x') = (1 + x.x')^2
# which is a dot product for only the original (x[0], x[1])
SVC(kernel="poly").fit(X, y)
```

Although the polynomial order 2 is finite and computationally feasible, some kernels, such as the radial basis function kernel, would be infinite-dimensional.
