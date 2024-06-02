---
tags:
  - python
  - ml
  - swe
  - jax
  - bayesian
folder: learning
share: true
title: pytensor
date created: Saturday, June 1st 2024, 11:35:13 am
date modified: Sunday, June 2nd 2024, 6:56:14 pm
---

[PyTensor](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_pytensor.html) is the backend to PyMC.

But it is better thought of as a **tensor library** that can:

- compile to **multiple backends** (jax, numba, etc)
- convert mathematical expressions into graphs which can be optimised through **graph rewrites**.

## graph rewrites

PyTensor takes a **generative graph representation** to and **compiles** to an optimised graph. This allows us to write the maths as we see it, and allow the rewrites to optimise for computational stability and efficiency.

Here is an example with a pointless log-exp statement. The generative graph looks like

```python
y = pm.Normal.dist(tau=pt.log(pt.exp(x)))
pytensor.dprint(y) # print the computational graph

# normal_rv{0, (0, 0), floatX, False}.1 [id A]
# ├─ RandomGeneratorSharedVariable(<Generator(PCG64) at 0x1761E1540>) [id B] 
# ├─ [] [id C]
# ├─ 11 [id D]
# ├─ 0 [id E]
# └─ Mul [id F] 
#    ├─ Pow [id G] 
#    │ ├─ Abs [id H] 
#    │ │  └─ Log [id I] 
#    │ │     └─ Exp [id J] 
#    │ │        └─ x [id K] 
#    │ └─ ExpandDims{axis=0} [id L] 
#    │    └─ -0.5 [id M]
#    └─ Sign [id N] 
#       └─ Log [id I] 
#          └─ ···
```

When we compile the graph, we get

```python
# prettier printing
mode = pytensor.compile.mode.get_default_mode().excluding("fusion")

# pytensor.function creates callable object so that we can push values through the compiled graph
# pytensor.function([inputs], outputs)
fn = pytensor.function([x], y, mode=mode)
pytensor.dprint(fn)

# normal_rv{0, (0, 0), floatX, False}.1 [id A]
# ├─ RandomGeneratorSharedVariable(<Generator(PCG64) at 0x1761E1540>) [id B] 
# ├─ [] [id C]
# ├─ 11 [id D]
# ├─ 0 [id E]
# └─ Mul [id F] 4
#    ├─ Reciprocal [id G] 3 
#    │ └─ Sqrt [id H] 2
#    │    └─ Abs [id I] 1
#    │       └─ x [id J]
#    └─ Sign [id K] 0 
#       └─ x [id J]
```

where the log-exp has been removed and the `pow(x, -0.5)` has been replaced with a `1/sqrt(x)` (probably for stability).

### logp PyMC model

A particular case of rewrites are in PyMC in evaluating `logp`. PyMC distributions are written in a single **cannoncial form**. For example, regardless of whether the user specifies `cov`, `chol`, or `tau` in `pm.Normal()`, PyMC will always convert to `chol` for `logp` evaluation, before PyTensor optimises the graph.

For an example PyMC model, the generative graph for `logp`:

```python
with pm.Model() as m:
	tau = pm.InverseGamma("tau", 1, 1)
	y = pm.Normal("y", 0, tau=tau, shape=(3,))

logp_y = m.logp(vars=[y], sum=False)
pytensor.dprint(logp_y)

# Check{sigma > 0} [id A] 'y_logprob' 
#  ├─ Sub [id B] 
#  │ ├─ Sub [id C] 
#  │ │ ├─ Mul [id D] 
#  │ │ │ ├─ ExpandDims{axis=0} [id E] 
#  │ │ │ │ └─ -0.5 [id F] 
#  │ │ │ └─ Pow [id G] 
#  │ │ │    ├─ True_div [id H]
#  │ │ │    │ ├─ Sub [id I]
#  │ │ │    │ │ ├─ y [id J] 
#  │ │ │    │ │ └─ ExpandDims{axis=0} [id K]
#  │ │ │    │ │    └─ 0 [id L]
#  │ │ │    │ └─ ExpandDims{axis=0} [id M]
#  │ │ │    │    └─ Mul [id N]
#  │ │ │    │       ├─ Pow [id O]
#  │ │ │    │       │ ├─ Abs [id P]
#  │ │ │    │       │ │  └─ Exp [id Q]
#  │ │ │    │       │ │     └─ tau_log__ [id R]
#  │ │ │    │       │ └─ -0.5 [id S]
#  │ │ │    │       └─ Sign [id T]
#  │ │ │    │          └─ Exp [id Q]
#  │ │ │    │             └─ ···
#  │ │ │    └─ ExpandDims{axis=0} [id U]
#  │ │ │       └─ 2 [id V]
#  │ │ └─ ExpandDims{axis=0} [id W]
#  │ │    └─ Log [id X]
#  │ │       └─ Sqrt [id Y]
#  │ │          └─ 6.283185307179586 [id Z]
#  │ └─ ExpandDims{axis=0} [id BA]
#  │    └─ Log [id BB]
#  │       └─ Mul [id N]
#  │          └─ ···
#  └─ All{axes=None} [id BC]
#     └─ MakeVector{dtype='bool'} [id BD]
#        └─ All{axes=None} [id BE]
#           └─ Gt [id BF]
#              ├─ Mul [id N]
#              │  └─ ···
#              └─ 0 [id BG]
```

And after optimisation

```python
y_value = pt.vector("y_value", shape=(3,))
logp_y = pm.logp(y, y_value)
fn = pytensor.function([y_value, tau], logp_y, mode=mode)
pytensor.dprint(fn)

# Check{sigma > 0} [id A] 'y_logprob' 13
#  ├─ Sub [id B] 12
#  │ ├─ Sub [id C] 11
#  │ │ ├─ Mul [id D] 10
#  │ │ │ ├─ [-0.5] [id E]
#  │ │ │ └─ Sqr [id F] 9
#  │ │ │    └─ True_div [id G] 7
#  │ │ │       ├─ y_value [id H]
#  │ │ │       └─ ExpandDims{axis=0} [id I] 6
#  │ │ │          └─ Mul [id J] 4
#  │ │ │             ├─ Reciprocal [id K] 3
#  │ │ │             │ └─ Sqrt [id L] 2
#  │ │ │             │ └─ Abs [id M] 1
#  │ │ │             │ └─ tau [id N]
#  │ │ │             └─ Sign [id O] 0
#  │ │ │                └─ tau [id N]
#  │ │ └─ [0.91893853] [id P]
#  │ └─ Log [id Q] 8
#  │    └─ ExpandDims{axis=0} [id I] 6
#  │       └─ ···
#  └─ Gt [id R] 5
#     ├─ Mul [id J] 4
#     │  └─ ···
#     └─ 0 [id S]
```

where, amongst other rewrites, the term `log(sqrt(c))` has now become a constant throughout all `logp` evaluations.
