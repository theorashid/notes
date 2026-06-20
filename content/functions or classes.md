---
tags:
  - python
  - swe
folder: learning
title: functions or classes
date created: Wednesday, January 31st 2024, 11:20:36 am
date modified: Monday, June 8th 2026, 4:44:30 pm
share: true
---

[Rémi Louf asks](https://twitter.com/remilouf/status/1725957553937408490)

> I noticed that over the years I've adopted a more functional style of programming in Python. If anything because it's easier to write tests.
> 
> What are **good arguments in favour of using classes instead of functions**?

## why not classes

Eric Ma replies

> Mostly to store data, I think. I pretty much only use classes for: 
> 
> 1. Data classes,
> 2. More conveniently parameterised functions (Equinox/PyTorch-style).

[Eric uses](https://ericmjl.github.io/blog/2023/12/12/classes-functions-both/) **objects for holding data**, and **functions for processing the data**. 

Point 2 refers to implementing configurable functions using a class with the `__call__` function. This allows the pattern often used in neural network libraries:

```python
layer = Dense(in_dims, out_dims)
out = layer(x)
```

Here, the `in_dims` and `out_dims` parameters are attached to the object directly as part of the initialisation. This is useful when we configure reusable data associated with the function – here, the parameters of the layer.

Patrick Kidger adds

> Classes are fine, just don't mutate them :) This still counts as "functional". Basically, use frozen dataclasses.
> 
> Classes:
> - are containers for data
> - provide abstract interfaces (=single dispatch)
> - offer namespacing

Patrick's [Equinox](https://docs.kidger.site/equinox/pattern/) library is designed around the base class `eqx.Module`, where **every subclass is either abstract** (it can be subclassed, but not instantiated); **or final** (it can be instantiated, but not subclassed). This means the `__init__` method, and all dataclass fields, are defined once in one class.

Everything is abstract until the final class.

This promotes composing classes over class hierarchies. We *never* use `super().__init__()`.

Using lots of classes could lead to spaghetti code, but using the abstract/final design pattern helps avoid this because `__init__` always makes it clear when the class is final.

## why classes

It is sometimes useful to put methods on a class to [aid discoverability](https://twitter.com/PatrickKidger/status/1726003518895693860) Think PyTorch `tensor.pow(2).mean(dim=1).sqrt()` vs NumPy `np.sqrt(np.mean(array**2, axis=1))` (although a pipe operator could help here).

Classes are easily extensible.

For non-production data science work, it's useful to have the data attached when doing stuff interactively and be able to tab complete.

## summary

[Use (data)classes for holding data, and functions for processing the data](https://ericmjl.github.io/blog/2023/12/12/classes-functions-both/).

Coming from a maths background, functions have always clearer in my head as a way of transforming data over OOP, so I'll try and implement this pattern wherever possible.
