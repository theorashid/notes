---
tags:
  - jax
  - ml
  - python
  - neural-network
folder: learning
share: true
title: jax sharding
date created: Friday, December 5th 2025, 5:42:11 pm
date modified: Friday, December 5th 2025, 6:27:16 pm
---

Every `jax.Array` has an associated [`jax.sharding.Sharding`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") object describing which [shard](https://docs.jax.dev/en/latest/sharded-computation.html) of the global data is required by each global device.

Most of the time, you can just write a function as if you’re operating on the full dataset, and [[./jax.jit|jax.jit]] will split that computation across multiple devices for automatic parallelism.

But if you don't want to use the `jax.jit` heuristics, you can parallelise manually using [`jax.shard_map()`](https://docs.jax.dev/en/latest/notebooks/shard_map.html) (also deprecates `jax.pmap`) to write a function that will handle a single shard of data. For example, a sharded `sum` will sum up the shards of the array on each device.

## types of parallelism

Single Program, Multiple Data (SPMD) model of parallelism: write a single program that runs on multiple devices, using annotations to specify which part of the data each device is responsible for.

- `jax.sharding.Mesh` is an arrangement of all our accelerators into a `numpy.ndarray`
-  `jax.P` specifies which axes of our array are to be sharded over which axes of devices (with standard broadcasting rules)

Some common regimes for [neural networks](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#examples-neural-networks):

- [Data parallel](https://docs.jax.dev/en/latest/the-training-cookbook.html#data-parallel) (performant for small models). Each accelerator stores a complete copy of the model parameters, and we shard activations along the batch axis to split the computation of the gradients.
- [Fully-sharded data parallel](https://docs.jax.dev/en/latest/the-training-cookbook.html#fully-sharded-data-parallel-fsdp). Shard both the model and the parameters.
- [Tensor parallel](https://docs.jax.dev/en/latest/the-training-cookbook.html#tensor-parallel). Structure the model so that the forward pass computation can be performed in parallel. For example, shard along the heads of a [[./multi-headed self-attention|multi-headed self-attention]] network.
These can also be scaled to a [kubernetes cluster](<jax on kubernetes cluster multi-host: https://docs.jax.dev/en/latest/multi_process.html>).
