---
tags:
  - python
  - jax
  - swe
folder: learning
share: true
title: splitting keys in jax
date created: Friday, June 21st 2024, 9:59:57 am
date modified: Friday, June 21st 2024, 10:12:32 am
---

For sequential data generation, the official way to [split keys](https://jax.readthedocs.io/en/latest/jep/263-prng.html).

```python
for i in range(N):
	key, subkey = jax.random.split(key)
	jax.random.normal(subkey)
```

Or we can build a [[./python generators|generator]].

```python
# with a limit to avoid infinite `for` loop
def key_gen(seed, N=10):
	key = jax.random.PRNGKey(seed)
	i = 0
	while i < N:
		yield key
		key, _ = jax.random.split(key)
		i += 1

key_iter = key_gen(10, N=3)
for key in key_iter:
	print(jax.random.normal(key))

# or define without limit and use `next()`
def key_gen(seed):
	key = jax.random.PRNGKey(seed)
	while True:
		yield key
		key, _ = jax.random.split(key)

keys = key_gen(10)

print(jax.random.normal(next(keys)))
print(jax.random.normal(next(keys)))
```

For non-sequential data, it is best to vectorise using [[./jax.vmap|jax.vmap]].

```python
def f(key):
	return jax.random.normal(key)

keys = jax.random.split(key, N)
jax.vmap(f)(keys)
```
