---
tags:
  - python
  - swe
folder: learning
share: true
title: python generators
date created: Friday, June 21st 2024, 9:42:17 am
date modified: Friday, June 21st 2024, 9:59:49 am
---

A generator in python is a function that returns an **iterator** using the `yield` keyword. This means we can use it with `next()` or in a `for` loop. This is common for `dataloader`. It is an object you can (infinitely) **pull** from.

```python
def fib(limit):
	a, b = 0, 1
	while a < limit:
		yield a
		a, b = b, a + b

# create a generator object
x = fib(5)

print(next(x)) # 0
print(next(x)) # 1
print(next(x)) # 1
print(next(x)) # 2
print(next(x)) # 3

# use in for loop
for i in fib(5):
	print(i)
```
