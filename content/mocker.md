---
tags:
  - python
  - swe
folder: learning
share: true
title: mocker
date created: Friday, March 22nd 2024, 8:51:46 pm
date modified: Sunday, March 24th 2024, 4:03:40 pm
---

Use `pytest-mocker` so you don't need to run a function in your code during testing.

For example, a function which writes to disk.

```python
def delete_table():
	...
	write_database_to_disk()

	return
```

Use a `mocker` to skip over this function in the test.

```python
def test_delete_table(mocker):
	mocker.patch(path.to.where.function.is.used.write_database_to_disk)

	...
```

Note, it is important to know [where to mock](https://docs.python.org/3/library/unittest.mock.html#where-to-patch). This must be **where the function is called**, which is not necessarily where it is defined.

More notes [here](https://pytest-with-eric.com/mocking/pytest-mocking/) and [here](https://changhsinlee.com/pytest-mock/).
