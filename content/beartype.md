---
tags:
  - python
  - swe
folder: learning
share: true
title: beartype
date created: Saturday, October 12th 2024, 9:51:21 pm
date modified: Saturday, October 12th 2024, 10:34:37 pm
---

[beartype](https://beartype.readthedocs.io/en/latest/) **enforces type hints**. It can be used manually with `@beartype`, or at the top of `{your_package}.__init__` submodule put

```python
from beartype.claw import beartype_this_package

beartype_this_package()
```

Code will fail if you pass incorrect types to a function.

beartype integrates with [jaxtyping](https://docs.kidger.site/jaxtyping/) for arrays, [[./pandera|pandera]] for dataframes, and [more](https://beartype.readthedocs.io/en/latest/faq/#how-do-i-type-check).
