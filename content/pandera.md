---
tags:
  - python
  - swe
folder: learning
share: true
title: pandera
date created: Saturday, October 12th 2024, 10:32:58 pm
date modified: Saturday, October 12th 2024, 10:43:08 pm
---

[pandera](https://pandera.readthedocs.io/en/stable/) performs data **validation on dataframe**-like objects (pandas, polars, dask modin, spark etc).

```python
import pandera as pa
from pandera.typing import Series

# pydantic-style syntax
class Schema(pa.DataFrameModel):
    column1: int = pa.Field(le=10)
    column2: float = pa.Field(lt=-1.2)
    column3: str = pa.Field(str_startswith="value_")

	# more specific custom rule
	@pa.check("column3")
    def column_3_check(cls, series: Series[str]) -> Series[bool]:
        """Check that column3 values have two elements after being split with '_'"""
        return series.str.split("_", expand=True).shape[1] == 2

Schema.validate(df)
```

pandera integrates with [[./beartype|beartype]].
