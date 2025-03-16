---
tags:
  - python
  - ml
folder: learning
share: true
title: mlflow log array metric
date created: Tuesday, March 11th 2025, 7:42:47 pm
date modified: Tuesday, March 11th 2025, 7:49:12 pm
---

[[./mlflow|mlflow]] `log_metric` is designed for simple unidimensional metrics. Perhaps you want to look at the metric across different lead times. You can use the `step` parameter for this, which is normally used for looking how a metric varies over training steps or epochs.

```python
def log_array_metric(metric_name: str, metric_array: np.ndarray, start_step=0) -> None:
	for step, value in enumerate(metric_array, start=start_step):
		mlflow.log_metric(metric_name, value, step=step)
```
