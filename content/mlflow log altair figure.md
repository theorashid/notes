---
tags:
  - python
  - ml
folder: learning
share: true
title: mlflow log altair figure
date created: Tuesday, March 11th 2025, 7:44:09 pm
date modified: Tuesday, March 11th 2025, 7:53:44 pm
---

[[./mlflow|mlflow]] `log_figure` only supports matplotlib or plotly. We can save an altair chart (or other library that saves `.png`) to a tempfile and use [pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html) `Image` to open the image, which is compatible with `mlflow.log_image()`.

```python
def save_altair_chart_to_mlflow(chart: alt.Chart, artifact_file: str) -> None:
	with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
		temp_filename = tmpfile.name

	try:
		chart.save(temp_filename)
		img = Image.open(temp_filename)
		mlflow.log_image(img, artifact_file)

	finally:
		# ensure the temporary file is removed
		os.unlink(temp_filename)
```
