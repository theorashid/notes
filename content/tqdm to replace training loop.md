---
tags:
  - python
  - swe
folder: learning
share: true
title: tqdm to replace training loop
date created: Friday, May 3rd 2024, 3:16:47 pm
date modified: Friday, May 3rd 2024, 3:20:04 pm
---

For a nicer progress bar, replace `for _ in range(1000)` with `tqdm`.

```python
losses = []
with tqdm.notebook.tqdm(range(1000)) as tepoch:
	for _ in tepoch:
		tepoch.set_description(f"Epoch")
		optimizer.zero_grad()
		output = model(X_train)
		loss = -mll(output, y_train)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		tepoch.set_postfix(loss=loss.item())
```
