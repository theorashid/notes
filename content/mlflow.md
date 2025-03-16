---
tags:
  - python
  - swe
  - ml
folder: learning
share: true
title: mlflow
date created: Thursday, May 30th 2024, 3:13:37 pm
date modified: Tuesday, March 11th 2025, 7:44:48 pm
---

[mlflow](https://mlflow.org/docs/latest/index.html) is a tool for experiment tracking and model registry. Below is an example script, which also uses [[./Typer to replace argparse|Typer]]. This uses a local [database](https://mlflow.org/docs/latest/tracking/tutorials/local-database.html) to keep track of experiments and models, and the mlflow ui can be accessed using `mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db`. These can both be replaced with cloud servers.

```python
from typing import Annotated

import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext
import numpy as np
from sklearn.model_selection import train_test_split
import typer

app = typer.Typer()


def train(X, y, model, loss_fn, metrics_fn, optimizer):
    model.train()
    for _ in range(len(y)):
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            mlflow.log_metric("loss", loss.item(), step=(batch // 100))


@app.command()
def main(
	seed: Annotated[int, typer.Option(help="Seed for random state.")] = 2,
	test_size: Annotated[float, typer.Option(help="Proportion of the test dataset.")] = 0.2,
	n_iterations: Annotated[int, typer.Option(help="Number of iterations to train the model.")] = 1000,
	lr: Annotated[float, typer.Option(help="Learning rate.")] = 0.1,
):
	uri: str = "sqlite:///mlruns.db"
	experiment_name: str = "test_model"
	run_name: str = "test_run"

	X, y = load_data()
	X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

	model = Model()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	dataset_params = {
		"seed": seed,
		"test_size": test_size,
	}

	model_params = {
		"optimizer": "Adam",
		"learning_rate": lr,
		"n_iterations": n_iterations,
	}
	
	with mlflow.start_run(run_name=run_name):
		mlflow.set_tag("info", "Some words")

		mlflow.log_params(dataset_params)
		mlflow.log_params(model_params)

		# any `mlflow.log` statements needs to be within `with`
		train(X_train, y_train, model, optimizer, n_iterations)
		
		test_pred = model.predict(X_test)

		mlflow.log_metric("Test MSE", mean_squared_error(test_pred, y_test))

		# generic type, but most use cases have a log model (see docs)
		mlflow.pyfunc.log_model(artifact_path="", python_model=model)

if __name__ == "__main__":
	app()
```

There is in-built [autologging](https://mlflow.org/docs/latest/tracking/autolog.html#automatic-logging) support for scikit-learn, pytorch lightning, keras, etc. More general model support is available, with an example for [pymc](https://gist.github.com/juanitorduz/8f34be9aeb5269fd2f12b578e64e4925) and for [pytorch](https://mlflow.org/docs/latest/deep-learning/pytorch/guide/index.html).
