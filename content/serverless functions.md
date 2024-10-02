---
tags:
  - swe
  - python
folder: learning
share: true
title: serverless functions
date created: Wednesday, September 4th 2024, 6:01:41 pm
date modified: Wednesday, September 4th 2024, 6:57:58 pm
---

Functions as a service. Triggered by an **event** (e.g. click).

- scales with demand automatically
- do not pay for idle
- eliminates server maintenance

But:

- timeouts
- latency issues, having to deal with cold starts each time

Main options:

- AWS Lambda (annoying to set up, hardware limitations, expensive, not easy to get big machines/GPU)
- [[serverless functions#Coiled|Coiled]]
- [[serverless functions#Modal|Modal]]

## Coiled

- [Coiled](https://docs.coiled.io/user_guide/index.html) creates a VM, `uv pip install` local software packages, cloud credentials, files, etc.., [runs your script, and then shuts down the VM](https://medium.com/coiled-hq/easy-heavyweight-serverless-functions-1983288c9ebc)
- use any of the major cloud providers
	- based on Dask clusters
- [support](https://docs.coiled.io/user_guide/mlops-with-mlflow.html#putting-it-all-together) for [[./mlflow|mlflow]]

```sh
pip install coiled "dask[complete]"
coiled login
```

- run a **script** using CLI

```sh
coiled run --region europe-west1 --gpu python script.py
```

- run a **single function** in python

```python
import coiled  
  
@coiled.function(
	# see https://cloud.coiled.io/settings/setup/infrastructure
	workspace="devs", # 
	region="europe-west1",
	vm_type="g2-standard-4",
	# or can specify specific GPU type
	# gpu=True, # interpreted as 1
	# worker_gpu_type="nvidia-tesla-t4",
	idle_timeout="5 minutes", # shutdown after 5 minutes of idleness
)
def train_expensive_model(**kwargs):
	return fit(**kwargs)
  
train_expensive_model()
```

## Modal

- [modal](https://modal.com/) runs it in its own cloud environment (so no AWS or other cloud providers yet) but with [rapid spin-up times](https://erikbern.com/2022/12/07/what-ive-been-working-on-modal) (Rust rather than Docker/kubernetes)
- support for [scheduling cron jobs](https://modal.com/docs/guide/cron#scheduling-remote-cron-jobs) in the cloud, [timeouts](https://modal.com/docs/guide/timeouts), [retries](https://modal.com/docs/guide/retries#failures-and-retries)
- support for [web endpoints](https://modal.com/docs/guide/webhooks) `@modal.web_endpoint(method="POST)`

```sh
pip install modal
modal setup
```

- [specify images](https://modal.com/docs/guide/custom-container) in python (rather than Docker) through method chaining
	- can also use `mamba install` or `uv`
	- can access images from registry or `.from_dockerfile("Dockerfile")`

```python
image = (
	modal.Image.debian_slim(python_version="3.10")
	.apt_install("git")
	.pip_install("torch==2.2.1")
	.run_commands("git clone https://github.com/modal-labs/agi && echo 'ready to go!'")
)
```

- run a **single function** in python

```python
import modal

app = modal.App("app-name")

@app.function(
	image=image,
	# add specific GPU, could even do modal.gpu.A100(size="80GB", count=8)
	gpu="A100",
	cpu=8.0, # ask for 8 CPU cores
	memory=(1024, 32768), # in MB, ask for at least 1GB RAM (32GB limit)
)
def train_expensive_model(**kwargs):
	return fit(**kwargs)

@app.local_entrypoint()
def main():
	train_expensive_model.local() # run the function locally
	train_expensive_model.remote() # run the function remotely on Modal
	train_expensive_model.map()
```
