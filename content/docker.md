---
tags:
  - swe
  - python
folder: learning
title: docker
date created: Thursday, February 8th 2024, 6:10:14 pm
date modified: Friday, January 24th 2025, 7:47:12 pm
share: true
---

Deployment requirements

- **software** requirements (python package dependencies, python version)
- **OS** requirements (operating system, system packages, config)
- **hardware**/resource requirements (CPU, RAM, storage, GPU, networking – ports, load balancing etc)

1. `Dockerfile` recipe
2. `docker build .` package the code
3. `docker run <image>` run the container

## cheat sheet

| command                                                                                      | task                                      |
| -------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `docker build .`                                                                             | build a docker image                      |
| `docker build -t <image-name>:<tag> .`                                                       | add image name and tag                    |
| `docker tag <image-id> <account>.dkr.ecr.<region>.amazonaws.com/<folder>/<image-name>:<tag>` | prepare local image for upload to AWS ECR |
| `docker push <account>.dkr.ecr.<region>.amazonaws.com/<folder>/<image-name>:<tag>`           | push tagged image to AWS ECR              |
| `docker image ls`                                                                            | list docker images                        |
| `docker pull <image-name>:<tag>`                                                             | run image:tag                             |
| `docker run <image-name>:<tag>`                                                              | run image:tag                             |
| `docker run -it <image-name>:<tag> sh`                                                       | run `sh` on image:tag interactively       |
| `docker ps`                                                                                  | show running containers                   |
| `docker stop <container-id>`                                                                 | stop container                            |
| `docker contain ls`                                                                          | list running containers                   |
| `docker-compose up`                                                                          | spin up docker compose                    |
| `docker-compose up -d`                                                                       | spin up in detached mode                  |
| `docker-compose down`                                                                        | spin down docker compose                  |
| `docker system prune`                                                                        | remove stopped and dangling images        |

## basic Dockerfile

`FROM`, `COPY`, `RUN`, `CMD`

```Dockerfile
FROM python:3
COPY requirements.txt . # copy file from local to current
RUN pip install -r requirements.txt
COPY cool.py .
CMD ["univocrn", "cool:app", "--reload"]
```

## scaling

Docker compose lets you run multiple containers. For small workloads, Amazon EC2 and docker compose works well.

Docker orchestrators deal with discoverability, auto-scaling, bin packing (distributing across multiple servers given CPU RAM constraints). These range from docker swarm, to kubernetes (industry standard).

Consider [modal](https://modal.com/) ([blog post](https://erikbern.com/2022/12/07/what-ive-been-working-on-modal#fn:3)) as an alternative to these files for python jobs where we can specify images, cron jobs, GPUs etc where code executes in the cloud but is printed locally.

## minimal sagemaker dockerfile

Sagemaker allows you to [bring your own container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html). To use a `image_uri` along with the `source_dir` and `entry_point` arguments in `sagemaker.estimator.Estimator`, the Dockerfile needs two extra `ENV` variables:

- `ENV PATH="/opt/program:${PATH}"`
	- SageMaker AI runs `docker run <image> train`
	- `PATH` identifies the location of the `train` and `serve` programs when the container is invoked
	- `sagemaker-training` install creates a train executable file that uses `!#` to execute in python, which calls `from sagemaker_training.cli.train import main`, which [calls](https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/trainer.py) `sagemaker_training.trainer.train()`
- `ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code`
	- `sagemaker-training` script uploads the files in `source_dir` to s3 before downloading the tar from s3 and copying the files to `/opt/ml/code`

```docker
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for faster python dependency resolve
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# essential to have sagemaker-training installed
RUN uv pip install --system "xgboost-ray==0.1.19" "numpy==1.26.4" "ray[all]>=2.0.0" "scipy>=1.7.0" "modin[ray]==0.32.0" "sagemaker==2.226.1" "sagemaker-training==4.8.3"

# this environment variable is used by the SageMaker container to determine our user code directory for `source_dir`
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV PATH="/opt/program:${PATH}"

# avoid buffering python standard output (useful for logging)
ENV PYTHONUNBUFFERED=TRUE

# PYTHONDONTWRITEBYTECODE keeps python from writing the .pyc files
ENV PYTHONDONTWRITEBYTECODE=TRUE
```
