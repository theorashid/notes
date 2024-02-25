---
tags:
  - swe
  - python
folder: learning
title: docker
date created: Thursday, February 8th 2024, 6:10:14 pm
date modified: Sunday, February 25th 2024, 7:57:39 pm
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

| command | task |
| ---- | ---- |
| `docker build .` | build a docker image |
| `docker build -t <image-name>:<tag>` | add image name and tag |
| `docker images` | list docker images |
| `docker pull <image-name>:<tag>` | run image tag |
| `docker run <image-name>:<tag>` | run image:tag |
| `docker contain ls` | list running containers |
| `docker-compose up` | spin up docker compose |
| `docker-compose up -d` | spin up in detached mode |
| `docker-compose down` | spin down docker compose |

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
