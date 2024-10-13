---
tags: 
folder: learning
title: index
date created: Wednesday, January 31st 2024, 10:57:03 am
date modified: Monday, August 12th 2024, 5:14:24 pm
share: true
---

These are a set of minimal notes and code snippets. I wrote these notes for myself for quick reference. *These notes are a work in progress*.

---

## topics

### ssm

These notes focus on my use cases of state space models. These are not the linear deterministic kind of SSM used in big bad LLMs like [Mamba](https://github.com/state-spaces/mamba), but instead **Bayesian inference in stochastic (non)linear dynamical models**.

- [[./state space model|state space model]]
- [[./linear gaussian ssm|linear gaussian ssm]]
- [[./nonlinear gaussian ssm|nonlinear gaussian ssm]]
- [[./kalman filtering and smoothing|kalman filtering and smoothing]]
- [[./ssm in dynamax|ssm in dynamax]]
- [[./online learning using ssm|online learning using ssm]]
- [[./structural time series models|structural time series models]]
- [[./state space gaussian process|state space gaussian process]]
- [[./ssm resources|ssm resources]]

### llm

Read Ravin's [GenAI guidebook](https://ravinkumar.com/GenAiGuidebook/model_basics/SimpleLinRegFlax.html) or Vicki Boykis' [normcore-llm.md](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e) instead.

- [[./language modelling|language modelling]]
- [[./transformer architecture|transformer architecture]]
- [[./multi-headed self-attention|multi-headed self-attention]]
- [[./llm pre-training|llm pre-training]]
- [[./llm fine-tuning|llm fine-tuning]]
- [[./word2vec|word2vec]]
- [[./retrieval augmented generation|retrieval augmented generation]]

### research engineering

| File                                                                                                                               | date modified                 | date created                 |
| ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------- |
| [[./pandera\|pandera]]                                                                                             | 10:43 PM - October 12, 2024   | 10:32 PM - October 12, 2024  |
| [[./beartype\|beartype]]                                                                                           | 10:34 PM - October 12, 2024   | 9:51 PM - October 12, 2024   |
| [[./docker\|docker]]                                                                                               | 9:51 PM - October 12, 2024    | 10:10 AM - February 08, 2024 |
| [[./d3.js\|d3.js]]                                                                                                 | 5:17 PM - October 02, 2024    | 2:23 PM - September 25, 2024 |
| [[./observable plot\|observable plot]]                                                                             | 5:09 PM - October 02, 2024    | 4:42 PM - October 02, 2024   |
| [[./python versioning, virtual environments and packaging\|python versioning, virtual environments and packaging]] | 11:54 PM - September 09, 2024 | 8:00 AM - January 30, 2024   |
| [[./mac setup\|mac setup]]                                                                                         | 4:00 PM - September 08, 2024  | 7:59 AM - January 30, 2024   |
| [[./serverless functions\|serverless functions]]                                                                   | 6:58 PM - September 04, 2024  | 6:01 PM - September 04, 2024 |
| [[./jax.vmap\|jax.vmap]]                                                                                           | 5:17 PM - August 30, 2024     | 4:30 PM - February 04, 2024  |
| [[./splitting keys in jax\|splitting keys in jax]]                                                                 | 10:12 AM - June 21, 2024      | 9:59 AM - June 21, 2024      |
| [[./python generators\|python generators]]                                                                         | 9:59 AM - June 21, 2024       | 9:42 AM - June 21, 2024      |
| [[./pytensor\|pytensor]]                                                                                           | 6:56 PM - June 02, 2024       | 11:35 AM - June 01, 2024     |
| [[./mlflow\|mlflow]]                                                                                               | 4:58 PM - May 30, 2024        | 3:13 PM - May 30, 2024       |
| [[./infrastructure as code\|infrastructure as code]]                                                               | 5:22 PM - May 03, 2024        | 4:47 PM - May 03, 2024       |
| [[./tqdm to replace training loop\|tqdm to replace training loop]]                                                 | 3:20 PM - May 03, 2024        | 3:16 PM - May 03, 2024       |
| [[./Typer to replace argparse\|Typer to replace argparse]]                                                         | 11:57 AM - April 13, 2024     | 11:42 AM - April 13, 2024    |
| [[./populate a PostgreSQL database with SQLModel\|populate a PostgreSQL database with SQLModel]]                   | 11:40 AM - April 13, 2024     | 6:03 PM - March 03, 2024     |
| [[./sqlmodel (sqlalchemy) cascade\|sqlmodel (sqlalchemy) cascade]]                                                 | 11:39 AM - April 13, 2024     | 11:14 AM - April 13, 2024    |
| [[./mocker\|mocker]]                                                                                               | 4:03 PM - March 24, 2024      | 8:51 PM - March 22, 2024     |
| [[./shorten FastAPI app.py with APIRouter\|shorten FastAPI app.py with APIRouter]]                                 | 8:38 PM - March 11, 2024      | 10:23 PM - March 07, 2024    |


## other

| File                                                                                     | date modified                 | date created                |
| ---------------------------------------------------------------------------------------- | ----------------------------- | --------------------------- |
| [[./treescope\|treescope]]                                               | 11:06 PM - September 01, 2024 | 2:37 PM - August 31, 2024   |
| [[./neural networks in jax\|neural networks in jax]]                     | 12:54 PM - September 01, 2024 | 3:02 PM - August 29, 2024   |
| [[./linear model as a neural network\|linear model as a neural network]] | 7:31 PM - August 29, 2024     | 1:32 PM - February 04, 2024 |
| [[index\|index]]                                                       | 5:14 PM - August 12, 2024     | 10:57 AM - January 31, 2024 |
| [[./fixed point iteration\|fixed point iteration]]                       | 3:14 PM - August 03, 2024     | 11:43 AM - July 30, 2024    |
| [[./implicit function theorem\|implicit function theorem]]               | 3:09 PM - August 03, 2024     | 11:46 AM - August 03, 2024  |
| [[./archive\|archive]]                                                   | 3:08 PM - August 03, 2024     | 11:07 AM - January 31, 2024 |
| [[./automatic differentiation\|automatic differentiation]]               | 3:07 PM - August 03, 2024     | 1:45 PM - August 03, 2024   |
| [[./newton's method\|newton's method]]                                   | 3:07 PM - August 03, 2024     | 4:43 PM - July 29, 2024     |
| [[./explicit and implicit layers\|explicit and implicit layers]]         | 2:55 PM - August 03, 2024     | 11:48 AM - August 03, 2024  |
| [[./INLA\|INLA]]                                                         | 5:40 PM - May 10, 2024        | 5:22 PM - May 03, 2024      |
| [[./pg_dump\|pg_dump]]                                                   | 3:58 PM - March 24, 2024      | 8:47 PM - March 22, 2024    |
| [[./getting these notes online\|getting these notes online]]             | 9:53 PM - February 25, 2024   | 8:53 PM - February 25, 2024 |
| [[./kernel trick\|kernel trick]]                                         | 7:58 PM - February 25, 2024   | 8:00 PM - February 14, 2024 |


---

[[./archive|archive]]
