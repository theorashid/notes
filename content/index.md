---
tags: 
folder: learning
title: index
date created: Wednesday, January 31st 2024, 10:57:03 am
date modified: Wednesday, July 2nd 2025, 3:43:30 pm
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

| File                                                                                                                               | date modified                 | date created                  |
| ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ----------------------------- |
| [[./ray on sagemaker\|ray on sagemaker]]                                                                           | 5:02 PM - March 15, 2025      | 9:36 PM - March 13, 2025      |
| [[./mlflow\|mlflow]]                                                                                               | 7:44 PM - March 11, 2025      | 7:13 AM - May 30, 2024        |
| [[./docker\|docker]]                                                                                               | 7:47 PM - January 24, 2025    | 10:10 AM - February 08, 2024  |
| [[./infrastructure as code\|infrastructure as code]]                                                               | 10:07 PM - January 13, 2025   | 8:47 AM - May 03, 2024        |
| [[./kubernetes\|kubernetes]]                                                                                       | 10:07 PM - January 13, 2025   | 7:52 PM - January 13, 2025    |
| [[./mac setup\|mac setup]]                                                                                         | 7:53 PM - January 13, 2025    | 11:59 PM - January 29, 2024   |
| [[./beartype\|beartype]]                                                                                           | 5:08 PM - November 10, 2024   | 9:51 PM - October 12, 2024    |
| [[./pandera\|pandera]]                                                                                             | 10:43 PM - October 12, 2024   | 10:32 PM - October 12, 2024   |
| [[./d3.js\|d3.js]]                                                                                                 | 9:17 AM - October 02, 2024    | 6:23 AM - September 25, 2024  |
| [[./observable plot\|observable plot]]                                                                             | 9:09 AM - October 02, 2024    | 8:42 AM - October 02, 2024    |
| [[./python versioning, virtual environments and packaging\|python versioning, virtual environments and packaging]] | 3:54 PM - September 09, 2024  | 12:00 AM - January 30, 2024   |
| [[./serverless functions\|serverless functions]]                                                                   | 10:58 AM - September 04, 2024 | 10:01 AM - September 04, 2024 |
| [[./jax.vmap\|jax.vmap]]                                                                                           | 9:17 AM - August 30, 2024     | 8:30 AM - February 04, 2024   |
| [[./splitting keys in jax\|splitting keys in jax]]                                                                 | 2:12 AM - June 21, 2024       | 1:59 AM - June 21, 2024       |
| [[./python generators\|python generators]]                                                                         | 1:59 AM - June 21, 2024       | 1:42 AM - June 21, 2024       |
| [[./pytensor\|pytensor]]                                                                                           | 10:56 AM - June 02, 2024      | 3:35 AM - June 01, 2024       |
| [[./tqdm to replace training loop\|tqdm to replace training loop]]                                                 | 7:20 AM - May 03, 2024        | 7:16 AM - May 03, 2024        |
| [[./Typer to replace argparse\|Typer to replace argparse]]                                                         | 3:57 AM - April 13, 2024      | 3:42 AM - April 13, 2024      |
| [[./populate a PostgreSQL database with SQLModel\|populate a PostgreSQL database with SQLModel]]                   | 3:40 AM - April 13, 2024      | 10:03 AM - March 03, 2024     |
| [[./sqlmodel (sqlalchemy) cascade\|sqlmodel (sqlalchemy) cascade]]                                                 | 3:39 AM - April 13, 2024      | 3:14 AM - April 13, 2024      |


## other

| File                                                                                     | date modified                | date created                 |
| ---------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------- |
| [[./archive\|archive]]                                                   | 3:43 PM - July 02, 2025      | 3:07 AM - January 31, 2024   |
| [[index\|index]]                                                       | 3:43 PM - July 02, 2025      | 2:57 AM - January 31, 2024   |
| [[./time series features\|time series features]]                         | 4:01 PM - May 09, 2025       | 5:10 PM - April 30, 2025     |
| [[./xarray zarr s3\|xarray zarr s3]]                                     | 9:31 PM - March 15, 2025     | 10:42 PM - March 10, 2025    |
| [[./mlflow log altair figure\|mlflow log altair figure]]                 | 7:53 PM - March 11, 2025     | 7:44 PM - March 11, 2025     |
| [[./mlflow log array metric\|mlflow log array metric]]                   | 7:49 PM - March 11, 2025     | 7:42 PM - March 11, 2025     |
| [[./getting these notes online\|getting these notes online]]             | 7:01 PM - December 14, 2024  | 12:53 PM - February 25, 2024 |
| [[./probability integral transform\|probability integral transform]]     | 4:03 PM - December 14, 2024  | 1:22 PM - December 14, 2024  |
| [[./copula\|copula]]                                                     | 2:32 PM - December 14, 2024  | 12:01 PM - December 14, 2024 |
| [[./automatic differentiation\|automatic differentiation]]               | 6:02 PM - December 01, 2024  | 5:45 AM - August 03, 2024    |
| [[./treescope\|treescope]]                                               | 3:06 PM - September 01, 2024 | 6:37 AM - August 31, 2024    |
| [[./neural networks in jax\|neural networks in jax]]                     | 4:54 AM - September 01, 2024 | 7:02 AM - August 29, 2024    |
| [[./linear model as a neural network\|linear model as a neural network]] | 11:31 AM - August 29, 2024   | 5:32 AM - February 04, 2024  |
| [[./fixed point iteration\|fixed point iteration]]                       | 7:14 AM - August 03, 2024    | 3:43 AM - July 30, 2024      |
| [[./implicit function theorem\|implicit function theorem]]               | 7:09 AM - August 03, 2024    | 3:46 AM - August 03, 2024    |
| [[./newton's method\|newton's method]]                                   | 7:07 AM - August 03, 2024    | 8:43 AM - July 29, 2024      |
| [[./explicit and implicit layers\|explicit and implicit layers]]         | 6:55 AM - August 03, 2024    | 3:48 AM - August 03, 2024    |
| [[./INLA\|INLA]]                                                         | 9:40 AM - May 10, 2024       | 9:22 AM - May 03, 2024       |
| [[./pg_dump\|pg_dump]]                                                   | 8:58 AM - March 24, 2024     | 1:47 PM - March 22, 2024     |
| [[./kernel trick\|kernel trick]]                                         | 11:58 AM - February 25, 2024 | 12:00 PM - February 14, 2024 |


---

[[./archive|archive]]
