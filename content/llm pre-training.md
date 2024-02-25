---
tags:
  - llm
  - ml
  - neural-network
folder: llm
title: llm pre-training
date created: Sunday, February 4th 2024, 2:23:13 pm
date modified: Sunday, February 25th 2024, 7:59:18 pm
share: true
---

Training LLMs in a **self-supervised** manner (i.e. using only the raw text itself) is known as **pre-training**. This makes it easy to scale train data by just finding more text rather than labelling data.

Pre-trained models are also called **foundation models**.

Pre-trained models:

- do not follow instructions well
- quite readily emit toxic content

Pre-trained models can be tailored to a specific tasks using [[./llm fine-tuning|llm fine-tuning]].
