---
tags:
  - llm
  - ml
  - neural-network
folder: llm
title: llm fine-tuning
date created: Sunday, February 4th 2024, 2:24:39 pm
date modified: Sunday, February 25th 2024, 7:59:14 pm
share: true
---

Once we have done [[./llm pre-training|llm pre-training]] to understand language (general), we use **transfer learning** to [fine-tune](https://ravinkumar.com/GenAiGuidebook/language_models/finetuning.html) the model to a **specific** task.

Loss is typically calculated over the entire output phrase.

We can make LLMs easier to interact with using human input:

- **Instruction** (supervised) fine-tuning: the model is fine-tuned on thousands of instruction prompt-completion pairs that were **human-labeled**.
- Reinforcement Learning with Human feedback: humans indicated what they *prefer*.

Updating the entire set of weights is computationally inefficient. Use **parameter efficient fine tuning** to update *some* of the weights:

- Prompt tuning: add a new set of weights between the prompt and the model.
- Low-rank adaptation (LORA): reduce the number of parameters that need to be trained using singular value decomposition.
	- Creating a new set of LORA parameters that are much smaller than the others, but are shaped in a way that can be easily combined without changing the input and output sizes of certain parts of the model
