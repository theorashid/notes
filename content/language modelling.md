---
tags:
  - llm
  - ml
folder: llm
title: language modelling
date created: Sunday, February 4th 2024, 1:52:36 pm
date modified: Sunday, February 25th 2024, 7:59:10 pm
share: true
---

*Taken from Ravin Kumar's [GenAI Guidebook](https://ravinkumar.com/GenAiGuidebook/language_models/small_language_model_basics.html#wikipedia).*

The task of predicting the next logical word in a sequence is called **language modelling**.

Formally,

> A language model is a **probability distribution over sequences of words**.

Generating text given a prompt is also referred to as **conditional generation**. In classical language modelling, we are often only concerned with predicting the next word. For a text to text model, the intuition is, if I have to predict a word, what is the most likely one. For example, when fixing a typo. "name my is..."

$$ P(\text{next word = my} | \text{current word = name}) \approx 0 $$

This configuration is very unlikely and so should be changed.

Taking the token with the highest probability as our prediction is known as **greedy** sampling. We can introduce **stochasticity** to our generations by sampling from the probability distribution instead of being greedy. Increasing *temperature* makes our model take more risks when selecting from the probability distribution and thus be more *creative*.

We are often not only trying to predict given only the previous word. This is where more complex, modern neural networks like the [[./transformer architecture|transformer architecture]] come in.
