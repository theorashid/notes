---
tags:
  - llm
  - ml
folder: learning
share: true
title: structured generation
date created: Sunday, December 1st 2024, 6:02:10 pm
date modified: Sunday, December 8th 2024, 11:23:46 pm
---

Algorithm described in [paper](https://arxiv.org/abs/2307.09702)/[blog](https://blog.dottxt.co/coalescence.html#orga963f9a) and implemented in [outlines](https://dottxt-ai.github.io/outlines/latest/reference/generation/structured_generation_explanation/).

For a sequence of tokens $S_{t} = (s_1, ..., s_{t})$, predict the next token $s_{t+1}$

$$
\begin{align}
\mathbf{\alpha} &= \text{LLM}(S_{t,}\mathbf{\theta}) \\
s_{t+1} &= \text{Categorical}(\mathbf{\alpha})
\end{align}
$$

where we can perform multinomial sampling from the probability distribution of output logits.

The idea of **guided generation** is to **restrict support** of the possibilities of the vocabulary. Forbidding sequences that do not respect the structure of the possible vocabulary.

## naive approach

- apply a boolean mask $m$ that restricts the support of the original distribution
- renormalise output logits
- perform multinomial sampling

$O(N)$ cost of deciding whether each possible token is in restricted vocabulary at each step.

## efficient approach (finite state machine)

Use a finite state machine tuple, $(Q, \Sigma, \delta, q_{0}, F)$, to efficiently **continue the state machine** without reading from the beginning of the growing sample sequence each time.

- $Q$, finite set of states
- $\Sigma$, finite alphabet
- $\delta: Q \times \Sigma \rightarrow Q$, transition function
- $q_0$, start state
- $F$, set of accept states

Need to **pre-process** the vocabulary using the **regular expression**’s FSM and build an index (hash map for mask $m$ with $O(1)$) to find subsequences of FSM that accept each string:

- consider starting in every viable FSM state
- determine the valid next tokens if transition function produces a viable state.
