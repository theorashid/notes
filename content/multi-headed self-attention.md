---
tags:
  - llm
  - ml
  - neural-network
folder: llm
title: multi-headed self-attention
date created: Sunday, February 4th 2024, 2:12:04 pm
date modified: Sunday, February 25th 2024, 7:59:24 pm
share: true
---

## attention

The model figures out what parts of its input it should “care about”. The [attention](https://lilianweng.github.io/posts/2018-06-24-attention/) mechanism creates shortcuts between the context vector and the **entire source input**.

## self-attention

Attention mechanism relating different positions of a **single sequence** in order to compute a representation of the same sequence. For example, a LSTM learns the correlation between the current words and the previous part of the sentence.

The core parts of the attention mechanism are  **key**-**value** $(\mathbf{K}, \mathbf{V})$ pairs (encoder hidden states, dimension $n$), and a **query** ($\mathbf{Q}$, dimension $m$), which is compressed from the previous output.

We define similarity between the query and each of the key-value pairs using a **scaled-dot product**.

$\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

We then extract the hash key with maximum weight to get the next output.

## multi-headed self-attention

Performing `n_head` separate attention computations (in parallel). Different heads can learn different things about the sentence (ensembling always helps).
