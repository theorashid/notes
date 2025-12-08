---
tags:
  - llm
  - ml
  - neural-network
folder: llm
title: multi-headed self-attention
date created: Sunday, February 4th 2024, 2:12:04 pm
date modified: Tuesday, November 25th 2025, 5:16:45 pm
share: true
---

## attention

The model figures out what parts of its input it should “care about”. The [attention](https://lilianweng.github.io/posts/2018-06-24-attention/) mechanism creates shortcuts between the context vector and the **entire source input**.

## self-attention

Attention mechanism relating different positions of a **single sequence** in order to compute a representation of the same sequence. For example, a LSTM learns the correlation between the current words and the previous part of the sentence.

The core parts of the attention mechanism are  **key**-**value** $(\mathbf{K}, \mathbf{V})$ pairs (encoder hidden states, dimension $n$), and a **query** ($\mathbf{Q}$, dimension $m$), which is compressed from the previous output.

We define **similarity** between the query and each of the key-value pairs using a **scaled-dot product**.

$\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

We then extract the hash key with maximum weight to get the next output.

Andrej Kaparthy in [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) ([repo](https://github.com/karpathy/ng-video-lecture)) implements a single head of self-attention as

```python
# input of size (batch, time-step, channels)
# output of size (batch, time-step, head size)
B, T, C = x.shape

# learnable parameters
# apply to embedding
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # (B, T, head_size)
q = query(x) # (B, T, head_size)

# compute attention scores ("affinities")
# how much is each word relevant to each other
wei = q @ k.transpose(-2,-1) # (B, T, head_size) @ (B, hs, T) -> (B, T, T)
tril = tril(ones(T, T)) # lower diagonal of ones
wei = wei.masked_fill(tril == 0, float("-inf")) # (B, T, T), get rid of upper diagonal
# this prevents later tokens from influencing earlier ones during training
wei = F.softmax(wei, dim=-1) # (B, T, T), apply along columns

# perform the weighted aggregation of the values
v = value(x) # (B, T, head_size)
out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
```

The vector `x` contains the private information about the token:

- `query(x)` is what I'm interested in, and I'm asking the other `x`
- `key(x)` is what I have
- `value(x)`, if you find me interesting (dot product, high affinity), here is what I will communicate to you

All best described by [3b1b](https://www.youtube.com/watch?v=eMlx5fFNoYc).

## multi-headed self-attention

Performing `n_head` separate attention computations (in parallel). Different heads can learn different things about the sentence (ensembling always helps).
