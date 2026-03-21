---
tags:
  - llm
  - ml
  - neural-network
  - jax
  - python
folder: llm
title: transformer architecture
date created: Sunday, February 4th 2024, 2:01:52 pm
date modified: Tuesday, February 3rd 2026, 5:15:17 pm
share: true
---

## before transformers

Classical approaches to [[./language modelling|language modelling]], such as Markov chain approaches, were not flexible enough to learn nuances the same way neural network can.

Previous neural network architectures had scaling or gradient problems.

[Transformers](https://ravinkumar.com/GenAiGuidebook/language_models/large_language_model_basics.html) are structured in a manner that allows them to be trained readily and stably. Originally introduced in the [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)paper. Best-explained by [3b1b](https://www.youtube.com/watch?v=wjZofJX0v4M).

## high-level transformer architecture

At a high level, the GPT architecture has three sections:

- Text and positional **embeddings**
- A transformer **decoder stack**
- A **projection to vocab** step

High-level architecture (*from [GPT in 60 lines of Numpy](https://jaykmody.com/blog/gpt-from-scratch/). Also see [Transformers in Bare-Metal JAX](https://sdbuchanan.com/blog/jax-2/)*):

```python
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head): # [n_seq] -> [n_seq, n_vocab]
	# token + positional embeddings
	x = wte[inputs] + wpe[range(len(inputs))] # [n_seq] -> [n_seq, n_embd]
	
	# decoder stack: forward pass through n_layer transformer blocks
	for block in blocks:
		x = transformer_block(x, **block, n_head=n_head) # [n_seq, n_embd] -> [n_seq, n_embd]

	# projection to vocab
	x = layer_norm(x, **ln_f) # [n_seq, n_embd] -> [n_seq, n_embd]
	return x @ wte.T # [n_seq, n_embd] -> [n_seq, n_vocab]
```

### inputs

The inputs to the model are tokens. These are created by using a tokeniser to break a string down and map to integers.

Token are converted to vectors via an embedding matrix.

```python
# wte converts token to vector via embedding matrix
# wpe encode positional information into our inputs
x = wte[inputs] + wpe[range(len(inputs))]
```

### decoder architecture

```python
# decoder stack: forward pass through n_layer transformer blocks
for block in blocks:
	x = transformer_block(x, **block, n_head=n_head) # [n_seq, n_embd] -> [n_seq, n_embd]
```

where 

```python
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head): # [n_seq, n_embd] -> [n_seq, n_embd]
	# multi-head causal self attention
	x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head) # [n_seq, n_embd] -> [n_seq, n_embd]

	# position-wise feed forward network – standard dense network
	x = x + ffn(layer_norm(x, **ln_2), **mlp) # [n_seq, n_embd] -> [n_seq, n_embd]

	return x
```

Layer normalisation ensures that the inputs for each layer are always within a consistent range, which helps speed up and stabilise the training process.

`mha` are [[./multi-headed self-attention|multi-headed self-attention]] blocks, which facilitate the communication between the inputs. Nowhere else in the network does the model allow inputs to *see* each other.

### outputs

```python
# projection to vocab
# reusing the embedding matrix `wte` for the projection
# output logits rather than softmax-transformed
x = layer_norm(x, **ln_f) # [n_seq, n_embd] -> [n_seq, n_embd]
return x @ wte.T # [n_seq, n_embd] -> [n_seq, n_vocab]
```

Normally apply softmax transform (convert set of real numbers to probabilities) over the last axis of the input. But softmax is monotonic and logits are more stable so we output them.

See [[./language modelling|language modelling]] for how to predict.

Details on [scaling LLMs on TPUs](https://jax-ml.github.io/scaling-book/) via tensor sharding.
