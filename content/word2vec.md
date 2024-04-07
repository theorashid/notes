---
tags:
  - llm
  - ml
  - neural-network
folder: llm
title: word2vec
date created: Saturday, February 17th 2024, 1:24:31 pm
date modified: Monday, April 1st 2024, 9:31:43 pm
share: true
---

**Encode semantics** in a meaningful way by **representing words in a vector space** (see [3b1b](https://www.youtube.com/watch?v=wjZofJX0v4M)). For example,

> king - man + woman = queen

In a [simplified set up](https://www.youtube.com/watch?v=viZrOnJclY0), we can **train a neural network** so it correctly **predicts the** next **word** in the corpus:

$$

\begin{align*}

\underbrace{
\begin{pmatrix}
0 & 0 & 1 & 0 & 0 \\
\end{pmatrix}}_{\text{one hot current word}}
\cdot
\mathbf{E}

\rightarrow

\underbrace{

\begin{pmatrix}

... \\

\end{pmatrix}

}_{\text{embeddings}}
\cdot
\mathbf{W}
\rightarrow_{\text{softmax}}


\underbrace{
\begin{pmatrix}
0 & 1 & 0 & 0 & 0 \\
\end{pmatrix}}_{\text{predicted next word}}

\end{align*}

$$

Dimensions:

- One-hot current word has shape `(1, N_words)`
- **Embeddings matrix**, $\mathbf{E}$, has shape `(N_words, N_dims)`
- Latent embeddings have shape `(1, N_dims)`
- Second trainable weight matrix, $\mathbf{W}$, has length `(N_dims, N_words)`
- Predicted next word has shape `(1, N_words)`

We use **cross-entry loss** to compare the predicted word to the true word. We can [vectorise this over the entire corpus](https://jaketae.github.io/study/word2vec/) by having the current word shape as `(length_corpus, N_words)`.

[word2vec](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) (2013) has an embedding dimension (`N_dims`) dimension of around 300 dimensions for word vectors. OpenAI Ada has 1536.

word2vec **uses more context than** just the **next word**:

- Continuous bag of words (CBOW) uses the surrounding context words to predict the centre word (maximise the probability of every context word given every corresponding centre word)
- Skip-gram uses the word in the middle to predict surrounding words
