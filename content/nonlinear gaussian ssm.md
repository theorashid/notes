---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: nonlinear gaussian ssm
date created: Sunday, February 4th 2024, 4:22:44 pm
date modified: Sunday, February 25th 2024, 7:58:38 pm
share: true
---

Moving from [[./linear gaussian ssm|linear gaussian ssm]], we relax the assumptions of linear dynamics ($\mathbf{F}_t \mathbf{z}_t \rightarrow \mathbf{f}(\mathbf{z}_t)$) and linear observation models ($\mathbf{H}_{t}\mathbf{z}_t \rightarrow \mathbf{h}(\mathbf{z}_t)$).

$$
\begin{align}
\mathbf{z}_{t+1} &= \mathbf{f}(\mathbf{z}_{t}, \mathbf{u}_{t}) + \mathbf{q}_t, \qquad \mathbf{q}_t \sim \mathcal{N}(0, \mathbf{Q}_{t}) \\
\mathbf{y}_t &= \mathbf{h}(\mathbf{z}_{t}, \mathbf{u}_{t}) + \mathbf{r}_t, \qquad  \mathbf{r}_t \sim \mathcal{N}(0, \mathbf{R}_t)
\end{align}
$$

This model can be used to [track an object performing nonlinear motion](https://probml.github.io/dynamax/notebooks/nonlinear_gaussian_ssm/ekf_ukf_spiral.html) or in [[./online learning using ssm|online learning using ssm]] for [neural networks](https://probml.github.io/dynamax/notebooks/nonlinear_gaussian_ssm/ekf_mlp.html). 
