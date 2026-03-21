---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: online learning using ssm
date created: Sunday, February 4th 2024, 4:26:15 pm
date modified: Monday, March 16th 2026, 1:38:24 pm
share: true
---

## online linear regression

Sequential Bayesian inference for the [parameters of a linear regression model](https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/kf_linreg.html). Treat the **parameters of the model as the unknown hidden states** ($\mathbf{z}_t = \mathbf{\theta}_t^T$). The parameters are updated with each new measurement.

This is a [[./linear gaussian ssm|linear gaussian ssm]] with $\mathbf{F} = \mathbb{I}$, $\mathbf{Q} = 0$, $\mathbf{H}_t = \mathbf{x}_t^T$, $\mathbf{R} = \sigma^2$.

$$
\begin{align}
\mathbf{\theta}_t &= \mathbf{\theta}_{t-1} \\
y_t &= \mathbf{x}_t^{T} \mathbf{\theta}_t + r_{t}, \qquad r_t \sim N(0, \sigma^2)
\end{align}
$$

## online learning for a neural network

Sequential Bayesian inference for the [parameters of a multi-layer dense neural network](https://probml.github.io/dynamax/notebooks/nonlinear_gaussian_ssm/ekf_mlp.html) (perceptron) for regression. Treat the parameters (weights and biases) of the network as the unknown hidden states.

$$
\begin{align}
\mathbf{\theta}_t &= \mathbf{\theta}_{t-1} \\
y_t &= h(\theta_t, x_t) + r_{t}, \qquad r_t \sim N(0, \sigma^2)
\end{align}
$$

This is a [[./nonlinear gaussian ssm|nonlinear gaussian ssm]], where $h$ is the **nonlinear observation model defined by a neural network**. Add a small amount of Gaussian drift for numerical stability $\mathbf{q}_t \sim N(0, 0.01 \mathbb{I})$. Inference can be done using the extended Kalman filter.

The parameters are updated with each new measurement ([video](https://github.com/probml/probml-data/blob/main/data/ekf_mlp_demo.mp4) of training).

The model can be generalised to other likelihoods, such as Bernoulli for classification, which is then a [[./generalised gaussian ssm|generalised gaussian ssm]].
