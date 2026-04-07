---
tags:
  - ssm
  - ml
  - bayesian
folder: ssm
title: continuous-discrete state space model
date created: Tuesday, March 31st 2026, 5:56:18 pm
date modified: Wednesday, April 1st 2026, 12:01:37 pm
share: true
---

A **continuous-discrete** [[./state space model|state space model]] has latent states that evolve in *continuous time* via a [[./stochastic calculus|stochastic differential equation]] (SDE), but observations arrive at *discrete* times.

## model

This is the general SDE from [[./stochastic calculus#Stochastic differential equations|stochastic calculus]], $dX = a\,dt + b\,dW$, applied to a multivariate latent state $\mathbf{z}(t) \in \mathbb{R}^{d_z}$. The scalar drift $a$ becomes a vector-valued function $\boldsymbol{\mu}$, and the scalar diffusion $b$ becomes a matrix $\mathbf{L} \in \mathbb{R}^{d_z \times d_w}$ multiplying a $d_w$-dimensional Brownian motion:

$$d\mathbf{z}(t) = \boldsymbol{\mu}(\mathbf{z}(t), \mathbf{u}(t), t)\, dt + \mathbf{L}(\mathbf{z}(t), \mathbf{u}(t), t)\, d\mathbf{W}(t)$$

$\boldsymbol{\mu}$ is the **drift** (deterministic tendency), $\mathbf{L}$ is the **diffusion coefficient** (noise scaling). Observations at times $t_1, t_2, \ldots$:

$$\mathbf{y}_{t_k} \sim p(\mathbf{y} | \mathbf{z}(t_k), \mathbf{u}(t_k), t_k)$$

| Component           | Continuous-time                                                         | Discrete-time equivalent                                 |
| ------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- |
| Dynamics            | $\boldsymbol{\mu}(\mathbf{z}, t)$ – rate of change                      | $\mathbf{f}(\mathbf{z}_t)$ – next state                  |
| Noise               | $\mathbf{L}(\mathbf{z}, t)$ – diffusion coefficient, shape $(d_z, d_w)$ | $\mathbf{Q}_t$ – per-step covariance, shape $(d_z, d_z)$ |
| Noise relation      | Instantaneous covariance $\mathbf{L}\mathbf{L}^T\, dt$                  | Fixed covariance $\mathbf{Q}_t$ per step                 |
| Brownian motion dim | $d_w$ (can differ from $d_z$)                                           |                                                          |

The diffusion $\mathbf{L}$ has shape $(d_z, d_w)$ where $d_w$ is the Brownian motion dimension. When $d_w < d_z$, noise enters only some state dimensions. Under Euler-Maruyama discretisation, the per-step covariance is $\mathbf{Q} = \mathbf{L}\mathbf{L}^T\, \Delta t$ (for the linear case, the exact $\mathbf{Q}$ involves an integral – see [[continuous-discrete state space model#Exact discretisation (linear case)|below]]).

### special cases

**Linear SDE (LTI).** $d\mathbf{z} = (\mathbf{A}\mathbf{z} + \mathbf{B}\mathbf{u} + \mathbf{b})\, dt + \mathbf{L}\, d\mathbf{W}$ with linear-Gaussian observations. Exact discretisation via matrix exponentials gives a [[./linear gaussian ssm|linear gaussian ssm]] – see [[./state space gaussian process|state space gaussian process]].

**ODE (no noise).** $\mathbf{L} = 0$: a deterministic system observed with noise. Drift is a plain vector field, the state follows a trajectory. Standard ODE solvers (e.g. Tsit5 in [diffrax](https://docs.kidger.site/diffrax/)) integrate between observation times.

**Potential-based drift.** $\boldsymbol{\mu}(\mathbf{z}) = -\nabla_\mathbf{z} V(\mathbf{z})$ for a scalar potential $V$. This is **Langevin dynamics** – the state descends an energy landscape with stochastic perturbation. Common in physics-inspired models.

## discretisation

To apply discrete-time filters ([[./kalman filtering and smoothing|Kalman]], EKF, particle), we need the transition density $p(\mathbf{z}_{t_{k+1}} | \mathbf{z}_{t_k})$. This requires discretising the SDE.

### Euler-Maruyama

First-order Itô-Taylor approximation: integrate the SDE and freeze $\boldsymbol{\mu}$, $\mathbf{L}$ at the left endpoint (stochastic forward Euler). The noise term becomes $\mathbf{L}\,\Delta W$ (from $dW = \sqrt{dt}\,\mathcal{N}(0,1)$ ([[./stochastic calculus#Itô calculus|Itô]])), giving covariance $\mathbf{L}\,\mathbb{E}[\Delta W \Delta W^T]\,\mathbf{L}^T = \mathbf{L}\mathbf{L}^T \Delta t$. For time step $\Delta t = t_{k+1} - t_k$:

$$\mathbf{z}_{t_{k+1}} \sim \mathcal{N}\!\left(\mathbf{z}_{t_k} + \boldsymbol{\mu}(\mathbf{z}_{t_k}, t_k)\,\Delta t,\; \mathbf{L}\mathbf{L}^T \Delta t\right)$$

Simple and effective for small $\Delta t$, biased for large steps. The result is a [[./nonlinear gaussian ssm|nonlinear gaussian ssm]] transition that any discrete-time filter can consume.

### exact discretisation (linear case)

For $d\mathbf{z} = \mathbf{A}\mathbf{z}\, dt + \mathbf{L}\, d\mathbf{W}$, the solution is exact via variation of constants (the matrix analogue of $dx/dt = ax \implies x(t) = e^{at}x(0)$):

$$\mathbf{z}_{t_{k+1}} = \exp(\mathbf{A}\,\Delta t)\,\mathbf{z}_{t_k} + \int_{t_k}^{t_{k+1}} \exp(\mathbf{A}(t_{k+1} - s))\,\mathbf{L}\,dW(s)$$

The integral is a Gaussian (deterministic integrand times $dW$), giving $\mathbf{z}_{t_{k+1}} \sim \mathcal{N}(\mathbf{F}\,\mathbf{z}_{t_k},\, \mathbf{Q})$ where:

$$
\begin{align}
\mathbf{F} &= \exp(\mathbf{A}\,\Delta t) \\
\mathbf{Q} &= \int_0^{\Delta t} \exp(\mathbf{A}\,s)\, \mathbf{L}\mathbf{L}^T\, \exp(\mathbf{A}\,s)^T\, ds
\end{align}
$$

An exact [[./linear gaussian ssm|linear gaussian ssm]] – no approximation error. This is the basis for [[./state space gaussian process|state-space GPs]]. Nonlinear SDEs have no closed-form solution, requiring Euler-Maruyama or numerical solvers.

## continuous-discrete filtering

The filter alternates between two phases:

1. **Predict (continuous).** Propagate the filtering distribution forward from $t_k$ to $t_{k+1}$. For Gaussian filters, this means integrating **moment ODEs** for the mean and covariance (ODEs, not the SDE itself).
2. **Update (discrete).** Incorporate the observation $\mathbf{y}_{t_{k+1}}$ using the standard Kalman/EKF/particle update.

### Moment ODEs

For any linear SDE $d\mathbf{z} = \mathbf{A}\mathbf{z}\,dt + \mathbf{L}\,d\mathbf{W}$, the moment ODEs are read off directly from $\mathbf{A}$ and $\mathbf{L}$:

$$\frac{d\mathbf{m}}{dt} = \mathbf{A}\,\mathbf{m}, \qquad \frac{d\mathbf{P}}{dt} = \mathbf{A}\,\mathbf{P} + \mathbf{P}\,\mathbf{A}^T + \mathbf{L}\mathbf{L}^T$$

The covariance is the continuous Lyapunov equation.

### Example: scalar OU process

$dz = -\lambda z\, dt + \sigma, dW$, observed as $y_k = z(t_k) + r_k$, $r_k \sim \mathcal{N}(0, R)$. The filter maintains $p(z(t) | y_{1:k}) = \mathcal{N}(m(t), P(t))$.

**Predict:** read off $A = -\lambda$, $L = \sigma$ and substitute into the moment ODEs:

$$\frac{dm}{dt} = -\lambda\, m, \qquad \frac{dP}{dt} = -2\lambda\, P + \sigma^2$$

For this linear case, the solutions are closed-form and recover the exact discretisation. For nonlinear drift, these ODEs have no closed form and require a numerical ODE solver.

**Update:** at $t_{k+1}$, apply the standard [[./kalman filtering and smoothing|Kalman update]] with $y_{k+1}$.

## Simulation

Forward sampling from a continuous-time model requires an SDE solver:

- **ODE** ($\mathbf{L} = 0$): standard adaptive solvers (Tsit5, Dopri5)
- **SDE** ($\mathbf{L} \neq 0$): stochastic solvers (Euler-Maruyama, Heun) with a Brownian motion source
Implementations:
- [cd-dynamax](https://github.com/HD-UQ/cd_dynamax): continuous-discrete filters (KF, EKF, UKF, EnKF, DPF)
- [diffrax](https://docs.kidger.site/diffrax/): ODE/SDE solvers in JAX

*See Särkkä & Solin, [Applied Stochastic Differential Equations](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf) (2019), Chapters 9-12.*
