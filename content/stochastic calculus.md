---
tags:
  - bayesian
  - ml
folder: learning
share: true
title: stochastic calculus
date created: Tuesday, March 17th 2026, 7:59:05 pm
date modified: Tuesday, March 31st 2026, 6:00:26 pm
---

*Simplified from [Ji-Ha Kim's blog post](https://jiha-kim.github.io/posts/introduction-to-stochastic-calculus/).*

## Brownian motion

The **Wiener process** $W(t)$ describes the position of a particle undergoing **Brownian motion**:

$$W(t) \sim N(0, t)$$

An increment over $(s, t)$ is $\Delta W(s, t) \sim N(0, t - s)$, and in the infinitesimal limit:

$$\Delta W(t, t + dt) \sim N(0, dt) = \sqrt{dt}\, N(0, 1)$$

The sample path $t \mapsto W(t)$ is continuous but **nowhere differentiable**. The rate of change over a small interval $dt$:

$$\lim_{dt \to 0} \frac{\Delta W(t, t+dt)}{dt} = \lim_{dt \to 0} \frac{1}{\sqrt{dt}} N(0,1)$$

which diverges. This rules out standard calculus.

## Itô calculus

The **stochastic differential** is

$$dW = \sqrt{dt}\, N(0, 1)$$

with $\mathbb{E}[dW] = 0$ and $\text{Var}(dW) = \mathbb{E}[(dW)^2] = dt$.

### $(dW)^2 \approx dt$

$\mathbb{E}[(dW)^2] = dt$ and $\text{Var}[(dW)^2] = 2\,dt^2$, negligible as $dt \to 0$. So $(dW)^2 \approx dt$ in the mean-square sense. Unlike ordinary calculus where $(dx)^2$ vanishes, $(dW)^2$ is on the same scale as $dt$.

### Itô's lemma

For $f(t, W(t))$, ordinary calculus gives $df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial W} dW$. Brownian motion's roughness requires a second-order correction. Taylor-expand:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial W} dW + \frac{1}{2} \frac{\partial^2 f}{\partial W^2} (dW)^2 + \text{higher order terms}$$

As $dt \to 0$: $dt^2$ and $dt\,dW$ vanish, but $(dW)^2 \approx dt$ stays. This gives **Itô's lemma** (the stochastic chain rule):

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial W} dW + \frac{1}{2} \frac{\partial^2 f}{\partial W^2} dt$$

The extra $\frac{1}{2} \frac{\partial^2 f}{\partial W^2} dt$ captures the curvature from Brownian motion.

### Itô's lemma for a general SDE

For $dX = a\,dt + b\,dW$ and $f(t, X(t))$, since $dX = O(dW)$ keep terms up to $dX^2 = O(dW^2)$:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial X} dX + \frac{1}{2} \frac{\partial^2 f}{\partial X^2} dX^2$$

$$= \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial X}(a\,dt + b\,dW) + \frac{1}{2} \frac{\partial^2 f}{\partial X^2}(a\,dt + b\,dW)^2$$

$$= \left(\frac{\partial f}{\partial t} + a\,\frac{\partial f}{\partial X} + \frac{1}{2} b^2 \frac{\partial^2 f}{\partial X^2}\right) dt + b\,\frac{\partial f}{\partial X}\,dW$$

## Stochastic differential equations

**SDEs** blend deterministic behaviour with stochastic noise:

$$dX(t) = a(t, X(t))\, dt + b(t, X(t))\, dW(t)$$

$a$ is the **drift** (average direction), $b$ is the **diffusion** (strength of random jitter). If $b = 0$, this is a standard ODE; if $a = 0$, pure scaled Brownian motion.

### Constant drift and diffusion

Setting $a = \mu$ and $b = \sigma$ (constants, independent of $t$ and $X$):

$dX(t) = \mu\,dt + \sigma\,dW(t)$ with $X(0) = 0$:

$$X(t) = \int_0^t \mu\,ds + \int_0^t \sigma\,dW(s) = \mu t + \sigma W(t) \sim N(\mu t, \sigma^2 t)$$

A process drifting linearly with noise spreading over time.

### Geometric Brownian motion

For systems where changes scale with size (e.g. stock prices):

$$dS(t) = \mu\,S(t)\,dt + \sigma\,S(t)\,dW(t)$$

The percentage change $dS/S = \mu\,dt + \sigma\,dW$ has a trend and randomness. To solve, apply Itô's lemma with $f = \ln S$, so $f_t = 0$, $f_S = 1/S$, $f_{SS} = -1/S^2$:

$$d(\ln S) = \frac{1}{S}(\mu S\,dt + \sigma S\,dW) + \frac{1}{2}\left(-\frac{1}{S^2}\right)\sigma^2 S^2\,dt = \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma\,dW$$

Integrate from $0$ to $t$:

$$S(t) = S(0) \exp\left(\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W(t)\right)$$

The drift is adjusted by $-\frac{1}{2}\sigma^2$ from the Itô correction. This underlies the **Black-Scholes** model.

## Itô vs Stratonovich

The **Itô integral** evaluates the integrand at the **left endpoint** of each interval -- non-anticipating (only uses information up to the current time). Natural in finance.

**Stratonovich calculus** evaluates at the **midpoint** instead. This preserves the ordinary chain rule with no second-derivative correction, because the midpoint evaluation absorbs the $\frac{1}{2} b_X\, b$ correction into the integral definition.

Conversion between Itô drift $a$ and Stratonovich drift $\tilde{a}$:

$$a = \tilde{a} + \frac{1}{2} b_X\, b$$

where $b_X = \partial b / \partial X$. The diffusion term is the same in both.

Stratonovich suits physical systems where noise has slight smoothness or continuity -- the **Wong-Zakai theorem** shows that smooth noise approximations converge to Stratonovich SDEs in the limit. Itô dominates in finance for its non-anticipating, martingale-friendly properties.

## SDEs and state space models

SDEs are the continuous-time backbone of the [[./continuous-discrete state space model|continuous-discrete state space model]]: the latent state evolves via an SDE (drift = dynamics, diffusion = noise), but observations arrive at discrete times. The SDE must be **discretised** (e.g. Euler-Maruyama, or exactly via matrix exponentials for linear SDEs) before standard filters can be applied.
