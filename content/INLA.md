---
tags:
  - ml
  - bayesian
  - jax
folder: learning
share: true
title: INLA
date created: Friday, May 3rd 2024, 5:22:53 pm
date modified: Friday, May 10th 2024, 5:40:17 pm
---

Approximating the full joint posterior distribution using a Gaussian distribution is inaccurate. An alternative is to **approximate the marginal posterior distribution of some subset of the parameters**, referred to as the marginal Laplace approximation.

Then, integrate out the remaining parameters using another method.

- **Integrated**. Using numerical integration
- **Nested**. Because we need $p(\theta | y)$ to get $p(u|y)$
- **Laplace approximations**. The methods used to obtain parameters for the Gaussian approximation.

## When is it useful?

Wherever we have a **latent Gaussian**. Almost always.

Following [Dan Simpson](https://dansblog.netlify.app/posts/2022-03-22-a-linear-mixed-effects-model/a-linear-mixed-effects-model), we can write a **linear mixed model** as

$$
y = X\beta + Zb + \epsilon,
$$

where 

- Fixed effects. $X$ contain the covariates, $\beta$ is a vector of unknown regression coefficients with a Gaussian prior $\beta \sim N(0, R)$
- Random effects. $Z$ is a known matrix that describes the random effects, $b \sim N(0, \Sigma_b)$

We can put fixed and random effects together $A = [X \vdots Z]$, with a **latent field** (or latent Gaussian component) $u = (\beta^T, b^T)^T$.

This allows us to re-write the model as a **three-stage hierarchical** model

$$
p(y, u, \theta) = p(y | u, \theta) p(u | \theta) p(\theta),
$$

where $\mathbf{u} = (u_1, \ldots, u_N)$ is the latent field, and $\boldsymbol{\mathbf{\theta}} = (\theta_1, \ldots, \theta_m)$ are the hyperparameters. For the latent field, $p(u | \theta)$, we can write

$$
u \mid \theta \sim N(0, Q(\theta)^{-1})
$$

and parametrise by a **precision matrix**

$$
Q(\theta) = \begin{pmatrix} \Sigma_b^{-1} & 0 \\ 0 & R^{-1}\end{pmatrix}.
$$

## What am I approximating?

Marginalise out $u$ and then use [standard inference techniques](https://arxiv.org/abs/2004.12550) on $\theta$.

The posterior can be written as

$$

p(\theta|y) = \frac{p(y|u, \theta) p(u|\theta) p(\theta)}{p(u|y, \theta)} \times \frac{1}{p(y)}

$$

- GMRF specifies $p(u|\theta)$
- prior $p(\theta)$ is supplied
- measurement model gives us $p(y|u,\theta)$
- normalising constant $p(y)$ is ignored since $y$ is fixed

```python
log_prior_fn(params) + log_marginal_likelihood(params, y)
```

The other term in the denominator, $p(u| y, \theta)$, is **estimated** with [Laplace's method](https://en.wikipedia.org/wiki/Laplace%27s_method), $p(u| y, \theta) \approx \mathcal{N}(u^*, \Sigma^*)$ where $u^*$ matches the mode and $[\Sigma^*]^{-1}$ the curvature

The likelihood

$$
p(y|\theta) = \frac{p(y|u, \theta) p(u|\theta)}{p(u|y, \theta)}
$$

```python
# evaluated at x = \hat{x}_0 below
log_laplace_approx = self._conditional_gaussian_approximation(
	x, y, gaussian
).logpdf(x)
log_marginal_likelihood = (    # P(y | params) =
	gaussian.logpdf(x)         # P(x | params)
	+ self.f(x, y).sum()       # * P(y | x, params)
	- log_laplace_approx       # / P(x | y, params)
)
```

### Laplace approximation

Denoting $f(x) = \log(p(y|u))$, we have the following approximation around any given $\hat x$:

$$
\begin{align}
\log(p(x|y, \theta)) &= \log( p(y|x, \theta)) + \log(p(x|\theta)) + \text{const} \\
&= f(x) -\frac 12 (x-\mu)^T Q(x-\mu) + \frac 12\log\det(Q) + \text{const} \\
&\approx f(\hat x) + (x-\hat x)f'(\hat x) + \frac 12 (x-\hat x)^2 f''(\hat x) -\frac 12 (x-\mu)^T Q(x-\mu) + \frac 12\log\det(Q) +
\text{const} \\
&= -\frac 12 x^T(-f''(\hat x) + Q)x + x^T(Q\mu + f'(\hat x) -
\hat xf''(\hat x)) + \frac 12\log\det(Q) + \text{const}
\end{align}
$$

In the last line we group the quadratic and linear terms.

```python
def _conditional_gaussian_approximation(
      self, x, y, unconditional_gaussian
  ) -> gmrf.Gaussian:
    """The gaussian approximation of P(x | y, params) at the given x."""
    fpp = self.fpp(x, y)
    precision = unconditional_gaussian.precision.add_diag(-fpp)
    linear_part = (
	    # self.precision @ self.mean
        unconditional_gaussian.information_vector + 
        self.fp(x, y) -
        x * fpp
    )
    # Gaussian with mean -fpp + precision
    return gmrf.Gaussian(precision, precision.solve(linear_part))
```

The constant is then be determined by using the fact that this must be a probability distribution in $x$ (i.e. use the value of the constant from the log probability function of the gaussian with the same quadratic and linear terms).

Evaluate at $\hat{x}_0 = \mu$ to maximise. Find this $\hat x$ with Newton's method, which amounts to iteratively solving for the maximum in the very same quadratic approximation. Iteratively solve for $\hat x_{i+1}$ in

$$

(Q - f''(\hat x_i))\hat x_{i+1} = Q\mu + f'(\hat x_i) - \hat x_i f''(\hat x_i)

$$

until stabilisation.

```python
def newton_step(x, y, gaussian):
	return self._conditional_gaussian_approximation(x, y, gaussian).mean

fpi = jaxopt.FixedPointIteration(newton_step)
x = fpi.run(gaussian.mean, y, gaussian).params
```

## Where can sparsity help?

Looking at the precision matrix again

$$
Q(\theta) = \begin{pmatrix} \Sigma_b^{-1} & 0 \\ 0 & R^{-1}\end{pmatrix}
$$

- Often we make a Markov assumption on $\Sigma_b^{-1}$, for example in ICAR model in spatial settings where we only assign covariance to nearest neighbours. This makes the precision matrix sparse.
	- This is the **Gaussian Markov Random Field** (GMRF)
- Usually the random effect $b$ size dominates the covariates $\beta$.

This means normally $Q$ is sparse. The four steps that are needed in sparse implementation are:

- Adding to the diagonal. $Q - f''(\hat x_i)$
- Multiplying a vector $Q\mu$
- Solving linear systems (Newton step)
- Computing $\log\det(Q)$

More details in [Dan's blog](https://dansblog.netlify.app/posts/2022-03-22-a-linear-mixed-effects-model/a-linear-mixed-effects-model) and the README of the jax [implementation](https://github.com/geraschenko/gmrfs/tree/main/gmrfs).

Resources used:  

- Adam Howes' [thesis chapter](https://athowes.github.io/thesis/naomi-aghq.html) on (R-)INLA
- Dan Simpson's [blog](https://dansblog.netlify.app/posts/2022-03-22-a-linear-mixed-effects-model/a-linear-mixed-effects-model)
- Junpeng Lao's [attempt](https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/main/Ports/Laplace%20approximation%20in%20pymc3.ipynb) in pymc3
- That Stan team [paper](https://arxiv.org/abs/2004.12550)
- INLA from [scratch](https://stefansiegert.net/inla-project/inla-from-scratch)
- jax [implementation](https://github.com/geraschenko/gmrfs/tree/main/gmrfs)

And the relevant pymc [issue](https://github.com/pymc-devs/pymc/issues/6992).
