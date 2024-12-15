---
tags:
  - bayesian
  - python
folder: learning
share: true
title: copula
date created: Saturday, December 14th 2024, 12:01:59 pm
date modified: Saturday, December 14th 2024, 2:32:13 pm
---

[Copula](https://twiecki.io/blog/2018/05/03/copulas/) is a multivariate distribution $C(U_1,U_2,....,U_n)$ such that marginalising gives $U_{i} \sim \text{Uniform}(0,1)$.

Used to describe some complex distribution $P(a, b)$ where $a$ and $b$ are correlated using only the **marginal distributions**. For example, if we don't have access to individual-level microdata but we have different slices of a population by deprivation or age.

- we observe the marginals
- we do not observe the correlated joint distribution
	- *Gaussian* copula (assumed below): use correlated MVN space to model correlation between variables

Make use of the [[./probability integral transform|probability integral transform]] to transform (CDF) from correlated joint distribution to uniform marginals to (inverse CDF) correlated (arbitrary) marginal distributions and back.

## copula data generation

- MVN space: simulate MVN, CDF -> 
- uniform marginals, inverse CDF -> 
- observation space: transformed (correlated) marginal distributions

```python
# simulate correlated variables a, b
x = multivariate_normal(mu, cov).rvs(n_samples, random_state=rng)
a_norm = x[:, 0]
b_norm = x[:, 1]

# transform: correlated space -> uniform space
# apply CDF transform, correlated normal -> uniform marginals
a_unif = norm(loc=0, scale=1).cdf(a_norm)
b_unif = norm(loc=0, scale=1).cdf(b_norm)

# transform: uniform space -> observation space
# apply(arbitrary) inverse CDF to get the marginal distributions we want
a = θ["a_dist"].ppf(a_unif) # e.g. stats.gumbel_l()
b = θ["b_dist"].ppf(b_unif) # e.g. stats.beta(a=10, b=2)

# joint distribution of θ["a_dist"].rvs(10000) and θ["b_dist"].rvs(10000) not correlated
# joint distribution of a and b is correlated
```

## copula inference process

- observation space, CDF ->
- uniform marginals, inverse CDF -> 
- MVN space: simulate MVN

```python
# observation -> uniform space
a1 = θ["a_dist"].cdf(a)
b1 = θ["b_dist"].cdf(b)

# uniform -> MvNormal space
a2 = norm(loc=0, scale=1).ppf(a1)
b2 = norm(loc=0, scale=1).ppf(b1)
```

An [example pymc model](https://www.pymc.io/projects/examples/en/latest/howto/copula-estimation.html):

```python
# 1. estimate marginals by observing data
coords = {"obs_id": np.arange(len(a))}
with pm.Model(coords=coords) as marginal_model:
    a_mu = pm.Normal("a_mu", mu=0, sigma=1)
    a_sigma = pm.Exponential("a_sigma", lam=0.5)
    pm.Normal("a", mu=a_mu, sigma=a_sigma, observed=a, dims="obs_id")

    b_scale = pm.Exponential("b_scale", lam=0.5)
    pm.Exponential("b", lam=1 / b_scale, observed=b, dims="obs_id")

# 2. transform marginals to uniform space based on knowledge of data generating process and inferred prarameters from step 1
# 3. transform uniform marginals to MVN space to create `data`
data, a_mu, a_sigma, b_scale = transform_data(marginal_idata)

# 4. infer correlation structure by observing this MVN `data`
coords.update({"param": ["a", "b"], "param_bis": ["a", "b"]})
with pm.Model(coords=coords) as copula_model:
    # prior on covariance of the multivariate normal
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol",
        n=2,
        eta=2.0,
        sd_dist=pm.Exponential.dist(1.0),
        compute_corr=True,
    )
    cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("param", "param_bis"))
    pm.MvNormal("N", mu=0.0, cov=cov, observed=data, dims=("obs_id", "param"))
```
