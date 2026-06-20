---
tags:
  - bayesian
folder: learning
share: true
title: discrete choice models
date created: Monday, June 8th 2026, 4:44:30 pm
date modified: Friday, June 19th 2026, 5:37:25 pm
---

*See [Train Ch.3](https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf) and [PyMC example](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-discrete-choice_models.html). Thanks to (mostly) [Paula Navarrete Díaz](https://cheerstopaula.github.io/site/home.html) for writing these notes.*

## setup

Decision maker $n$ faces $J$ alternatives. Utility of alternative $i$ for person $n$:

$$U_{ni} = V_{ni} + \varepsilon_{ni}$$

- $V_{ni}$ — **representative utility** (observed, parameterised: $V_{ni} = \beta^T x_{ni}$)
- $\varepsilon_{ni}$ — **unobserved** (random from researcher's perspective)

$n$ chooses $i$ iff $U_{ni} > U_{nj}$ for all $j \neq i$.

$$P_{ni} = P(\varepsilon_{nj} - \varepsilon_{ni} < V_{ni} - V_{nj} \quad \forall j \neq i)$$

Only the *difference* in observed utility, $V_{ni} - V_{nj}$, matters, not absolute values. Same for $\varepsilon$.

## logit

Assume $\varepsilon_{ni}$ are i.i.d. **Gumbel** (type I extreme value):

$$F(\varepsilon_{ni}) = e^{-e^{-\varepsilon_{ni}}} \qquad f(\varepsilon_{ni}) = e^{-\varepsilon_{ni}} e^{-e^{-\varepsilon_{ni}}}$$

**Gumbel** is an extreme value distribution — natural for argmax (largest of many unobserved factors). The difference of two Gumbels is logistic:

$$F_{\varepsilon_{ni} - \varepsilon_{nj}}(t) = \int f(\varepsilon_{ni}) F(\varepsilon_{ni} - t) \; d\varepsilon_{ni} = \frac{1}{1 + e^{-t}}$$

which is the standard logistic CDF. This gives closed-form choice probabilities, unlike normal errors (probit, which needs simulation).

**Derivation of $P_{ni}$.** Condition on $\varepsilon_{ni}$, $i$ is chosen if every other $\varepsilon_{nj} < \varepsilon_{ni} + V_{ni} - V_{nj}$. Since $\varepsilon_{nj}$ are independent Gumbel:

$$P_{ni} \mid \varepsilon_{ni} = \prod_{j \neq i} e^{-e^{-(\varepsilon_{ni} + V_{ni} - V_{nj})}}$$

Integrate (*Train 3.10*) over all $\varepsilon_{ni}$ weighted by its density:

$$P_{ni} = \frac{e^{V_{ni}}}{\sum_j e^{V_{nj}}}$$

i.e. **softmax** over representative utilities.

## IIA and limitations

**Independence of irrelevant alternatives.** The ratio $P_{ni} / P_{nk} = e^{V_{ni}} / e^{V_{nk}}$ depends only on $i$ and $k$ — not on any other alternative.

Follows directly from i.i.d. assumption — $\varepsilon$ have no correlation structure across alternatives.

i.i.d. Gumbel **fails with correlated alternatives** (close substitutes), taste variation across individuals (random coefficients needed), repeated choices / panel data (unobserved individual effects). These require nested logit (correlated errors within nests), mixed logit (random coefficients, $\beta$ varies across $n$), or normal errors (probit (multivariate normal $\varepsilon)$).

## interpretation

The point where an increase in $V_{ni}$ has the greatest effect on $P_{ni}$ is when $P_{ni} \approx 0.5$. At high probabilities further increases in representative utility have little effect on the choice probability. Marginal effect:

$$\frac{\partial P_{ni}}{\partial V_{ni}} = P_{ni}(1 - P_{ni})$$

which is maximised at $P_{ni}=0.5$.

## reconstructing the counterfactual

We only observe the decision maker's choice from a constrained set but want counterfactual propensities over the full set. e.g. true choice set $\{A, B, C, D, E\}$ but only $\{C, D, E\}$ shown; $D$ chosen. What are the propensities under the full set?

In general: full set $M$; shown set $C \subset M$. Decision maker chooses $c \in C$. Unshown alternatives $\bar{C} = M \setminus C$.

The revealed-but-rejected alternatives $C \setminus \{c\}$ are like Monty Hall doors opened to reveal goats — except here the mass flows *to* the chosen door rather than away from it (Backwards Monty Hall). $c$ absorbs all probability mass from the shown-but-not-chosen alternatives $C \setminus \{c\}$. $\bar{C}$ (unshown) retain their ex-ante probability. Consequence of Gumbel max-stability.

### counterfactual posterior

Define the ex-ante softmax (as above) over the full set:

$$P(y=k) = \frac{e^{V_k}}{\sum_{j \in M} e^{V_j}}$$

Conditioning on the observed choice $c$ and the shown-but-not-chosen alternatives $C \setminus \{c\}$ being rejected:

$$P(y=k \mid \tilde{y}=c) = 0 \qquad (k \in C \setminus \{c\})$$

$$P(y=k \mid \tilde{y}=c) = P(y=k) \qquad (k \in \bar{C})$$

$$P(y=c \mid \tilde{y}=c) = \textstyle\sum_{h \in C} P(y=h)$$

The shown-but-not-chosen alternatives drop to zero; their mass transfers entirely to $c$. The unshown alternatives $\bar{C}$ are unaffected — their conditional probability equals their ex-ante probability. This is a special property of the Gumbel: the distribution of maxima over disjoint subsets factorises (max-stability), i.e. the joint probability splits into the product of marginals — so knowing $c$ beat $C \setminus \{c\}$ tells you nothing about how $c$ compares to $\bar{C}$.

When $C = M$ (unconstrained), $\bar{C} = \varnothing$ and $P(y=c \mid \tilde{y}=c) = 1$.

#### proof via Gumbel max-stability

Intuitively: max-stability means the maximum of Gumbel variables is itself Gumbel. Splitting the choice set into $\bar{C}$ and $C$, the events "unshown $k$ is overall best" and "shown $c$ is best among shown" involve maxima over disjoint sets. Max-stability makes these factorise ($P(y=k \land \tilde{y}=c) = P(y=k)\,P(\tilde{y}=c)$, the events are independent).

$U_j = V_j + \varepsilon_j$ with $\varepsilon_j \sim \text{Gumbel}(0,1)$ i.i.d.

**Fact 1 (Max-stability).** $\max_j X_j \sim \text{Gumbel}(\log \sum e^{\mu_j}, 1)$ for independent $X_j \sim \text{Gumbel}(\mu_j, 1)$.

**Fact 2 (Softmax).** $P(X_j \text{ is max}) = e^{\mu_j} / \sum_k e^{\mu_k}$.

**Unshown best is independent of constrained best.** For $k \in \bar{C}$, the events $\{y=k\}$ (unshown $k$ is overall best) and $\{\tilde{y}=c\}$ (shown $c$ is best among shown) are independent:

$$P(y=k \land \tilde{y}=c) = P(y=k)\,P(\tilde{y}=c)$$

*Proof.* The joint event is $\{U_k > \bar{U},\; U_k > U_c,\; U_c > U'\}$ where $\bar{U} = \max_{j \in \bar{C} \setminus \{k\}} U_j$, $U' = \max_{h \in C \setminus \{c\}} U_h$. By Fact 1, $\bar{U} \sim \text{Gumbel}(\log \sum_{j \in \bar{C} \setminus \{k\}} e^{V_j}, 1)$ and $U' \sim \text{Gumbel}(\log \sum_{h \in C \setminus \{c\}} e^{V_h}, 1)$.

Condition on $U_c = u$ then $U_k = v$:

$$\int_{-\infty}^{\infty} \int_{u}^{\infty} F_{\bar{U}}(v)\,F_{U'}(u)\,f_{U_k}(v)\,f_{U_c}(u)\;dv\,du$$

Substituting $F_X(x) = \exp(-e^{-x} e^{\mu_X})$, the Gumbel densities, and $t = e^{-u}$, $z = e^{-v}$ factors the integral into:

$$\frac{e^{V_k}}{\sum_{j \in M} e^{V_j}} \cdot \frac{e^{V_c}}{\sum_{h \in C} e^{V_h}} = P(y=k)\,P(\tilde{y}=c)$$

**Deriving the counterfactual posterior.** The independence result gives $P(y=k \land \tilde{y}=c) = P(y=k)\,P(\tilde{y}=c)$ for $k \in \bar{C}$. By the definition of conditional probability:

$$P(y=k \mid \tilde{y}=c) = \frac{P(y=k \land \tilde{y}=c)}{P(\tilde{y}=c)} = \frac{P(y=k)\,P(\tilde{y}=c)}{P(\tilde{y}=c)} = P(y=k)$$

So **unshown alternatives are unaffected by the constrained observation**.

Since $c$ was chosen over every other shown alternative, $U_c > U_j$ for all $j \in C \setminus \{c\}$, giving $P(y=j \mid \tilde{y}=c) = 0$ — shown-but-not-chosen get zero mass.

Finally, conditional probabilities over the full set $M$ must sum to 1:

$$\underbrace{\sum_{k \in C \setminus \{c\}} P(y=k \mid \tilde{y}=c)}_{\text{shown-but-not-chosen } = 0} \;+\; \underbrace{\sum_{k \in \bar{C}} P(y=k \mid \tilde{y}=c)}_{\text{unshown}} \;+\; \underbrace{P(y=c \mid \tilde{y}=c)}_{\text{shown-and-chosen}} = 1$$

Substituting $P(y=k \mid \tilde{y}=c) = P(y=k)$ for $k \in \bar{C}$:

$$\sum_{k \in \bar{C}} P(y=k) + P(y=c \mid \tilde{y}=c) = 1$$

Partition $M$ into $C$ and $\bar{C}$: $\sum_{j \in C} P(y=j) + \sum_{k \in \bar{C}} P(y=k) = 1$ (one alternative in $M$ is chosen), so $\sum_{k \in \bar{C}} P(y=k) = 1 - \sum_{j \in C} P(y=j)$:

$$P(y=c \mid \tilde{y}=c) = \sum_{j \in C} P(y=j)$$

i.e. the chosen alternative absorbs all the mass from the shown-but-not-chosen alternatives.
