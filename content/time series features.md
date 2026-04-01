---
tags:
  - ml
folder: learning
share: true
title: time series features
date created: Wednesday, April 30th 2025, 5:10:52 pm
date modified: Monday, March 30th 2026, 7:33:41 pm
---

Yearly seasonalities need several years of data to learn patterns. The basic approaches are **dummy variables**, **Fourier features** (periodic) and **splines** (not periodic).

![[./notes/learning/assets/fourier.svg|fourier.svg]]

![[./notes/learning/assets/bspline.svg|bspline.svg]]Below are some more complex feature designs.

*See [this gist](https://gist.github.com/kforeman/17c78b7f50530ebc7cca07deec5df30a) for the code.*

*The [gloria package](https://e-dyn.github.io/gloria/get_started/events.html) also has support for some features, also repeating events.*

## radial basis (bump)

We want to generate a Gaussian-like curve around a specific date.

$$ f_{x_{0}, \epsilon}(x) = \exp(-(\frac{x - x_0}{\epsilon})^2)$$

The width of the bump function is controlled by $\epsilon$. Rather than a dummy variable, the influence of a single day seep into adjacent days.

This can be repeated, e.g. for day-of-week effect.

*See [Ronan](https://www.ronanlaker.com/linear-models-demystified/#radial-basis-functions) and [Vincent Warmerdam](https://youtu.be/68ABAU_V8qI?si=gHJaD7s88sX3YpWE).*

## asymmetric bump

Sometimes the effect before and after the seasonal variable is not the same, e.g. after Christmas there might be a strong effect before and a drop after.

$$
\begin{equation}
  g_{x_{0}, \epsilon}(x) = \left\{
  \begin{array}{@{}ll@{}}
    a_{-} f_{x_{0}, \epsilon_{-}}(x), & x \leq x_0 \\
    a_{+} f_{x_{0}, \epsilon_{+}}(x), & x \gt x_0
  \end{array}\right.
\end{equation}
$$

Now, there can be different widths and different amplitudes before and after the event. Note, the amplitude can also be negative here.

*From [Juan's blog](https://juanitorduz.github.io/bump_func/).*

## eigen features

For longer seasonalities where we do not know the underlying pattern, but we want that the effect starts at zero and ends at zero but behaves like Fourier series otherwise. e.g. two week school holiday.

Suppose the seasonality lasts for $T_0$ days (the *span*, equally we can use the number of points we want between these dates). Define two matrices, $\mathbf{M}$ and $\mathbf{A}$, of size $(T_{0}+ 1) \times (T_{0}+ 1)$ where

$$
\begin{equation}
  M_{ij} = \left\{
  \begin{array}{@{}ll@{}}
    1 - 1 / (T_{0}+ 1), & i = j \\
    - 1 / (T_{0}+ 1), & \text{otherwise}
  \end{array}\right.
\end{equation}
$$

and

$$
\begin{equation}
  A_{ij} = \left\{
  \begin{array}{@{}ll@{}}
    1, & i \geq j \\
    0, & \text{otherwise}
  \end{array}\right.
\end{equation}
$$

$\mathbf{M}$ is a circulant matrix, a special type of Toeplitz/diagonal-constant matrix where every row is the same as the previous row, just shifted to the right by 1. The [eigenvectors of circulant matrices can be written in terms of the roots of unity](https://web.mit.edu/18.06/www/Spring17/Circulant-Matrices.pdf), $\omega_{n}= e^{\frac{2\pi i}{n}}$, which are the solutions to $z^{n} = 1$. The $k$-th eigenvector for any $n \times n$ circulant matrix is

$$
v_{k}= (\omega_{n}^{0k} \quad \omega_{n}^{1k} \quad ... \quad \omega_{n}^{(n-1)k})^T
$$

and multiplying by the matrix whose columns are the eigenvectors $\mathbf{F}$ where $F_{ij} = \omega_n^{jk}$ is the discrete Fourier Transform (DFT). (Aside: this matrix factorises into $\sim n\log{n}$ sparse matrices for the fast Fourier Transform (FFT)).

$\mathbf{A}$ is a lower diagonal matrix representing cumulative sums. The combination $\mathbf{A}\mathbf{M}\mathbf{A^T}$ is therefore acts to *envelop* the Fourier series terms.

For our feature, let $(w_i,v_i)$ be the eigenvalue-eigenvector (normalised) pairs of $\mathbf{A}\mathbf{M}\mathbf{A^T}/(T_0/4)$. Then, the oscillatory components are $\sqrt{w_i}v_i$.

```python
# fill out the matrices
AMA = A.dot(M).dot(A.T) / (span / 4)
w, v = np.linalg.eig(AMA)

basis = np.concatenate(
	[
		# the first time point should be zero
		np.zeros((1, order * 2)),
		# the final element of v_i is zero since the last column
		# and row of AMA are zero
		np.sqrt(w[: order * 2].real * v[:, : order * 2]),
	]
)
```

![[./notes/learning/assets/eigen.svg|eigen.svg]]

*Credit to [Ulrich Mueller](https://www.princeton.edu/~umueller/).*

## monotonic

Monotone functions can be modelled using [I-splines](https://projecteuclid.org/journals/statistical-science/volume-3/issue-4/Monotone-Regression-Splines-in-Action/10.1214/ss/1177012761.full), which are the [integral of non-negative M-splines](https://stats.stackexchange.com/questions/519465/how-to-correctly-use-i-splines-for-monotone-non-decreasing-increasing-regressio), with strictly positive or strictly negative coefficients. GAM packages like `mgcv` or `pyGAM` use a [likelihood penalty for monotonic constraints](https://rdrr.io/cran/mgcv/man/pcls.html).

*python [implementation](https://gist.github.com/naturale0/9ecb645eb4abc0412a839a83fd278f9c).*

## extending to multivariate

The seasonalities can have the same shape for each stratum but different magnitude.

$$ + \sum_{l=1}^{n_s}{\lambda_{lj}}\sum_{i=1}^{n_l}{\beta_{li}}x_{tli} $$

where we have $l = 1, ..., n_s$ seasonalities of this type (e.g. holidays), we have $j$ strata (e.g. store), we have $i = 1, ..., n_l$ regressors (basis terms). The $\beta_{li}$ control the shape and are shared among strata, and the $\lambda_{lj}$ vary the magnitude of the effect between strata.

## holiday day-of-week damping interaction

During holidays with variable day-of-week alignment (e.g. Christmas modelled with eigen features), the weekly seasonality amplitude should shrink — people don't know what day of the week it is. A per-holiday damping parameter attenuates the day-of-week effect and lets the holiday feature shape take over.

Fixed day-of-week holidays (e.g. Thanksgiving Thu, Easter Sun) should not be damped. Their holiday features already learn the effect relative to the fixed weekly pattern, so adding damping creates an identifiability issue.

Define a `seasonal_scale` of shape `(n_time, n_series)`, equal to `1.0` on normal days and `< 1.0` on damped holidays. The suppression amount is learned per stratum with a prior (`1.0` is full suppression). In the scan step:

```python
+ seasonal_scale[t] * previous_seasonality[0]
```

## variation between years

The coefficients of these basis functions might vary slightly between years. Use a random walk prior on $\lambda_{lj}$ and $\beta_{li}$ over years.
