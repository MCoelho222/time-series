# Wald-Wolfowitz Test for Serial Correlation

Performs the non-parametric Wald-Wolfowitz test for assessing whether
the ordering of a sequence of observations is consistent with a random
process (independence).

The implementation follows the serial-correlation test introduced by
Wald & Wolfowitz (1943), using the circular definition of the
statistic. It is the test adopted by the RHIS *independence* component
and is applied on the ranks of the series by default.

## Function

```python
wald_wolfowitz(ts, alpha=0.05, *, on_ranks=False, ties=True)
```

## Parameters

### `ts`

- **Type:** `array-like`
- **Description:** Ordered numerical observations for which randomness
  of the ordering is to be assessed.
- **Notes:** Non-finite values are rejected. The series must have at
  least 3 observations and cannot be constant. Degenerate inputs raise
  a `ValueError`.

### `alpha`

- **Type:** `float`
- **Default:** `0.05`
- **Description:** Significance level used for the hypothesis test.

### `on_ranks`

- **Type:** `bool`
- **Default:** `False`
- **Description:** If `True`, the test is applied on the ranks of the
  series instead of the raw values, making the test distribution-free.

### `ties`

- **Type:** `bool`
- **Default:** `True`
- **Description:** If `True` and `on_ranks` is `True`, ranks are
  replaced by the average rank within each group of tied values.

## Returns

A test result containing:

- **`statistic`** (`float`): The circular serial-correlation statistic
  $r$ (see below).
- **`p_value`** (`float`): Two-sided p-value.
- **`reject`** (`bool`): `True` when the null hypothesis is rejected.

## Hypotheses

- **Null hypothesis ($H_0$):** The ordering is random; observations are
  independent.
- **Alternative hypothesis ($H_1$):** The ordering is not random; the
  sequence exhibits serial dependence.

## Test Statistic

Let $n$ be the number of observations and $\bar{x}$ their mean. The
series is mean-centered before computing the statistic.

The circular serial-correlation statistic is:

$$
r =
\sum_{i=1}^{n-1} (x_i - \bar{x})(x_{i+1} - \bar{x})
+ (x_1 - \bar{x})(x_n - \bar{x})
$$

The last term closes the sequence into a circle, so the first and last
observations are treated as neighbours. This is the form considered by
the 1943 paper.

Let $S_2 = \sum_{i=1}^{n} (x_i - \bar{x})^2$ and
$S_4 = \sum_{i=1}^{n} (x_i - \bar{x})^4$. Under the null hypothesis,
the expected value and variance of $r$ are:

$$
E(r) = -\frac{S_2}{n - 1}
$$

$$
\operatorname{Var}(r)
=
\frac{S_2^2 - S_4}{n - 1}
+
\frac{S_2^2 - 2 S_4}{(n - 1)(n - 2)}
-
\frac{S_2^2}{(n - 1)^2}
$$

The standardized statistic is computed as:

$$
Z = \frac{|r - E(r)|}{\sqrt{\operatorname{Var}(r)}}
$$

Under $H_0$, $Z$ approximately follows a standard normal distribution,
and the two-sided p-value is:

$$
p = 2 \left( 1 - \Phi(Z) \right)
$$

The null hypothesis is rejected when $p < \alpha$ (equivalently, when
$|Z| > z_{1 - \alpha/2}$, with $z_{0.975} \approx 1.96$ for
$\alpha = 0.05$).

## Adaptations

The implementation introduces the following adaptations relative to the
plain textbook formulation:

- **Circular closure:** the term $(x_1 - \bar{x})(x_n - \bar{x})$ wraps
  the sequence so that the first and last observations are neighbours,
  as in the original 1943 treatment. The statistic returned is the raw
  circular sum $r$, not the standardized $Z$.
- **Normal approximation only:** the asymptotic normal approximation is
  always used; no exact small-sample critical tables are applied.
  Monte Carlo checks show the two-sided tail stays close to the nominal
  level for $n \geq 10$.
- **Ranks option:** with `on_ranks=True`, the statistic is computed on
  the ranks of the series, making the test distribution-free. When
  `ties=True`, tied values receive their average rank.
- **Degenerate inputs:** a series with non-finite values, fewer than
  three observations, constant values, or yielding a near-zero variance
  for $r$ raises a `ValueError` instead of returning an undefined
  result. Inside the RHIS pipeline (`Rhis.calculate_rhis`), this error
  is caught and a `NaN` p-value is recorded for the affected slice, so
  downstream processing continues without interruption.

## Interpretation

Small values of $|Z|$ are consistent with a random ordering, while large
values of $|Z|$ indicate serial dependence (positive autocorrelation of
successive terms makes adjacent deviations alike, increasing $r$;
negative autocorrelation makes them alternate, decreasing it).

A failure to reject $H_0$ does not establish that the sequence is
random. It indicates that the observed sequence does not provide
sufficient evidence against the independence hypothesis at the selected
significance level.

## Requirements and Assumptions

- The observations must form an ordered sequence.
- All observations must be finite numeric values.
- The series must contain at least three observations and at least two
  distinct values.
- Under $H_0$, the ordering is assumed to be random.
- The p-value relies on the normal approximation, which improves as the
  sample size grows.

## Notes

The Wald-Wolfowitz *Runs test* (Wald & Wolfowitz, 1940) is a different
procedure: it counts runs of observations belonging to two categories
(e.g., relative to a fixed cut point such as the median) and is
described elsewhere in the literature. The serial-correlation test
documented here operates directly on the numerical values (or their
ranks) and preserves their order and magnitude.

## References

- Wald, A., & Wolfowitz, J. (1943). *An exact test for randomness in
  the non-parametric case based on serial correlation*. Annals of
  Mathematical Statistics, 14(4), 378–388.

- Wald, A., & Wolfowitz, J. (1940). *On a test whether two samples are
  from the same population*. Annals of Mathematical Statistics, 11(2),
  147–162.

- Naghettini, M., & Pinto, E. J. A. (2007). *Hidrologia Estatística*.
  CPRM, Belo Horizonte.