# Mann-Kendall Test for Trend

Performs the non-parametric Mann-Kendall test for monotonic trend
(Gilbert, 1987; Helsel & Hirsch, 2002). It is the test adopted by the
RHIS *stationarity* component (the `S` test).

Every pair of observations is compared and classified as an increase, a
decrease, or a tie. A preponderance of increases indicates an upward
trend; a preponderance of decreases a downward trend. The test makes no
distributional assumptions about the data, only about the ordering.

## Function

```python
mann_kendall(ts, alpha=0.05, alternative='two-sided')
```

## Parameters

### `ts`

- **Type:** `array-like`
- **Description:** An ordered time series for which a monotonic trend
  is to be assessed.
- **Notes:** Non-finite values are not handled explicitly and should be
  removed beforehand. The normal approximation used for the p-value is
  recommended for series with 10 or more observations.

### `alpha`

- **Type:** `float`
- **Default:** `0.05`
- **Description:** Significance level used for the hypothesis test.

### `alternative`

- **Type:** `str`
- **Default:** `'two-sided'`
- **Description:** Direction of the alternative hypothesis:
  - `'two-sided'`: a monotonic trend is present;
  - `'less'`: a monotonic downward trend is present;
  - `'greater'`: a monotonic upward trend is present.

## Returns

A test result containing:

- **`statistic`** (`float`): The Mann-Kendall statistic $S$.
- **`p_value`** (`float`): The p-value computed from the normal
  approximation, directional according to `alternative`.
- **`reject`** (`bool`): `True` when the null hypothesis is rejected.
- **`alternative`** (`str`): The alternative hypothesis used for the
  test (`'two-sided'`, `'greater'`, or `'less'`).

## Hypotheses

- **Null hypothesis ($H_0$):** There is no monotonic trend; the
  observations are randomly ordered over time.
- **Alternative hypothesis ($H_1$):** A monotonic trend is present
  (`'two-sided'`), monotonic downward (`'less'`), or monotonic upward
  (`'greater'`).

## Test Statistic

Let $x_1, \ldots, x_n$ be the observations. The statistic is

$$
S = \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}
\operatorname{sign}(x_j - x_i)
$$

that is, the number of increasing pairs minus the number of decreasing
pairs; ties contribute zero.

When the observations contain tied groups of sizes $t_1, t_2, \ldots$,
the variance under $H_0$ is

$$
\operatorname{Var}(S) = \frac{1}{18}
\left[n(n-1)(2n+5)
- \sum_{p} t_p(t_p-1)(2t_p+5)\right]
$$

The continuity-corrected standardized statistic is

$$
Z =
\begin{cases}
\dfrac{S-1}{\sqrt{\operatorname{Var}(S)}} & S > 0 \\[6pt]
0 & S = 0 \\[6pt]
\dfrac{S+1}{\sqrt{\operatorname{Var}(S)}} & S < 0
\end{cases}
$$

Under $H_0$, $Z$ approximately follows a standard normal distribution.
The p-value is directional: for the two-sided test
$p = 2(1 - \Phi(|Z|))$, and for the one-sided alternatives it is the
tail on the side of the observed trend (large, above 0.5, when the
series moves in the direction opposite to the alternative). The null
hypothesis is rejected when $p < \alpha$.

## Adaptations

- **Trend measure:** the reported statistic is $S$ itself (a count of
  pairwise comparisons), not the normalized Kendall tau. Its sign gives
  the direction and its magnitude grows with series length.
- **Always normal approximation:** the asymptotic normal approximation
  is used unconditionally. It is reliable from about 10 observations
  upward (Gilbert, 1987); for very short series an exact distribution
  would be preferable.
- **Continuity correction:** the 0.5 continuity correction is always
  applied, which makes the test slightly conservative.
- **Ties:** groups of tied values enter the variance through the
  $t(t-1)(2t+5)$ correction, shrinking the standard deviation relative
  to the tie-free case.
- **Direction-aware one-sided p-values:** the reported one-sided
  p-value reflects whether the trend is upward or downward, so a small
  p-value is never paired with a non-rejection when the direction
  contradicts the alternative.
- **Degenerate inputs:** a constant series has $S = 0$ with zero
  variance; the test returns `statistic=0`, `p_value=1.0`
  (two-sided), and `reject=False`.

## Interpretation

A large positive $S$ indicates a tendency for later observations to
exceed earlier ones (upward trend); a large negative $S$ the reverse.
A failure to reject $H_0$ indicates that the observed ordering of
increases and decreases is consistent with randomness at the selected
significance level. Because the test only uses the signs of pairwise
differences, it is insensitive to outliers in magnitude but still
detects monotonic movement.

## Requirements and Assumptions

- The observations are recorded in their natural (e.g., chronological)
  order.
- The observations should be numeric and finite.
- The p-value relies on the normal approximation, which improves with
  the sample size (at least 10 observations are recommended).

## Example

The worked example from Gilbert (1987), page 212:

```
20, 20, 20, 20, 15, 20, 20, 30, 27, 26, 23, 35, 25, 28, 70, 26,
24, 34, 32, 23, 50, 30
```

The test yields $S = 111$, standard deviation $35.02$, and
$Z = 3.14$ (the book reports 3.1), giving a two-sided p-value of
approximately `p = 0.0017`; the hypothesis of no trend is rejected at
the 5% level.

## References

- Gilbert, R. O. (1987). *Statistical Methods for Environmental
  Pollution Monitoring*. Chapter 16 - Detecting and Estimating Trends.
  Van Nostrand Reinhold.
- Helsel, D. R., & Hirsch, R. M. (2002). *Statistical Methods in Water
  Resources*. Techniques of Water-Resources Investigations of the United
  States Geological Survey, Book 4, Chapter A3.
- Mann, H. B. (1945). *Nonparametric tests against trend*.
  Econometrica, 13(3), 245-259.
- Kendall, M. G. (1975). *Rank Correlation Methods*. 4th edition.
  Charles Griffin.