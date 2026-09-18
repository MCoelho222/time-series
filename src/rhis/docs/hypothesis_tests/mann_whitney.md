# Mann-Whitney U Test

Performs the non-parametric Mann-Whitney U test for comparing two
independent groups of observations, following the presentation of
Helsel & Hirsch (2002; 2020), *Statistical Methods in Water Resources*
— Chapter on *Differences between two independent groups*.

The Mann-Whitney test is also known as the Wilcoxon Rank-Sum test. It is
the test adopted by the RHIS *homogeneity* component: when comparing a
single series, the series is split into two halves and their
distributions are compared.

The implementation uses the large-sample normal approximation; no exact
small-sample critical tables are applied.

## Function

```python
mann_whitney(x, y=None, alpha=0.05, alternative='two-sided',
             *, continuity=True, ties=True)
```

## Parameters

### `x`

- **Type:** `array-like`
- **Description:** Ordered numerical observations of the first group.
  When `y` is `None`, `x` is split into its first and second halves with
  `split_into_parts(x, 2)` and the two resulting groups are compared.

### `y`

- **Type:** `array-like`, optional
- **Default:** `None`
- **Description:** Ordered numerical observations of the second group.
  When omitted, `x` is compared against its own second half (used by the
  RHIS homogeneity component).

### `alpha`

- **Type:** `float`
- **Default:** `0.05`
- **Description:** Significance level used for the hypothesis test.

### `alternative`

- **Type:** `str`
- **Default:** `'two-sided'`
- **Description:** Direction of the alternative hypothesis:
  - `'two-sided'`: the two groups differ (`prob[x > y] != 0.5`);
  - `'greater'`: the first group tends to be larger (`prob[x > y] > 0.5`);
  - `'less'`: the first group tends to be smaller (`prob[x > y] < 0.5`).

### `continuity`

- **Type:** `bool`
- **Default:** `True`
- **Description:** If `True`, applies the 0.5 continuity correction when
  standardizing the statistic.

### `ties`

- **Type:** `bool`
- **Default:** `True`
- **Description:** If `True`, tied values receive their average rank and
  the sampling variance is corrected for ties. If `False`, ties are
  ignored and raw consecutive ranks are used.

## Returns

A test result containing:

- **`statistic`** (`float`): The smaller of the two U statistics
  $U = \min(U_1, U_2)$.
- **`p_value`** (`float`): The p-value computed from the normal
  approximation, directional according to `alternative`.
- **`reject`** (`bool`): `True` when the null hypothesis is rejected.
- **`alternative`** (`str`): The alternative hypothesis used for the
  test (`'two-sided'`, `'greater'`, or `'less'`).

## Hypotheses

- **Null hypothesis ($H_0$):** `prob[x > y] = 0.5`; both groups are drawn
  from distributions with the same central tendency (identical shape is
  assumed).
- **Alternative hypothesis ($H_1$):** `prob[x > y] != 0.5` (two-sided),
  `prob[x > y] > 0.5` (greater), or `prob[x > y] < 0.5` (less).

## Test Statistic

Let the two groups have sizes $n_1$ and $n_2$, with $n = n_1 + n_2$, and
rank all $n$ observations together. Let $R_1$ and $R_2$ be the sums of
the ranks of each group (with ties replaced by their average rank when
`ties=True`).

The two U statistics are:

$$
U_1 = n_1 n_2 + \frac{n_1 (n_1 + 1)}{2} - R_1
$$

$$
U_2 = n_1 n_2 + \frac{n_2 (n_2 + 1)}{2} - R_2
$$

which satisfy $U_1 + U_2 = n_1 n_2$. The reported statistic is
$U = \min(U_1, U_2)$.

Under $H_0$, the expected value of each U statistic is:

$$
E(U) = \frac{n_1 n_2}{2}
$$

and its variance, in the absence of ties, is:

$$
\operatorname{Var}(U) = \frac{n_1 n_2 (n + 1)}{12}
$$

### Correction for ties

With average ranks, the variance is corrected to:

$$
\operatorname{Var}(U)
=
\frac{n_1 n_2}{n (n - 1)}
\sum_{i=1}^{n} R_i^2
-
\frac{n_1 n_2 (n + 1)^2}{4 (n - 1)}
$$

This is algebraically equivalent to the more common form based on the
sizes $t$ of the tied groups:

$$
\operatorname{Var}(U)
=
\frac{n_1 n_2}{12}
\left(
n + 1 - \frac{\sum_{t} t(t^2 - 1)}{n (n - 1)}
\right)
$$

### Standardized statistic and p-value

The chosen U statistic $U^*$ is: $U_2$ for `alternative='greater'`,
$U_1$ for `alternative='less'`, and $\max(U_1, U_2)$ for the two-sided
test. The standardized statistic is:

$$
Z = \frac{U^* - E(U) - \delta}{\sqrt{\operatorname{Var}(U)}}
$$

where $\delta = 0.5$ when `continuity=True` and $0$ otherwise. The
continuity correction is always *subtracted* (regardless of the observed
direction), which nudges the p-value upward, as done by `scipy`.

Under $H_0$, $Z$ approximately follows a standard normal distribution.
The p-value is a survival (right-tail) probability:

$$
p = f \cdot \left( 1 - \Phi\left(Z\right) \right),
\qquad
f =
\begin{cases}
2 & \text{two-sided} \\
1 & \text{one-sided}
\end{cases}
$$

clamped to a maximum of $1$. Choosing $U^*$ per alternative makes the
one-sided p-values depend on the observed direction of the difference:
with `alternative='greater'` and $x$ tending larger, $Z$ is positive and
$p$ is small; if the data point the opposite way, $Z$ is negative and $p$
approaches $1$.

The null hypothesis is rejected when $p < \alpha$.

## Adaptations

- **Univariate usage:** when `y` is `None`, the input series is split in
  half and the two halves are compared. This is how the RHIS homogeneity
  test consumes the test (`Rhis.calculate_rhis`).
- **Normal approximation only:** the asymptotic standard normal
  approximation is always used. It is recommended for groups with more
  than about 10 observations in each sample, although small groups are
  accepted.
- **Average ranks for ties:** with `ties=True`, tied values share the
  mean of their rank positions and the variance is corrected
  accordingly.
- **Constant input:** when the two groups contain a single identical
  value only, the statistic is undefined; the test returns
  `statistic=0` and `p_value=1.0` (no rejection) instead of raising.
- **scipy compatibility:** the returned p-values match
  `scipy.stats.mannwhitneyu(x, y, method='asymptotic')` for all
  alternatives and both continuity settings, including tied data. Note
  that `scipy` reports as its statistic $U = R_1 - n_1(n_1+1)/2$ (the
  rank-based U of the first group), while this implementation reports
  $\min(U_1, U_2)$; the two orientations do not affect the p-value.

## Interpretation

Large values of $U$ (close to $n_1 n_2$) indicate that the first group
tends to have smaller values than the second; small values of $U$
indicate the opposite. The test compares the central location of the two
populations, assuming the two distributions have the same shape.

A failure to reject $H_0$ does not establish that the two groups are
identical. It indicates that the observed difference in ranks does not
provide sufficient evidence of a location shift at the selected
significance level.

## Requirements and Assumptions

- The two samples are independent and each is randomly selected from the
  population it represents.
- The measured variable (which is subsequently ranked) is effectively
  continuous.
- The underlying distributions are identical in shape; the test then
  compares the location (median) of the two groups.
- All observations must be finite numeric values.
- The p-value relies on the normal approximation, which improves as the
  sample sizes grow; the groups should ideally exceed 10 observations.

## Example

The dataset used in the tests is the groundwater-quality example from
Helsel & Hirsch, comparing 10 observations from an industrial site
against 10 from a residential site:

```
x = [0.59, 0.87, 1.1, 1.1, 1.2, 1.3, 1.6, 1.7, 3.2, 4.0]  # industrial
y = [0.3, 0.36, 0.5, 0.7, 0.7, 0.9, 0.92, 1., 1.3, 9.7]  # residential
```

Group medians are 1.25 and 0.8. The two-sided test yields
approximately `p = 0.0491`, matching the value reported in the book.

## References

- Helsel, D. R., & Hirsch, R. M. (2002). *Statistical Methods in Water
  Resources*. Techniques of Water-Resources Investigations, Book 4,
  Chapter A3. U.S. Geological Survey. Chapter 5 — Differences between two
  independent groups, p. 118.
  https://pubs.usgs.gov/tm/04/a03/tm4a3.pdf

- Mann, H. B., & Whitney, D. R. (1947). *On a test of whether one of two
  random variables is stochastically larger than the other*. Annals of
  Mathematical Statistics, 18(1), 50–60.

- Wilcoxon, F. (1945). *Individual comparisons by ranking methods*.
  Biometrics Bulletin, 1(6), 80–83.