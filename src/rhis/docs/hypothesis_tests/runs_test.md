# Runs Test (Single Sample)

Performs the non-parametric single-sample runs test for randomness of a
sequence of observations, following Sheskin (2004, Test 10).

The sequence is split into observations above and below the median, and
the observations between rises and falls are grouped into *runs* (maximal
sequences of consecutive observations on the same side of the median).
A random sequence produces neither too few nor too many runs.

## Function

```python
runs_test(ts, alpha=0.05, alternative='two-sided', *, continuity=True)
```

## Parameters

### `ts`

- **Type:** `array-like`
- **Description:** Ordered numerical observations for which randomness
  of the ordering is to be assessed. Values equal to the median are
  ignored.
- **Notes:** A non-empty series must contain observations strictly
  above and strictly below the median for the statistic to be defined.

### `alpha`

- **Type:** `float`
- **Default:** `0.05`
- **Description:** Significance level used for the hypothesis test.

### `alternative`

- **Type:** `str`
- **Default:** `'two-sided'`
- **Description:** Direction of the alternative hypothesis:
  - `'two-sided'`: the ordering is non-random;
  - `'less'`: non-randomness due to too few runs;
  - `'greater'`: non-randomness due to too many runs.

### `continuity`

- **Type:** `bool`
- **Default:** `True`
- **Description:** If `True`, applies the 0.5 continuity correction when
  standardizing the number of runs.

## Returns

A test result containing:

- **`statistic`** (`float`): The number of runs $R$.
- **`p_value`** (`float`): The p-value computed from the normal
  approximation, directional according to `alternative`.
- **`reject`** (`bool`): `True` when the null hypothesis is rejected.
- **`alternative`** (`str`): The alternative hypothesis used for the
  test (`'two-sided'`, `'greater'`, or `'less'`).

## Hypotheses

- **Null hypothesis ($H_0$):** The events are arranged randomly; careful
  mixing or clustering is absent.
- **Alternative hypothesis ($H_1$):** The events are arranged
  non-randomly, due to too few runs (`'less'`), too many runs
  (`'greater'`), or either (`'two-sided'`).

## Test Statistic

Let the median be $m$. Each observation $x_i$ is converted into a sign

$$
s_i =
\begin{cases}
+1 & x_i > m \\
-1 & x_i < m
\end{cases}
$$

Observations with $x_i = m$ are excluded. Let $n_1$ and $n_2$ be the
number of $+1$ and $-1$ signs, respectively. A *run* is a maximal
sequence of consecutive equal signs. The number of runs is

$$
R = 1 + \sum_{i=2}^{n_1 + n_2} \mathbb{1}[s_i \neq s_{i-1}]
$$

Under $H_0$, the expected value and variance of $R$ are

$$
E(R) = \frac{2 n_1 n_2}{n_1 + n_2} + 1
$$

$$
\operatorname{Var}(R) =
\frac{2 n_1 n_2 (2 n_1 n_2 - n_1 - n_2)}
{(n_1 + n_2)^2 (n_1 + n_2 - 1)}
$$

The standardized statistic is

$$
Z = \frac{|R - E(R)| - \delta}{\sqrt{\operatorname{Var}(R)}}
$$

where $\delta = 0.5$ when `continuity=True` and $0$ otherwise. Under
$H_0$, $Z$ approximately follows a standard normal distribution. The
p-value is directional: for the two-sided test
$p = 2(1 - \Phi(|Z|))$, and for the one-sided alternatives it is the
tail on the side of the observed number of runs (large, above 0.5, when
the series has a run count on the opposite side of the expectation).
The null hypothesis is rejected when $p < \alpha$.

## Adaptations

- **Median cut:** randomness is assessed relative to the median of the
  observed series. Values equal to the median are excluded, following
  the standard procedure, so the effective sample size may be smaller
  than the series length.
- **Normal approximation with continuity:** the asymptotic normal
  approximation is always used, with a user-selectable 0.5 continuity
  correction.
- **Direction-aware one-sided p-values:** the reported one-sided
  p-value reflects whether the series has too few or too many runs, so a
  small p-value is never paired with a non-rejection when the direction
  contradicts the alternative.
- **Degenerate inputs:** when the series has no observations strictly
  above or strictly below the median, the number of runs is undefined;
  the test returns `statistic=0`, `p_value=0.0`, and `reject=True`.

## Interpretation

Clustered series (e.g. a trend or a slow drift) produce too few runs,
because values on the same side of the median appear together for long
stretches. Periodic or systematically alternating series produce too
many runs. A failure to reject $H_0$ indicates that the observed number
of runs is consistent with a random arrangement at the selected
significance level.

## Requirements and Assumptions

- The observations are recorded in their natural order.
- All observations must be finite numeric values.
- The series must contain at least one observation strictly above and at
  least one strictly below the median.
- The p-value relies on the normal approximation, which improves with
  the sample size.

## Example

The dataset used in the tests is the milk-dispensing quality-control
example from Sheskin (2004, Example 10.2), 21 consecutive containers:

```
1.90, 1.99, 2.00, 1.78, 1.77, 1.76, 1.98, 1.90, 1.65, 1.76, 2.01,
1.78, 1.99, 1.76, 1.94, 1.78, 1.67, 1.87, 1.91, 1.91, 1.89
```

The median is 1.89. The two-sided test yields $R = 11$ runs and
approximately `p = 0.82`, matching the value reported in the book; the
randomness hypothesis is not rejected.

## References

- Sheskin, D. J. (2004). *Handbook of Parametric and Nonparametric
  Statistical Procedures*. Test 10 - Run Test for Randomness. 3rd
  edition. Chapman & Hall/CRC.