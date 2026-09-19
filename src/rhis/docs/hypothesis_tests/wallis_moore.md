# Wallis-Moore Test for Randomness

Performs the non-parametric Wallis-Moore *phase test* (Wallis & Moore,
1941) for randomness of a sequence of observations. It is the test
adopted by the RHIS *randomness* component (the `R` test).

The test counts the *phase runs* of the series: consecutive observations
are compared pairwise and each pair is classified as a rise (`+`) or a
fall (`-`). A phase is a maximal sequence of consecutive equal signs.
Under randomness the expected number of phases is known and the observed
count is compared against it using a normal approximation.

## Function

```python
wallis_moore(ts, alpha=0.05, alternative='two-sided')
```

## Parameters

### `ts`

- **Type:** `array-like`
- **Description:** Ordered numerical observations for which randomness
  of the ordering is to be assessed.
- **Notes:** Non-finite values are not handled explicitly and should be
  removed beforehand. A constant series is treated as a degenerate
  input (see Adaptations).

### `alpha`

- **Type:** `float`
- **Default:** `0.05`
- **Description:** Significance level used for the hypothesis test.

### `alternative`

- **Type:** `str`
- **Default:** `'two-sided'`
- **Description:** Direction of the alternative hypothesis:
  - `'two-sided'`: the observations are not random;
  - `'less'`: non-randomness due to too few phases;
  - `'greater'`: non-randomness due to too many phases.

## Returns

A test result containing:

- **`statistic`** (`float`): The number of phase runs $V$ (see below).
- **`p_value`** (`float`): The p-value computed from the normal
  approximation, directional according to `alternative`.
- **`reject`** (`bool`): `True` when the null hypothesis is rejected.
- **`alternative`** (`str`): The alternative hypothesis used for the
  test (`'two-sided'`, `'greater'`, or `'less'`).

## Hypotheses

- **Null hypothesis ($H_0$):** The observations are random; the order of
  rises and falls shows no systematic pattern.
- **Alternative hypothesis ($H_1$):** The observations are non-random,
  with too few phases (`'less'`), too many phases (`'greater'`), or
  either (`'two-sided'`).

## Test Statistic

Let $x_1, \ldots, x_n$ be the observations and define the first
differences $d_i = \operatorname{sign}(x_{i+1} - x_i)$ with
$i = 1, \ldots, n - 1$, where `+` denotes a rise and `-` a fall.

Ties ($d_i = 0$) are handled by counting the phase runs twice:
once treating equal values as rises and once treating them as falls.
Let $V_{+}$ be the number of phase runs in the first grouping and $V_{-}$
the number in the second. The reported statistic is the average:

$$
V = \frac{V_{+} + V_{-}}{2}
$$

When there are no ties, $V_{+} = V_{-}$ and $V$ is simply the number of
phase runs.

Under $H_0$, the expected number of phase runs and its standard
deviation are

$$
E(V) = \frac{2n - 1}{3}
\qquad
\sigma(V) = \sqrt{\frac{16n - 29}{90}}
$$

with $n$ the number of observations. The standardized statistic is

$$
Z = \frac{V - E(V)}{\sigma(V)}
$$

Under $H_0$, $Z$ approximately follows a standard normal distribution.
The p-value is directional: for the two-sided test
$p = 2(1 - \Phi(|Z|))$, and for the one-sided alternatives it is the
tail on the side of the observed number of phases (large, above 0.5,
when the series has a phase count on the opposite side of the
expectation). The null hypothesis is rejected when $p < \alpha$.

## Adaptations

- **Ties:** equal observations break the rise/fall classification. The
  implementation follows the standard device of averaging the number of
  phase runs obtained when equal values are counted as rises and when
  they are counted as falls.
- **Normal approximation only:** the asymptotic normal approximation is
  always used. Monte Carlo checks confirm that for iid continuous
  series the empirical mean and variance of $V$ match $E(V)$ and
  $\sigma(V)^2$ closely.
- **Direction-aware one-sided p-values:** the reported one-sided
  p-value reflects whether the series has too few or too many phases, so
  a small p-value is never paired with a non-rejection when the
  direction contradicts the alternative.
- **Degenerate inputs:** a constant series has no distinct phases; the
  test returns `statistic=0`, `p_value=0.0`, and `reject=True`.

## Interpretation

A monotone series has a single phase: too few phases indicate a
systematic trend (successive observations consistently rise or fall). An
alternating series oscillates and produces too many phases, indicating
anti-persistence. A failure to reject $H_0$ indicates that the observed
sequence of rises and falls is consistent with a random ordering at the
selected significance level.

## Requirements and Assumptions

- The observations are recorded in their natural order.
- The observations should be numeric and finite.
- The p-value relies on the normal approximation, which improves with
  the sample size.

## Example

The dataset used in the tests is the milk-dispensing quality-control
example from Sheskin (2004, Example 10.2), 21 consecutive containers:

```
1.90, 1.99, 2.00, 1.78, 1.77, 1.76, 1.98, 1.90, 1.65, 1.76, 2.01,
1.78, 1.99, 1.76, 1.94, 1.78, 1.67, 1.87, 1.91, 1.91, 1.89
```

The two-sided test yields $V = 12$ phase runs and approximately
`p = 0.37`; the randomness hypothesis is not rejected.

## References

- Wallis, W. A., & Moore, G. H. (1941). *A significance test for time
  series analysis*. Journal of the American Statistical Association,
  36(215), 401-409.