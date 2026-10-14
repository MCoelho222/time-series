# RHIS

## What is RHIS?

RHIS stands for **R**andomness, **H**omogeneity, **I**ndependence and **S**tationarity. These are the four basic assumptions a time series is expected to satisfy to be considered *statistically representative* (Naghettini, 2007; Coelho et al., 2017; Coelho et al., 2019). They underpin almost every frequency analysis, design flood or trend study: if these assumptions do not hold, the results may not reflect current conditions.

Each assumption is checked with a well-known non-parametric test:

* **Randomness (R)** — Wallis-Moore phase test
* **Homogeneity (H)** — Mann-Whitney U test
* **Independence (I)** — Wald-Wolfowitz serial-correlation test
* **Stationarity (S)** — Mann-Kendall trend test

When one or more of these hypotheses are rejected, the record probably contains a variability pattern rather than pure noise. A few everyday examples:

* **Urban growth** — impervious surfaces speed up rainfall runoff, so streamflow tends to rise year after year (a trend).
* **A new reservoir upstream** — flows downstream settle to a new level once the dam starts operating (a shift).
* **Changes in land use or climate** — the seasonal and year-to-year behaviour changes, adding cycles or irregular swings.
* **The Nile** — the example shipped with this package is a classic record with a clear change point.

## What it does?

RHIS finds the **longest recent stretch of a time series that passes all four tests** — in other words, the longest part of the record you can "safely" use.

It tests every window ending at the most recent observation, starting with the longest possible window (the entire time series). It then shortens the window by removing one observation from the beginning at a time, until only the 10 most recent values remain. Ten observations is the minimum sample size for the hypothesis tests, based on the normal approximation.

For each window, RHIS calculates a p-value for each of the four hypotheses. It then takes the minimum of these four p-values to obtain RHIS-min. This ensures that all four hypotheses are considered when determining whether a window passes the tests.

The windows are evaluated from longest to shortest, meaning the starting point moves from oldest to newest. The first time RHIS-min reaches or exceeds the significance level (\alpha) (0.05 by default), all four hypotheses are no longer rejected. This identifies the boundary between the RHIS-compliant recent stretch and the earlier observations.

The results are returned as a simple dictionary containing the p-values for each hypothesis and the (start, end) indexes of the RHIS-compliant stretch.

For example, a time series with 15 elements produces six p-values for each hypothesis, corresponding to windows of sizes 15, 14, 13, 12, 11, and 10.

```py
p_evol = {
    'series_0': {
        'R': [0.0000, 0.1772, 0.0523, 0.0014, 0.0272, 0.0541],
        'H': [0.0021, 0.0727, 0.3645, 0.0033, 0.3372, 0.2641],
        'I': [0.1000, 0.0645, 0.9234, 0.2219, 0.1275, 0.5433],
        'S': [0.0670, 0.0145, 0.0023, 0.2943, 0.7369, 0.6182],
        'RHIS-min': [0.0000, 0.0145, 0.0023, 0.0014, 0.0272, 0.0541]
        }
}

# For R:
# 0.0000 -> p-value for the complete series (N=15)
# 0.1772 -> p-value with N - 1 observations
# 0.0523 -> p-value with N - 2 observations
# 0.0014 -> p-value with N - 3 observations
# 0.0272 -> p-value with N - 4 observations
# 0.0541 -> p-value with N - 5 observations (10 most recent values)

idxs = {
    'series_0': (5, 15)
    }

# 5  -> index of the first observation in the RHIS-compliant stretch
# 15 -> exclusive end index (the last observation is at index 14)
```

Reading the RHIS-min values from left to right, we can see how the series becomes RHIS-compliant as the window gets shorter. With (\alpha = 0.05), the first window that passes all four tests is the one containing the 10 most recent observations, where RHIS-min = 0.0541.

Therefore, the RHIS-compliant stretch starts at index 5 and ends at index 15 (exclusive), corresponding to the last 10 observations of the series.

## What do these tests actually test?

Every test returns a **p-value**: the probability of seeing a pattern at least this strong if the assumption were really true. A small p-value is evidence against the assumption. In RHIS an assumption is rejected when its p-value falls below the significance level (alpha), which is 0.05 by default. Here is what each of the four tests looks at.

### Randomness

A random series has no systematic order. The randomness test (Wallis-Moore) counts the "ups" and "downs" in the series: a random series changes direction often, a series with too few changes keeps moving the same way (trend-like), and one with too many changes keeps zig-zagging. Randomness is rejected when this pattern is statistically significant at the chosen significance level.

### Homogeneity

A homogeneous series behaves like a single population from beginning to end. The test (Mann-Whitney) splits the record into two halves and compares them. Homogeneity is rejected if one half is systematically larger or smaller than the other at the chosen significance level — a sign of a shift.

### Independence

A series is independent when one observation tells you nothing about the next. In practice values often carry memory: after a wet period, groundwater keeps feeding the river, so flows stay high and then decay slowly. The test (Wald-Wolfowitz) checks whether neighbouring values are too similar (positive correlation) or too different (negative correlation). Independence is rejected when this dependence is statistically significant at the chosen significance level.

### Stationarity

A stationary series keeps its statistical behaviour over time and, in particular, has no lasting upward or downward trend. The test (Mann-Kendall) compares every pair of observations and asks whether increases clearly outnumber decreases (or the reverse). Stationarity is rejected when such a monotonic trend is statistically significant at the chosen significance level.

## What to keep in mind

RHIS gives you statistics; you bring the context. A few things are worth knowing before you act on a result.

### What kind of series it works with

RHIS takes a table of numbers in time order, oldest row first, one column per variable — annual streamflow, monthly rainfall, daily water levels, water-quality concentrations, and so on. You can hand it that table in either of two ways:

- a `pandas.DataFrame` (every row is a time step, so the row index must not be a MultiIndex; the columns may use a MultiIndex, in which case the series are keyed by the column-label tuples), or
- a mapping (for example a `dict`) of series names to sequences of numbers, read positionally — a `pandas.Series` index is ignored, and series of different lengths are padded with NaN internally. The reported indexes are always relative to each original series, never to the padded table.

Each column needs at least 10 numeric values to be considered, and every window it tests needs a handful of values (5, or 10 for long records). Missing or non-numeric entries are skipped, not filled in, so gaps quietly reduce how much data each window really has.

### What the tests can and cannot tell you

These are statistical tests, not detectors of causes. A "fail" tells you a pattern is unlikely to be pure chance; it does not tell you why. A "pass" is not a certificate that the data are sound. On short records a real trend or shift can go unnoticed for lack of evidence; on very long records even a tiny, physically meaningless change can look significant. The threshold (alpha, 0.05 by default) is a convention, not a law of nature.

### Outliers

An outlier is a value far from the rest — sometimes real (a record flood) and sometimes a mistake (a typo, a sensor glitch, a unit mix-up). These tests work with the order, ranks, or signs of the values rather than their size, so they tolerate outliers better than a simple average. Even so, one extreme value (say 999.9 instead of 99.9) can still shift the rankings and change the chosen period. RHIS does not detect or remove outliers: look at them first, fix genuine data-entry errors, and use judgement for the real extremes.

### RHIS cannot see your time step

The tests read a sequence of numbers, not dates. They do not know whether your data are annual, monthly, daily, or irregular, and they do not see gaps. Three practical consequences:

- A trend in monthly data is a different statement from a trend in annual data, and a long daily record can flag smaller changes than a short annual one, simply because it has more observations.
- Seasonality is not removed. Monthly values with a strong seasonal cycle can look "non-random" because of the seasons alone. When that matters, aggregate to a suitable step (for example, annual values) or use seasonally adjusted data.
- Missing or irregularly spaced records are treated as if the sequence simply continued, so a gap is invisible and can mislead.

### Interpret in context

A small p-value means "this pattern is unlikely to be chance" — not "the river has changed". A large p-value means "no evidence of a problem here" — not "the data are correct". Always read the numbers alongside what you know: new construction or a dam, land-use change, a relocated gauge, a new instrument or analyst, a change in sampling. Treat the representative period as a well-supported suggestion for analysis, not a promise about future conditions.

## References

Coelho, M.; Fernandes, C. V. S.; Detzel, D. H. M.; Mannich, M. [Statistical validity of water quality time series in urban watersheds](https://www.scielo.br/j/rbrh/a/KyMFpKdRKYHvzh8pjQhvGCM/?lang=en). Brazilian Journal of Water Resources, v.22, e51, 2017.

Coelho, M.; Fernandes, C. V. S.; Detzel, D. H. M. [Uncertainty analysis in the detection of trends, cycles, and shifts in water resources time series](https://link.springer.com/article/10.1007/s11269-019-02210-1). Water Resources Management, v.27, p.1 - 16, 2019.

HELSEL & HIRSCH (2002). [Techniques of Water Resources investigations fo the United States Geological Survey. Chapter 5 - Statistical Methods in Water Resources](https://pubs.usgs.gov/tm/04/a03/tm4a3.pdf).

NAGHETTINI, M.; PINTO, E. J. A. Hidrologia e estatística. Geological Survey of Brazil. Belo Horizonte: CPRM, 2007.

Wald, A., & Wolfowitz, J. (1943). [An exact test for randomness in the non-parametric case based on serial correlation](https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2Faoms%2F1177731358). Annals of Mathematical Statistics, 14(4), 378-388.

## Install and Run

```bash
python -m venv .venv
```

```bash
.venv/scripts/activate
```

```bash
pip install -e .[dev]
```

To see an example:

```bash
rhis
```

## Using RHIS

Give `Rhis` either a `pandas.DataFrame` or a mapping of series names to value sequences.

```py
import pandas as pd
from rhis import Rhis

# From a DataFrame: each column is a series, each row a time step.
df = pd.read_csv("data.csv")

rhis = Rhis(df)

# ... or from a dict of series name -> sequence of values.
rhis = Rhis({
    "flow": [1100, 1160, 1160, 813, 1120, 1150],
    "rain": [420, 390, 505, 460, 480, 510],
})
```

Both forms expose the same API:

```py
# 1. The RHIS evolution, as a nested dict: one list of p-values per hypothesis
#    ("R", "H", "I", "S") plus their row-wise minimum ("RHIS-min"). Entry i is
#    the p-value of the period that starts at observation i and runs to the
#    most recent one.
evolutions = rhis.build_rhis_evol_dict()
evol = evolutions["flow"]

# 2. The representative series, as positional (start, end) indexes:
#    {"flow": (start, end)}, ready to slice with .iloc[start:end].
repr_idxs = rhis.get_repr_idxs()

start, end = repr_idxs["flow"]
representative = df.iloc[start:end]
```

If the DataFrame columns are a MultiIndex, the series keys are the column-label tuples (for example `("flow", "m3s")`), and you look them up with that tuple.
