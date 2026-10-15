# RHIS-TS

## What is RHIS-TS?

RHIS stands for **R**andomness, **H**omogeneity, **I**ndependence and **S**tationarity. TS stands for **T**ime **S**eries. These are the four basic assumptions a time series is expected to satisfy to be considered *statistically representative* (Naghettini, 2007; Coelho et al., 2017; Coelho et al., 2019). They underpin almost every frequency analysis, design flood or trend study: if these assumptions do not hold, the results may not reflect current conditions.

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

RHIS-TS finds the **longest recent stretch of a time series that passes all four tests** — in other words, the longest part of the record you can "safely" use.

It tests every window ending at the most recent observation, starting with the longest possible window (the entire time series). It then shortens the window by removing one observation from the beginning at a time, until only the 8 most recent values remain. Eight observations is the minimum sample size: the smallest length at which all four tests are still able to reach significance.

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

## The Nile example

The Nile dataset is the annual river flow measured at Aswan, Egypt, from 1871 to 1970 — 100 years in units of 10⁸ m³. Time-series analyses (going back to Cobb, 1978) find an abrupt changepoint around 1898: the mean annual flow fell from roughly 109 billion m³ (1871–1897) to about 86 billion m³ after the turn of the century. The series is a classic in statistics, econometrics, and data science — shipped as the `Nile` dataset in R and in Statsmodels — and is widely used for ARIMA modeling, structural-break detection, and general time-series work.

The drop was mainly climatic: the summer monsoons over the Ethiopian highlands, which supply about 80% of the river's water, weakened when the tropical rain belt stopped shifting so far north, and an extreme El Niño (1896–1899) finished the job. The first Aswan Dam started that same year, but it was small, not yet finished, and could not affect water falling upstream in Ethiopia, so the 20–25% fall was natural. Eltahir (1996) confirmed the link statistically, matching Aswan records (1872–1972) with Pacific sea-surface temperatures: ENSO explained about 25% of the year-to-year variation. During El Niño, shifted atmospheric circulation brings less rain over the Ethiopian Highlands, so less water reaches the Blue Nile and flow downstream drops.

### RHIS-TS applied on Nile time series

![Nile](/assets/RHIS%20flow.PNG)

This figure shows the results of the **rhis-ts** package on the Nile series. **Top:** the full annual flow series together with the RHIS-compliant stretch, `flow_repr` (1896–1970). **Bottom:** the evolution of the four RHIS p-values. Reading from left to right, the first point comes from the tests applied to the full series; each following point re-runs the tests on the series from that time step onward, so the last point is the test on the most recent 10 values. Under RHIS-min, all p-values start below alpha, and homogeneity and stationarity stop being rejected around 1885. Independence, however, keeps being rejected until 1896, when all four hypotheses are finally non-rejected. The stretch from 1896 to 1970 is therefore the most recent period in which all four hypotheses hold — a single population that can be safely used for frequency analysis, matching the changepoint reported by Cobb (1978) and the climatic explanation of Eltahir (1996).

Data source: [Flow of the River Nile](https://www.kaggle.com/datasets/lsind18/flow-of-the-river-nile)

## What do these tests actually test?

Every test returns a **p-value**: the probability of seeing a pattern at least this strong if the assumption were really true. A small p-value is evidence against the assumption. In RHIS-TS an assumption is rejected when its p-value falls below the significance level (alpha), which is 0.05 by default. Here is what each of the four tests looks at.

### Randomness

A random series has no systematic order. The randomness test (Wallis-Moore) counts the "ups" and "downs" in the series: a random series changes direction often, a series with too few changes keeps moving the same way (trend-like), and one with too many changes keeps zig-zagging. Randomness is rejected when this pattern is statistically significant at the chosen significance level. Seasonality is a classic source of non-randomness in the "too many changes" direction: a seasonal series turns up and down on a fixed schedule, packing far more direction changes into the record than chance alone would produce, and the test flags that regular zig-zag as non-random.

### Homogeneity

A homogeneous series behaves like a single population from beginning to end. The test (Mann-Whitney) splits the record into two halves and compares them. Homogeneity is rejected if one half is systematically larger or smaller than the other at the chosen significance level — a sign of a shift. A gradual trend causes the same problem: it makes the second half systematically higher (or lower) than the first, so the two halves differ even though there is no abrupt break, and the result can look very similar to what the stationarity test (Mann-Kendall) reports.

### Independence

A series is independent when one observation tells you nothing about the next. In practice values often carry memory: after a wet period, groundwater keeps feeding the river, so flows stay high and then decay slowly. The test (Wald-Wolfowitz) checks whether neighbouring values are too similar (positive correlation) or too different (negative correlation). Independence is rejected when this dependence is statistically significant at the chosen significance level.

### Stationarity

A stationary series keeps its statistical behaviour over time and, in particular, has no lasting upward or downward trend. The test (Mann-Kendall) compares every pair of observations and asks whether increases clearly outnumber decreases (or the reverse). Stationarity is rejected when such a monotonic trend is statistically significant at the chosen significance level.

## What to keep in mind

RHIS-TS gives you statistics; you bring the context. A few things are worth knowing before you act on a result.

### What kind of series it works with

RHIS-TS takes a table of numbers in time order, oldest row first, one column per variable — annual streamflow, monthly rainfall, daily water levels, water-quality concentrations, and so on. You can hand it that table in either of two ways:

- a `pandas.DataFrame` (every row is a time step, so the row index must not be a MultiIndex; the columns may use a MultiIndex, in which case the series are keyed by the column-label tuples), or
- a mapping (for example a `dict`) of series names to sequences of numbers, read positionally — a `pandas.Series` index is ignored, and series of different lengths are padded with NaN internally. The reported indexes are always relative to each original series, never to the padded table.

**Each column needs at least 8 numeric values** to be considered, and every stretch it tests is also at least 8 observations long. Missing or non-numeric entries are skipped, not filled in, so gaps quietly reduce how much data each window really has.

### What the tests can and cannot tell you

These are statistical tests, not detectors of causes. A "fail" tells you a pattern is unlikely to be pure chance; it does not tell you why. A "pass" is not a certificate that the data are sound. On short records a real trend or shift can go unnoticed for lack of evidence; on very long records even a tiny, physically meaningless change can look significant. The threshold (alpha, 0.05 by default) is a convention, not a law of nature.

### Outliers

An outlier is a value far from the rest — sometimes real (a record flood) and sometimes a mistake (a typo, a sensor glitch, a unit mix-up). These tests work with the order, ranks, or signs of the values rather than their size, so they tolerate outliers better than a simple average. Even so, one extreme value (say 999.9 instead of 99.9) can still shift the rankings and change the chosen period. RHIS-TS does not detect or remove outliers: look at them first, fix genuine data-entry errors, and use judgement for the real extremes.

### RHIS-TS cannot see your time step

The tests read a sequence of numbers, not dates. They do not know whether your data are annual, monthly, daily, or irregular, and they do not see gaps. Three practical consequences:

- A trend in monthly data is a different statement from a trend in annual data, and a long daily record can flag smaller changes than a short annual one, simply because it has more observations.
- Seasonality is not removed. Monthly values with a strong seasonal cycle can look "non-random" because of the seasons alone. When that matters, aggregate to a suitable step (for example, annual values) or use seasonally adjusted data.
- Missing or irregularly spaced records are treated as if the sequence simply continued, so a gap is invisible and can mislead.

### Interpret in context

A small p-value means "this pattern is unlikely to be chance" — not "the river has changed". A large p-value means "no evidence of a problem here" — not "the data are correct". Always read the numbers alongside what you know: new construction or a dam, land-use change, a relocated gauge, a new instrument or analyst, a change in sampling. Treat the representative period as a well-supported suggestion for analysis, not a promise about future conditions.

## References

Coelho, M.; Fernandes, C. V. S.; Detzel, D. H. M.; Mannich, M. [Statistical validity of water quality time series in urban watersheds](https://www.scielo.br/j/rbrh/a/KyMFpKdRKYHvzh8pjQhvGCM/?lang=en). Brazilian Journal of Water Resources, v.22, e51, 2017.

Coelho, M.; Fernandes, C. V. S.; Detzel, D. H. M. [Uncertainty analysis in the detection of trends, cycles, and shifts in water resources time series](https://link.springer.com/article/10.1007/s11269-019-02210-1). Water Resources Management, v.27, p.1 - 16, 2019.

Eltahir, A. B. E. [El Niño and the natural variability in the flow of the Nile River](https://web.mit.edu/people/eltahir/www/Publications_files/1996%20Eltahir%20flow%20Nile%20River%20WRR.pdf). Water Resources Research, Vol. 32, No. 1, Pages 131–137, January 1996.

George W. Cobb. [The problem of the Nile: Conditional solution to a changepoint problem](https://doi.org/10.1093/biomet/65.2.243), Biometrika, Volume 65, Issue 2, August 1978, Pages 243–251.

Helsel & Hirsch (2002). [Techniques of Water Resources investigations fo the United States Geological Survey. Chapter 5 - Statistical Methods in Water Resources](https://pubs.usgs.gov/tm/04/a03/tm4a3.pdf).

Naghettini, M.; Pinto, E. J. A. Hidrologia e estatística. Geological Survey of Brazil. Belo Horizonte: CPRM, 2007.

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
rhis-ts
```

## Using RHIS-TS

Give `Rhis` either a `pandas.DataFrame` or a mapping of series names to value sequences.

```py
import pandas as pd
from rhis_ts import Rhis

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
