# RHIS

## What is RHIS?

RHIS stands for **R**andomness, **H**omogeneity, **I**ndependence and **S**tationarity. These are the four basic assumptions a time series is expected to satisfy to be considered *statistically representative* (Naghettini, 2007; Coelho et al., 2017; Coelho, 2019). They underpin almost every frequency analysis, design flood or trend study: if these assumptions do not hold, the results may not reflect current conditions.

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

RHIS finds the **longest recent stretch of a time series that passes all four tests** — in other words, the longest part of the record you can safely use.

It tests every window that ends at the most recent observation: it starts with the smallest usable window (the last 5 or 10 values) and grows it one observation at a time until the complete series has been tested. Every window gets a p-value per hypothesis, and the results come back as a simple dict. Read from the oldest to the most recent starting point, the p-values show where the series first becomes fully RHIS-compliant — and that point is the selection boundary.

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

RHIS takes a table of numbers in time order, oldest row first, one column per variable — annual streamflow, monthly rainfall, daily water levels, water-quality concentrations, and so on. Each column needs at least 10 numeric values to be considered, and every window it tests needs a handful of values (5, or 10 for long records). Missing or non-numeric entries are skipped, not filled in, so gaps quietly reduce how much data each window really has.

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

## Scientific foundations

These tests were used in my doctoral thesis to assess the representativeness of water quality time series in the Alto Iguaçu Watershed in southern Brazil.

If you are interested, please check out the article below.

[Uncertainty analysis in the detection of trends, cycles, and shifts in water resources time series](https://link.springer.com/article/10.1007/s11269-019-02210-1)

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

```py
import pandas as pd
from rhis import Rhis

df = pd.read_csv("data.csv")

rhis = Rhis(df)

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
representative_df = df.iloc[start:end]
```
