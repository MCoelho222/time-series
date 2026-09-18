# Wald-Wolfowitz Test for Independence and Stationarity

Performs the non-parametric Wald-Wolfowitz test for independence
and stationarity.

## Usage

    ww.test(x)

## Arguments

### `x`

A vector or a time series object of class `"ts"`.

## Details

Let \(x_1, x_2, \ldots, x_n\) denote the sampled data. The test
statistic of the Wald-Wolfowitz test is calculated as:

\[
R = \sum_{i=1}^{n-1} x_i x_{i+1} + x_1 x_n
\]

The expected value of \(R\) is:

\[
E(R) = \frac{s_1^2 - s_2}{n-1}
\]

The expected variance of \(R\) is:

\[
V(R)
=
\frac{s_2^2 - s_4}{n-1}
-
E(R)^2
+
\frac{
s_1^4 - 4s_1^2s_2 + 4s_1s_3 + s_2^2 - 2s_4
}{
(n-1)(n-2)
}
\]

where:

\[
s_t = \sum_{i=1}^{n} x_i^t,
\qquad t = 1,2,3,4
\]

For \(n > 10\), the test statistic is normally distributed, with:

\[
z = \frac{R - E(R)}{\sqrt{V(R)}}
\]

`ww.test()` calculates p-values from the standard normal
distribution for the two-sided case.

## Value

An object of class `"htest"` with the following components:

### `method`

A character string indicating the chosen test.

### `data.name`

A character string giving the name(s) of the data.

### `statistic`

The Wald-Wolfowitz z-value.

### `alternative`

A character string describing the alternative hypothesis.

### `p.value`

The p-value for the test.

## Note

`NA` values are omitted.

## References

- R. K. Rai, A. Upadhyay, C. S. P. Ojha and L. M. Lye (2013).
  Statistical analysis of hydro-climatic variables. In:
  R. Y. Surampalli, T. C. Zhang, C. S. P. Ojha, B. R. Gurjar,
  R. D. Tyagi and C. M. Kao (eds.), *Climate Change Modelling,
  Mitigation, and Adaptation*. Reston, VA: ASCE.
  doi: https://doi.org/10.1061/9780784412718

- A. Wald and J. Wolfowitz (1943). An exact test for randomness
  in the non-parametric case based on serial correlation.
  *Annals of Mathematical Statistics*, 14, 378–388.

- WMO (2009). *Guide to Hydrological Practices, Volume II:
  Management of Water Resources and Application of Hydrological
  Practices*. WMO-No. 168.