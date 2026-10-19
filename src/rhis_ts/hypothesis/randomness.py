from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rhis_ts.custom_types import RunsTestResults, WallisMooreResults
from rhis_ts.utils import p_value_normal_directional

if TYPE_CHECKING:
    from rhis_ts.custom_types import TimeSeriesFlex


def runs_test(
        ts: TimeSeriesFlex,
        alternative: str = 'two-sided',
        *,
        continuity: bool = True,
) -> RunsTestResults:
    """
    Apply the single-sample runs test on a time series.

    Uses the median as the cut point: observations above the median get
    a '+' sign, below it a '-' sign, and values equal to the median are
    ignored. The statistic returned is the number of runs (maximal
    sequences of consecutive equal signs).

    Hypotheses

        Null hypothesis
            H0: The events in the underlying population represented by the
                sample series are distributed randomly.

        Alternative hypothesis
            H1: (two-sided): The events in the underlying population
                represented by the sample series are distributed
                nonrandomly.
            H1: (less): Non-randomness due to too few runs.
            H1: (greater): Non-randomness due to too many runs.

    References
    ----------
        SHESKIN (2004). Handbook of Parametric and Nonparametric
        Statistical Procedures - Test 10. 3rd edition.

    Parameters
    ----------
        ts
            The time series (1D list or numpy ndarray).
        alternative
            One of the alternative hypotheses:
                two-sided
                greater
                less
        continuity
            If True, applies the correction for continuity for the normal
            approximation.

    Return
    ------
        A namedtuple
            ('RunsTestResults', ['statistic', 'p_value', 'alternative'])

    See Also
    --------
        src/rhis_ts/docs/hypothesis_tests/runs_test.md
            Full description of the statistic, its distribution, and the
            interpretation of the results.
    """
    arr = np.asarray(ts, dtype=float)
    if np.all(arr == arr[0]):
        return RunsTestResults(np.nan, np.nan, alternative)

    median = np.median(arr)
    signs = np.where(arr > median, 1, np.where(arr < median, -1, 0))
    signs = signs[signs != 0]

    if signs.size == 0:
        return RunsTestResults(0, 0.0, alternative)

    n1 = float(np.count_nonzero(signs > 0))
    n2 = float(signs.size - n1)
    stat = float(1 + np.count_nonzero(signs[1:] != signs[:-1]))

    try:
        stat_mean = (((2. * n1 * n2) / (n1 + n2)) + 1.)
        var_num = (2. * n1 * n2 * (2. * n1 * n2 - n1 - n2))
        var_den = ((n1 + n2) ** 2 * (n1 + n2 - 1.))
        num_z = (abs(stat - stat_mean) - 0.5) if continuity else stat - stat_mean
        z = num_z / ((var_num / var_den) ** 0.5)
    except ZeroDivisionError:
        return RunsTestResults(0, 0.0, alternative)

    result = p_value_normal_directional(stat, stat_mean, z, alternative)
    return RunsTestResults(stat, result.p_value, alternative)


def wallis_moore(
    ts: TimeSeriesFlex,
    *,
    alternative: str = 'two-sided',
) -> WallisMooreResults:
    """
    Apply the Wallis and Moore (1941) phase test for randomness.

    Consecutive observations are compared and each pair is classified as
    a rise ('+') or a fall ('-'). Ties are handled by counting the phase
    runs twice - once treating equal values as rises and once treating
    them as falls - and averaging the two counts. The reported statistic
    is that average number of phase runs. Under randomness its mean is
    (2n-1)/3 and its standard deviation sqrt((16n-29)/90), with n the
    number of observations.

    Hypotheses

        Null hypothesis
            H0: The observations are random.
        Alternative hypothesis
            H1: (two-sided): The observations are not random.
            H1: (less): Non-randomness due to too few phases.
            H1: (greater): Non-randomness due to too many phases.

    References
    ----------
        Wallis, W. A., & Moore, G. H. (1941). A significance test for
        time series analysis. Journal of the American Statistical
        Association, 36(215), 401-409.

    Parameters
    ----------
        ts
            A time series (1D list or numpy ndarray).
        alternative
            One of the alternative hypotheses:
                two-sided
                greater
                less

    Return
    ------
        A namedtuple
            ('WallisMooreResults', ['statistic', 'p_value', 'alternative'])

    See Also
    --------
        src/rhis_ts/docs/hypothesis_tests/wallis_moore.md
            Full description of the phase statistic, its distribution,
            and the interpretation of the results.
    """
    ts_arr = np.asarray(ts, dtype=float)
    if np.all(ts_arr == ts_arr[0]):
        return WallisMooreResults(np.nan, np.nan, alternative)

    diffs = np.diff(ts_arr)

    signs1 = np.where(diffs >= 0, 1, -1)
    signs2 = np.where(diffs > 0, 1, -1)

    up_runs_ones_sum = np.count_nonzero(signs1[1:] != signs1[:-1]) + 1
    down_runs_ones_sum = np.count_nonzero(signs2[1:] != signs2[:-1]) + 1

    runs = (up_runs_ones_sum + down_runs_ones_sum) / 2.

    n = ts_arr.size
    expected_runs = (2. * n - 1.) / 3.
    sigma = ((16. * n - 29.) / 90.) ** 0.5
    z = (runs - expected_runs) / sigma

    result = p_value_normal_directional(runs, expected_runs, z, alternative)
    return WallisMooreResults(runs, result.p_value, alternative)


if __name__ == "__main__":
    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    p_value = wallis_moore(ts).p_value
    print(f"p-value: {p_value}")
