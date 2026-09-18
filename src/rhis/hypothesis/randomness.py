from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rhis.custom_types import RunsTestResults, WallisMooreResults
from rhis.utils import test_decision_normal

if TYPE_CHECKING:
    from rhis.custom_types import TimeSeriesFlex


def runs_test(  # noqa: C901
        ts: TimeSeriesFlex,
        alpha: float = 0.05,
        alternative: str = 'two-sided',*,
        continuity: bool = True
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
        alpha
            The significance level for the test.
        continuity
            If True, applies the correction for continuity for the normal
            approximation.

    Return
    ------
        A namedtuple
            ('RunsTestResults', ['statistic', 'p_value', 'reject',
            'alternative'])
            The parameter 'reject' is of type bool. 'True' means the null
            hypothesis was rejected.

    See Also
    --------
        src/rhis/docs/hypothesis_tests/runs_test.md
            Full description of the statistic, its distribution, and the
            interpretation of the results.
    """
    ts = np.array(ts) if isinstance(ts, list) else ts

    median = np.median(np.array(ts))
    up_runs_ones = []
    down_runs_ones = []
    signs = [] # +1 (higher than median); -1 (lower than median)
    runs_per_group = []
    for element in ts:
        # Values equal to the median do not count
        if element > median:
            signs.append(1)
        if element < median:
            signs.append(-1)

    if not signs:
        reject = True
        return RunsTestResults(0, 0.0, reject, alternative)

    for i in range(1, len(signs)):
        el = signs[i]
        next_el = signs[i - 1]
        if el < next_el:
            up_runs_ones.append(1)
        if el > next_el:
            down_runs_ones.append(1)

    if signs[0] > 0:
        runs_per_group.append(np.sum(up_runs_ones))
        runs_per_group.append(np.sum(down_runs_ones) + 1)
    if signs[0] < 0:
        runs_per_group.append(np.sum(up_runs_ones) + 1)
        runs_per_group.append(np.sum(down_runs_ones))

    signs1 = np.array(signs)

    positives = signs1[signs1 > 0]
    negatives = signs1[signs1 < 0]

    n1 = float(len(positives))
    n2 = float(len(negatives))
    stat = float(np.sum(runs_per_group))

    try:
        stat_mean = (((2. * n1 * n2) / (n1 + n2)) + 1.)
        var_num = (2. * n1 * n2 * (2. * n1 * n2 - n1 - n2))
        var_den = ((n1 + n2) ** 2 * (n1 + n2 - 1.))
        num_z = (abs(stat - stat_mean) - 0.5) if continuity else stat - stat_mean
        z = num_z / ((var_num / var_den) ** 0.5)
    except ZeroDivisionError:
        reject = True
        return RunsTestResults(0, 0.0, reject, alternative)

    decision = test_decision_normal(stat, stat_mean, z, alternative, alpha)
    return RunsTestResults(stat, decision.p_value, decision.reject, alternative)


def wallis_moore(
        ts: TimeSeriesFlex,
        alpha: float = 0.05,
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
        alpha
            The significance level for the test.
        alternative
            One of the alternative hypotheses:
                two-sided
                greater
                less

    Return
    ------
        A namedtuple
            ('WallisMooreResults', ['statistic', 'p_value', 'reject',
            'alternative'])
            The parameter 'reject' is of type bool. 'True' means the null
            hypothesis was rejected.

    See Also
    --------
        src/rhis/docs/hypothesis_tests/wallis_moore.md
            Full description of the phase statistic, its distribution,
            and the interpretation of the results.
    """
    ts_arr = np.array(ts)
    if np.all(ts_arr == ts_arr[0]):
        reject = True
        return WallisMooreResults(0, 0., reject, alternative)

    #Group 1 (pluses for zeros)
    signs1 = []
    pluses1 = []
    minuses1 = []
    up_runs_ones = []

    #Group 2 (minuses for zeros)
    signs2 = []
    pluses2 = []
    minuses2 = []
    down_runs_ones = []

    for i in range(1, len(ts)):

        if ts[i] < ts[i - 1]:
            signs1.append(-1)
            signs2.append(-1)
            minuses1.append(1)
            minuses2.append(1)

        if ts[i] > ts[i - 1]:
            signs1.append(1)
            signs2.append(1)
            pluses1.append(1)
            pluses2.append(1)

        if ts[i] == ts[i - 1]:
            signs1.append(1)
            signs2.append(-1)
            pluses1.append(1)
            minuses2.append(1)

    for i in range(1, len(ts) - 1):
        if signs1[i] != signs1[i - 1]:
            up_runs_ones.append(1)
        if signs2[i] != signs2[i - 1]:
            down_runs_ones.append(1)

    #Group 1
    up_runs_ones_arr = np.array(up_runs_ones)
    up_runs_ones_sum = np.sum(up_runs_ones_arr) + 1

    #Group 2
    down_runs_ones_arr = np.array(down_runs_ones)
    down_runs_ones_sum = np.sum(down_runs_ones_arr) + 1

    runs = (up_runs_ones_sum + down_runs_ones_sum) / 2.

    n = len(ts)
    expected_runs = (2. * n - 1.) / 3.
    sigma = ((16. * n - 29.) / 90.) ** 0.5
    z = (runs - expected_runs) / sigma

    decision = test_decision_normal(runs, expected_runs, z, alternative, alpha)
    return WallisMooreResults(runs, decision.p_value, decision.reject, alternative)


if __name__ == "__main__":
    from rhis.plotting import plot_test

    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    p_value = wallis_moore(ts).p_value
    plot_test(ts, p_value, filename='randomness', title='Randomness Test Example')
    print(f"p-value: {p_value}")
