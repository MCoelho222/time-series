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
    Apply the Single-Sample Runs Test in on a time series. Uses the median as a
    criteria for defining runs (up or down).

    hypotheses

        Null hypothesis
            H0: The events in the underlying population represented by the sample
                series are distributed randomly.

        Alternative hypothesis
            H1: (two-sided): The events in the underlying population represented by
                the sample series are distributed nonrandomly.
            H1: (less): The events in the underlying population represented by the
                sample series are distributed non-randomly due to few runs.
            H1: (greater): The events in the underlying population represented by
                the sample series are distributed non-randomly due to too many runs.

    References

        SHESKIN (2004). Handbook of Parametric and Nonparametric Statistical
        Procedures - Test 10. 3rd edition.

    Parameters
    ----------
        ts
            The time series (1D list or numpy ndarray).
        alternative
            One of the alternative hypo:
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
            ('RunsTestResults', ['statistic', 'p_value', 'reject', 'alternative'])
            The parameter 'reject' is of type bool. 'True' means the null hypothesis
            was reject.
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
    return RunsTestResults(stat, round(decision.p_value, 4), decision.reject, alternative)


def wallismoore(
        ts: TimeSeriesFlex,
        alpha: float = 0.05,
        alternative: str = 'two-sided',
    ) -> WallisMooreResults:
    """
    Applies the Wallis and Moore (1941) runtest for randomness.

    Reference
    ---------
        SHESKIN (2004). Handbook of Parametric and Nonparametric Statistical
        Procedures - Test 10. 3rd edition.

    Parameters
    ----------
        ts
            1D list or numpy array.
        interval
            1D list or tuple with length 2. The first object is the index referent
            to the sample number to start the time series. The second number is last
            sample number.
        alpha
            The significance level for the test.

    Return
    -------
        A namedtuple
            ('WallisMooreResults', ['statistic', 'p_value', 'reject', 'alternative'])
            The parameter 'reject' is of type bool. 'True' means the null hypothesis
            was reject.
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
    return WallisMooreResults(runs, round(decision.p_value, 4), decision.reject, alternative)


if __name__ == "__main__":
    from rhis.plotting import plot_test

    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    p_value = wallismoore(ts).p_value
    plot_test(ts, p_value, filename='randomness', title='Randomness Test Example')
    print(f"p-value: {p_value}")
