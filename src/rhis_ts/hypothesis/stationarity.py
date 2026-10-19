from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sts

from rhis_ts.custom_types import MannKendallResults
from rhis_ts.utils import p_value_normal_directional

if TYPE_CHECKING:
    from rhis_ts.custom_types import TimeSeriesFlex

MIN_TAU_LENGTH = 3
MIN_PAIR_LENGTH = 2


def mann_kendall(
        ts: TimeSeriesFlex,
        *,
        alternative: str = 'two-sided',
) -> MannKendallResults:
    """
    Apply the Mann-Kendall test for monotonic trend using the normal
    approximation, which is valid for series with 10 or more elements
    (GILBERT, 1987).

    The statistic is computed from all pairwise differences: each pair
    of observations (i, j) with i < j contributes +1 if x_j is greater
    than x_i, -1 if it is smaller, and 0 if they are tied. The trend is
    measured by the sum S of these contributions.

    Hypotheses

        Null hypothesis
            H0: There is no monotonic trend in the series; the
                observations are randomly ordered over time.

        Alternative hypothesis
            H1: (two-sided): A monotonic trend is present.
            H1: (less): A monotonic downward trend is present.
            H1: (greater): A monotonic upward trend is present.

    References
    ----------
        GILBERT, R. O. (1987). Statistical Methods for Environmental
        Pollution Monitoring.

        HELSEL & HIRSCH (2002). Techniques of Water Resources
        investigations of the United States Geological Survey. Chapter 3
        - Statistical Methods in Water Resources.

    Parameters
    ----------
        ts
            A time series to be tested (1D list or numpy ndarray).

        alternative
            One of the alternative hypotheses:
                two-sided
                greater
                less

    Return
    ------
        namedtuple
            ('MannKendallResults', ['statistic', 'p_value', 'alternative'])
            'alternative' reflects the alternative hypothesis used in
            the test.

    See Also
    --------
        src/rhis_ts/docs/hypothesis_tests/mann_kendall.md
            Full description of the statistic, its distribution, and the
            interpretation of the results.
    """
    arr = np.asarray(ts, dtype=float)
    n = arr.size
    if np.all(arr == arr[0]):
            return MannKendallResults(np.nan, np.nan, alternative)

    _, counts = np.unique(arr, return_counts=True)
    ties_factor = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))

    if n < MIN_TAU_LENGTH:
        test_s = 0.0 if n < MIN_PAIR_LENGTH else float(np.sign(arr[1] - arr[0]))
    else:
        n0 = n * (n - 1) / 2
        ties_pairs = float(np.sum(counts * (counts - 1)) / 2)
        tau = sts.kendalltau(np.arange(n), arr, method='asymptotic').statistic
        test_s = float(round(tau * np.sqrt((n0 - ties_pairs) * n0)))

    sigma = ((1 / 18) * ((n * (n - 1.) * (2. * n + 5.)) - ties_factor)) ** 0.5

    condition_value = 0.
    if test_s > condition_value:
        z = abs((test_s - 1.)/sigma)
    elif test_s < condition_value:
        z = abs((test_s + 1.)/sigma)
    else:
        z = condition_value

    result = p_value_normal_directional(test_s, condition_value, z, alternative)

    return MannKendallResults(test_s, result.p_value, alternative)

if __name__ == "__main__":
    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]

    p_value = mann_kendall(ts).p_value
    print(f"p-value: {p_value}")
