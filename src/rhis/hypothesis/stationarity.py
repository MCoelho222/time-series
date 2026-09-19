from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rhis.custom_types import MannKendallResults
from rhis.utils import ranks_with_ties_corrected, test_decision_normal

if TYPE_CHECKING:
    from rhis.custom_types import TimeSeriesFlex


def mann_kendall(
        ts: TimeSeriesFlex,
        alpha: float = 0.05,
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

        alpha
            The significance level for the test. Default is 0.05.

    Return
    ------
        namedtuple
            ('MannKendallResults', ['statistic', 'p_value', 'reject',
            'alternative'])
            The parameter 'reject' is of type bool. 'True' means the
            null hypothesis was rejected. 'alternative' reflects the
            alternative hypothesis used in the test.

    See Also
    --------
        src/rhis/docs/hypothesis_tests/mann_kendall.md
            Full description of the statistic, its distribution, and the
            interpretation of the results.
    """
    n = len(ts)
    ts = np.array(ts)
    signs = []

    for i in range(n - 1):
        s = ts[i + 1] - ts[:i + 1]
        signs.extend(np.sign(s))

    signs_array = np.array(signs)
    test_s = float(len(signs_array[signs_array > 0]) - len(signs_array[signs_array < 0]))

    ties_data = ranks_with_ties_corrected(ts, ties_data=True)['ties_groups_count']

    ties_factor = 0
    for value in ties_data:
        ties_factor += (value * (value - 1) * (2 * value + 5))

    sigma = ((1 / 18) * ((n * (n - 1.) * (2. * n + 5.)) - ties_factor)) ** 0.5

    condition_value = 0.
    if test_s > condition_value:
        z = abs((test_s - 1.)/sigma)
    elif test_s < condition_value:
        z = abs((test_s + 1.)/sigma)
    else:
        z = condition_value

    decision = test_decision_normal(test_s, condition_value, z, alternative, alpha)

    return MannKendallResults(test_s, decision.p_value, decision.reject, alternative)

if __name__ == "__main__":
    from rhis.plotting import plot_test

    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    p_value = mann_kendall(ts).p_value
    plot_test(ts, p_value, filename='stationarity', title='Stationarity Test Example')
    print(f"p-value: {p_value}")
