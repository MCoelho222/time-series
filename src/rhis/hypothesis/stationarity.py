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
    Apply the Mann-Kendall test using the normal approximation,
    which is valid for series with 10 or more elements (GILBERT, 1987).

    References
    ----------
        GILBERT, R. O. (1987). Statistical Methods for Environmental Pollution
        Monitoring.

        HELSEL & HIRSCH (2002). Techniques of Water Resources investigations of
        the United States Geological Survey. Chapter 3 - Statistical Methods in
        Water Resources.

    Parameters
    ----------
        ts
            A time series to be tested.

        alternative
            'two-sided', 'greater', or 'less'.

        alpha
            The significance level for the test. Default is 0.05.

    Return
    ------
        namedtuple
            ('MannKendallResults', ['statistic', 'p_value', 'reject'])

            'reject' is boolean. If True, the null hypothesis was reject.
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
    if test_s == condition_value:
        z = condition_value
    if test_s < condition_value:
        z = abs((test_s + 1.)/sigma)

    decision = test_decision_normal(test_s, condition_value, z, alternative, alpha)

    return MannKendallResults(test_s, decision.p_value, decision.reject, alternative)

if __name__ == "__main__":
    from rhis.plotting import plot_test

    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    p_value = mann_kendall(ts).p_value
    plot_test(ts, p_value, filename='stationarity', title='Stationarity Test Example')
    print(f"p-value: {p_value}")
