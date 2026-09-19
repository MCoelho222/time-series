from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sts

from rhis.custom_types import WaldWolfowitzResults
from rhis.utils import ranks_with_ties_corrected, to_ranks

if TYPE_CHECKING:
    from rhis.custom_types import TimeSeriesFlex

MIN_SERIES_LENGTH = 3


def wald_wolfowitz(
        ts: TimeSeriesFlex,
        alpha: float = 0.05,*,
        on_ranks: bool = False,
        ties: bool = True,
        ) -> WaldWolfowitzResults:
    """
    Wald & Wolfowitz test for serial correlation.

    Tests the hypothesis that x1, ..., xN are independent observations from the
    same population, using the circular serial correlation statistic of Wald &
    Wolfowitz (1943). Because the series is centered and closed into a circle
    (the first and last observations are neighbours), the statistic captures
    both positive and negative serial dependence. Under the null hypothesis the
    standardized statistic approximately follows a standard normal
    distribution, and the null hypothesis is rejected when p_value < alpha.

    Parameters
    ----------
        ts
            A time series to be tested. It must contain at least three finite
            numeric values and at least two distinct values.
        alpha
            The significance level for the test. Default is 0.05.
        on_ranks
            If True, the test will be applied on the ranks of the series,
            making it distribution-free.
        ties
            If True and on_ranks is True, the ranks will be corrected for ties.

    Returns
    -------
        A namedtuple
            ('WaldWolfowitzResults', ['statistic', 'p_value', 'reject'])
            The parameter 'reject' is of type bool. 'True' means the null
            hypothesis was rejected.

    Raises
    ------
        ValueError
            If the series has fewer than three observations, contains
            non-finite values, is constant, or yields a degenerate
            (near-zero) variance for the serial correlation statistic.

    See Also
    --------
        src/rhis/docs/hypothesis_tests/wald_wolfowitz.md
            Full description of the statistic, its distribution, and the
            adaptations used here (circular closure, mean-centering, normal
            approximation, ranks and ties handling).

    References
    ----------
        Wald, A., & Wolfowitz, J. (1943). An exact test for randomness in the
        non-parametric case based on serial correlation. Annals of Mathematical
        Statistics, 14(4), 378-388.
    """
    arr = np.array(ts, dtype=float)
    if not np.all(np.isfinite(arr)):
        msg = "The time series must contain only finite numeric values."
        raise ValueError(msg)

    if len(arr) < MIN_SERIES_LENGTH:
        msg = f"The time series must have at least {MIN_SERIES_LENGTH} observations."
        raise ValueError(msg)

    if np.all(arr == arr[0]):
        msg = "The time series must contain at least two distinct values."
        raise ValueError(msg)

    if on_ranks and not ties:
        arr = np.array(to_ranks(arr))
    if on_ranks and ties:
        arr = ranks_with_ties_corrected(arr)

    avg = np.mean(arr)
    arr = arr - avg
    n = len(arr)

    r = np.sum(arr[:-1] * arr[1:]) + arr[0] * arr[-1]

    s2 = float(np.sum(arr ** 2))
    s4 = float(np.sum(arr ** 4))

    e_r = - s2 / (n - 1)

    a = (s2 ** 2 - s4) / (n - 1)
    b = (s2 ** 2 - 2 * s4) / ((n - 1) * (n - 2))

    c =  s2 ** 2 / (n - 1) ** 2
    var_r = a + b - c
    var_lim = 0.00001
    if abs(var_r) < var_lim:
        msg = "The variance of the serial correlation statistic is too small for the test to be meaningful."
        raise ValueError(msg)

    z = abs((r - e_r) / np.sqrt(var_r))
    p = 2 * (1 - sts.norm.cdf(z))

    reject = p < alpha

    return WaldWolfowitzResults(r, p, reject)

if __name__ == "__main__":
    from rhis.plotting import plot_test

    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]

    ts_p = wald_wolfowitz(ts, on_ranks=False).p_value
    ranks_p = wald_wolfowitz(ts, on_ranks=True).p_value

    plot_test(
        ts,
        ts_p,
        ranks_p_value=ranks_p,
        filename='independence',
        title='Independence Test Example',
    )

    print(f"p-value (series): {ts_p}")
    print(f"p-value (ranks): {ranks_p}")

