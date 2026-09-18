from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sts

from rhis.custom_types import WaldWolfowitzResults
from rhis.utils import ranks_with_ties_corrected, to_ranks

if TYPE_CHECKING:
    from rhis.custom_types import TimeSeriesFlex


def wald_wolfowitz(
        ts: TimeSeriesFlex,
        alpha: float = 0.05,*,
        on_ranks: bool = False,
        ties: bool = True,
        ) -> WaldWolfowitzResults:
    """
    Wald & Wolfowitz test for serial correlation.

    Test the hypothesis that x1, ..., xN are independent observations from the
    same population.

    References
        Wald A. and Wolfowitz J. (1943). An exact test for randomness in the
        non-parametric case based on serial correlation.

    Parameters
    ----------
        ts
            A time series to be tested.
        alpha
            The significance level for the test. Default is 0.05.
        on_ranks
            If True, the test will be applied on the ranks.
        ties
            If True and on_ranks is True, the ranks will be corrected for ties.

    Return
    ------
        A namedtuple
            ('WaldWolfowitzResults', ['statistic', 'p_value', 'reject'])
            The parameter 'reject' is of type bool. 'True' means the null
            hypothesis was reject.
    """
    arr = np.array(ts)
    if np.all(arr == arr[0]):
        reject = True
        return WaldWolfowitzResults(0, 0., reject)

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
        reject = True
        return WaldWolfowitzResults(0, 0., reject)

    z = abs((r - e_r) / np.sqrt(var_r))
    p = 2 * (1 - sts.norm.cdf(z))

    reject = p < alpha

    return WaldWolfowitzResults(r, p, reject)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from rhis.utils import to_ranks

    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]

    ts_ranks = to_ranks(ts)
    ts_p = wald_wolfowitz(ts, on_ranks=False).p_value
    ranks_p = wald_wolfowitz(ts, on_ranks=True).p_value

    ts_name = 'series'
    ts_color = 'blue'
    ranks_color = 'k'
    alpha_color = 'red'
    alpha = 0.05

    fig, pvalue_ax = plt.subplots(figsize=(8, 6))
    series_ax = pvalue_ax.twinx()
    series_ax.scatter(range(len(ts)), ts, color=ts_color, label=ts_name)
    series_ax.scatter(range(len(ts_ranks)), ts_ranks, color=ranks_color, label='ranks')

    pvalue_ax.axhline(ts_p, color=ts_color, linestyle='--', label=f'p-value ({ts_name}): {round(ts_p, 4)}')
    pvalue_ax.axhline(ranks_p, color=ranks_color, linestyle='--', label=f'p-value (ranks): {round(ranks_p, 4)}')

    pvalue_ax.axhline(alpha, color=alpha_color, label=f'alpha: {alpha}')

    pvalue_ax.set_xlabel('Time')
    pvalue_ax.set_ylabel('p-value')
    pvalue_ax.set_ylim(0, 1)
    series_ax.set_ylabel(ts_name)

    pvalue_handles, pvalue_labels = pvalue_ax.get_legend_handles_labels()
    series_handles, series_labels = series_ax.get_legend_handles_labels()

    pvalue_ax.legend(
                pvalue_handles + series_handles,
                pvalue_labels + series_labels,
                loc='upper left',
            )
    fig.suptitle('Independence Test Example', fontsize=12)
    fig.tight_layout()

    plots_dir = Path('hypothesis_testing_plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'independence.PNG', bbox_inches="tight")

    # print(wald_wolfowitz(ts, on_ranks=True, ties=True).p_value)
    print(f"p-value ({ts_name}): {ts_p}")
    print(f"p-value (ranks): {ranks_p}")
    # print(wald_wolfowitz(ts, on_ranks=False, ties=True).p_value)

