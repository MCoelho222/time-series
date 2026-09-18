from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import mannwhitneyu

from rhis.custom_types import MannWhitneyResults
from rhis.utils import ranks_with_ties_corrected, split_into_parts, test_decision_normal

if TYPE_CHECKING:
    from rhis.custom_types import TimeSeriesFlex


def mann_whitney(  # noqa: PLR0913
        x: TimeSeriesFlex,
        y: TimeSeriesFlex | None = None,
        alpha: float = 0.05,
        alternative: str = 'two-sided',
        *,
        continuity: bool = True,
        ties: bool = True,
        ) -> MannWhitneyResults:
    """
    Compare two independent groups of data using the Mann-Whitney U test.

    This implementation applies the large sample approximation (applicable
    if x, y > 10 elements). The Mann-Whitney test is also known as The
    Rank-Sum test or Wilcoxon Rank-Sum.

    Assumptions

        - Each sample has been randomnly selected from the population it
          represents.
        - The two samples are independent of one another.
        - The original variable observed (which is subsequently ranked) is
          a continuous random variable.
        - The underlying distributions from which the samples are derived
          are identical in shape.

    Null and Alternative hypotheses

        H0: prob[x > y] = 0.5

        H1: prob[ x > y] != 0.5 (two-sided)
        H2: prob[ x > y] > 0.5 (greater)
        H3: prob[ x > y] < 0.5 (less)

    References
    ----------
        HELSEL & HIRSCH (2002). Techniques of Water Resources
        investigations fo the United States Geological Survey. Chapter 5 -
        Statistical Methods in Water Resources, p.118.
        Source: https://pubs.usgs.gov/tm/04/a03/tm4a3.pdf

    Parameters
    ----------
        x
            A list of floats or integers.
        y
            A list of floats or integers.
        alternative
            two-sided: x != y
            greater: x > y
            less: x < y
        alpha
            The significance level (0.05 by default).
        continuity
            If True, applies correction for continuity.
        ties
            If True, applies correction for ties.

    Returns
    -------
        A namedtuple
            ('MannWhitneyResults', ['statistic', 'p_value', 'reject'])
            The parameter 'reject' is of type bool. 'True' means the null
            hypothesis was reject.
    """
    if y is None:
        data = split_into_parts(x, 2)
        x = data[0]
        y = data[1]

    g1 = list(x)
    g2 = list(y)

    gs_concat = g1 + g2
    gs_sorted = np.sort(gs_concat)

    if np.all(gs_sorted == gs_sorted[0]):
        reject = False
        return MannWhitneyResults(0, 1., reject, alternative)

    n = len(gs_concat)
    ranks = np.sort(ranks_with_ties_corrected(gs_concat)) if ties else [i + 1 for i in range(n)]

    ranks_dict = dict(zip(gs_sorted, ranks, strict=True))

    g1_ranks = [ranks_dict[value] for value in g1]
    g2_ranks = [ranks_dict[value] for value in g2]

    rank_sum1 = sum(g1_ranks)
    rank_sum2 = sum(g2_ranks)

    n1 = len(g1)
    n2 = len(g2)
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2 - rank_sum1
    u2 = n1 * n2 + (n2 * (n2 + 1)) / 2 - rank_sum2

    stat = min(u1, u2)

    mean_stat = (n1 * n2) / 2
    var = (n1 * n2 * (n1 + n2 + 1)) / 12

    if ties:
        var = ((n1 * n2) / ((n) * (n - 1))) * np.sum(np.array(ranks) ** 2) \
            - ((n1 * n2 * (n + 1) ** 2) / (4 * (n - 1)))

    z = abs(stat - mean_stat) / np.sqrt(var)

    if continuity:
        z = (abs(stat - mean_stat) - 0.5) / np.sqrt(var)

    decision = test_decision_normal(rank_sum1, rank_sum2, z, alternative, alpha)

    return MannWhitneyResults(stat, decision.p_value, decision.reject, alternative)


if __name__ == "__main__":
    from rhis.plotting import plot_test

    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    ts_splitted = split_into_parts(ts, 2)
    ts1 = ts_splitted[0]
    ts2 = ts_splitted[1]

    p_value = mann_whitney(ts).p_value
    plot_test(ts, p_value, filename='homogeneity', title='Homogeneity Test Example')

    print(f"p-value: {p_value}")
    print(mannwhitneyu(ts1, ts2, method='asymptotic').pvalue)
