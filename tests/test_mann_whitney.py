from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from rhis.hypothesis import mann_whitney
from rhis.utils import ranks_with_ties_corrected, split_into_parts

P_VALUE_TOL = 1e-4
SCIPY_TOL = 1e-6
ALPHA = 0.05
P_HALF = 0.5

X = [0.59, 0.87, 1.1, 1.1, 1.2, 1.3, 1.6, 1.7, 3.2, 4.0] # industrial site
Y = [0.3, 0.36, 0.5, 0.7, 0.7, 0.9, 0.92, 1., 1.3, 9.7] # residential site


def test_mann_whitney():
    """
    Test the Mann-Whitney hypothesis test.

    Example from the book Statistical Methods in Water Resources.

    Author: Helsel & Hirsch
    Year: 2002
    Source: https://pubs.usgs.gov/tm/04/a03/tm4a3.pdf

    Chapter 5 - Differences between two independent groups

    Compares 10 observations from an industrial site against 10 from a
    residential site. The test asserts that:

    - the group medians match those reported in the book (1.25 and 0.8);
    - the two-sided p-value equals the book value (0.0491) within a
      tolerance of 1e-4;
    - the p-value agrees with the independent asymptotic reference
      implementation scipy.stats.mannwhitneyu (method='asymptotic').

    See also: src/rhis/docs/hypothesis_tests/mann_whitney.md
    """
    x = X
    y = Y

    median_x = 1.25
    median_y = 0.8
    expected_p = 0.0491

    result_a = mann_whitney(x=x, y=y)
    result_b = mannwhitneyu(x=x, y=y, method='asymptotic')

    assert np.median(np.array(x)) == median_x
    assert np.median(np.array(y)) == median_y
    assert result_a.p_value == pytest.approx(expected_p, abs=P_VALUE_TOL)
    assert result_a.p_value == pytest.approx(result_b.pvalue, abs=P_VALUE_TOL)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("continuity", [True, False])
def test_mann_whitney_matches_scipy(alternative, continuity):
    """
    The p-value matches scipy for every alternative and continuity
    setting on data with ties.
    """
    x = X
    y = Y

    result = mann_whitney(x, y, alternative=alternative, continuity=continuity)
    reference = mannwhitneyu(
        x, y,
        alternative=alternative,
        method='asymptotic',
        use_continuity=continuity,
        )

    assert result.p_value == pytest.approx(reference.pvalue, abs=SCIPY_TOL)
    assert result.reject == (result.p_value < ALPHA)
    assert result.alternative == alternative


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_mann_whitney_without_ties_matches_scipy(alternative):
    """
    With distinct (tie-free) observations, the ties=False path agrees
    with scipy for every alternative.
    """
    x = [1., 2., 3., 4., 5., 6., 7., 8.]
    y = [10., 11., 12., 13., 14., 15.]

    result = mann_whitney(x, y, alternative=alternative, ties=False)
    reference = mannwhitneyu(x, y, alternative=alternative, method='asymptotic')

    assert result.p_value == pytest.approx(reference.pvalue, abs=SCIPY_TOL)


def test_mann_whitney_statistic_is_minimum_u():
    """
    The reported statistic is the minimum of U1 and U2 computed from the
    average ranks of each group.
    """
    x = X
    y = Y
    gs_sorted = np.sort(x + y)
    ranks = np.sort(ranks_with_ties_corrected(x + y))
    ranks_dict = dict(zip(gs_sorted, ranks, strict=True))

    n1 = len(x)
    n2 = len(y)
    rank_sum1 = sum(ranks_dict[value] for value in x)
    rank_sum2 = sum(ranks_dict[value] for value in y)

    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2 - rank_sum1
    u2 = n1 * n2 + (n2 * (n2 + 1)) / 2 - rank_sum2

    result = mann_whitney(x, y)

    assert result.statistic == min(u1, u2)


def test_mann_whitney_one_sided_direction():
    """
    One-sided p-values reflect the observed direction: the industrial
    site has a larger median, so 'greater' rejects while 'less' does not.
    """
    greater = mann_whitney(X, Y, alternative='greater')
    less = mann_whitney(X, Y, alternative='less')

    assert greater.p_value == pytest.approx(0.0246, abs=P_VALUE_TOL)
    assert greater.reject
    assert not less.reject
    assert less.p_value > P_HALF


def test_mann_whitney_one_sided_wrong_direction():
    """
    When the data contradict the one-sided alternative, the p-value
    approaches 1 and the null hypothesis is not rejected.
    """
    result = mann_whitney(Y, X, alternative='greater')

    assert result.p_value > P_HALF
    assert not result.reject


def test_mann_whitney_univariate_split():
    """
    With y=None the series is split into two halves, which are compared;
    this agrees with calling scipy directly on the two halves.
    """
    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    halves = split_into_parts(ts, 2)

    result = mann_whitney(ts)
    reference = mannwhitneyu(halves[0], halves[1], method='asymptotic')

    assert result.p_value == pytest.approx(reference.pvalue, abs=P_VALUE_TOL)


def test_mann_whitney_constant_input():
    """
    When both groups contain a single identical value, the test is
    undefined; a p-value of 1.0 (no rejection) is returned instead of
    raising.
    """
    result = mann_whitney([5.0] * 7, [5.0] * 5, alternative='two-sided')

    assert result.statistic == 0
    assert result.p_value == 1.0
    assert result.reject is False
    assert result.alternative == 'two-sided'



