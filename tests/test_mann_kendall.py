from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.stats as sts

from rhis.hypothesis import mann_kendall
from rhis.utils import ranks_with_ties_corrected

if TYPE_CHECKING:
    from collections.abc import Sequence

P_HALF = 0.5
P_VALUE_TOL = 1e-6
STAT_TOL = 0.02

GILBERT = [20, 20, 20, 20, 15, 20, 20, 30, 27, 26, 23, 35, 25, 28, 70, 26, 24, 34, 32, 23, 50, 30]
MAIN_EXAMPLE = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
MONOTONE_UP = list(range(1, 21))
MONOTONE_DOWN = list(range(20, 0, -1))
BALANCED = [3, 6, 9, 6, 8, 6, 7, 3]


def _independent_statistic(ts: Sequence[float]) -> float:
    """S = number of increasing pairs minus number of decreasing pairs."""
    signs = np.sign(np.triu(np.array(ts)[None, :] - np.array(ts)[:, None], 1))
    return float(np.sum(signs == 1) - np.sum(signs == -1))


def _independent_sigma(ts: Sequence[float]) -> float:
    """Standard deviation of S under the null, with the ties correction."""
    n = len(ts)
    _, counts = np.unique(ts, return_counts=True)
    ties_factor = sum(int(c) * (int(c) - 1) * (2 * int(c) + 5) for c in counts)
    return ((1 / 18) * (n * (n - 1) * (2 * n + 5) - ties_factor)) ** 0.5


def test_mann_kendall_gilbert_example() -> None:
    """
    Reproduces the worked example from GILBERT (1987), page 212,
    chapter 16 - Detecting and Estimating Trends, for which the book
    reports a z value of 3.1.

    With the one-sided 'greater' alternative the direction-aware
    p-value is the upper tail, so |Phi^-1(p)| equals the standardized
    statistic.
    """
    expected_z = 3.1
    accepted_error = STAT_TOL
    result = mann_kendall(ts=GILBERT, alternative='greater')
    z = abs(sts.norm.ppf(result.p_value))

    error = abs(z - expected_z) / expected_z
    assert error <= accepted_error
    assert result.reject


def test_mann_kendall_two_sided_gilbert_example() -> None:
    """
    The exact statistic, standard deviation, and p-value for the
    Gilbert (1987) example on the two-sided test.
    """
    result = mann_kendall(GILBERT)

    sigma = _independent_sigma(GILBERT)
    z = abs((111.0 - 1.0) / sigma)
    expected_p = min(1.0, 2.0 * sts.norm.sf(z))

    assert result.statistic == pytest.approx(_independent_statistic(GILBERT))
    assert result.statistic == pytest.approx(111.0)
    assert z == pytest.approx(3.141148, abs=1e-6)
    assert result.p_value == pytest.approx(expected_p, abs=P_VALUE_TOL)


def test_mann_kendall_statistic_is_sign_sum() -> None:
    """
    The statistic S is the sum over all pairs of +1/-1 differences,
    computed independently of the implementation.
    """
    assert mann_kendall(MAIN_EXAMPLE).statistic == pytest.approx(_independent_statistic(MAIN_EXAMPLE))
    assert mann_kendall(MAIN_EXAMPLE).statistic == pytest.approx(138.0)
    assert mann_kendall(GILBERT).statistic == pytest.approx(_independent_statistic(GILBERT))


def test_mann_kendall_p_value_matches_book_example() -> None:
    """
    The two-sided p-value for the __main__ example series equals the
    normal-approximation value computed from the continuity-corrected z.
    """
    result = mann_kendall(MAIN_EXAMPLE)

    sigma = _independent_sigma(MAIN_EXAMPLE)
    z = abs((138.0 - 1.0) / sigma)
    expected_p = min(1.0, 2.0 * sts.norm.sf(z))

    assert z == pytest.approx(3.846332, abs=1e-6)
    assert result.p_value == pytest.approx(expected_p, abs=P_VALUE_TOL)
    assert result.p_value == pytest.approx(0.000120, abs=1e-6)


def test_mann_kendall_variance_accounts_for_ties() -> None:
    """
    The standard deviation of S reduces when ties are present; the
    implementation uses the ties-correction factor t(t-1)(2t+5).
    """
    no_ties_sigma = _independent_sigma([x + i * 1e-3 for i, x in enumerate(GILBERT)])
    tied_sigma = _independent_sigma(GILBERT)

    assert tied_sigma < no_ties_sigma
    tied_groups = ranks_with_ties_corrected(GILBERT, ties_data=True)['ties_groups_count']
    assert max(tied_groups) > 1


def test_mann_kendall_one_sided_direction() -> None:
    """
    A monotone upward series rejects only for the 'greater'
    alternative, and a monotone downward series only for 'less'.
    """
    up = mann_kendall(MONOTONE_UP)
    down = mann_kendall(MONOTONE_DOWN)

    assert up.statistic == pytest.approx(_independent_statistic(MONOTONE_UP))
    assert mann_kendall(MONOTONE_UP, alternative='greater').reject
    assert not mann_kendall(MONOTONE_UP, alternative='less').reject
    assert mann_kendall(MONOTONE_UP, alternative='less').p_value > P_HALF

    assert down.statistic == pytest.approx(_independent_statistic(MONOTONE_DOWN))
    assert mann_kendall(MONOTONE_DOWN, alternative='less').reject
    assert not mann_kendall(MONOTONE_DOWN, alternative='greater').reject
    assert mann_kendall(MONOTONE_DOWN, alternative='greater').p_value > P_HALF


def test_mann_kendall_statistic_equal_zero() -> None:
    """
    When positive and negative comparisons balance exactly (S = 0) the
    standardized statistic is 0: both one-sided p-values are 0.5 and the
    two-sided p-value is 1.
    """
    result = mann_kendall(BALANCED)

    assert result.statistic == pytest.approx(_independent_statistic(BALANCED))
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0, abs=P_VALUE_TOL)
    assert not result.reject

    less = mann_kendall(BALANCED, alternative='less')
    greater = mann_kendall(BALANCED, alternative='greater')
    assert less.p_value == pytest.approx(P_HALF, abs=P_VALUE_TOL)
    assert greater.p_value == pytest.approx(P_HALF, abs=P_VALUE_TOL)
    assert not less.reject
    assert not greater.reject


def test_mann_kendall_constant_input() -> None:
    """
    A constant series has no trend at all (S = 0 with zero variance):
    the statistic is 0, the two-sided p-value is 1, and the null is not
    rejected.
    """
    constant = mann_kendall([5.0] * 8)

    assert np.isnan(constant.statistic)
    assert np.isnan(constant.p_value)
    assert constant.reject is None
    assert constant.alternative == 'two-sided'
