from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from rhis.hypothesis import wald_wolfowitz
from rhis.utils import ranks_with_ties_corrected, to_ranks

CONSTANT_SERIES = [5.0] * 10
SHORT_SERIES = [1.0, 2.0]
DEGENERATE_VARIANCE_SERIES = [1.0, 1.0, 5.0]
ALPHA_DEFAULT = 0.05
ALPHA_LOOSE = 0.10
STRONG_DEPENDENCE_P = 0.01


def test_wald_wolfowitz():
    """
    Test the Wald-Wolfowitz hypothesis test.

    Example from the book Hydrological Statistic (Hidrologia Estatística - Brazil)
    Chapter 7 - Hypothesis Tests - example 7.6, p. 267
    Auhtor: Naghettini & Pinto
    Year: 2007
    """
    ts1 = [104.3, 97.9, 89.2, 92.7, 98, 141.7, 81.1, 97.3, 72,\
            93.9, 83.8, 122.8, 87.6, 101, 97.8, 59.9, 49.4, 57,\
            68.2, 83.2, 60.6, 50.1, 68.7, 117.1, 80.2, 43.6, 66.8,\
            118.4, 110.4, 99.1, 71.6]

    ts2 = [62.6, 61.2, 46.8, 79, 96.3, 77.6, 69.3, 67.2, 72.4, 78,\
            141.8, 100.7, 87.4, 100.2, 166.9, 74.8, 133.4, 85.1, 78.9,\
            76.4, 64.2, 53.2, 112.2, 110.8, 82.2, 88.1, 80.9, 89.8, 114.9,\
            63.6, 57.3]

    ts = ts1 + ts2

    expected_stat = 8254
    expected_p = 0.059
    expected_reject = False

    result = wald_wolfowitz(ts)

    stat_err = abs(result.statistic - expected_stat) / expected_stat
    p_err = abs(result.p_value - expected_p) / expected_p

    accepted_stat_err = 0.001
    accepted_p_err = 0.1

    assert stat_err <= accepted_stat_err
    assert p_err <= accepted_p_err
    assert expected_reject == result.reject


def test_wald_wolfowitz_statistic_formula() -> None:
    """
    The statistic, p-value, and reject flag follow the documented
    formulas computed independently for the series [2, 5, 1, 8].
    """
    series = [2.0, 5.0, 1.0, 8.0]
    arr = np.array(series) - np.mean(series)
    n = len(arr)

    r = np.sum(arr[:-1] * arr[1:]) + arr[0] * arr[-1]

    s2 = float(np.sum(arr ** 2))
    s4 = float(np.sum(arr ** 4))

    e_r = -s2 / (n - 1)
    var_r = (s2 ** 2 - s4) / (n - 1) \
        + (s2 ** 2 - 2 * s4) / ((n - 1) * (n - 2)) \
        - s2 ** 2 / (n - 1) ** 2

    z = abs((r - e_r) / np.sqrt(var_r))
    p = 2 * (1 - norm.cdf(z))

    result = wald_wolfowitz(series)

    assert result.statistic == pytest.approx(r, abs=1e-9)
    assert result.p_value == pytest.approx(p, abs=1e-9)
    assert result.reject == (p < ALPHA_DEFAULT)


def test_wald_wolfowitz_alpha_threshold() -> None:
    """
    The p-value does not depend on alpha; only the rejection decision
    does. The book example has p ~ 0.0595, so it is rejected at
    alpha = 0.10 but not at alpha = 0.05.
    """
    ts1 = [104.3, 97.9, 89.2, 92.7, 98, 141.7, 81.1, 97.3, 72,\
            93.9, 83.8, 122.8, 87.6, 101, 97.8, 59.9, 49.4, 57,\
            68.2, 83.2, 60.6, 50.1, 68.7, 117.1, 80.2, 43.6, 66.8,\
            118.4, 110.4, 99.1, 71.6]
    ts2 = [62.6, 61.2, 46.8, 79, 96.3, 77.6, 69.3, 67.2, 72.4, 78,\
            141.8, 100.7, 87.4, 100.2, 166.9, 74.8, 133.4, 85.1, 78.9,\
            76.4, 64.2, 53.2, 112.2, 110.8, 82.2, 88.1, 80.9, 89.8, 114.9,\
            63.6, 57.3]
    ts = ts1 + ts2

    at_default = wald_wolfowitz(ts, alpha=ALPHA_DEFAULT)
    at_loose = wald_wolfowitz(ts, alpha=ALPHA_LOOSE)

    assert at_loose.p_value == pytest.approx(at_default.p_value, abs=1e-9)
    assert not at_default.reject
    assert at_loose.reject


def test_wald_wolfowitz_on_ranks_equals_explicit_ranks() -> None:
    """
    Applying the test with on_ranks=True is equivalent to feeding the
    precomputed ranks with on_ranks=False, both with and without the
    correction for ties.
    """
    series = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0, 11.0, 3.0]

    with_ties = wald_wolfowitz(series, on_ranks=True, ties=True)
    explicit_ties = wald_wolfowitz(ranks_with_ties_corrected(series), on_ranks=False)
    assert with_ties.statistic == pytest.approx(explicit_ties.statistic, abs=1e-9)
    assert with_ties.p_value == pytest.approx(explicit_ties.p_value, abs=1e-9)

    without_ties = wald_wolfowitz(series, on_ranks=True, ties=False)
    explicit_ranks = wald_wolfowitz(to_ranks(series), on_ranks=False)
    assert without_ties.statistic == pytest.approx(explicit_ranks.statistic, abs=1e-9)
    assert without_ties.p_value == pytest.approx(explicit_ranks.p_value, abs=1e-9)


def test_wald_wolfowitz_minimum_valid_length() -> None:
    """
    The shortest series that yields a non-degenerate statistic has four
    observations; every three-observation series, even with distinct
    values, has a near-zero variance and is rejected.
    """
    result = wald_wolfowitz([2.0, 5.0, 1.0, 8.0])

    assert 0.0 <= result.p_value <= 1.0

    with pytest.raises(ValueError, match='too small'):
        wald_wolfowitz([1.0, 2.0, 4.0])


def test_wald_wolfowitz_strong_positive_dependence() -> None:
    """
    A strictly increasing series exhibits strong positive serial
    dependence, so the independence hypothesis is rejected.
    """
    result = wald_wolfowitz(list(np.arange(1.0, 21.0)))

    assert result.p_value < STRONG_DEPENDENCE_P
    assert result.reject


def test_wald_wolfowitz_raises_on_constant_series() -> None:
    with pytest.raises(ValueError, match='at least two distinct values'):
        wald_wolfowitz(CONSTANT_SERIES)


def test_wald_wolfowitz_raises_on_non_finite_values() -> None:
    series = [np.nan, 1.0, 2.0, 3.0, 4.0]

    with pytest.raises(ValueError, match='only finite numeric values'):
        wald_wolfowitz(series)


def test_wald_wolfowitz_raises_on_short_series() -> None:
    with pytest.raises(ValueError, match='at least 3 observations'):
        wald_wolfowitz(SHORT_SERIES)


def test_wald_wolfowitz_raises_on_degenerate_variance() -> None:
    with pytest.raises(ValueError, match='too small'):
        wald_wolfowitz(DEGENERATE_VARIANCE_SERIES)
