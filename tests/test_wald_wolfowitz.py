from __future__ import annotations

import numpy as np
import pytest

from rhis.hypothesis import wald_wolfowitz

CONSTANT_SERIES = [5.0] * 10
SHORT_SERIES = [1.0, 2.0]
DEGENERATE_VARIANCE_SERIES = [1.0, 1.0, 5.0]


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
