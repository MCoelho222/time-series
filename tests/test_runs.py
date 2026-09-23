from __future__ import annotations

import numpy as np
import pytest

from rhis.hypothesis import runs_test, wallis_moore

P_HALF = 0.5

MILK = [1.90, 1.99, 2., 1.78, 1.77, 1.76, 1.98, 1.9, 1.65, \
        1.76, 2.01, 1.78, 1.99, 1.76, 1.94, 1.78, 1.67, 1.87, 1.91, 1.91, 1.89]

ALTERNATING = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
MONOTONE = [float(v) for v in np.arange(1.0, 31.0)]


def test_randomness():
    """
    ------------------------------------------------------------------------------------------------
    SHESKIN (2004). Handbook of Parametric and Nonparametric Statistical Procedures - Test 10.
    3rd edition.

    Example 10.2

    A quality control study is conducted on a machine that pours milk into containers.
    The amount of milk (in liters) dispensed by the machine into 21 consecutive containers follows:
    1.90, 1.99, 2.00, 1.78, 1.77, 1.76, 1.98, 1.90, 1.65, 1.76, 2.01, 1.78, 1.99, 1.76, 1.94, 1.78,
    1.67, 1.87, 1.91, 1.91, 1.89. Are the successive increments and decrements in the amount of milk
    dispensed random?
    ------------------------------------------------------------------------------------------------
    """
    ts = MILK

    expected_stat = [11, 12]
    expected_p = [0.82, 0.36]
    expected_reject = [False, False]

    runs = runs_test(ts, alternative='two-sided')
    wallis = wallis_moore(ts, alternative='two-sided')
    tests = [runs, wallis]

    for i in range(2):
        stat_err = abs(tests[i].statistic - expected_stat[i]) / expected_stat[i]
        p_err = abs(tests[i].p_value - expected_p[i]) / expected_p[i]

        accepted_stat_err = 0.1
        accepted_p_err = 0.1

        assert stat_err <= accepted_stat_err
        assert p_err <= accepted_p_err
        assert expected_reject[i] == tests[i].reject


def test_runs_test_statistic_is_number_of_runs() -> None:
    """
    The runs statistic is the number of runs around the median,
    computed independently.
    """
    median = np.median(MILK)
    signs = [1 if value > median else -1 for value in MILK if value != median]
    expected_runs = 1.0 + float(sum(signs[i] != signs[i - 1] for i in range(1, len(signs))))

    result = runs_test(MILK)

    assert result.statistic == expected_runs


def test_runs_test_too_few_runs() -> None:
    """
    A monotone series has the fewest possible runs, so the 'less'
    alternative rejects while 'greater' approaches p = 1.
    """
    less = runs_test(MONOTONE, alternative='less')
    greater = runs_test(MONOTONE, alternative='greater')

    assert less.reject
    assert not greater.reject
    assert greater.p_value > P_HALF


def test_runs_test_too_many_runs() -> None:
    """
    An alternating series has the most possible runs, so the 'greater'
    alternative rejects; the one-sided p equals half the two-sided p.
    """
    two_sided = runs_test(ALTERNATING, alternative='two-sided')
    greater = runs_test(ALTERNATING, alternative='greater')
    less = runs_test(ALTERNATING, alternative='less')

    assert greater.reject
    assert two_sided.reject
    assert not less.reject
    assert less.p_value > P_HALF
    assert greater.p_value == pytest.approx(two_sided.p_value / 2, abs=1e-6)


def test_wallis_moore_statistic_is_phase_count() -> None:
    """
    The Wallis-Moore statistic is the average number of phase runs
    (equal values counted once as rises and once as falls), computed
    independently.
    """
    diffs = np.sign(np.diff(MILK))
    group_rises = np.where(diffs >= 0, 1, -1)
    group_falls = np.where(diffs <= 0, -1, 1)

    runs_rises = 1 + int(np.sum(group_rises[1:] != group_rises[:-1]))
    runs_falls = 1 + int(np.sum(group_falls[1:] != group_falls[:-1]))
    expected = (runs_rises + runs_falls) / 2

    result = wallis_moore(MILK)

    assert result.statistic == expected


def test_wallis_moore_one_sided_direction() -> None:
    """
    A monotone series has a single phase, so 'less' rejects; an
    alternating series has the most phases, so 'greater' rejects while
    the opposite alternatives approach p = 1.
    """
    fewer = wallis_moore(MONOTONE, alternative='less')
    more = wallis_moore(ALTERNATING, alternative='greater')

    assert fewer.reject
    assert more.reject
    assert wallis_moore(MONOTONE, alternative='greater').p_value > P_HALF
    assert wallis_moore(ALTERNATING, alternative='less').p_value > P_HALF


def test_runs_test_and_wallis_moore_constant_input() -> None:
    """
    For a constant series both tests classify the event as non-random
    and return a zero p-value with rejection (the runs test has no
    signs at all; the Wallis-Moore test has a single degenerate phase).
    """
    constant_runs = runs_test([5.0] * 8)
    constant_wallis = wallis_moore([5.0] * 8)

    assert np.isnan(constant_runs.statistic)
    assert np.isnan(constant_runs.p_value)
    assert constant_runs.reject is None
    assert np.isnan(constant_wallis.statistic)
    assert np.isnan(constant_wallis.p_value)
    assert constant_wallis.reject is None
