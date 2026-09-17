from __future__ import annotations

from rhis.hypothesis import runs_test, wallismoore


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
    ts = [1.90, 1.99, 2., 1.78, 1.77, 1.76, 1.98, 1.9, 1.65, \
          1.76, 2.01, 1.78, 1.99, 1.76, 1.94, 1.78, 1.67, 1.87, 1.91, 1.91, 1.89]

    expected_stat = [11, 12]
    expected_p = [0.82, 0.36]
    expected_reject = [False, False]

    runs = runs_test(ts, alternative='two-sided')
    wallis = wallismoore(ts, alternative='two-sided')
    tests = [runs, wallis]

    for i in range(2):
        stat_err = abs(tests[i].statistic - expected_stat[i]) / expected_stat[i]
        p_err = abs(tests[i].p_value - expected_p[i]) / expected_p[i]

        accepted_stat_err = 0.1
        accepted_p_err = 0.1

        assert stat_err <= accepted_stat_err
        assert p_err <= accepted_p_err
        assert expected_reject[i] == tests[i].reject
