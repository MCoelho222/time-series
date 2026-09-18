from __future__ import annotations

import scipy.stats as sts

from rhis.custom_types import TestDecisionNormal


def p_value_normal(z: float) -> float:
    """
    Calculate the p_value from the normal distribution.

    Parameters
    ----------
        z
            The z value from a test that uses normal approximation.

    Returns
    -------
        The p_value.
    """
    return 1. - sts.norm.cdf(abs(z))


def test_decision_normal(
        stat: float,
        stat_mean: float,
        z: float,
        alternative: str,
        alpha: float
        ) -> TestDecisionNormal:
    """
    Decide about rejection of the null hypothesis using normal
    approximation.

    Parameters
    ----------
        stat
            The value of the test statistic.
        stat_mean
            The expected value of the test statistic.
        z
            The value of the normalized test statistic.
        alpha
            The significance level of the test.
        alternative
            The alternative hypothesis: 'two-sided', 'greater',
            or 'less'.

    Return
    ------
        A namedtuple
            ('TestDecisionNormal', ['p_value', 'alpha', 'reject'
            , 'alternative'])
            The parameter 'reject' is of type bool. 'True' means
            the null hypothesis was reject.
    """
    p = p_value_normal(z)

    if alternative == 'two-sided':
        p = min(1.0, p * 2.)
        reject = p < alpha
    if alternative == 'less':
        reject = stat < stat_mean and p < alpha
    if alternative == 'greater':
        reject = stat > stat_mean and p < alpha

    return TestDecisionNormal(p, alpha, reject, alternative)


