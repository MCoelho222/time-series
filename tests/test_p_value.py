from __future__ import annotations

import pytest

from rhis.utils import p_value_normal
from rhis.utils import test_decision_normal as decision_normal

ALPHA = 0.05
Z_NULL = 0.0
Z_MILD = 1.5
Z_SIGNIFICANT = 1.96
Z_STRONG = 2.0
P_VALUE_TOL = 1e-4
HALF_P_VALUE = 0.5


# =====================================================================
# p_value_normal
# =====================================================================


def test_p_value_normal_is_half_at_null_z() -> None:
    assert p_value_normal(Z_NULL) == pytest.approx(HALF_P_VALUE)


def test_p_value_normal_known_values() -> None:
    assert p_value_normal(Z_MILD) == pytest.approx(0.0668, abs=P_VALUE_TOL)
    assert p_value_normal(Z_SIGNIFICANT) == pytest.approx(0.0250, abs=P_VALUE_TOL)
    assert p_value_normal(Z_STRONG) == pytest.approx(0.0228, abs=P_VALUE_TOL)


def test_p_value_normal_is_symmetric_for_negative_z() -> None:
    assert p_value_normal(-Z_SIGNIFICANT) == p_value_normal(Z_SIGNIFICANT)


def test_p_value_normal_always_within_unit_interval() -> None:
    for z in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        assert 0.0 < p_value_normal(z) <= 1.0


def test_p_value_normal_decreases_as_z_grows() -> None:
    assert p_value_normal(Z_MILD) > p_value_normal(Z_SIGNIFICANT)
    assert p_value_normal(Z_SIGNIFICANT) > p_value_normal(Z_STRONG)


# =====================================================================
# test_decision_normal
# =====================================================================


def test_decision_normal_two_sided_rejects_when_p_below_alpha() -> None:
    result = decision_normal(0.0, 0.0, Z_STRONG, 'two-sided', ALPHA)
    assert result.reject
    assert result.p_value == pytest.approx(0.0455, abs=P_VALUE_TOL)


def test_decision_normal_two_sided_accepts_when_p_above_alpha() -> None:
    result = decision_normal(0.0, 0.0, Z_MILD, 'two-sided', ALPHA)
    assert not result.reject
    assert result.p_value == pytest.approx(0.1336, abs=P_VALUE_TOL)


def test_decision_normal_two_sided_clamps_p_value_to_one() -> None:
    result = decision_normal(0.0, 0.0, Z_NULL, 'two-sided', ALPHA)
    assert result.p_value == 1.0
    assert not result.reject


def test_decision_normal_uses_provided_alpha_for_rejection() -> None:
    result = decision_normal(0.0, 0.0, Z_STRONG, 'two-sided', 0.01)
    assert result.p_value == pytest.approx(0.0455, abs=P_VALUE_TOL)
    assert not result.reject


@pytest.mark.parametrize(
    ('stat', 'stat_mean', 'expected_reject'),
    [
        (1.0, 2.0, True),
        (2.0, 2.0, False),
        (3.0, 2.0, False),
    ],
)
def test_decision_normal_less_requires_stat_below_mean(stat, stat_mean, expected_reject) -> None:
    result = decision_normal(stat, stat_mean, Z_STRONG, 'less', ALPHA)
    assert result.reject == expected_reject


@pytest.mark.parametrize(
    ('stat', 'stat_mean', 'expected_reject'),
    [
        (3.0, 2.0, True),
        (1.0, 2.0, False),
        (2.0, 2.0, False),
    ],
)
def test_decision_normal_greater_requires_stat_above_mean(stat, stat_mean, expected_reject) -> None:
    result = decision_normal(stat, stat_mean, Z_STRONG, 'greater', ALPHA)
    assert result.reject == expected_reject


def test_decision_normal_one_sided_keeps_single_tailed_p_value() -> None:
    result = decision_normal(1.0, 2.0, Z_STRONG, 'less', ALPHA)
    assert result.p_value == pytest.approx(0.0228, abs=P_VALUE_TOL)
    assert result.reject


def test_decision_normal_one_sided_opposite_side_has_large_p_value() -> None:
    """
    When the statistic lies on the opposite side of the mean, the
    one-sided p-value is large (above 0.5) instead of misleadingly
    small, and the null hypothesis is not rejected.
    """
    less = decision_normal(3.0, 2.0, Z_STRONG, 'less', ALPHA)
    greater = decision_normal(1.0, 2.0, Z_STRONG, 'greater', ALPHA)

    assert less.p_value == pytest.approx(1.0 - 0.0228, abs=P_VALUE_TOL)
    assert less.p_value > HALF_P_VALUE
    assert not less.reject
    assert greater.p_value == pytest.approx(1.0 - 0.0228, abs=P_VALUE_TOL)
    assert greater.p_value > HALF_P_VALUE
    assert not greater.reject


def test_decision_normal_returns_all_fields() -> None:
    result = decision_normal(3.0, 2.0, Z_STRONG, 'greater', ALPHA)
    assert result.p_value == pytest.approx(0.0228, abs=P_VALUE_TOL)
    assert result.alpha == ALPHA
    assert result.reject
    assert result.alternative == 'greater'
