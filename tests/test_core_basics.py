from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from rhis.core import Rhis
from rhis.exceptions import RhisEvolNotCalledError

DEFAULT_ALPHA = 0.05
SHORT_SERIES_LEN = 5
LONG_SERIES_LEN = 101
INITIAL_LEN_FOR_LONG_SERIES = 10
TOTAL_P_VALUES_TOL = 1e-9

MONTHLY_PERIOD_START = '2020-01-01'
MONTHLY_PERIOD_END = '2026-01-01'
MONTHLY_FREQ = 'MS'
DROPPED_MONTHS = ['2020-06-01', '2021-12-01', '2023-03-01', '2025-08-01']
TEXT_MARKERS = {
    '2021-03-01': 'N/A',
    '2022-01-01': 'NA',
    '2022-09-01': '2-5',
    '2023-08-01': '<LD',
    '2024-01-01': '-',
}
MISSING_VALUES = {'2020-02-01': np.nan, '2020-10-01': None}
OUTLIERS = {'2020-11-01': 999.9, '2024-06-01': -12.7}
FLOW_OUTLIER_HIGH = 999.9
FLOW_OUTLIER_LOW = -12.7

SPARSE_TS_NON_NUMERIC = 8
SPARSE_TS_NUMERIC = 3


def _make_df(n_rows: int = 60, n_cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({f'series_{i}': rng.normal(50, 15, n_rows) for i in range(n_cols)})


def _make_messy_monthly_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range(MONTHLY_PERIOD_START, MONTHLY_PERIOD_END, freq=MONTHLY_FREQ)
    values = (np.sort(rng.normal(50, 10, len(index))).round(1) + 20 * np.linspace(0, 1, len(index))).astype(object)

    df = pd.DataFrame({'flow': values}, index=index, dtype=object)
    df = df.drop(pd.to_datetime(DROPPED_MONTHS))

    for date, value in TEXT_MARKERS.items():
        df.loc[date, 'flow'] = value
    for date, value in MISSING_VALUES.items():
        df.loc[date, 'flow'] = value
    for date, value in OUTLIERS.items():
        df.loc[date, 'flow'] = value

    return df


@pytest.mark.parametrize(
    'invalid_df',
    [
        [1, 2, 3],
        {'a': 1, 'b': 2},
        pd.Series([1, 2, 3]),
    ],
)
def test_constructor_raises_value_error_when_not_dataframe(invalid_df) -> None:
    with pytest.raises(ValueError, match=r'non-MultiIndex pandas\.DataFrame'):
        Rhis(invalid_df)


def test_constructor_raises_value_error_when_multiindex() -> None:
    index = pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
    df = pd.DataFrame(np.ones((4, 1)), index=index)
    with pytest.raises(ValueError, match=r'non-MultiIndex pandas\.DataFrame'):
        Rhis(df)


def test_constructor_sets_default_attributes() -> None:
    df = _make_df()
    rhis = Rhis(df)
    pd.testing.assert_frame_equal(rhis.orig_df, df)
    assert rhis.rhis_df is None
    assert rhis.rhis_stats_included is False
    assert rhis.is_rhis_complete is False
    assert rhis.alpha == DEFAULT_ALPHA


def test_constructor_length_init_ts_for_short_series() -> None:
    rhis = Rhis(_make_df(n_rows=SHORT_SERIES_LEN))
    assert rhis.length_init_ts == SHORT_SERIES_LEN


def test_constructor_length_init_ts_for_long_series() -> None:
    rhis = Rhis(_make_df(n_rows=LONG_SERIES_LEN))
    assert rhis.length_init_ts == INITIAL_LEN_FOR_LONG_SERIES


def test_constructor_accepts_messy_monthly_df() -> None:
    df = _make_messy_monthly_df()
    rhis = Rhis(df)

    full_months = pd.date_range(MONTHLY_PERIOD_START, MONTHLY_PERIOD_END, freq=MONTHLY_FREQ)
    assert len(df) == len(full_months) - len(DROPPED_MONTHS)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.index.to_series().diff().dropna().nunique() > 1
    assert df.index[0] == pd.Timestamp(MONTHLY_PERIOD_START)
    assert df.index[-1] == pd.Timestamp(MONTHLY_PERIOD_END)

    assert df['flow'].isin(TEXT_MARKERS.values()).sum() == len(TEXT_MARKERS)

    nums = pd.to_numeric(df['flow'], errors='coerce')
    expected_numeric = len(df) - len(TEXT_MARKERS) - len(MISSING_VALUES)
    assert int(nums.notna().sum()) == expected_numeric
    assert float(nums.max()) == FLOW_OUTLIER_HIGH
    assert float(nums.min()) == FLOW_OUTLIER_LOW

    rhis = Rhis(df)
    pd.testing.assert_frame_equal(rhis.orig_df, df)
    assert rhis.length_init_ts == SHORT_SERIES_LEN


def test_constructor_does_not_raise_for_sparse_numeric_column() -> None:
    df = pd.DataFrame({'flow': ['<LD'] * SPARSE_TS_NON_NUMERIC + [1.0, 2.0, 3.0]})
    rhis = Rhis(df)
    assert len(rhis.orig_df) == SPARSE_TS_NON_NUMERIC + SPARSE_TS_NUMERIC


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_evol_sets_state_and_structure(make_df) -> None:
    df = make_df()
    rhis = Rhis(df)
    result = rhis.evol()

    assert rhis.is_rhis_complete is True
    assert rhis.rhis_stats_included is True
    assert rhis.rhis_df is result


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_evol_p_values_are_within_unit_interval(make_df) -> None:
    rhis = Rhis(make_df())
    rhis.evol()

    assert rhis.rhis_df is not None
    values = rhis.rhis_df.to_numpy(dtype=float)
    finite = np.isfinite(values)
    assert np.all(values[finite] >= 0.0)
    assert np.all(values[finite] <= 1.0 + TOTAL_P_VALUES_TOL)


@pytest.mark.parametrize(
    'make_df',
    [partial(_make_df, n_rows=60, n_cols=1), _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_evol_produces_expected_nan_padding(make_df) -> None:
    df = make_df()
    rhis = Rhis(df)
    rhis.evol()

    assert rhis.rhis_df is not None
    for column in df.columns:
        for hyp in ['R', 'H', 'I', 'S']:
            series = rhis.rhis_df[(column, hyp)].to_numpy()
            assert np.count_nonzero(np.isnan(series)) >= rhis.length_init_ts - 1


def test_calculate_rhis_returns_four_p_values() -> None:
    ts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = Rhis.calculate_rhis(ts, alpha=DEFAULT_ALPHA)

    assert set(result) == {'R', 'H', 'I', 'S'}
    assert all(isinstance(p_value, float) for p_value in result.values())


def test_calculate_rhis_ignores_non_numeric_and_missing() -> None:
    df = _make_messy_monthly_df()
    result = Rhis.calculate_rhis(df['flow'], alpha=DEFAULT_ALPHA)

    assert set(result) == {'R', 'H', 'I', 'S'}
    assert all(np.isfinite(p_value) for p_value in result.values())


def test_add_rhis_compliant_raises_before_evol() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    with pytest.raises(RhisEvolNotCalledError, match=r'Rhis\.evol\(\) should be run before adding'):
        rhis.add_rhis_compliant_to_df()


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_add_rhis_compliant_includes_repr_columns(make_df) -> None:
    df = make_df()
    orig_cols = df.columns.tolist()
    rhis = Rhis(df)
    rhis.evol()
    result = rhis.add_rhis_compliant_to_df()

    for column in orig_cols:
        repr_name = column + '_repr'
        assert repr_name in result.columns

        repr_series = pd.to_numeric(result[repr_name], errors='coerce')
        orig_series = pd.to_numeric(result[column], errors='coerce')
        valid = repr_series.notna()

        assert valid.sum() <= len(result)
        pd.testing.assert_series_equal(repr_series[valid], orig_series[valid], check_names=False)


def test_calculate_repr_rhis_pvalues_raises_before_evol() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    with pytest.raises(RhisEvolNotCalledError, match=r'Rhis\.evol\(\) should be run before adding'):
        rhis.calculate_repr_rhis_pvalues()


def test_calculate_repr_rhis_pvalues_raises_without_repr_columns() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    rhis.evol()
    with pytest.raises(ValueError, match=r'No RHIS representative series found'):
        rhis.calculate_repr_rhis_pvalues()


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_calculate_repr_rhis_pvalues_returns_calculate_rhis_p_values(make_df) -> None:
    df = make_df()
    orig_cols = df.columns.tolist()
    rhis = Rhis(df)
    rhis.evol()
    rhis.add_rhis_compliant_to_df()

    result = rhis.calculate_repr_rhis_pvalues()

    assert set(result) == {column + '_repr' for column in orig_cols}
    for column in orig_cols:
        repr_name = column + '_repr'
        expected = Rhis.calculate_rhis(rhis.orig_df[repr_name].to_numpy(), alpha=rhis.alpha)
        assert result[repr_name] == pytest.approx(expected)


def test_is_all_rhis_compliant_true_for_compliant_repr() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    rhis.evol()
    rhis.add_rhis_compliant_to_df()

    assert rhis.is_all_rhis_compliant()


def test_is_all_rhis_compliant_false_and_logs_rejection() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    rhis.evol()

    rhis.orig_df['series_0_repr'] = list(range(60))

    records: list = []
    sink_id = logger.add(records.append, level='DEBUG')
    try:
        result = rhis.is_all_rhis_compliant()
    finally:
        logger.remove(sink_id)

    assert not result
    assert any("'series_0_repr'" in str(record) and 'rejected' in str(record) for record in records)


def test_is_all_rhis_compliant_raises_without_repr_columns() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    rhis.evol()
    with pytest.raises(ValueError, match=r'No RHIS representative series found'):
        rhis.is_all_rhis_compliant()


def test_is_all_rhis_compliant_raises_before_evol() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    with pytest.raises(RhisEvolNotCalledError, match=r'Rhis\.evol\(\) should be run before adding'):
        rhis.is_all_rhis_compliant()


def _retrieve_idxs(pvalues: list[float]) -> tuple[int, int]:
    rhis = Rhis.__new__(Rhis)
    return rhis._find_rhis_compliant_idxs(np.array(pvalues, dtype=float), alpha=DEFAULT_ALPHA)


def test_find_rhis_compliant_idxs_full_series_compliant() -> None:
    """
    When the p-value of the complete series is at or above alpha, the
    representative slice is the entire series.
    """
    pvalues = [0.8, 0.9, 0.7] * 5 + [np.nan] * 5
    assert _retrieve_idxs(pvalues) == (0, len(pvalues))


def test_find_rhis_compliant_idxs_alpha_boundary_keeps_full_series() -> None:
    """
    A first p-value exactly equal to alpha fails to reject, so the whole
    series is kept (the 'at or above alpha' boundary is inclusive).
    """
    pvalues = [DEFAULT_ALPHA] + [0.9] * 6 + [np.nan] * 3
    assert _retrieve_idxs(pvalues) == (0, len(pvalues))


def test_find_rhis_compliant_idxs_alpha_boundary_stops_in_interior() -> None:
    """
    An interior p-value exactly equal to alpha also fails to reject (the
    boundary is inclusive for interior stretches, just like the first
    entry), so the walk stops there.
    """
    pvalues = [0.03, 0.02, DEFAULT_ALPHA, 0.9] + [np.nan] * 6
    assert _retrieve_idxs(pvalues) == (2, len(pvalues))


def test_find_rhis_compliant_idxs_longest_compliant_stretch() -> None:
    """
    The first stretch that fails to reject is the longest compliant one,
    even when a shorter stretch rejects again.
    """
    pvalues = [0.03, 0.02, 0.06, 0.9, 0.04] + [np.nan] * 5
    assert _retrieve_idxs(pvalues) == (2, len(pvalues))


def test_find_rhis_compliant_idxs_stops_at_untestable_stretch() -> None:
    """
    When every testable stretch rejects, the boundary lands at the first
    NaN (stretch too short to test).
    """
    pvalues = [0.01, 0.02] + [np.nan] * 8
    assert _retrieve_idxs(pvalues) == (2, len(pvalues))


def test_find_rhis_compliant_idxs_leading_nan_keeps_full_series() -> None:
    """
    An undefined p-value (NaN) cannot reject the hypothesis, so a NaN
    first entry leaves the whole series as the representative slice.
    """
    pvalues = [np.nan, 0.9, 0.8] + [np.nan] * 7
    assert _retrieve_idxs(pvalues) == (0, len(pvalues))


def test_find_rhis_compliant_idxs_empty_series() -> None:
    """An empty p-value series yields an empty representative slice."""
    assert _retrieve_idxs([]) == (0, 0)
