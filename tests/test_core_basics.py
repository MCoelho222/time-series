from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhis.core import Rhis
from rhis.hypothesis import wald_wolfowitz

DEFAULT_ALPHA = 0.05
STRICTER_ALPHA = 0.01
SHORT_SERIES_LEN = 5
LONG_SERIES_LEN = 101
INITIAL_LEN_FOR_LONG_SERIES = 10

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

    for date, marker in TEXT_MARKERS.items():
        df.loc[date, 'flow'] = marker
    for date, missing in MISSING_VALUES.items():
        df.loc[date, 'flow'] = missing
    for date, outlier in OUTLIERS.items():
        df.loc[date, 'flow'] = outlier

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
    assert rhis.alpha == DEFAULT_ALPHA
    assert rhis.repr_idxs is None
    assert rhis.pvalues_alpha == {}
    assert set(rhis.pvalues) == set(df.columns)
    for col in df.columns:
        assert set(rhis.pvalues[col]) == {'R', 'H', 'I', 'S', 'RHIS-min'}


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


def test_calculate_rhis_returns_four_p_values() -> None:
    ts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = Rhis.calculate_rhis(ts, alpha=DEFAULT_ALPHA)

    assert set(result) == {'R', 'H', 'I', 'S'}
    assert all(isinstance(p_value, float) for p_value in result.values())


def test_calculate_rhis_ignores_non_numeric_and_missing() -> None:
    df = _make_messy_monthly_df()
    result = Rhis.calculate_rhis(df['flow'].to_list(), alpha=DEFAULT_ALPHA)

    assert set(result) == {'R', 'H', 'I', 'S'}
    assert all(np.isfinite(p_value) for p_value in result.values())


def test_calculate_rhis_returns_nan_for_too_short_series() -> None:
    result = Rhis.calculate_rhis([1.0, 2.0], alpha=DEFAULT_ALPHA)

    assert set(result) == {'R', 'H', 'I', 'S'}
    assert all(np.isnan(p_value) for p_value in result.values())


def test_calculate_rhis_independence_is_tested_on_ranks() -> None:
    ts = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.6, 5.3, 8.2, 1.9, 6.0]
    result = Rhis.calculate_rhis(ts, alpha=DEFAULT_ALPHA)

    on_ranks = wald_wolfowitz(ts, alpha=DEFAULT_ALPHA, on_ranks=True).p_value
    on_values = wald_wolfowitz(ts, alpha=DEFAULT_ALPHA, on_ranks=False).p_value

    assert result['I'] == pytest.approx(on_ranks)
    assert result['I'] != pytest.approx(on_values)


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_build_rhis_evol_dict_shape_and_nan_tail(make_df) -> None:
    df = make_df()
    rhis = Rhis(df)
    col = df.columns[0]

    evolutions = rhis.build_rhis_evol_dict(col)
    evol = evolutions[col]

    assert set(evolutions) == {col}
    assert set(evol) == {'R', 'H', 'I', 'S', 'RHIS-min'}
    for hyp in ('R', 'H', 'I', 'S', 'RHIS-min'):
        ps = np.asarray(evol[hyp], dtype=float)
        assert len(ps) == len(rhis.numeric_orig_df)
        assert len(ps) == len(rhis.orig_df)
        assert np.isfinite(ps[0])
        assert np.isnan(ps[-(rhis.length_init_ts - 1):]).all()
    assert rhis.pvalues[col] == evol
    assert rhis.pvalues_alpha[col] == rhis.alpha


def test_build_rhis_evol_dict_all_columns_returns_nested_dict() -> None:
    df = _make_df()
    rhis = Rhis(df)

    evolutions = rhis.build_rhis_evol_dict()

    assert set(evolutions) == set(rhis.numeric_orig_df.columns)
    for col, evol in evolutions.items():
        assert set(evol) == {'R', 'H', 'I', 'S', 'RHIS-min'}
        assert rhis.pvalues[col] == evol
        assert rhis.pvalues_alpha[col] == rhis.alpha


def test_build_rhis_evol_dict_constant_series_yields_nan() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        'constant': [5.0] * 60,
        'trend': np.linspace(0.0, 100.0, 60) + rng.normal(size=60),
    })
    rhis = Rhis(df)

    constant_evol = rhis.build_rhis_evol_dict('constant')['constant']
    trend_evol = rhis.build_rhis_evol_dict('trend')['trend']

    assert np.isnan(constant_evol['I']).all()
    assert np.isfinite(trend_evol['I'][:-(rhis.length_init_ts - 1)]).all()
    assert np.isnan(trend_evol['I'][-(rhis.length_init_ts - 1):]).all()


def test_build_rhis_evol_dict_recomputes_when_alpha_differs() -> None:
    rhis = Rhis(_make_df(n_rows=60, n_cols=1))
    col = 'series_0'

    default_evol = rhis.build_rhis_evol_dict(col)[col]
    assert rhis.pvalues_alpha[col] == rhis.alpha

    stricter_evol = rhis.build_rhis_evol_dict(col, alpha=STRICTER_ALPHA)[col]
    assert rhis.pvalues_alpha[col] == STRICTER_ALPHA
    assert rhis.pvalues[col] == stricter_evol
    assert stricter_evol is not default_evol

    cached_evol = rhis.build_rhis_evol_dict(col, alpha=STRICTER_ALPHA)[col]
    assert cached_evol is stricter_evol


def test_build_rhis_evol_dict_builds_when_pvalues_empty_even_if_alpha_cached() -> None:
    rhis = Rhis(_make_df(n_rows=60, n_cols=1))
    col = 'series_0'

    rhis.pvalues_alpha[col] = rhis.alpha

    evol = rhis.build_rhis_evol_dict(col)[col]

    assert len(evol['RHIS-min']) == len(rhis.numeric_orig_df)
    assert rhis.pvalues[col] == evol


def test_add_rhis_min_to_evol_is_min_over_hypotheses() -> None:
    rhis = Rhis(_make_df())
    evol = {'R': [0.2, 0.8], 'H': [0.6, 0.9], 'I': [0.1, 0.7], 'S': [0.4, 0.3]}

    rhis._add_rhis_min_to_evol(evol)

    assert set(evol) == {'R', 'H', 'I', 'S', 'RHIS-min'}
    np.testing.assert_allclose(evol['RHIS-min'], [0.1, 0.3])


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_get_repr_idxs_returns_dict_of_index_tuples(make_df) -> None:
    df = make_df()
    rhis = Rhis(df)

    result = rhis.get_repr_idxs()

    assert set(result) == set(rhis.numeric_orig_df.columns)
    for start, end in result.values():
        assert 0 <= start <= end == len(rhis.numeric_orig_df)
    assert rhis.repr_idxs == result


def test_get_repr_idxs_single_column_populates_pvalues() -> None:
    rhis = Rhis(_make_df(n_rows=60, n_cols=2))

    result = rhis.get_repr_idxs('series_0')

    assert set(result) == {'series_0'}
    assert rhis.repr_idxs == result
    evol = rhis.pvalues['series_0']
    assert set(evol) == {'R', 'H', 'I', 'S', 'RHIS-min'}
    assert len(evol['RHIS-min']) == len(rhis.numeric_orig_df)
    assert rhis.pvalues_alpha['series_0'] == rhis.alpha

    assert rhis.get_repr_idxs('series_0') == result


def test_get_repr_idxs_matches_manual_evolution() -> None:
    rhis = Rhis(_make_df(n_rows=60, n_cols=1))
    col = 'series_0'

    result = rhis.get_repr_idxs(col)

    evol = rhis.build_rhis_evol_dict(col)[col]
    expected = rhis._find_rhis_compliant_idxs(np.asarray(evol['RHIS-min'], dtype=float), rhis.alpha)
    assert result == {col: expected}


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


# TODO: include test for Rhis.numeric_orig_df (keeps all numeric values from all columns, at the same order (rows and columns), converts all non-numeric to nans, etc.)