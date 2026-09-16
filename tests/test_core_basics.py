from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rhis.core import Rhis
from src.rhis.exceptions import RhisEvolNotCalledError

DEFAULT_ALPHA = 0.05
SHORT_SERIES_LEN = 5
LONG_SERIES_LEN = 101
INITIAL_LEN_FOR_LONG_SERIES = 10


def _make_df(n_rows: int = 60, n_cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({f'series_{i}': rng.normal(50, 15, n_rows) for i in range(n_cols)})


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


def test_evol_sets_state_and_structure() -> None:
    df = _make_df(n_rows=60)
    rhis = Rhis(df)
    result = rhis.evol()

    assert rhis.is_rhis_complete is True
    assert rhis.rhis_stats_included is True
    assert rhis.rhis_df is result


def test_evol_p_values_are_within_unit_interval() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    rhis.evol()

    assert rhis.rhis_df is not None
    values = rhis.rhis_df.to_numpy(dtype=float)
    nan_mask = np.isnan(values)
    assert np.all(values[~nan_mask] >= 0.0)
    assert np.all(values[~nan_mask] <= 1.0)


def test_evol_produces_expected_nan_padding() -> None:
    df = _make_df(n_rows=60, n_cols=1)
    rhis = Rhis(df)
    rhis.evol()

    assert rhis.rhis_df is not None
    for hyp in ['R', 'H', 'I', 'S']:
        series = rhis.rhis_df[('series_0', hyp)].to_numpy()
        assert np.count_nonzero(np.isnan(series)) == rhis.length_init_ts - 1


def test_calculate_rhis_returns_four_p_values() -> None:
    ts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = Rhis.calculate_rhis(ts)

    assert set(result) == {'R', 'H', 'I', 'S'}
    assert all(isinstance(p_value, float) for p_value in result.values())
    assert all((p_value >= 0.0 and p_value <= 1.0) for p_value in result.values())


def test_add_rhis_compliant_raises_before_evol() -> None:
    rhis = Rhis(_make_df(n_rows=60))
    with pytest.raises(RhisEvolNotCalledError, match=r'Rhis\.evol\(\) should be run before adding rhis compliant'):
        rhis.add_rhis_compliant_to_df()


def test_add_rhis_compliant_includes_repr_columns() -> None:
    df = _make_df(n_rows=60)
    orig_cols = df.columns.tolist()
    rhis = Rhis(df)
    rhis.evol()
    result = rhis.add_rhis_compliant_to_df()

    for column in orig_cols:
        repr_name = column + '_repr'
        assert repr_name in result.columns
        assert result[repr_name].notna().any()
