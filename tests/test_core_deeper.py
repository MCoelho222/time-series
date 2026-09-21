from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhis.core import Rhis
from tests.test_core_basics import _make_df, _make_messy_monthly_df

CUSTOM_LENGTH_INIT_TS = 15


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_build_rhis_evol_df_rhis_min_column_is_min_over_rhis(make_df) -> None:
    df = make_df()
    rhis = Rhis(df)
    rhis.build_rhis_evol_df()

    assert rhis.rhis_df is not None
    for column in df.columns:
        hyps = np.column_stack(
            [rhis.rhis_df[(column, hyp)].to_numpy(dtype=float) for hyp in 'RHIS']
        )
        all_present = ~np.isnan(hyps).any(axis=1)

        actual = rhis.rhis_df[(column, 'RHIS-min')].to_numpy(dtype=float)
        expected = np.full(len(actual), np.nan)
        expected[all_present] = hyps[all_present].min(axis=1)

        np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_build_rhis_evol_df_without_rhis_min() -> None:
    rhis = Rhis(_make_df())
    rhis.build_rhis_evol_df(rhis_min=False)

    assert rhis.rhis_df is not None
    assert ('series_0', 'RHIS-min') not in rhis.rhis_df.columns


def test_build_rhis_evol_df_with_custom_length_init_ts() -> None:
    rhis = Rhis(_make_df())
    rhis.build_rhis_evol_df(length_init_ts=CUSTOM_LENGTH_INIT_TS)

    assert rhis.length_init_ts == CUSTOM_LENGTH_INIT_TS


def test_build_rhis_evol_df_handles_constant_series_with_nan() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        'constant': [5.0] * 60,
        'trend': np.linspace(0.0, 100.0, 60) + rng.normal(size=60),
    })

    rhis = Rhis(df)
    rhis.build_rhis_evol_df()

    assert rhis.rhis_df is not None
    fill = rhis.length_init_ts - 1
    constant_i = rhis.rhis_df[('constant', 'I')].to_numpy()
    trend_i = rhis.rhis_df[('trend', 'I')].to_numpy()

    assert np.isnan(constant_i).all()
    assert np.isfinite(trend_i[:-fill]).all()
    assert np.isnan(trend_i[-fill:]).all()


def test_calculate_rhis_returns_nan_for_constant_series() -> None:
    result = Rhis.calculate_rhis([5.0] * 10)

    assert np.isnan(result['I'])
    assert all(np.isfinite(result[hyp]) for hyp in 'RHS')


def test_build_rhis_compliant_df_derives_stat_without_rhis_evol() -> None:
    """The repr df is derivable without building rhis_df at all."""
    rhis = Rhis(_make_df(n_rows=60, n_cols=1))

    repr_df = rhis.build_rhis_compliant_df('min')

    assert rhis.rhis_df is None
    assert set(repr_df.columns) == {'series_0'}
