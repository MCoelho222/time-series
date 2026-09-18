from __future__ import annotations

import numpy as np
import pytest

from rhis.core import Rhis
from tests.test_core_basics import _make_df, _make_messy_monthly_df

STATS_METHODS = [('min', np.min), ('avg', np.mean), ('med', np.median), ('max', np.max)]
CUSTOM_LENGTH_INIT_TS = 15


@pytest.mark.parametrize(
    'make_df',
    [_make_df, _make_messy_monthly_df],
    ids=['clean_series', 'messy_monthly'],
)
def test_evol_stats_columns_are_aggregates_over_rhis(make_df) -> None:
    df = make_df()
    rhis = Rhis(df)
    rhis.evol()

    assert rhis.rhis_df is not None
    for column in df.columns:
        hyps = np.column_stack(
            [rhis.rhis_df[(column, hyp)].to_numpy(dtype=float) for hyp in 'RHIS']
        )
        all_present = ~np.isnan(hyps).any(axis=1)

        for name, method in STATS_METHODS:
            actual = rhis.rhis_df[(column, name)].to_numpy(dtype=float)
            expected = np.full(len(actual), np.nan)
            expected[all_present] = method(hyps[all_present], axis=1)

            np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_evol_without_rhis_stats() -> None:
    rhis = Rhis(_make_df())
    rhis.evol(include_rhis_stats=False)

    assert rhis.rhis_df is not None
    assert ('series_0', 'min') not in rhis.rhis_df.columns


def test_evol_with_custom_length_init_ts() -> None:
    rhis = Rhis(_make_df())
    rhis.evol(length_init_ts=CUSTOM_LENGTH_INIT_TS)

    assert rhis.length_init_ts == CUSTOM_LENGTH_INIT_TS


def test_add_rhis_compliant_includes_stats_when_missing() -> None:
    rhis = Rhis(_make_df())
    rhis.evol(include_rhis_stats=False)
    rhis.add_rhis_compliant_to_df('min')

    assert rhis.rhis_df is not None
    assert ('series_0', 'min') in rhis.rhis_df.columns
    assert 'series_0_repr' in rhis.orig_df.columns
