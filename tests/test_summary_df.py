from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from rhis.core import SUMMARY_DF_COLUMNS, Rhis
from rhis.exceptions import RhisSummaryNotBuiltError
from tests.test_core_basics import _make_df, _make_messy_monthly_df

DEFAULT_ALPHA = 0.05
MESSY_NON_NUMERIC_COUNT = 7
YEAR_PERIOD_PATTERN = re.compile(r'^\d{4}-\d{4}$')


def _row_int(summary: pd.DataFrame, column: str, key: str) -> int:
    return int(cast('Any', summary.loc[column, key]))


def _row_float(summary: pd.DataFrame, column: str, key: str) -> float:
    return float(cast('Any', summary.loc[column, key]))


def _row_str(summary: pd.DataFrame, column: str, key: str) -> str:
    return str(cast('Any', summary.loc[column, key]))


def test_summary_df_raises_before_build() -> None:
    rhis = Rhis(_make_df())

    with pytest.raises(RhisSummaryNotBuiltError, match='build_rhis_compliant_df'):
        _ = rhis.summary_df


def test_summary_df_structure_and_values() -> None:
    df = _make_df(n_rows=60, n_cols=2)
    rhis = Rhis(df)
    repr_df = rhis.build_rhis_compliant_df()
    summary = rhis.summary_df

    assert isinstance(summary, pd.DataFrame)
    assert list(summary.index) == list(df.columns)
    assert list(summary.columns) == list(SUMMARY_DF_COLUMNS)

    for column in df.columns:
        n = len(df)
        representative_length = int(repr_df[column].notna().sum())

        assert _row_int(summary, column, 'original_length') == n
        assert _row_int(summary, column, 'representative_length') == representative_length
        assert _row_str(summary, column, 'stat') == 'min'
        assert _row_float(summary, column, 'alpha') == pytest.approx(DEFAULT_ALPHA)
        assert _row_str(summary, column, 'original_period') == f'0-{n - 1}'
        assert _row_str(summary, column, 'representative_period') == f'0-{n - 1}'
        assert _row_str(summary, column, 'most_rejected_hypothesis') in ('R', 'H', 'I', 'S', 'none')
        assert _row_float(summary, column, 'discarded_percentage') + _row_float(
            summary, column, 'remaining_percentage'
        ) == pytest.approx(100.0)


def test_summary_df_reflects_requested_stat() -> None:
    df = _make_df(n_rows=60, n_cols=1)
    rhis = Rhis(df)
    rhis.build_rhis_compliant_df(stat='S')

    assert _row_str(rhis.summary_df, 'series_0', 'stat') == 'S'


def test_summary_df_years_periods() -> None:
    df = pd.DataFrame({'flow': np.arange(1871, 1971, dtype=float)}, index=np.arange(1871, 1971))
    rhis = Rhis(df)
    repr_df = rhis.build_rhis_compliant_df()
    summary = rhis.summary_df

    assert _row_str(summary, 'flow', 'original_period') == '1871-1970'
    representative_period = _row_str(summary, 'flow', 'representative_period')
    assert re.fullmatch(YEAR_PERIOD_PATTERN, representative_period)
    assert representative_period.endswith('-1970')
    assert _row_int(summary, 'flow', 'representative_length') == int(repr_df['flow'].notna().sum())


def test_summary_df_messy_monthly_counts_and_periods() -> None:
    rhis = Rhis(_make_messy_monthly_df())
    repr_df = rhis.build_rhis_compliant_df('min')
    summary = rhis.summary_df

    assert _row_int(summary, 'flow', 'non_numeric_excluded') == MESSY_NON_NUMERIC_COUNT
    assert _row_int(summary, 'flow', 'representative_length') == int(repr_df['flow'].notna().sum())
    assert _row_str(summary, 'flow', 'original_period') == '2020-2026'
    assert _row_str(summary, 'flow', 'representative_period').endswith('-2026')
    assert _row_str(summary, 'flow', 'most_rejected_hypothesis') in ('R', 'H', 'I', 'S')


def test_summary_df_trending_series_rejects_some_hypothesis() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({'trend': np.linspace(0.0, 100.0, 90) + rng.normal(scale=2, size=90)})
    rhis = Rhis(df)
    rhis.build_rhis_compliant_df()

    assert _row_str(rhis.summary_df, 'trend', 'most_rejected_hypothesis') in ('R', 'H', 'I', 'S')
