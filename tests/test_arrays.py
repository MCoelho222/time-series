from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from rhis_ts.utils import clean_numeric_array

# =====================================================================
# clean_numeric_array
# =====================================================================


def test_clean_numeric_array_keeps_finite_numerics_in_order() -> None:
    values: list[Any] = [1.0, '2', None, np.nan, 'N/A', 3.5]
    assert clean_numeric_array(values).tolist() == [1.0, 2.0, 3.5]


def test_clean_numeric_array_converts_numeric_strings() -> None:
    numeric_strings: list[Any] = ['4', '-2.5', 1]
    assert clean_numeric_array(numeric_strings).tolist() == [4.0, -2.5, 1.0]


def test_clean_numeric_array_drops_infinite_values() -> None:
    assert clean_numeric_array([1.0, float('inf'), float('-inf')]).tolist() == [1.0]


def test_clean_numeric_array_accepts_tuple_input() -> None:
    assert clean_numeric_array((3, 4.0)).tolist() == [3.0, 4.0]


def test_clean_numeric_array_accepts_pandas_series() -> None:
    assert clean_numeric_array(pd.Series([1.0, 9.0])).tolist() == [1.0, 9.0]  # type: ignore[arg-type]


def test_clean_numeric_array_returns_float_ndarray() -> None:
    result = clean_numeric_array([1, 2])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    'values',
    [
        [],
        ['2-5', '<LD', '-', 'NA'],
        [np.nan],
        [None],
    ],
)
def test_clean_numeric_array_returns_empty_array_when_no_finite_values(values) -> None:
    result = clean_numeric_array(values)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == []
