from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from rhis.utils import clean_numeric_array, nans_nums_from_array

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
def test_clean_numeric_array_raises_when_no_finite_values(values) -> None:
    with pytest.raises(ValueError, match='contains no finite numeric values'):
        clean_numeric_array(values)


# =====================================================================
# nans_nums_from_array
# =====================================================================


MIXED_ARRAY = np.array([0.1, np.nan, 0.3, np.nan, 0.5])
NO_NAN_ARRAY = np.array([0.1, 0.2])
ALL_NAN_ARRAY = np.array([np.nan, np.nan])


def test_nans_nums_from_array_returns_only_numeric_by_default() -> None:
    assert np.asarray(nans_nums_from_array(MIXED_ARRAY)).tolist() == [0.1, 0.3, 0.5]


def test_nans_nums_from_array_splits_numeric_and_nan() -> None:
    nums, nans = nans_nums_from_array(MIXED_ARRAY, only_nums=False)
    assert nums.tolist() == [0.1, 0.3, 0.5]
    assert nans.size == np.count_nonzero(np.isnan(MIXED_ARRAY))
    assert np.isnan(nans).all()


def test_nans_nums_from_array_without_nans() -> None:
    nums, nans = nans_nums_from_array(NO_NAN_ARRAY, only_nums=False)
    assert nums.tolist() == [0.1, 0.2]
    assert nans.size == np.count_nonzero(np.isnan(NO_NAN_ARRAY))


def test_nans_nums_from_array_all_nans() -> None:
    nums, nans = nans_nums_from_array(ALL_NAN_ARRAY, only_nums=False)
    assert nums.size == np.count_nonzero(~np.isnan(ALL_NAN_ARRAY))
    assert nans.size == ALL_NAN_ARRAY.size
    assert np.isnan(nans).all()


def test_nans_nums_from_array_empty() -> None:
    nums, nans = nans_nums_from_array(np.array([]), only_nums=False)
    assert nums.size == 0
    assert nans.size == 0


def test_nans_nums_from_array_accepts_integer_array() -> None:
    assert np.asarray(nans_nums_from_array(np.array([1, 2]))).tolist() == [1, 2]
