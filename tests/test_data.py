from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rhis.utils import slice_init, slices_to_evol, split_into_parts

SLICE_INIT_MAX_SHORT_SIZE = 100
SLICE_INIT_LONG_SIZE = 101
SHORT_INIT_LEN = 5
LONG_INIT_LEN = 10
SPLIT_INTO_TWO = 2
SPLIT_INTO_FOUR = 4


# =====================================================================
# split_into_parts
# =====================================================================


def test_split_into_parts_even_split() -> None:
    assert split_into_parts([1, 2, 3, 4, 5, 6], SPLIT_INTO_TWO) == [[1, 2, 3], [4, 5, 6]]


def test_split_into_parts_uneven_split_balances_leading_parts() -> None:
    result = split_into_parts([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
    assert result == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]


def test_split_into_parts_preserves_order_and_length() -> None:
    ts = list(range(1, 11))
    parts = 3
    result = split_into_parts(ts, parts)
    assert len(result) == parts
    assert [item for part in result for item in part] == ts


def test_split_into_parts_single_part() -> None:
    assert split_into_parts([1, 2, 3], 1) == [[1, 2, 3]]


def test_split_into_parts_raises_when_more_parts_than_items() -> None:
    with pytest.raises(ValueError, match='cannot be greater than the series length'):
        split_into_parts([1, 2], SPLIT_INTO_FOUR)


def test_split_into_parts_allows_parts_equal_to_length() -> None:
    assert split_into_parts([1, 2, 3], 3) == [[1], [2], [3]]


def test_split_into_parts_accepts_numpy_array() -> None:
    result = split_into_parts(np.array([1.0, 2.0, 3.0, 4.0]), SPLIT_INTO_TWO)
    assert len(result) == SPLIT_INTO_TWO
    assert np.array_equal(result[0], np.array([1.0, 2.0]))
    assert np.array_equal(result[1], np.array([3.0, 4.0]))


# =====================================================================
# slice_init
# =====================================================================


def test_slice_init_uses_short_length_up_to_limit() -> None:
    for size in [1, 50, SLICE_INIT_MAX_SHORT_SIZE]:
        assert slice_init(size) == SHORT_INIT_LEN


def test_slice_init_uses_long_length_above_limit() -> None:
    assert slice_init(SLICE_INIT_LONG_SIZE) == LONG_INIT_LEN


# =====================================================================
# slices_to_evol
# =====================================================================


def test_slices_to_evol_produces_growing_slices() -> None:
    result = slices_to_evol([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]


def test_slices_to_evol_number_of_slices() -> None:
    ts = list(range(10))
    start = 3
    result = slices_to_evol(ts, start)
    assert len(result) == len(ts) - start + 1


def test_slices_to_evol_last_slice_is_entire_series() -> None:
    ts = [1, 2, 3, 4, 5]
    result = slices_to_evol(ts, 2)
    assert result[-1] == ts


def test_slices_to_evol_start_equal_to_length() -> None:
    assert slices_to_evol([1, 2, 3], 3) == [[1, 2, 3]]


def test_slices_to_evol_start_greater_than_length() -> None:
    assert slices_to_evol([1, 2], 5) == []


def test_slices_to_evol_works_with_non_numeric_items() -> None:
    non_numeric: list[Any] = ['a', 'b', 'c']
    assert slices_to_evol(non_numeric, 2) == [['a', 'b'], ['a', 'b', 'c']]


def test_slices_to_evol_accepts_numpy_array() -> None:
    result = slices_to_evol(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.array_equal(result[0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(result[-1], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
