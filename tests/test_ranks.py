from __future__ import annotations

import numpy as np
import pytest

from src.rhis.utils import get_ties_index, ranks_with_ties_corrected, to_ranks

NO_TIES_TS = [4.0, 7.0, 9.0]
TIE_AT_START_TS = [4.0, 4.0, 8.0, 9.0]
MIDDLE_TIE_TS = [10.0, 3.0, 3.5, 4.0, 4.0, 5.0, 22.5]
END_TIE_TS = [10.0, 3.0, 3.5, 4.0, 4.0, 5.0, 22.5, 22.5, 22.5]
ALL_EQUAL_TS = [2.0, 2.0, 2.0, 2.0]
MULTI_TIES_TS = [1.0, 1.0, 1.0, 3.0, 3.5, 4.0, 4.0, 5.0, 22.5, 22.5, 22.5]


# =====================================================================
# get_ties_index
# =====================================================================


def test_no_ties_returns_empty_list() -> None:
    assert get_ties_index(NO_TIES_TS) == []


def test_input_does_not_need_to_be_sorted() -> None:
    assert get_ties_index([4.0, 9.0, 4.0], start=0) == [1, 2]


@pytest.mark.parametrize(
    ('ts', 'start', 'expected'),
    [
        (TIE_AT_START_TS, 0, [1, 2]),
        (MIDDLE_TIE_TS, 2, [3, 4]),
        (END_TIE_TS, 6, [7, 8, 9]),
        (ALL_EQUAL_TS, 0, [1, 2, 3, 4]),
        ([1.5, 1.5], 0, [1, 2]),
    ],
)
def test_tie_ranks_for_consecutive_ties(ts, start, expected) -> None:
    assert get_ties_index(ts, start) == expected


def test_start_must_point_to_first_index_of_tie_group() -> None:
    assert get_ties_index(MIDDLE_TIE_TS, start=3) == []


def test_start_skips_to_later_tie_group() -> None:
    assert get_ties_index(MULTI_TIES_TS, start=5) == [6, 7]


def test_start_at_beginning_reports_first_tie_group() -> None:
    assert get_ties_index(MULTI_TIES_TS, start=0) == [1, 2, 3]


@pytest.mark.parametrize('ts', [[], [5.0]])
def test_insufficient_data_raises_index_error(ts) -> None:
    with pytest.raises(IndexError):
        get_ties_index(ts)


# =====================================================================
# ranks_with_ties_corrected
# =====================================================================


def test_no_ties_returns_unmodified_ranks() -> None:
    result = ranks_with_ties_corrected(NO_TIES_TS)
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_middle_tie_group_corrected_to_mean_rank() -> None:
    result = ranks_with_ties_corrected(MIDDLE_TIE_TS)
    assert result.tolist() == [6.0, 1.0, 2.0, 3.5, 3.5, 5.0, 7.0]


def test_end_tie_group_corrected_to_mean_rank() -> None:
    result = ranks_with_ties_corrected(END_TIE_TS)
    assert result.tolist() == [6.0, 1.0, 2.0, 3.5, 3.5, 5.0, 8.0, 8.0, 8.0]


def test_multiple_tie_groups_corrected_independently() -> None:
    result = ranks_with_ties_corrected(MULTI_TIES_TS)
    assert result.tolist() == [2.0, 2.0, 2.0, 4.0, 5.0, 6.5, 6.5, 8.0, 10.0, 10.0, 10.0]


def test_all_values_equal_collapses_ranks_to_single_value() -> None:
    result = ranks_with_ties_corrected(ALL_EQUAL_TS)
    assert result.tolist() == [2.5, 2.5, 2.5, 2.5]


def test_corrected_ranks_keep_original_order_as_float_array() -> None:
    result = ranks_with_ties_corrected(MIDDLE_TIE_TS)
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.tolist() == [6.0, 1.0, 2.0, 3.5, 3.5, 5.0, 7.0]


def test_ties_data_without_ties() -> None:
    result = ranks_with_ties_corrected(NO_TIES_TS, ties_data=True)

    assert set(result) == {'corrected_ranks', 'tied_ranks', 'ties_count', 'ties_groups_count'}
    assert result['corrected_ranks'].tolist() == [1.0, 2.0, 3.0]
    assert result['tied_ranks'] == []
    assert result['ties_groups_count'] == [1, 1, 1]
    assert result['ties_count'] == len(result['ties_groups_count'])


def test_ties_data_reports_single_tie_group() -> None:
    result = ranks_with_ties_corrected(MIDDLE_TIE_TS, ties_data=True)

    assert result['corrected_ranks'].tolist() == [6.0, 1.0, 2.0, 3.5, 3.5, 5.0, 7.0]
    assert result['tied_ranks'] == [[3, 4]]
    assert result['ties_groups_count'] == [1, 1, 2, 1, 1, 1]
    assert result['ties_count'] == len(result['ties_groups_count'])


def test_ties_data_reports_multiple_tie_groups() -> None:
    result = ranks_with_ties_corrected(MULTI_TIES_TS, ties_data=True)

    assert result['corrected_ranks'].tolist() == [2.0, 2.0, 2.0, 4.0, 5.0, 6.5, 6.5, 8.0, 10.0, 10.0, 10.0]
    assert result['tied_ranks'] == [[1, 2, 3], [6, 7], [9, 10, 11]]
    assert result['ties_groups_count'] == [3, 1, 1, 2, 1, 3]
    assert result['ties_count'] == len(result['ties_groups_count'])


# =====================================================================
# to_ranks
# =====================================================================


def test_to_ranks_preserve_original_order() -> None:
    assert to_ranks(MIDDLE_TIE_TS) == [6, 1, 2, 3, 4, 5, 7]


def test_to_ranks_do_not_require_sorted_input() -> None:
    assert to_ranks([9.0, 4.0, 7.0]) == [3, 1, 2]


def test_to_ranks_tied_values_get_successive_ranks() -> None:
    assert to_ranks([4.0, 9.0, 4.0]) == [1, 3, 2]


def test_to_ranks_tied_values_skip_rank_for_single_ties() -> None:
    assert to_ranks([1.5, 1.5, 2.0]) == [1, 2, 3]


def test_to_ranks_all_equal_values() -> None:
    assert to_ranks(ALL_EQUAL_TS) == [1, 2, 3, 4]


def test_to_ranks_single_element() -> None:
    assert to_ranks([5.0]) == [1]


def test_to_ranks_supports_negative_and_zero() -> None:
    assert to_ranks([-2, 0, 3]) == [1, 2, 3]


def test_to_ranks_empty_series() -> None:
    assert to_ranks([]) == []
