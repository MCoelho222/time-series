"""Methods for ties correction."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from scipy.stats import rankdata

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rhis_ts.custom_types import TiesData, TimeSeriesFlex


def get_ties_index(ts: TimeSeriesFlex, start: int = 0) -> list[int]:
    """
    Checks if there are equal numbers in sequence and gets their ranks.

    Parameters
    ----------
        ts
            A list of integers or floats.
        start
            The index from where the verification should start checking.

    Returns
    -------
        A list with ranks where ties are present.
    """
    n = len(ts)
    ts_sorted = np.sort(ts)
    ranks = list(range(1, n + 1))

    tie_ranks = []
    i = start
    while ts_sorted[i] == ts_sorted[i + 1]:
        tie_ranks.append(ranks[i])
        i += 1
        if i == n - 1:
            break

    if len(tie_ranks) > 0:
        tie_ranks.append(tie_ranks[-1] + 1)

    return tie_ranks


@overload
def ranks_with_ties_corrected(
    ts: TimeSeriesFlex,
    *,
    ties_data: Literal[True],
) -> TiesData:
    ...


@overload
def ranks_with_ties_corrected(
    ts: TimeSeriesFlex,
    *,
    ties_data: Literal[False] = False,
) -> NDArray[np.int64 | np.float64]:
    ...


@overload
def ranks_with_ties_corrected(
    ts: TimeSeriesFlex,
    *,
    ties_data: bool,
) -> NDArray[np.int64 | np.float64] | TiesData:
    ...


def ranks_with_ties_corrected(
    ts: TimeSeriesFlex,
    *,
    ties_data: bool = False,
    ) -> NDArray[np.int64 | np.float64] | TiesData:
    """
    Apply correction for ties.

    Parameters
    ----------
        ts
            A list of integers or floats.

        ties_data
            If True, information about ties will be returned.

    Returns
    -------
        A list with ranks corrected for ties, or a dictionary with
        information about ties, including the list with ranks. The
        ranks will be in the original time series order.
    """
    arr = np.asarray(ts, dtype=float)
    ranks = np.asarray(rankdata(arr, method='average'), dtype=float)

    if not ties_data:
        return ranks

    _, counts = np.unique(arr, return_counts=True)
    starts = np.cumsum(counts) - counts + 1
    ties_index = [
        list(range(int(start), int(start + count)))
        for start, count in zip(starts, counts, strict=True)
        if count > 1
    ]

    ties_result: TiesData = {
        'corrected_ranks': ranks,
        'tied_ranks': ties_index, # The indexes where ties are present
        'ties_count': int(counts.size), # The number of ties groups
        'ties_groups_count': counts.tolist(), # The number of elements in each tie group
    }

    return ties_result


def to_ranks(ts: TimeSeriesFlex) -> list[int | float]:
    """
    Transform the original series in a series of ranks.

    Parameters
    ----------
        ts
            A list or array with numbers.

    Return
    ------
        A list with the original data replaced by their ranks.
    """
    return rankdata(np.asarray(ts), method='ordinal').astype(int).tolist()

if __name__ == "__main__":
    ts1 = [10, 3, 3.5, 4., 4., 5., 22.5]
    ts2 = [10, 3, 3.5, 4., 4., 5., 22.5, 22.5, 22.5]
    ts3 = [1, 1, 1, 3, 3.5, 4., 4., 5., 22.5, 22.5, 22.5]
    ts4 = [2, 2, 2, 2]

    print(to_ranks(ts1))
    print(to_ranks(ts2))
    print(to_ranks(ts3))
    print(to_ranks(ts4))

    print()

    print(ranks_with_ties_corrected(ts1, ties_data=True))
    print(ranks_with_ties_corrected(ts2, ties_data=True))
    print(ranks_with_ties_corrected(ts3, ties_data=True))
    print(ranks_with_ties_corrected(ts4, ties_data=True))
