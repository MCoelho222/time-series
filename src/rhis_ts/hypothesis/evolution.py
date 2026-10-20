"""Incremental RHIS evolution over the suffixes of a single series."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from rhis_ts.hypothesis.homogeneity import _mann_whitney_from_ranks
from rhis_ts.hypothesis.independence import _wald_wolfowitz_from_array
from rhis_ts.hypothesis.randomness import _wallis_moore_from_runs
from rhis_ts.hypothesis.stationarity import _mann_kendall_from_s

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from rhis_ts.custom_types import TimeSeriesFlex

_HYPOTHESES = ('R', 'H', 'I', 'S')
_MIN_RUN_UPDATE_LENGTH = 3


def _count_inversions(values: NDArray[np.float64]) -> int:
    """
    Count the discordant pairs of a numeric array with a Fenwick tree.

    A discordant pair is a pair (i, j), i < j, with values[i] > values[j].
    """
    _, inverse = np.unique(values, return_inverse=True)
    size = int(inverse.max()) + 1 if inverse.size else 0
    tree = np.zeros(size + 1, dtype=np.int64)
    total = 0
    for idx in inverse[::-1]:
        i = int(idx)
        j = i
        while j > 0:
            total += int(tree[j])
            j -= j & -j
        j = i + 1
        while j <= size:
            tree[j] += 1
            j += j & -j
    return total


class SuffixEvolution:
    """
    Incrementally test every suffix of a cleaned series for R, H, I and S.

    Starting from the complete series, each step drops the first observation
    and updates the ranks and the statistics in place, so every suffix is
    tested from the state left by the previous one instead of from scratch.

    The series must contain only finite numeric values; callers are expected
    to clean it first (see rhis_ts.utils.clean_numeric_array).

    Parameters
    ----------
        values
            A 1D array of finite numeric values.
        min_length
            The minimum suffix length to test.

    Attributes
    ----------
        n_suffixes
            Number of suffixes that can be tested (0 when the series is too
            short).
    """

    def __init__(
        self,
        values: TimeSeriesFlex,
        min_length: int,
    ) -> None:
        self._values = np.asarray(values, dtype=float)
        self._min_length = min_length
        self._total = self._values.size
        self._start = 0

        self.n_suffixes = max(0, self._total - min_length + 1)

        if self.n_suffixes == 0:
            self._ranks = np.empty(0, dtype=float)
            self._counts: dict[float, int] = {}
            self._distinct = 0
            self._ties_factor = 0.0
            self._inversions = 0
            self._ties_pairs = 0.0
            self._up_runs = 0
            self._down_runs = 0
            return

        self._ranks = np.asarray(rankdata(self._values, method='average'), dtype=float)

        uniques, counts = np.unique(self._values, return_counts=True)
        self._counts = dict(zip(uniques.tolist(), counts.tolist(), strict=True))
        self._distinct = int(uniques.size)
        self._ties_factor = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
        self._ties_pairs = float(np.sum(counts * (counts - 1)) / 2)
        self._inversions = _count_inversions(self._values)

        diffs = np.diff(self._values)
        signs_up = np.where(diffs >= 0, 1, -1)
        signs_down = np.where(diffs > 0, 1, -1)
        self._up_runs = int(np.count_nonzero(signs_up[1:] != signs_up[:-1]) + 1)
        self._down_runs = int(np.count_nonzero(signs_down[1:] != signs_down[:-1]) + 1)

    def __iter__(self) -> Iterator[tuple[int, dict[str, float]]]:
        """
        Yield ``(start, p_values)`` for every testable suffix, in order.

        The p-value dict has one entry per hypothesis ('R', 'H', 'I', 'S').
        """
        while self._start < self.n_suffixes:
            yield self._start, self._current_pvalues()
            self._slide()

    def _current_pvalues(self) -> dict[str, float]:
        """Compute the four p-values of the current suffix without mutating state."""
        n = self._total - self._start
        p_values = dict.fromkeys(_HYPOTHESES, np.nan)

        if self._distinct <= 1:
            return p_values

        ranks = self._ranks[self._start:]

        _, p_values['R'] = _wallis_moore_from_runs(
            n, self._up_runs, self._down_runs,
        )

        n1 = n // 2 + n % 2
        _, p_values['H'] = _mann_whitney_from_ranks(ranks, n1)

        try:
            _, p_values['I'] = _wald_wolfowitz_from_array(ranks)
        except ValueError:
            p_values['I'] = np.nan

        n0 = n * (n - 1) / 2
        test_s = n0 - self._ties_pairs - 2 * self._inversions
        p_values['S'] = _mann_kendall_from_s(n, test_s, self._ties_factor)

        return p_values

    def _slide(self) -> None:
        """Drop the first value of the current suffix and update every statistic."""
        start = self._start
        value = self._values[start]
        count = self._counts[float(value)]

        # The first value is the smallest of the suffix, so the number of
        # later values below it is its rank minus its share of the tie group.
        later_less = float(self._ranks[start]) - (count + 1) / 2.0
        self._inversions -= round(later_less)
        self._ties_pairs -= float(count - 1)

        if count == 1:
            del self._counts[float(value)]
            self._distinct -= 1
        else:
            self._counts[float(value)] = count - 1

        self._ties_factor += (
            (count - 1) * (count - 2) * (2 * (count - 1) + 5)
            - count * (count - 1) * (2 * count + 5)
        )

        rest_values = self._values[start + 1:]
        rest_ranks = self._ranks[start + 1:]
        rest_ranks[rest_values > value] -= 1.0
        rest_ranks[rest_values == value] -= 0.5

        n = self._total - start
        if n >= _MIN_RUN_UPDATE_LENGTH:
            d0 = self._values[start + 1] - self._values[start]
            d1 = self._values[start + 2] - self._values[start + 1]
            if (d0 >= 0) != (d1 >= 0):
                self._up_runs -= 1
            if (d0 > 0) != (d1 > 0):
                self._down_runs -= 1

        self._start += 1


def suffix_pvalues(
    values: TimeSeriesFlex,
    min_length: int,
) -> dict[str, NDArray[np.float64]]:
    """
    Compute the p-value evolution of every suffix of a cleaned series.

    Parameters
    ----------
        values
            A 1D array of finite numeric values (see SuffixEvolution).
        min_length
            The minimum suffix length to test.

    Returns
    -------
        A dict keyed by hypothesis ('R', 'H', 'I', 'S'), each holding an array
        with one entry per observation. Positions whose suffix is too short to
        test are NaN.
    """
    array = np.asarray(values, dtype=float)
    size = array.size
    result = {key: np.full(size, np.nan, dtype=float) for key in _HYPOTHESES}

    if size < min_length:
        return result

    for start, p_values in SuffixEvolution(array, min_length):
        for key in _HYPOTHESES:
            result[key][start] = p_values[key]

    return result
