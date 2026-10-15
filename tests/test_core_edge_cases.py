from __future__ import annotations

import numpy as np

from rhis_ts.core import MIN_SERIES_LENGTH, Rhis

ALL_EQUAL_SERIES = [7.0] * 20

DEGENERATE_SLICE = [3.0] + [7.0] * (MIN_SERIES_LENGTH - 1)

CONSTANT_TAIL_SERIES = [12.0, 9.5, 14.2, 8.8, 11.1, 13.7, 10.4, 9.9] + [7.0] * 12
DEGENERATE_START = 7


def test_calculate_rhis_all_equal_returns_all_nan() -> None:
    """
    Every hypothesis test is undefined on a constant series, so all four
    p-values (and, later, their minimum) must be NaN.
    """
    result = Rhis.calculate_rhis(ALL_EQUAL_SERIES)

    assert all(np.isnan(p_value) for p_value in result.values())


def test_calculate_rhis_marks_degenerate_slice_with_nan() -> None:
    """
    A slice long enough to test but almost flat (here one value followed by
    equal values) has no meaningful serial-correlation variance, and
    Wald-Wolfowitz raises for it. calculate_rhis must report that as a NaN
    independence p-value instead of letting the error escape.
    """
    result = Rhis.calculate_rhis(DEGENERATE_SLICE)

    assert np.isnan(result['I'])
    assert np.isfinite(result['R'])
    assert np.isfinite(result['H'])
    assert np.isfinite(result['S'])


def test_build_rhis_evol_dict_handles_degenerate_tail() -> None:
    """
    A series ending in a long run of equal values used to crash the
    evolution with a ValueError from Wald-Wolfowitz. The degenerate slices
    must instead appear as NaN, starting at the first untestable slice, and
    the untestable NaN must propagate into RHIS-min.
    """
    rhis = Rhis({'flow': CONSTANT_TAIL_SERIES})

    evol = rhis.build_rhis_evol_dict('flow')['flow']

    assert np.isfinite(np.asarray(evol['I'][:DEGENERATE_START])).all()
    assert np.isnan(np.asarray(evol['I'][DEGENERATE_START:])).all()
    assert np.isfinite(np.asarray(evol['RHIS-min'][:DEGENERATE_START])).all()
    assert np.isnan(np.asarray(evol['RHIS-min'][DEGENERATE_START:])).all()


def test_get_repr_idxs_does_not_crash_on_degenerate_tail() -> None:
    """
    Building the representative indexes on a series with a degenerate tail
    must not raise, must keep the slice ending at the latest observation,
    and must never start past the point where testing breaks down.
    """
    rhis = Rhis({'flow': CONSTANT_TAIL_SERIES})

    start, end = rhis.get_repr_idxs('flow')['flow']

    assert end == len(CONSTANT_TAIL_SERIES)
    assert 0 <= start <= DEGENERATE_START
