"""Methods for array manipulation."""
from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rhis.custom_types import TimeSeriesFlex


def clean_numeric_array(values: TimeSeriesFlex) -> NDArray[np.float64] | list:
    """Return finite numeric values from an arbitrary iterable."""
    numeric_values = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(numeric_value):
            numeric_values.append(numeric_value)

    if not numeric_values:
        return []
        # msg = 'The time series contains no finite numeric values.'
        # raise ValueError(msg)

    return np.asarray(numeric_values, dtype=np.float64)


def nans_nums_from_array(
    pvalue_ts: NDArray[np.float64], *, only_nums: bool = True
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    pvalue_ts_mask = np.isnan(pvalue_ts)
    pvalue_ts_nums = pvalue_ts[~pvalue_ts_mask]

    if only_nums:
        return pvalue_ts_nums

    pvalue_ts_nan = pvalue_ts[pvalue_ts_mask]

    return pvalue_ts_nums, pvalue_ts_nan
