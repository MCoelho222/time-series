"""Methods for array manipulation."""
from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rhis.custom_types import TimeSeriesFlex


def clean_numeric_array(values: TimeSeriesFlex) -> NDArray[np.float64]:
    """Return finite numeric values from an arbitrary iterable."""
    numeric_values = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(numeric_value):
            numeric_values.append(numeric_value)

    return np.asarray(numeric_values, dtype=np.float64)

