from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

TimeSeriesFlex = Sequence[int | float] | NDArray[np.int64 | np.float64]


class TiesData(TypedDict):
    corrected_ranks: NDArray[np.float64]
    tied_ranks: list[list[int]]
    ties_count: int
    ties_groups_count: list[int]
