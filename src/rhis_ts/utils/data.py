from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:

    from rhis_ts.custom_types import TimeSeriesFlex


def split_into_parts(ts: TimeSeriesFlex, parts: int) -> list[TimeSeriesFlex]:
    """
    Divide a series in multiple equal parts, as much as possible.

    Parameters
    ----------
        ts
            A list with items of any type.
        parts
            The number of parts to divide the series into.

    Return
    ------
        A list of lists with each part of the series respecting the original order.

    Raises
    ------
        ValueError
            If `parts` is greater than the length of the series.
    """
    size = len(ts)

    if parts > size:
        msg = f"The parameter 'parts' ({parts}) cannot be greater than the series length ({size})."
        logger.debug(msg)
        raise ValueError(msg)

    cut_index = size / parts

    if size % parts != 0:
        cut_index = int(np.ceil(cut_index))
    else:
        cut_index = int(cut_index)

    return [ts[cut_index * i: cut_index * (i + 1)] for i in range(parts)]


def slices_to_evol(ts: TimeSeriesFlex, start: int) -> list[TimeSeriesFlex]:
    """
    Break a flat series into its suffix slices.

    Each slice starts one position later than the previous one and runs to
    the very end of the series, so the number of elements decreases by one
    from one slice to the next. The first slice is the complete series and
    the last one has exactly `start` elements.

    Parameters
    ----------
        ts
            A list with integers or floats.
        start
            The minimum number of elements a slice must have to be tested.

    Returns
    -------
        A 2D list of lists: every suffix slice of `ts` with at least
        `start` elements, from the complete series down to the shortest one.
    """
    slices = []
    for i in range(len(ts) - (start - 1)):
        slices.append(ts[i:])

    return slices

if __name__ == "__main__":
    ts = [2, 4, 5, 1, 10, 12, 33, 22, 21, 15]
    print(slices_to_evol(ts, 5))

