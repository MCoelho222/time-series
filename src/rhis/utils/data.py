from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:

    from rhis.custom_types import TimeSeriesFlex


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


def slice_init(n: int) -> int:
    limit = 100
    return 10 if n > limit else 5


def slices_to_evol(ts: TimeSeriesFlex, start: int) -> list[TimeSeriesFlex]:
    """
    Break a flat list into a list of lists (2D).

    Each list will have an increasing number of elements.
    The element with index n will have one more element than element with index n-1.

    Parameters
    ----------
        ts
            A list with integers or floats.

    Returns
    -------
        A 2D list of lists. The lists are slices with increasing number of elements,
        so that the last is the complete timeseries.
    """
    slices = []
    for i in range(len(ts) - (start - 1)):
        slices.append(ts[: i + start])

    return slices

