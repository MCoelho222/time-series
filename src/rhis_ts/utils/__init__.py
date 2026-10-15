from __future__ import annotations

from rhis_ts.utils.arrays import clean_numeric_array
from rhis_ts.utils.data import slices_to_evol, split_into_parts
from rhis_ts.utils.p_value import p_value_normal, p_value_normal_directional
from rhis_ts.utils.ranks import get_ties_index, ranks_with_ties_corrected, to_ranks

