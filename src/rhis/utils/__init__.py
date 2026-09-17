from __future__ import annotations

from rhis.utils.arrays import clean_numeric_array, nans_nums_from_array
from rhis.utils.data import slice_init, slices_to_evol, split_into_parts
from rhis.utils.p_value import p_value_normal, test_decision_normal
from rhis.utils.ranks import get_ties_index, ranks_with_ties_corrected, to_ranks

