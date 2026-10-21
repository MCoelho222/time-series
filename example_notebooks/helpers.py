import numpy as np
import pandas as pd
from rhis_ts import Rhis


def _slice_and_pad(orig_ts, idx: tuple[int, int]):
    nums_ts = orig_ts[idx[0]:idx[1]]
    nan_init = np.full(idx[0], np.nan)
    nan_fin = np.full(len(orig_ts) - idx[1], np.nan)
    full_ts = np.append(nan_init, nums_ts)
    full_ts = np.append(full_ts, nan_fin)

    return full_ts


def build_rhis_evol_df(rhis: Rhis):
    frames = {
        col: pd.DataFrame(evol, index=rhis.orig_df.index)
        for col, evol in rhis.pvalues.items()
    }

    return pd.concat(frames, axis=1, names=['series', 'hypothesis'])


def build_repr_df(rhis):
    series = {}
    for col in rhis.numeric_orig_df.columns:
        orig_ts = rhis.numeric_orig_df[col]
        idx = rhis.repr_idxs[col]
        series[col] = _slice_and_pad(orig_ts, idx)

    return pd.DataFrame(series, index=rhis.numeric_orig_df.index)