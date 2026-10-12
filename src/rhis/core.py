from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis.hypothesis.homogeneity import mann_whitney
from rhis.hypothesis.independence import wald_wolfowitz
from rhis.hypothesis.randomness import wallis_moore
from rhis.hypothesis.stationarity import mann_kendall
from rhis.utils import clean_numeric_array, slice_init, slices_to_evol

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import Series

    from rhis.custom_types.data import TimeSeriesFlex

MIN_NUMERIC_VALUES = 10
MIN_TS_LENGTH_FOR_RHIS = 5
DEFAULT_ALPHA = 0.05

RHIS_HYPOTHESES = ('R', 'H', 'I', 'S')


class Rhis:
    def __init__(self, df: DataFrame) -> None:
        # TODO: We should accept MultiIndex and dict
        if (not isinstance(df, pd.DataFrame) or isinstance(df.index, pd.MultiIndex)):
            msg = "The parameter 'df' must be a non-MultiIndex pandas.DataFrame."
            logger.debug(msg)
            raise ValueError(msg)

        self.orig_df: DataFrame = df.copy()
        self.numeric_orig_df: DataFrame = self.orig_df.copy().apply(pd.to_numeric, errors='coerce')

        cols_to_drop = self.numeric_orig_df.columns[self.numeric_orig_df.isna().all()].to_list()

        for col in self.numeric_orig_df.columns:
            numeric_count = np.count_nonzero(np.isfinite(self.numeric_orig_df[col]))
            if numeric_count < MIN_NUMERIC_VALUES and col not in cols_to_drop:
                cols_to_drop.append(col)

        if len(cols_to_drop) > 0:
            msg = f"Dropping columns with all NaN or fewer than {MIN_NUMERIC_VALUES} numeric values: {cols_to_drop}"
            logger.debug(msg)
            self.numeric_orig_df = self.numeric_orig_df.drop(columns=cols_to_drop)

        self.alpha = DEFAULT_ALPHA
        self.length_init_ts = slice_init(len(self.orig_df))
        self.repr_idxs: dict[str, tuple[int, int]] | None = None

        self.pvalues: dict[str, dict[str, list[float]]] = {}
        for col in self.numeric_orig_df.columns:
            self.pvalues[col] = {'R': [], 'H': [], 'I': [], 'S': [], 'RHIS-min': []}


    def build_rhis_evol_dict(
        self,
        ts_name: str | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        cols = list(self.numeric_orig_df.columns) if ts_name is None else [ts_name]

        for col in cols:
            if len(self.pvalues[col]['RHIS-min']) == 0:
                evol_dict = self._evol_dict_from_ts(self.numeric_orig_df[col], self.length_init_ts)
                self.pvalues[col] = self._add_rhis_min_to_evol(evol_dict)

        return {col: self.pvalues[col] for col in cols}


    def get_repr_idxs(
        self,
        ts_name: str | None = None,
        alpha: float | None = None,
    ) -> dict[str, tuple[int, int]]:
        alpha = self.alpha if alpha is None else alpha
        evols = self.build_rhis_evol_dict(ts_name)

        self.repr_idxs = {
            col: self._find_rhis_compliant_idxs(np.asarray(evol['RHIS-min'], dtype=float), alpha)
            for col, evol in evols.items()
        }

        return self.repr_idxs


    def _find_rhis_compliant_idxs(self, pvalue_ts: NDArray[np.float64], alpha: float) -> tuple[int, int]:
        """
        Find the most recent stretch of the time series that is
        RHIS-compliant: the longest possible slice that still ends at
        the latest observation.

        The p-value series passed in is produced by the evolution
        process triggered by Rhis.build_rhis_evol_dict(); see the
        'pvalue_ts' description below for how to read it.

        Parameters
        ----------
            pvalue_ts
                The evolution of p-values computed by Rhis.build_rhis_evol_dict()
                for a single time series (or for a derived statistic, like
                'min', computed over its R, H, I and S p-values). The
                value at index i is the p-value of the slice that starts
                at position i and runs to the very end of the series.
                Index 0 tests the complete series, and the trailing
                entries are NaN because there are not enough
                observations left to run the tests.

            alpha
                The significance level. A p-value greater than or equal
                to alpha means that slice passed the tests (failed to
                reject the null hypothesis), i.e., it is RHIS-compliant.

        Returns
        -------
            A tuple (start, end) with indexes into the original series.
            The slice running from 'start' to the end of the series is
            the representative slice: the longest RHIS-compliant stretch
            that ends at the most recent observation.
        """
        pvalue_ts_last = len(pvalue_ts)
        if pvalue_ts_last == 0 or np.all(np.isnan(pvalue_ts)):
            return (0, 0)

        # The first entry tests the complete time series. When p >= alpha
        # the whole series passed every RHIS test (or the selected one),
        # so the representative slice is the entire series.
        if pvalue_ts[0] >= alpha:
            return (0, pvalue_ts_last)

        # The complete series failed the tests, so the representative
        # slice must be a slice that starts somewhere in the middle and
        # runs to the end (i.e., it always ends at the most recent
        # observation). These slices overlap in a simple way: the slice
        # starting at position 1 contains the slice starting at position
        # 2, which contains the one starting at 3, and so on. Each
        # longer slice has already been tested and failed, so the FIRST
        # slice that passes (p-value at or above alpha) is, by definition,
        # the LONGEST one that passes and still reaches the most recent
        # observation.
        #
        # NaN entries mark slices too short to test; they act as a
        # stopping boundary, just like a passing p-value. pvalue_ts[0]
        # is always a real number here (the complete series is long
        # enough to test), so we are guaranteed to find a stopping
        # point somewhere after index 0.
        fails_to_reject = ~(pvalue_ts < alpha)  # p >= alpha, or NaN
        idx = int(np.argmax(fails_to_reject))

        return idx, pvalue_ts_last


    @staticmethod
    def _evol_dict_from_ts(ts: Series, length_init_ts: int) -> dict[str, list[float]]:
        slices = slices_to_evol(ts.to_numpy(), length_init_ts)
        evol: dict[str, list[float]] = {'R': [], 'H': [], 'I': [], 'S': []}

        for sli in slices:
            rhis_dict = Rhis.calculate_rhis(sli)
            evol['R'].append(rhis_dict['R'])
            evol['H'].append(rhis_dict['H'])
            evol['I'].append(rhis_dict['I'])
            evol['S'].append(rhis_dict['S'])

        fill = np.full(length_init_ts - 1, np.nan)

        for hyp, ps in evol.items():
            evol[hyp] = list(np.append(ps, fill))

        return evol


    def _add_rhis_min_to_evol(self, evol_dict: dict[str, list[float]]) -> dict[str, list[float]]:
        rhis_values = list(evol_dict.values())
        evol_dict['RHIS-min'] = list(np.min(rhis_values, axis=0))

        return evol_dict


    @staticmethod
    def calculate_rhis(ts: TimeSeriesFlex) -> dict[str, float]:
        ts = clean_numeric_array(ts)
        if len(ts) < MIN_TS_LENGTH_FOR_RHIS:
            return {'R': np.nan, 'H': np.nan, 'I': np.nan, 'S': np.nan}

        result = {
            'R': wallis_moore(ts).p_value,
            'H': mann_whitney(ts).p_value,
            'I': wald_wolfowitz(ts, on_ranks=True).p_value,
            'S': mann_kendall(ts).p_value,
        }

        return result


