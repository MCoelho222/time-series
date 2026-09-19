from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame, Index

from rhis.exceptions import raise_if_no_rhis_run
from rhis.hypothesis.homogeneity import mann_whitney
from rhis.hypothesis.independence import wald_wolfowitz
from rhis.hypothesis.randomness import wallis_moore
from rhis.hypothesis.stationarity import mann_kendall
from rhis.plotting import plot_rhis_evolution
from rhis.utils import clean_numeric_array, slice_init, slices_to_evol

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from pandas import DataFrame, Series

    from rhis.custom_types import RhisCode, RhisStat
    from rhis.custom_types.data import TimeSeriesFlex


MIN_NUMERIC_VALUES = 10
DEFAULT_ALPHA = 0.05


class Rhis:
    def __init__(self, df: DataFrame) -> None:
        if (not isinstance(df, pd.DataFrame) or isinstance(df.index, pd.MultiIndex)):
            msg = "The parameter 'df' must be a non-MultiIndex pandas.DataFrame."
            logger.debug(msg)
            raise ValueError(msg)

        for column in df.columns:
            numeric_values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
            numeric_count = np.count_nonzero(np.isfinite(numeric_values))
            if numeric_count < MIN_NUMERIC_VALUES:
                msg = (f"Series {column} has fewer than 10 numeric values ({numeric_count}). "
                       "Statistical results will have no useful meaning.")
                logger.debug(msg)

        self.orig_df: DataFrame = df.copy()
        self.rhis_df: DataFrame | None = None
        self.rhis_stats_included = False
        self.is_rhis_complete = False
        self.alpha = DEFAULT_ALPHA

        self.length_init_ts = slice_init(len(self.orig_df))


    def _build_rhis_initial_df(self, df_cols: list[str], df_index: Index | None, *, include_rhis_stats: bool) -> DataFrame:
        rhis = ['R', 'H', 'I', 'S']
        if include_rhis_stats:
            rhis.extend(['min', 'avg', 'med', 'max'])
        cols = [(col, hyp) for col in df_cols for hyp in rhis]
        multi_index_cols = pd.MultiIndex.from_tuples(cols)
        result_df = pd.DataFrame(columns=multi_index_cols, index=df_index)

        return result_df


    def _include_rhis_compliant_ts_in_df(self, df: DataFrame, idx: tuple[int, int], df_col: str) -> None:
        orig_ts = df[df_col].to_numpy()
        nums_ts = orig_ts[idx[0]:idx[1]]
        nan_init = np.full(idx[0], np.nan)
        nan_fin= np.full(len(orig_ts) - idx[1], np.nan)
        full_ts = np.append(nan_init, nums_ts)
        full_ts = np.append(full_ts, nan_fin)

        df.loc[:, df_col + '_repr'] = full_ts


    def _find_rhis_compliant_idxs(self, pvalue_ts: NDArray[np.float64], alpha: float) -> tuple[int, int]:
        """
        Find the most recent stretch of the time series that is
        RHIS-compliant: the longest possible slice that still ends at
        the latest observation.

        The p-value series passed in is produced by the evolution
        process triggered by Rhis.evol(); see the 'pvalue_ts'
        description below for how to read it.

        Parameters
        ----------
            pvalue_ts
                The evolution of p-values computed by Rhis.evol() for a
                single time series (or for a derived statistic, like
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
        if pvalue_ts_last == 0:
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


    def _include_rhis_stats_in_df(self, df: DataFrame) -> None:
        col_groups = [df.columns[i:i + 4] for i in range(0, len(df.columns), 4)]
        for group in col_groups:
            df[(group[0][0], "min")] = df[group].min(axis=1)
            df[(group[0][0], "mean")] = df[group].mean(axis=1)
            df[(group[0][0], "median")] = df[group].median(axis=1)
            df[(group[0][0], "max")] = df[group].max(axis=1)


    @staticmethod
    def build_rhis_dict_from_timeseries(ts: Series, alpha: float, length_init_ts: int) -> dict[str, list[float]]:
        ts_np = ts.to_numpy()[::-1]
        slices = slices_to_evol(ts_np, length_init_ts)
        evol: dict[str, list[float]] = {'R': [], 'H': [], 'I': [], 'S': []}

        ts_clean = clean_numeric_array(ts_np)
        constant_series = bool(np.all(ts_clean == ts_clean[0]))
        if constant_series:
            msg = "Constant series detected; recording NaN independence p-values."
            logger.debug(msg)

        for sli in slices:
            rhis_dict = Rhis.calculate_rhis(sli, alpha, constant_series=constant_series)
            evol['R'].append(rhis_dict['R'])
            evol['H'].append(rhis_dict['H'])
            evol['I'].append(rhis_dict['I'])
            evol['S'].append(rhis_dict['S'])

        fill = np.full(length_init_ts - 1, np.nan)

        for hyp, ps in evol.items():
            evol[hyp] = list(np.append(ps[::-1], fill))

        return evol


    def _add_rhis_stats_to_evol(self, evol_dict: dict[str, list[float]]) -> dict[str, list[float]]:
        stats_dict: dict[str, Callable[..., NDArray[np.float64]]] = {
            'min': np.min,
            'med': np.median,
            'avg': np.mean,
            'max': np.max,
        }

        rhis_values = list(evol_dict.values())

        for name, method in stats_dict.items():
            evol_dict[name] = list(method(rhis_values, axis=0, keepdims=True).ravel())

        return evol_dict


    def _ts_evol(self, ts: Series,*, include_rhis_stats: bool) -> None:
        evol = self.build_rhis_dict_from_timeseries(ts, self.alpha, self.length_init_ts)

        if include_rhis_stats:
            evol = self._add_rhis_stats_to_evol(evol)

        if self.rhis_df is None:  # pragma: no cover - evol() always sets it before this loop
            msg = "RHIS dataframe has not been initialized."
            raise RuntimeError(msg)

        for hyp, ps in evol.items():
            self.rhis_df[(ts.name, hyp)] = ps


    def evol(
        self,
        cols: list[str] | None = None,
        length_init_ts: int | None = None,
        *,
        include_rhis_stats: bool = True,
    ) -> DataFrame:
        """
        Generate a dataframe (self.rhis_statistic_df or self.rhis_full_df) with the series from
        the evolutional application of the randomness, homogeneity, independence and
        stationarity (rhis) tests to the time series in the original dataframe
        (self.orig_df).

        Parameters
        ----------
            cols
                An Iterable with string representing the columns' names to be analyzed.
            stat
                One of ['min', 'med', 'max', None]. The statistic to be applied to the rhis
                evolution. For example, if 'min', the minimum p-value among the rhis p-values
                is used, and self.rhis_statistic_df is created.
            alpha
                The significance level.

        Return
        ------
            DataFrame with p-values evolution
        """
        if length_init_ts is not None:
            self.length_init_ts = length_init_ts

        msg = "Generating RHIS series..."
        logger.info(msg)

        evol_cols = cols if cols is not None else self.orig_df.columns.tolist()
        self.rhis_df = self._build_rhis_initial_df(evol_cols, self.orig_df.index, include_rhis_stats=include_rhis_stats)
        for col in evol_cols:
            ts = self.orig_df[col]
            self._ts_evol(ts, include_rhis_stats=include_rhis_stats)
        if include_rhis_stats:
            self.rhis_stats_included = True

        logger.info("RHIS completed successfully.")
        self.is_rhis_complete = True

        return self.rhis_df


    def add_rhis_compliant_to_df(self, rhis_stat: RhisStat | RhisCode = 'min') -> DataFrame:
        raise_if_no_rhis_run(is_rhis_complete=self.is_rhis_complete)

        if self.rhis_df is None:
            msg = 'RHIS dataframe has not been initialized.'
            raise RuntimeError(msg)

        cols_orig_df = self.orig_df.columns
        if rhis_stat in ['min', 'max', 'mean', 'median'] and not self.rhis_stats_included:
            self._include_rhis_stats_in_df(self.rhis_df)

        for col in cols_orig_df:
            target_col = (col, rhis_stat)
            rhis_series = self.rhis_df[target_col].to_numpy()
            cut_idxs = self._find_rhis_compliant_idxs(rhis_series, self.alpha)
            self._include_rhis_compliant_ts_in_df(self.orig_df, cut_idxs, col)

        logger.info("RHIS compliant data successfully included in the dataframe.")
        return self.orig_df


    def calculate_repr_rhis_pvalues(self) -> dict[str, dict[str, float]]:
        """
        Apply the RHIS tests to the RHIS-compliant (representative)
        series that were added to self.orig_df by
        add_rhis_compliant_to_df().

        For each representative series (the '<col>_repr' columns), this
        runs Rhis.calculate_rhis, stores the resulting p-values in a new
        dictionary, and returns it.

        Parameters
        ----------
            (none)

        Return
        ------
            A dictionary mapping every '<col>_repr' column name to its
            RHIS p-values {'R', 'H', 'I', 'S'}.

        Raises
        ------
            RhisEvolNotCalledError
                If Rhis.evol() has not been run yet.
            ValueError
                If no representative series are present; run
                Rhis.add_rhis_compliant_to_df() first.
        """
        raise_if_no_rhis_run(is_rhis_complete=self.is_rhis_complete)

        repr_cols = [col for col in self.orig_df.columns if col.endswith('_repr')]
        if not repr_cols:
            msg = ("No RHIS representative series found in the dataframe. "
                   "Run Rhis.add_rhis_compliant_to_df() to add them first.")
            logger.debug(msg)
            raise ValueError(msg)

        results: dict[str, dict[str, float]] = {}
        for repr_name in repr_cols:
            results[repr_name] = Rhis.calculate_rhis(self.orig_df[repr_name].to_numpy(), alpha=self.alpha)

        return results


    def is_all_rhis_compliant(self) -> bool:
        """
        Check whether every RHIS-compliant (representative) series
        passes all RHIS tests at the current significance level.

        For each '<col>_repr' column this inspects the p-values returned
        by calculate_repr_rhis_pvalues. A test is considered to have
        rejected the null hypothesis when its p-value is less than alpha;
        an undefined p-value (NaN, e.g. for the independence test on a
        constant series) is never a rejection.

        Parameters
        ----------
            (none)

        Return
        ------
            True when every hypothesis of every representative series
            fails to reject; False otherwise. For every rejected test a
            debug message is logged naming the column and the
            hypothesis.

        Raises
        ------
            RhisEvolNotCalledError
                If Rhis.evol() has not been run yet.
            ValueError
                If no representative series are present; run
                Rhis.add_rhis_compliant_to_df() first.
        """
        pvalues = self.calculate_repr_rhis_pvalues()

        all_compliant = True
        for repr_name, hyp_pvalues in pvalues.items():
            for hyp, p_value in hyp_pvalues.items():
                if p_value < self.alpha:
                    all_compliant = False
                    msg = (f"Column '{repr_name}' rejected hypothesis '{hyp}' "
                           f"(p = {p_value:.4f} < alpha = {self.alpha}).")
                    logger.debug(msg)

        return all_compliant


    def plot(self, *, show_repr: bool = True, figtitle: str | None = None) -> None:
        """
        Save one figure per analyzed time series to the `rhis_plots` directory.

        Each figure shows the series values (and its RHIS-compliant repr when
        `show_repr` is True) together with the evolution of the R, H, I and S
        p-values and the alpha line. To include the representative series, run
        `add_rhis_compliant_to_df()` before plotting.

        Parameters
        ----------
            show_repr
                Whether to plot the RHIS-compliant representative series when
                it has been added to the dataframe.
            figtitle
                An optional title for the saved figures. When given, it is used
                instead of the default `'RHIS <series>'`; see
                `plot_rhis_evolution`.

        Raises
        ------
            RhisEvolNotCalledError
                If `evol()` has not been run yet.
        """
        raise_if_no_rhis_run(is_rhis_complete=self.is_rhis_complete)

        if self.rhis_df is None:
            msg = 'RHIS dataframe has not been initialized.'
            raise RuntimeError(msg)

        plot_rhis_evolution(self.orig_df, self.rhis_df, self.alpha, show_repr=show_repr, figtitle=figtitle)


    @staticmethod
    def calculate_rhis(
        ts: TimeSeriesFlex,
        alpha: float = DEFAULT_ALPHA,
        *,
        constant_series: bool = False,
    ) -> dict[str, float]:
        ts = clean_numeric_array(ts)

        if constant_series or np.all(ts == ts[0]):
            independence_p_value = np.nan
        else:
            try:
                independence_p_value = wald_wolfowitz(ts, alpha=alpha, on_ranks=True).p_value
            except ValueError:
                msg = "Independence test undefined for this slice; recording NaN p-value."
                logger.debug(msg)
                independence_p_value = np.nan

        return {
            'R': wallis_moore(ts, alpha=alpha).p_value,
            'H': mann_whitney(ts, alpha=alpha).p_value,
            'I': independence_p_value,
            'S': mann_kendall(ts, alpha=alpha).p_value,
        }


