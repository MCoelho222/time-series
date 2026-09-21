from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame, Index

from rhis.exceptions import RhisSummaryNotBuiltError, raise_if_no_rhis_df_exists
from rhis.hypothesis.homogeneity import mann_whitney
from rhis.hypothesis.independence import wald_wolfowitz
from rhis.hypothesis.randomness import wallis_moore
from rhis.hypothesis.stationarity import mann_kendall
from rhis.plotting import _period_label, plot_rhis_evolution
from rhis.utils import clean_numeric_array, slice_init, slices_to_evol

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import Series

    from rhis.custom_types import RhisStat
    from rhis.custom_types.data import TimeSeriesFlex


MIN_NUMERIC_VALUES = 10
DEFAULT_ALPHA = 0.05

RHIS_HYPOTHESES = ('R', 'H', 'I', 'S')

SUMMARY_DF_COLUMNS: tuple[str, ...] = (
    'original_length',
    'representative_length',
    'stat',
    'discarded_percentage',
    'remaining_percentage',
    'non_numeric_excluded',
    'most_rejected_hypothesis',
    'alpha',
    'original_period',
    'representative_period',
)


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
        self.is_rhis_complete = False
        self.alpha = DEFAULT_ALPHA

        self._summary_df: DataFrame | None = None

        self.length_init_ts = slice_init(len(self.orig_df))


    def _build_rhis_initial_df(self, df_cols: list[str], df_index: Index | None, *, rhis_min: bool) -> DataFrame:
        rhis = ['R', 'H', 'I', 'S']
        if rhis_min:
            rhis.extend(['RHIS-min'])
        cols = [(col, hyp) for col in df_cols for hyp in rhis]
        multi_index_cols = pd.MultiIndex.from_tuples(cols)
        result_df = pd.DataFrame(columns=multi_index_cols, index=df_index)

        return result_df


    @staticmethod
    def _slice_and_pad(orig_ts: NDArray[np.float64], idx: tuple[int, int]) -> NDArray[np.float64]:
        """
        Build a full-length array from a (start, end) slice of the
        original series, padding the parts outside the slice with NaN so
        it stays aligned with the original index.
        """
        nums_ts = orig_ts[idx[0]:idx[1]]
        nan_init = np.full(idx[0], np.nan)
        nan_fin = np.full(len(orig_ts) - idx[1], np.nan)
        full_ts = np.append(nan_init, nums_ts)
        full_ts = np.append(full_ts, nan_fin)

        return full_ts


    def _find_rhis_compliant_idxs(self, pvalue_ts: NDArray[np.float64], alpha: float) -> tuple[int, int]:
        """
        Find the most recent stretch of the time series that is
        RHIS-compliant: the longest possible slice that still ends at
        the latest observation.

        The p-value series passed in is produced by the evolution
        process triggered by Rhis.build_rhis_evol_df(); see the
        'pvalue_ts' description below for how to read it.

        Parameters
        ----------
            pvalue_ts
                The evolution of p-values computed by Rhis.build_rhis_evol_df()
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


    def _add_rhis_min_to_evol(self, evol_dict: dict[str, list[float]]) -> dict[str, list[float]]:
        rhis_values = list(evol_dict.values())
        evol_dict['RHIS-min'] = list(np.min(rhis_values, axis=0))

        return evol_dict


    def _ts_evol(self, ts: Series, *, rhis_min: bool) -> None:
        evol = self.build_rhis_dict_from_timeseries(ts, self.alpha, self.length_init_ts)

        if rhis_min:
            evol = self._add_rhis_min_to_evol(evol)

        if self.rhis_df is None:  # pragma: no cover - _build_rhis_evol_df() always sets it before this loop
            msg = "RHIS dataframe has not been initialized."
            raise RuntimeError(msg)

        for hyp, ps in evol.items():
            self.rhis_df[(ts.name, hyp)] = ps


    def build_rhis_evol_df(
        self,
        cols: list[str] | None = None,
        length_init_ts: int | None = None,
        *,
        rhis_min: bool = True,
    ) -> DataFrame:
        """
        Build a dataframe (self.rhis_df) with the evolution of the p-values
        of the randomness, homogeneity, independence and stationarity (rhis)
        tests applied to the slices of the time series in the original
        dataframe (self.orig_df).

        Parameters
        ----------
            cols
                An Iterable with the columns' names to be analyzed. Defaults
                to all columns of self.orig_df.
            length_init_ts
                The minimum slice length for which the tests are defined. If
                not given, the value set at construction time is used.
            rhis_min
                When True, also adds the 'RHIS-min' p-value evolution, i.e.
                the pointwise minimum over the R, H, I and S evolutions.

        Return
        ------
            DataFrame with the p-values evolution.
        """
        if length_init_ts is not None:
            self.length_init_ts = length_init_ts

        msg = "Generating RHIS series..."
        logger.info(msg)

        evol_cols = cols if cols is not None else self.orig_df.columns.tolist()
        self.rhis_df = self._build_rhis_initial_df(evol_cols, self.orig_df.index, rhis_min=rhis_min)
        for col in evol_cols:
            ts = self.orig_df[col]
            self._ts_evol(ts, rhis_min=rhis_min)

        logger.info("RHIS evolution dataframe created successfully.")
        self.is_rhis_complete = True

        return self.rhis_df


    def build_rhis_compliant_df(self, stat: RhisStat = 'min') -> DataFrame:
        """
        Build a dataframe holding only the representative (RHIS-compliant)
        time series of each column of self.orig_df, without creating
        self.rhis_df.

        For every column, the R, H, I and S p-value evolutions are computed
        (see build_rhis_dict_from_timeseries); the compliance decision is
        then based either on the pointwise minimum of those evolutions
        ('min') or on the evolution of a single hypothesis. On top of that,
        the longest trailing slice that passes the tests at self.alpha is
        recovered via _find_rhis_compliant_idxs. Each column of the returned
        dataframe holds that slice, NaN-padded to the original index.

        As a side effect, this also builds self.summary_df, a per-series
        summary of the process (see the 'summary_df' property).

        Parameters
        ----------
            stat
                The basis for the compliance decision, either 'min' (the
                pointwise minimum over the R, H, I and S evolutions) or a
                single hypothesis among 'R', 'H', 'I' and 'S'. Defaults to
                'min'.

        Return
        ------
            A new DataFrame with the same index as self.orig_df and one
            column per original column, holding only the representative
            (RHIS-compliant) time series.

        Raises
        ------
            ValueError
                If 'stat' is not one of 'min', 'R', 'H', 'I' or 'S'.
        """
        valid_stats = (*RHIS_HYPOTHESES, 'min')
        if stat not in valid_stats:
            msg = f"Invalid stat '{stat}'; choose one of {list(valid_stats)}."
            logger.debug(msg)
            raise ValueError(msg)

        repr_df = DataFrame(index=self.orig_df.index)
        summary_rows: dict[str, dict[str, object]] = {}
        for col in self.orig_df.columns:
            ts = self.orig_df[col]
            evol = self.build_rhis_dict_from_timeseries(ts, self.alpha, self.length_init_ts)
            if stat == 'min':
                stat_pvalues = list(np.min(list(evol.values()), axis=0))
            else:
                stat_pvalues = evol[stat]
            cut_idxs = self._find_rhis_compliant_idxs(np.asarray(stat_pvalues, dtype=float), self.alpha)
            numeric_ts = pd.to_numeric(self.orig_df[col], errors='coerce').to_numpy(dtype=float)
            repr_df[col] = self._slice_and_pad(numeric_ts, cut_idxs)

            original_length = len(numeric_ts)
            representative_length = cut_idxs[1] - cut_idxs[0]
            original_period = _period_label(self.orig_df.index[0], self.orig_df.index[-1])
            representative_start = (
                self.orig_df.index[cut_idxs[0]] if cut_idxs[0] < original_length else self.orig_df.index[-1]
            )
            representative_period = _period_label(representative_start, self.orig_df.index[-1])

            summary_rows[col] = {
                'original_length': original_length,
                'representative_length': representative_length,
                'stat': stat,
                'discarded_percentage': round((original_length - representative_length) / original_length * 100, 2),
                'remaining_percentage': round(representative_length / original_length * 100, 2),
                'non_numeric_excluded': int(np.count_nonzero(np.isnan(numeric_ts))),
                'most_rejected_hypothesis': self._most_rejected_hypothesis(evol, self.alpha),
                'alpha': self.alpha,
                'original_period': original_period,
                'representative_period': representative_period,
            }

        self._summary_df = DataFrame.from_dict(summary_rows, orient='index').reindex(columns=list(SUMMARY_DF_COLUMNS))

        logger.info("RHIS-compliant dataframe built successfully.")

        return repr_df


    @staticmethod
    def _most_rejected_hypothesis(evol: dict[str, list[float]], alpha: float) -> str:
        """The R, H, I or S hypothesis rejected most often along its evolution."""
        best_hypothesis = 'none'
        best_count = 0
        for hypothesis in RHIS_HYPOTHESES:
            pvalues = np.asarray(evol[hypothesis], dtype=float)
            count = int(np.count_nonzero((~np.isnan(pvalues)) & (pvalues < alpha)))
            if count > best_count:
                best_count = count
                best_hypothesis = hypothesis

        return best_hypothesis


    @property
    def summary_df(self) -> DataFrame:
        """
        A per-series summary of the representative (RHIS-compliant)
        selection, built by Rhis.build_rhis_compliant_df().

        One row per original time series (indexed by the column name) with:

        original_length
            Number of observations in the original series.
        representative_length
            Number of observations kept in the representative series.
        stat
            The p-value series used to derive the representative indexes
            ('min', or one of 'R', 'H', 'I' or 'S').
        discarded_percentage
            Relative amount of observations discarded, as a percentage of
            the original length.
        remaining_percentage
            Relative amount of observations kept, as a percentage of the
            original length.
        non_numeric_excluded
            Number of non-numeric or missing values excluded from the
            statistical analyses.
        most_rejected_hypothesis
            The hypothesis among R, H, I and S rejected most often along
            its p-value evolution, or 'none' if nothing was rejected.
        alpha
            The significance level used for the decisions.
        original_period
            Label of the complete period covered by the original series.
        representative_period
            Label of the period covered by the representative series.

        Raises
        ------
            RhisSummaryNotBuiltError
                If `build_rhis_compliant_df()` has not been run yet.
        """
        if self._summary_df is None:
            msg = "Rhis.build_rhis_compliant_df() should be run before accessing Rhis.summary_df."
            logger.debug(msg)
            raise RhisSummaryNotBuiltError(msg)

        return self._summary_df


    def calculate_rhis_once_with_full_ts(self, repr_df: DataFrame) -> dict[str, dict[str, float]]:
        """
        Apply the RHIS tests to the representative series held in the
        dataframe returned by build_rhis_compliant_df().

        For each column of 'repr_df', this runs Rhis.calculate_rhis and
        stores the resulting p-values in a new dictionary.

        Parameters
        ----------
            repr_df
                A dataframe holding one representative (RHIS-compliant)
                series per column, NaN-padded to the original index.

        Return
        ------
            A dictionary mapping every column name of 'repr_df' to its
            RHIS p-values {'R', 'H', 'I', 'S'}.
        """
        results: dict[str, dict[str, float]] = {}
        for repr_name in repr_df.columns:
            results[repr_name] = Rhis.calculate_rhis(repr_df[repr_name].to_numpy(), alpha=self.alpha)

        return results


    def is_all_rhis_compliant(self, repr_df: DataFrame) -> bool:
        """
        Check whether every representative series passes all RHIS tests
        at the current significance level.

        For each column of 'repr_df' this inspects the p-values returned
        by calculate_rhis_once_with_full_ts. A test is considered to have
        rejected the null hypothesis when its p-value is less than alpha;
        an undefined p-value (NaN, e.g. for the independence test on a
        constant series) is never a rejection.

        Parameters
        ----------
            repr_df
                A dataframe holding one representative (RHIS-compliant)
                series per column, NaN-padded to the original index.

        Return
        ------
            True when every hypothesis of every representative series
            fails to reject; False otherwise. For every rejected test a
            debug message is logged naming the column and the
            hypothesis.
        """
        logger.info("Checking RHIS compliance...")
        pvalues = self.calculate_rhis_once_with_full_ts(repr_df)

        all_compliant = True
        for repr_name, hyp_pvalues in pvalues.items():
            for hyp, p_value in hyp_pvalues.items():
                if p_value < self.alpha:
                    all_compliant = False
                    msg = (f"Column '{repr_name}' rejected hypothesis '{hyp}' "
                           f"(p = {p_value:.4f} < alpha = {self.alpha}).")
                    logger.warning(msg)

        if all_compliant:
            logger.info("All RHIS compliant!")
        return all_compliant


    def plot_evolution(
        self,
        *,
        show_repr: bool = True,
        repr_df: DataFrame | None = None,
        figtitle: str | None = None,
    ) -> None:
        """
        Save one figure per analyzed time series to the `rhis_plots` directory.

        Each figure shows the series values (and its RHIS-compliant repr when
        `show_repr` is True) together with the evolution of the R, H, I and S
        p-values and the alpha line.

        Parameters
        ----------
            show_repr
                Whether to plot the RHIS-compliant representative series
                when a 'repr_df' is given.
            repr_df
                The dataframe returned by `build_rhis_compliant_df()`,
                holding the representative series. Only used when
                `show_repr` is True.
            figtitle
                An optional title for the saved figures. When given, it is used
                instead of the default `'RHIS <series>'`; see
                `plot_rhis_evolution`.

        Raises
        ------
            RhisEvolNotCalledError
                If `build_rhis_evol_df()` has not been run yet.
        """
        raise_if_no_rhis_df_exists(is_rhis_complete=self.is_rhis_complete)

        if self.rhis_df is None:
            msg = 'RHIS dataframe has not been initialized.'
            raise RuntimeError(msg)

        plot_rhis_evolution(
            self.orig_df,
            self.rhis_df,
            self.alpha,
            show_repr=show_repr,
            repr_df=repr_df,
            figtitle=figtitle,
        )


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


