"""The Rhis class: run the four RHIS-TS tests and locate the representative stretch."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis_ts.hypothesis.homogeneity import mann_whitney
from rhis_ts.hypothesis.independence import wald_wolfowitz
from rhis_ts.hypothesis.randomness import wallis_moore
from rhis_ts.hypothesis.stationarity import mann_kendall
from rhis_ts.utils import clean_numeric_array, slices_to_evol

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import Series

    from rhis_ts.custom_types.data import SeriesKey, TimeSeriesFlex

MIN_SERIES_LENGTH = 8
DEFAULT_ALPHA = 0.05

_SERIES_LIKE = (Sequence, np.ndarray, pd.Series)


def _is_series_like(values: object) -> bool:
    """Whether a mapping value looks like a sequence of observations."""
    return isinstance(values, _SERIES_LIKE) and not isinstance(values, str | bytes)


def _normalize_input(
    data: DataFrame | Mapping[str, TimeSeriesFlex | Series],
) -> tuple[DataFrame, dict[SeriesKey, int]]:
    """
    Normalize the accepted inputs into the DataFrame held internally.

    Returns the normalized DataFrame together with the original length of
    every series, which the ragged-mapping case needs to keep the reported
    indexes relative to each user series rather than to the padded frame.

    A DataFrame is used as is (its rows must not use a MultiIndex: each
    row is a time step, and its columns may be a MultiIndex). A mapping of
    series names to value sequences is materialized into a DataFrame whose
    columns are the mapping keys and whose values are taken positionally
    (any index carried by a pandas Series is discarded). Ragged mappings
    are padded with NaN up to the longest series.
    """
    if isinstance(data, DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            msg = (
                "The rows of the input DataFrame must not use a MultiIndex: "
                "each row is a time step."
            )
            logger.debug(msg)
            raise ValueError(msg)
        orig_df = data.copy()
        return orig_df, dict.fromkeys(orig_df.columns, len(orig_df))

    if isinstance(data, Mapping):
        if not data:
            msg = "The input mapping must contain at least one series."
            logger.debug(msg)
            raise ValueError(msg)

        series: dict[SeriesKey, Series] = {}
        lengths: dict[SeriesKey, int] = {}
        for name, values in data.items():
            if not _is_series_like(values):
                msg = (
                    f"The values for series {name!r} must be a sequence of "
                    "numbers, not a scalar."
                )
                logger.debug(msg)
                raise ValueError(msg)
            array = np.asarray(values)
            series[name] = pd.Series(array)
            lengths[name] = len(array)

        return DataFrame(series), lengths

    msg = (
        "The parameter 'data' must be a pandas.DataFrame (whose rows must not "
        "use a MultiIndex) or a mapping of series names to value sequences."
    )
    logger.debug(msg)
    raise ValueError(msg)


class Rhis:
    """
    Run the RHIS-TS representativeness analysis on a table of time series.

    RHIS-TS checks four assumptions - Randomness (R), Homogeneity (H),
    Independence (I) and Stationarity (S) - on every window that ends at the
    most recent observation. Starting from the complete series, the window is
    shortened one observation at a time; the first (and therefore longest)
    window that fails to reject all four hypotheses is the representative
    stretch.

    Building an instance only normalizes and stores the input; the analysis
    runs on demand through two public methods:

        build_rhis_evol_dict(ts_name=None)
            Return the p-value evolution for each hypothesis and each series.

        get_repr_idxs(ts_name=None, alpha=None)
            Return the (start, end) positional indexes of the representative
            stretch for each series.

    Parameters
    ----------
        data
            The time series to analyse, as a pandas.DataFrame or a mapping of
            series names to value sequences. See Rhis.__init__ for the
            accepted inputs and how each series is interpreted.

    Attributes
    ----------
        orig_df
            The input as a DataFrame; the mapping form is materialized and
            ragged series are NaN-padded.
        numeric_orig_df
            orig_df coerced to numeric, with unusable columns dropped.
        series_lengths
            Original length of every series.
        pvalues
            Per-series cache of the p-value evolution.
        repr_idxs
            Per-series representative-stretch indexes; None until
            get_repr_idxs() is called.
        alpha
            Significance level (0.05 by default).

    See Also
    --------
        Rhis.__init__ : accepted inputs and per-series interpretation.
        Rhis.build_rhis_evol_dict, Rhis.get_repr_idxs : the public analysis.
    """

    def __init__(self, data: DataFrame | Mapping[str, TimeSeriesFlex | Series]) -> None:
        """
        Prepare a table of time series for RHIS analysis.

        Parameters
        ----------
            data
                The time series to analyse, as either a pandas.DataFrame
                or a mapping of series names to value sequences.

                With a DataFrame, every column is a time series and every
                row is a time step, so the row index must not be a
                MultiIndex. The columns may use a MultiIndex; the series
                keys then are the column-label tuples.

                With a mapping, every key is a series name and every value
                is a sequence of numbers read in order (positional). The
                values are used positionally, so a pandas.Series index is
                ignored. Series of different lengths are padded with NaN
                up to the longest one for the analysis, but the reported
                indexes stay relative to each original series.

                Non-numeric entries are treated as missing and dropped
                window by window; columns with fewer than 8 numeric values
                are ignored entirely.
        """
        normalized_df, series_lengths = _normalize_input(data)
        self.orig_df: DataFrame = normalized_df
        self.series_lengths: dict[SeriesKey, int] = series_lengths
        self.numeric_orig_df: DataFrame = self.orig_df.copy().apply(pd.to_numeric, errors='coerce')
        # Columns with no usable numeric value at all (everything was non-numeric or missing);
        # partially numeric ones are filtered by length in the loop below.
        cols_to_drop = self.numeric_orig_df.columns[self.numeric_orig_df.isna().all()].to_list()

        for col in self.numeric_orig_df.columns:
            numeric_count = np.count_nonzero(np.isfinite(self.numeric_orig_df[col]))
            if numeric_count < MIN_SERIES_LENGTH and col not in cols_to_drop:
                cols_to_drop.append(col)

        if len(cols_to_drop) > 0:
            msg = f"Dropping columns with all NaN or fewer than {MIN_SERIES_LENGTH} numeric values: {cols_to_drop}"
            logger.debug(msg)
            self.numeric_orig_df = self.numeric_orig_df.drop(columns=cols_to_drop)

        self.alpha = DEFAULT_ALPHA
        # The package's final answer: for each series, the (start, end) positional indexes of
        # the RHIS-compliant representative stretch, meant to be sliced with .iloc[start:end].
        # None until get_repr_idxs() computes it (and cached there afterwards).
        self.repr_idxs: dict[SeriesKey, tuple[int, int]] | None = None

        # Per-series cache of the p-value evolution: one list per hypothesis ("R", "H", "I",
        # "S") plus their row-wise minimum ("RHIS-min"). Each list has one entry per starting
        # observation and is populated lazily by build_rhis_evol_dict(); empty lists mean "not
        # computed yet".
        self.pvalues: dict[SeriesKey, dict[str, list[float]]] = {}
        for col in self.numeric_orig_df.columns:
            self.pvalues[col] = {'R': [], 'H': [], 'I': [], 'S': [], 'RHIS-min': []}


    def build_rhis_evol_dict(
        self,
        ts_name: SeriesKey | None = None,
    ) -> dict[SeriesKey, dict[str, list[float]]]:
        """
        Compute the p-value evolution of every window ending at the latest observation.

        The evolution starts with the complete series (index 0) and then drops
        one observation from the beginning at a time, so the value at index i
        belongs to the slice that starts at position i and runs to the end of
        the series. Trailing entries are NaN because the remaining slices have
        fewer than MIN_SERIES_LENGTH observations to test; an earlier entry is
        NaN as well whenever the tests cannot be run on that slice, for instance
        when every value in it is equal.

        Results are computed once per series and cached in Rhis.pvalues;
        repeated calls return the very same lists.

        Parameters
        ----------
            ts_name
                The series to build. None (the default) builds every series.

        Returns
        -------
            A dict keyed by series name, holding one list of p-values per
            hypothesis ("R", "H", "I", "S") plus their row-wise minimum
            ("RHIS-min"). Every list has one entry per observation of the
            original series.

        Raises
        ------
            ValueError
                If 'ts_name' is not one of the known series names.
        """
        if ts_name is not None and ts_name not in self.pvalues:
            msg = f"The parameter 'ts_name' ({ts_name!r}) is not a known series name."
            logger.debug(msg)
            raise ValueError(msg)

        cols = list(self.numeric_orig_df.columns) if ts_name is None else [ts_name]

        for col in cols:
            if len(self.pvalues[col]['RHIS-min']) == 0:
                ts = self.numeric_orig_df[col].iloc[:self.series_lengths[col]]
                evol_dict = self._evol_dict_from_ts(ts)
                self.pvalues[col] = self._add_rhis_min_to_evol(evol_dict)

        return {col: self.pvalues[col] for col in cols}


    def get_repr_idxs(
        self,
        ts_name: SeriesKey | None = None,
        alpha: float | None = None,
    ) -> dict[SeriesKey, tuple[int, int]]:
        """
        Return the (start, end) indexes of the RHIS-compliant stretch of each series.

        The representative stretch is the longest slice that still ends at the
        most recent observation and fails to reject all four hypotheses, i.e.
        whose "RHIS-min" is at least 'alpha'. Because every candidate slice
        ends at the same observation, the first passing slice while walking
        from the longest (the complete series) to the shortest is by
        definition the longest passing one. Slices that could not be tested
        (shorter than MIN_SERIES_LENGTH, or with every value equal) appear
        as NaN and stop the search.

        The indexes are positional and meant to be sliced with .iloc[start:end].

        Parameters
        ----------
            ts_name
                The series to evaluate. None (the default) evaluates every
                series.
            alpha
                The significance level. None (the default) uses Rhis.alpha.

        Returns
        -------
            A dict keyed by series name, each value being a tuple
            (start, end) of positional indexes into the original series. The
            representative stretch is the empty slice (0, 0) when no testable
            slice passes.

        Raises
        ------
            ValueError
                If 'ts_name' is not one of the known series names.

        See Also
        --------
            Rhis.build_rhis_evol_dict : builds the p-value evolution used here.
            Rhis._find_rhis_compliant_idxs : the slice-selection logic.
        """
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
                'RHIS-min', computed over its R, H, I and S p-values). The
                value at index i is the p-value of the slice that starts
                at position i and runs to the very end of the series.
                Index 0 tests the complete series. The trailing entries
                are NaN because those slices are shorter than
                MIN_SERIES_LENGTH; an earlier entry may be NaN too, when
                the slice has enough observations but the tests cannot be
                computed on it, e.g. every value in it is equal.

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
        # the whole series passed every RHIS test, so the representative
        # slice is the entire series.
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
        # NaN entries mark slices that could not be tested: shorter than
        # MIN_SERIES_LENGTH, or long enough but degenerate (every value
        # equal, for instance). They act as a stopping boundary, just like
        # a passing p-value.
        #
        # For an evolution built by build_rhis_evol_dict(), pvalue_ts[0] is
        # a real number here: were it NaN, the complete series itself would
        # be untestable (too short, or all values equal), every slice would
        # be NaN too, and the check above would already have returned. Such
        # an evolution also always ends with MIN_SERIES_LENGTH - 1 NaN
        # entries, so a stopping point after index 0 is guaranteed.
        fails_to_reject = ~(pvalue_ts < alpha)  # p >= alpha, or NaN
        idx = int(np.argmax(fails_to_reject))

        return idx, pvalue_ts_last


    @staticmethod
    def _evol_dict_from_ts(ts: Series) -> dict[str, list[float]]:
        slices = slices_to_evol(ts.to_numpy(), MIN_SERIES_LENGTH)
        evol: dict[str, list[float]] = {'R': [], 'H': [], 'I': [], 'S': []}

        for sli in slices:
            rhis_dict = Rhis.calculate_rhis(sli)
            evol['R'].append(rhis_dict['R'])
            evol['H'].append(rhis_dict['H'])
            evol['I'].append(rhis_dict['I'])
            evol['S'].append(rhis_dict['S'])

        fill = np.full(MIN_SERIES_LENGTH - 1, np.nan)

        for hyp, ps in evol.items():
            evol[hyp] = list(np.append(ps, fill))

        return evol


    def _add_rhis_min_to_evol(self, evol_dict: dict[str, list[float]]) -> dict[str, list[float]]:
        rhis_values = list(evol_dict.values())
        evol_dict['RHIS-min'] = list(np.min(rhis_values, axis=0))

        return evol_dict


    @staticmethod
    def calculate_rhis(ts: TimeSeriesFlex) -> dict[str, float]:
        """
        Compute the four RHIS p-values of a single time series.

        Slices that cannot be tested get NaN instead of an error: a slice
        shorter than MIN_SERIES_LENGTH, or one whose values are so close
        together (up to and including a run of equal values) that the
        underlying statistic is degenerate. NaN is the package-wide signal
        for 'untestable'.

        Parameters
        ----------
            ts
                The time series to test. Non-numeric and non-finite values
                are ignored.

        Returns
        -------
            A dict with one p-value per hypothesis: 'R' (randomness),
            'H' (homogeneity), 'I' (independence) and 'S' (stationarity).
        """
        ts = clean_numeric_array(ts)
        if len(ts) < MIN_SERIES_LENGTH:
            return {'R': np.nan, 'H': np.nan, 'I': np.nan, 'S': np.nan}

        try:
            independence_p = wald_wolfowitz(ts, on_ranks=True).p_value
        except ValueError:
            # A slice with an extremely small spread (for example a long run
            # of equal values) has no meaningful serial-correlation variance,
            # so Wald-Wolfowitz refuses to test it. The slice is untestable,
            # and untestable slices are marked with NaN.
            independence_p = np.nan

        result = {
            'R': wallis_moore(ts).p_value,
            'H': mann_whitney(ts).p_value,
            'I': independence_p,
            'S': mann_kendall(ts).p_value,
        }

        return result


