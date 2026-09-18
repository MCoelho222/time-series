from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame, Index

from rhis.exceptions import raise_if_no_rhis_run
from rhis.hypothesis.homogeneity import mann_whitney
from rhis.hypothesis.independence import wald_wolfowitz
from rhis.hypothesis.randomness import wallismoore
from rhis.hypothesis.stationarity import mann_kendall
from rhis.plotting import plot_rhis_evolution
from rhis.utils import clean_numeric_array, nans_nums_from_array, slice_init, slices_to_evol

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


    def _retrieve_rhis_ts_idxs(self, pvalue_ts: NDArray[np.float64], alpha: float, length_init_ts: int) -> tuple[int, int]:
        pvalue_ts_nums = nans_nums_from_array(pvalue_ts)

        data = pvalue_ts[:]
        alpha_arr = np.full(len(data), alpha)
        idx = 0
        is_rejection = data <= alpha_arr

        while is_rejection[idx]:
            if idx == len(data) - 1:
                break
            idx += 1

        pvalue_ts_last = len(pvalue_ts_nums) + length_init_ts - 1
        if pvalue_ts[0] >= alpha:
            return (0, pvalue_ts_last)

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

        for sli in slices:
            rhis_dict = Rhis.calculate_rhis(sli, alpha)
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
            cut_idxs = self._retrieve_rhis_ts_idxs(rhis_series, self.alpha, self.length_init_ts)
            self._include_rhis_compliant_ts_in_df(self.orig_df, cut_idxs, col)

        logger.info("RHIS compliant data successfully included in the dataframe.")
        return self.orig_df


    def plot(self, *, show_repr: bool = True) -> None:
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

        Raises
        ------
            RhisEvolNotCalledError
                If `evol()` has not been run yet.
        """
        raise_if_no_rhis_run(is_rhis_complete=self.is_rhis_complete)

        if self.rhis_df is None:
            msg = 'RHIS dataframe has not been initialized.'
            raise RuntimeError(msg)

        plot_rhis_evolution(self.orig_df, self.rhis_df, self.alpha, show_repr=show_repr)


    @staticmethod
    def calculate_rhis(ts: TimeSeriesFlex, alpha: float = DEFAULT_ALPHA) -> dict[str, float]:
        ts = clean_numeric_array(ts)

        return  {
            'R': wallismoore(ts, alpha=alpha).p_value,
            'H': mann_whitney(ts, alpha=alpha).p_value,
            'I': wald_wolfowitz(ts, alpha=alpha, on_ranks=True).p_value,
            'S': mann_kendall(ts, alpha=alpha).p_value,
        }


