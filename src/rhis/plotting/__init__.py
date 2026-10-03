"""Plotting helpers for the hypothesis tests."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from matplotlib import font_manager

from rhis.utils import to_ranks

if TYPE_CHECKING:
    from pandas import DataFrame

    from rhis.custom_types import TimeSeriesFlex

PLOTS_DIR = 'hypothesis_testing_plots'
RHIS_PLOTS_DIR = 'rhis_plots'
DEFAULT_ALPHA = 0.05

FONTS_DIR = Path(__file__).parent / 'fonts'
for _font_path in sorted(FONTS_DIR.glob('*.ttf')):
    font_manager.fontManager.addfont(str(_font_path))
plt.rcParams['font.family'] = ['Inter', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'semibold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.edgecolor'] = 'white'
plt.rcParams['legend.facecolor'] = 'white'

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

COLORS_DEFAULT = {'R': '#c44e52', 'H': '#55a868', 'I': '#8172b3', 'S': '#4c72b0'}
ALPHA_COLOR = '#e11d48'
SERIES_COLOR = '#555555'
REPR_COLOR = "#000000"


def _year_label(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return str(value.year)
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value)


def _period_label(first: object, last: object) -> str:
    return f'{_year_label(first)}-{_year_label(last)}'


def plot_test(  # noqa: PLR0913
    ts: TimeSeriesFlex,
    p_value: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    series_name: str = 'series',
    ranks_p_value: float | None = None,
    filename: str = 'hypothesis_test',
    title: str = 'Hypothesis Test Example',
) -> None:
    """
    Plot a single time series together with the p-value(s) from a hypothesis test.

    The series is shown on one axis while the p-value line(s) and the alpha line are
    shown on a secondary axis. When `ranks_p_value` is given, the ranks of the series
    and their p-value line are also plotted. The figure is saved as a PNG inside the
    `hypothesis_testing_plots` directory, which is created if it does not exist yet.

    Parameters
    ----------
        ts
            The time series to plot.
        p_value
            The p-value of the hypothesis test applied to the series.
        alpha
            The significance level used by the test.
        series_name
            Label used for the series in the legend and on the y-axis.
        ranks_p_value
            Optional p-value of the test applied to the ranks of the series.
            When provided, the ranks and their p-value line are also plotted.
        filename
            The figure will be saved as `<filename>.PNG` inside the plots directory.
        title
            Title of the figure.
    """
    fig, pvalue_ax = plt.subplots(figsize=(8, 6))
    series_ax = pvalue_ax.twinx()

    series_color = '#4c72b0'
    ranks_color = '#333333'
    alpha_color = ALPHA_COLOR

    for ax in (series_ax, pvalue_ax):
        ax.spines[['top', 'right']].set_visible(False)

    series_ax.scatter(
        range(len(ts)),
        ts,
        marker='o',
        facecolor=series_color,
        edgecolors='white',
        linewidths=0.6,
        s=40,
        label=series_name,
    )
    pvalue_ax.axhline(
        p_value,
        color=series_color,
        linestyle='--',
        label=f'p-value ({series_name}): {round(p_value, 7)}',
    )

    if ranks_p_value is not None:
        ts_ranks = to_ranks(ts)
        series_ax.scatter(
            range(len(ts_ranks)),
            ts_ranks,
            marker='o',
            facecolor=ranks_color,
            edgecolors='white',
            linewidths=0.6,
            s=40,
            label='ranks',
        )
        pvalue_ax.axhline(
            ranks_p_value,
            color=ranks_color,
            linestyle='--',
            label=f'p-value (ranks): {round(ranks_p_value, 7)}',
        )

    pvalue_ax.axhline(alpha, color=alpha_color, label=f'alpha: {alpha}')

    pvalue_ax.grid(axis='y', color='0.9', linewidth=0.8)

    pvalue_ax.set_xlabel('Time')
    pvalue_ax.set_ylabel('p-value')
    pvalue_ax.set_ylim(0, 1)
    series_ax.set_ylabel(series_name)

    pvalue_handles, pvalue_labels = pvalue_ax.get_legend_handles_labels()
    series_handles, series_labels = series_ax.get_legend_handles_labels()
    pvalue_ax.legend(
        pvalue_handles + series_handles,
        pvalue_labels + series_labels,
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        edgecolor='white',
    )

    fig.suptitle(title, fontsize=13, fontweight='semibold')
    fig.tight_layout()

    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{filename}.PNG', dpi=300, bbox_inches='tight')


def plot_rhis_evolution(  # noqa: PLR0913
    orig_df: DataFrame,
    rhis_df: DataFrame,
    alpha: float,
    *,
    show_repr: bool = True,
    repr_df: DataFrame | None = None,
    figtitle: str | None = None,
) -> None:
    """
    Save one figure per analyzed time series to the `rhis_plots` directory.

    Each figure has two stacked subplots sharing the x-axis (time): the top
    subplot shows the series values (and the RHIS-compliant representative
    series when `show_repr` is True), while the bottom subplot shows the
    evolution of the R, H, I and S p-values, the RHIS-min envelope and the
    alpha line.

    Parameters
    ----------
        orig_df
            The original dataframe.
        rhis_df
            The dataframe with the R, H, I and S p-value evolutions produced by
            `Rhis.build_rhis_evol_df()`.
        alpha
            The significance level used by the tests.
        show_repr
            Whether to plot the RHIS-compliant representative series taken
            from `repr_df`, when one is provided.
        repr_df
            The dataframe returned by `Rhis.build_rhis_compliant_df()`,
            holding the representative series, NaN-padded to the original
            index. Only used when `show_repr` is True.
        figtitle
            An optional figure title. When given, it is used as the title of
            every saved figure; otherwise the title defaults to
            `'RHIS <series>'`.
    """
    orig_cols = [col for col in orig_df.columns if not col.endswith('_repr')]
    alpha_label = f"alpha={alpha}"
    hypotheses = ['R', 'H', 'I', 'S']

    for series_name in orig_cols:
        fig, (series_ax, pvalue_ax) = plt.subplots(
            2, 1, figsize=(9, 8.5), sharex=True
        )
        for ax in (series_ax, pvalue_ax):
            ax.spines[['top', 'right']].set_visible(False)

        series_period = _period_label(orig_df.index[0], orig_df.index[-1])
        repr_period = series_period

        if show_repr and repr_df is not None and series_name in repr_df.columns:
            repr_values = repr_df[series_name]
            valid = repr_values.notna()
            if valid.any():
                start = valid.idxmax()
                repr_period = _period_label(start, orig_df.index[-1])

        series_ax.scatter(
            orig_df.index,
            orig_df[series_name],
            marker='o',
            facecolor=SERIES_COLOR,
            edgecolors='white',
            # linewidths=0.6,
            alpha=0.85,
            s=45,
            label=f'{series_name} ({series_period})',
        )

        if show_repr and repr_df is not None and series_name in repr_df.columns:
            series_ax.plot(
                repr_df.index,
                repr_df[series_name],
                marker='o',
                markersize=6,
                # linewidth=1.5,
                # linestyle='--',
                color=REPR_COLOR,
                # markeredgecolor='white',
                # markeredgewidth=0.6,
                label=f'{series_name}_repr ({repr_period})',
            )

        series_ax.set_ylabel(series_name)
        series_ax.legend(
            loc='upper left',
            frameon=True,
            framealpha=0.9,
            edgecolor='white',
        )

        if (series_name, 'RHIS-min') in rhis_df.columns:
            pvalue_ax.plot(
                rhis_df.index,
                rhis_df[(series_name, 'RHIS-min')],
                color='#222222',
                linewidth=5,
                alpha=0.15,
                zorder=1,
                label='RHIS-min',
            )
        for hyp in hypotheses:
            pvalue_ax.plot(
                rhis_df.index,
                rhis_df[(series_name, hyp)],
                color=COLORS_DEFAULT[hyp],
                alpha=0.85,
                linewidth=1.5,
                zorder=3,
                label=hyp,
            )

        pvalue_ax.axhspan(0, alpha, color=ALPHA_COLOR, alpha=0.04, zorder=0)
        pvalue_ax.axhline(alpha, color=ALPHA_COLOR, linestyle='--', linewidth=1.2, zorder=2, label=alpha_label)

        pvalue_ax.grid(axis='y', color='0.9', linewidth=0.8)

        pvalue_ax.set_xlabel('Time')
        pvalue_ax.set_ylabel('p-value')
        pvalue_ax.set_ylim(0, 1)

        pvalue_ax.legend(
            loc='upper left',
            ncols=4,
            frameon=True,
            framealpha=0.9,
            edgecolor='white',
            columnspacing=0.8,
            handlelength=1.5,
        )

        if figtitle is not None:
            fig.suptitle(figtitle, fontsize=14, fontweight='semibold')
        else:
            fig.suptitle(f'RHIS {series_name}', fontsize=14, fontweight='semibold')
        fig.tight_layout()

        plots_dir = Path(RHIS_PLOTS_DIR)
        plots_dir.mkdir(exist_ok=True)
        safe_series_name = series_name.replace("/", "_")
        save_path = plots_dir / f"RHIS {safe_series_name}.PNG"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"A plot was saved to {save_path}")
        plt.close(fig)
