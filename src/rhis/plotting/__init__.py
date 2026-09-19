"""Plotting helpers for the hypothesis tests."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from rhis.utils import to_ranks

if TYPE_CHECKING:
    from pandas import DataFrame

    from rhis.custom_types import TimeSeriesFlex

PLOTS_DIR = 'hypothesis_testing_plots'
RHIS_PLOTS_DIR = 'rhis_plots'
DEFAULT_ALPHA = 0.05


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

    series_color = 'blue'
    ranks_color = 'k'
    alpha_color = 'red'

    series_ax.scatter(range(len(ts)), ts, color=series_color, label=series_name)
    pvalue_ax.axhline(
        p_value,
        color=series_color,
        linestyle='--',
        label=f'p-value ({series_name}): {round(p_value, 7)}',
    )

    if ranks_p_value is not None:
        ts_ranks = to_ranks(ts)
        series_ax.scatter(range(len(ts_ranks)), ts_ranks, color=ranks_color, label='ranks')
        pvalue_ax.axhline(
            ranks_p_value,
            color=ranks_color,
            linestyle='--',
            label=f'p-value (ranks): {round(ranks_p_value, 7)}',
        )

    pvalue_ax.axhline(alpha, color=alpha_color, label=f'alpha: {alpha}')

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
    )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{filename}.PNG', bbox_inches='tight')


def plot_rhis_evolution(
    orig_df: DataFrame,
    rhis_df: DataFrame,
    alpha: float,
    *,
    show_repr: bool = True,
    figtitle: str | None = None,
) -> None:
    """
    Save one figure per analyzed time series to the `rhis_plots` directory.

    Each figure shows the series values together with the evolution of the
    R, H, I and S p-values, the RHIS-min envelope and the alpha line.

    Parameters
    ----------
        orig_df
            The original dataframe. Optional `*_repr` columns are plotted when
            `show_repr` is True.
        rhis_df
            The dataframe with the R, H, I and S p-value evolutions produced by
            `Rhis.evol()`.
        alpha
            The significance level used by the tests.
        show_repr
            Whether to plot the RHIS-compliant representative series, added by
            `Rhis.add_rhis_compliant_to_df()`, when present.
        figtitle
            An optional figure title. When given, it is used as the title of
            every saved figure; otherwise the title defaults to
            `'RHIS <series>'`.
    """
    orig_cols = [col for col in orig_df.columns if not col.endswith('_repr')]
    alpha_label = f"alpha={alpha}"
    hypotheses = ['R', 'H', 'I', 'S']
    colors_default = {'R': 'black', 'H': 'cyan', 'I': 'green', 'S': 'blue'}

    for series_name in orig_cols:
        fig, pvalue_ax = plt.subplots(figsize=(8, 6))

        series_ax = pvalue_ax.twinx()

        series_ax.scatter(
            orig_df.index,
            orig_df[series_name],
            color='black',
            edgecolors='none',
            alpha=0.2,
            label=series_name,
        )

        if show_repr:
            repr_name = series_name + '_repr'
            if repr_name in orig_df.columns:
                series_ax.scatter(
                    orig_df.index,
                    orig_df[repr_name],
                    color='black',
                    edgecolors='none',
                    label=repr_name,
                )

        if (series_name, 'min') in rhis_df.columns:
            pvalue_ax.plot(
                rhis_df[(series_name, 'min')],
                color='black',
                linewidth=6,
                alpha=0.2,
                label='RHIS-min',
            )
        for hyp in hypotheses:
            pvalue_ax.plot(rhis_df[(series_name, hyp)], color=colors_default[hyp], alpha=0.5, label=hyp)

        pvalue_ax.axhline(alpha, color='red', linestyle='--', linewidth=1, label=alpha_label)

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
        )

        if figtitle is not None:
            fig.suptitle(figtitle, fontsize=14)
        else:
            fig.suptitle(f'RHIS {series_name}', fontsize=14)
        fig.tight_layout()

        plots_dir = Path(RHIS_PLOTS_DIR)
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / f"RHIS {series_name}.PNG", bbox_inches='tight')
        plt.close(fig)
