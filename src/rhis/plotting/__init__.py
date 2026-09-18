"""Plotting helpers for the hypothesis tests."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from rhis.utils import to_ranks

if TYPE_CHECKING:
    from rhis.custom_types import TimeSeriesFlex

PLOTS_DIR = 'hypothesis_testing_plots'
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
