from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rhis.core import Rhis

if TYPE_CHECKING:
    from pandas import DataFrame, Index


def generate_example_data() -> DataFrame:
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "series_A": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_B": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_C": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_D": np.clip(rng.normal(50, 15, 100), 0, 100),
    })

    # Forcing a trend on series_A and series_B
    df["series_A"] = np.sort(df["series_A"].to_numpy())
    df["series_B"] = np.sort(df["series_B"].to_numpy())

    return df


def plot_rhis(orig_df: DataFrame, rhis_df: DataFrame, orig_cols: Index[str],  alpha: float,*, show_repr: bool = True) -> None:
    alpha_label = f"alpha={alpha}"
    hypotheses = ['R', 'H', 'I', 'S']
    colors_default = {'R': 'black', 'H': 'cyan', 'I': 'green', 'S': 'blue'}

    for series_name in orig_cols:
        fig, pvalue_ax = plt.subplots(figsize=(8, 6))

        series_ax = pvalue_ax.twinx()

        series_ax.scatter(orig_df.index, orig_df[series_name], color='black', edgecolors='none', alpha=0.2, label=series_name)

        if show_repr:
            repr_name = series_name + "_repr"
            series_ax.scatter(orig_df.index, orig_df[repr_name], color='black', edgecolors='none', label=repr_name)

        pvalue_ax.plot(rhis_df[(series_name, 'min')], color='black', linewidth=6, alpha=0.2, label='RHIS-min')
        for hyp in hypotheses:
            pvalue_ax.plot(rhis_df[(series_name, hyp)], color=colors_default[hyp], alpha=0.5, label=hyp)

        pvalue_ax.axhline(alpha, color='red', linestyle='--', linewidth=1, label=alpha_label)

        pvalue_ax.set_xlabel('Time')
        pvalue_ax.set_ylabel('p-value')
        pvalue_ax.set_ylim(0, 1)
        series_ax.set_ylabel(series_name)
        series_ax.set_ylim(0, 100)
        series_ax.set_xlim(0, 100)

        pvalue_handles, pvalue_labels = pvalue_ax.get_legend_handles_labels()
        series_handles, series_labels = series_ax.get_legend_handles_labels()

        pvalue_ax.legend(
            pvalue_handles + series_handles,
            pvalue_labels + series_labels,
            loc='upper left',
        )

        fig.suptitle(f'RHIS {series_name}', fontsize=14)
        fig.tight_layout()

        filename = f"RHIS {series_name}.PNG"
        plots_dir = Path('RHIS_example_plots')
        plots_dir.mkdir(exist_ok=True)

        plt.savefig(plots_dir / filename, bbox_inches="tight")


def main() -> None:
    df = generate_example_data()
    orig_cols = df.columns
    rhis = Rhis(df)
    rhis.evol()
    rhis.add_rhis_compliant_to_df()

    orig_df = rhis.orig_df
    rhis_df = rhis.rhis_df
    alpha = rhis.alpha

    if rhis_df is not None:
        plot_rhis(orig_df, rhis_df, orig_cols, alpha)

if __name__ == "__main__":
    main()
