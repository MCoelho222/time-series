from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis.core import Rhis


def load_example_data() -> DataFrame:
    """
    Load the example environmental dataset: the annual flow of the
    River Nile at Aswan, 1871-1970.

    The measurements, in units of 10^8 m^3, are the classic time series
    published in Cobb (1978), Table 1, page 249, and distributed as the
    R 'datasets::Nile' object (also available in statsmodels). The
    series has an apparent changepoint near 1898, widely used in
    changepoint-detection literature, and is the reason the RHIS
    example recovers the post-1898 period as the representative slice.

    The record intentionally ends in 1970: it corresponds to Cobb's
    published table. The Aswan gauge itself has been read since 1869,
    and the Global Runoff Data Centre holds annual values up to about
    1984, but those are not freely downloadable as a simple CSV and
    would need GRDC registration. Going past the mid-1970s would also
    change the nature of the series, since after the Aswan High Dam
    closed, the flow at Aswan is reservoir-regulated rather than natural
    river flow.

    The values are bundled locally (as a CSV in src/rhis/data) so the
    example is self-contained and reproducible without a network
    connection.

    Returns
    -------
        A DataFrame with a 'year' index (1871-1970) and a 'flow' column
        holding the annual discharge in 10^8 m^3.
    """
    data_path = Path(__file__).parent / 'data' / 'nile_river_annual_flow.csv'
    df = pd.read_csv(data_path, index_col='year')
    return df


def main() -> None:
    df = load_example_data()
    rhis = Rhis(df)
    rhis.evol()
    rhis.add_rhis_compliant_to_df()
    rhis.plot(figtitle='Representative Series From Nile River Annual Flow (1871-1970)')

    repr_rhis = rhis.calculate_repr_rhis_pvalues()
    logger.info("RHIS p-values for each representative series:")
    for repr_name, pvalues in repr_rhis.items():
        logger.info(
            f"{repr_name}: R={pvalues['R']:.4f} | "
            f"H={pvalues['H']:.4f} | I={pvalues['I']:.4f} | S={pvalues['S']:.4f}"
        )

    if rhis.is_all_rhis_compliant():
        logger.info("The representative series from Nile River annual flows is RHIS-compliant at alpha = {}.", rhis.alpha)
    else:
        logger.warning("At least one representative series is not fully RHIS-compliant.")


if __name__ == "__main__":
    main()
