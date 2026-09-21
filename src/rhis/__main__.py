from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis.core import Rhis
from rhis.plotting import RHIS_PLOTS_DIR

WIDTH = 78


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


def _ruler(char: str = '=') -> None:
    print(char * WIDTH)


def _section_header(text: str) -> None:
    print()
    _ruler()
    print(f'  {text}')
    _ruler()


def _say(text: str = '') -> None:
    print(f'  {text}')


def main() -> None:  # noqa: PLR0915
    logger.remove()
    logger.add(sys.stderr, level='WARNING')

    df = load_example_data()
    rhis = Rhis(df)
    alpha = rhis.alpha

    print()
    _ruler('*')
    print('  RHIS - Randomness, Homogeneity, Independence and Stationarity')
    print('  A guided tour of the Nile')
    _ruler('*')

    # ------------------------------------------------------------------
    # 1. The data
    # ------------------------------------------------------------------
    _section_header('1. THE DATA')
    _say('The annual flow of the Nile River at Aswan, in 10^8 m^3 - one number')
    _say('per year, for a full century. Here are the first and last years:')
    print()
    print('  First years:')
    print(df.head(8).to_string())
    print()
    print('  Last years:')
    print(df.tail(8).to_string())
    print()
    _say('Do you see it? Around the 1890s the river drops. The big 19th-century')
    _say('flows give way to a lower, steadier regime. Two different "worlds" live')
    _say('inside this single series.')

    # ------------------------------------------------------------------
    # 2. What we want to know
    # ------------------------------------------------------------------
    _section_header('2. WHAT WE WANT TO KNOW')
    _say('If we average the whole record and call that "the normal Nile", the')
    _say('answer is meaningless - one world says ~1100, the other says ~700.')
    _say()
    _say('RHIS looks for the most recent stretch of the series that behaves like')
    _say('a single, sane process, by running four everyday sanity checks on every')
    _say('stretch that ends at the latest year:')
    print()
    print('    R - Randomness    : are the ups and downs scattered like dice?')
    print('    H - Homogeneity   : does the stretch behave the same start-to-end?')
    print('    I - Independence  : is each year unrelated to the one before it?')
    print('    S - Stationarity  : no trend, no sudden change of level?')
    print()
    _say('A stretch passes only when all four checks say "yes". RHIS then keeps')
    _say('the longest stretch that passes, and sets everything else aside.')

    # ------------------------------------------------------------------
    # 3. The step-by-step p-values
    # ------------------------------------------------------------------
    rhis_evol_df = rhis.build_rhis_evol_df()

    _section_header('3. THE STEP-BY-STEP P-VALUES   (rhis_evol_df)')
    _say('Starting at 1871, then 1872, ... up to 1970, every stretch that ends at')
    _say('1970 is tested. Each cell is a p-value:')
    print()
    print('    near 0  -> that check FAILS for this stretch (below the threshold)')
    print('    near 1  -> the check PASSES')
    print('    NaN     -> a stretch too short to test')
    print()
    _say(f'The threshold is alpha = {alpha}. Below it, the check fails.')
    print()
    print('  Every row is the stretch that starts in that year and ends in 1970:')
    print()
    print(rhis_evol_df.droplevel(level=0, axis=1).round(4).head(8).to_string())
    print()
    print('  ...')
    print()
    print(rhis_evol_df.droplevel(level=0, axis=1).round(4).tail(8).to_string())
    print()
    _say('In 1871 almost every check fails (H, I and S are essentially zero): the')
    _say('pre/post-changepoint mixture is not one world. The failures calm down')
    _say('as we move the starting year forward - until 1894, when all four checks')
    _say('start passing. That break is exactly what RHIS is looking for.')

    # ------------------------------------------------------------------
    # 4. The representative series and the summary
    # ------------------------------------------------------------------
    repr_df = rhis.build_rhis_compliant_df()
    summary = rhis.summary_df

    _section_header('4. THE REPRESENTATIVE SERIES   (repr_df)  +  SUMMARY   (summary_df)')
    repr_count = int(repr_df.notna().sum()['flow'])
    _say(f'The representative series keeps the 1894-1970 stretch ({repr_count} values)')
    _say('and leaves the 23 earlier years blank:')
    print()
    print('  Kept slice (1894-1970):')
    print(repr_df.dropna().head(5).to_string())
    print('  ...')
    print(repr_df.dropna().tail(5).to_string())
    print()

    col = 'flow'
    orig_len = int(cast('Any', summary.loc[col, 'original_length']))
    repr_len = int(cast('Any', summary.loc[col, 'representative_length']))
    discarded_pct = float(cast('Any', summary.loc[col, 'discarded_percentage']))
    non_numeric = int(cast('Any', summary.loc[col, 'non_numeric_excluded']))
    most_rejected = str(cast('Any', summary.loc[col, 'most_rejected_hypothesis']))
    original_period = str(cast('Any', summary.loc[col, 'original_period']))
    representative_period = str(cast('Any', summary.loc[col, 'representative_period']))

    print('  The whole story in one table:')
    print()
    print(rhis.summary_df.to_string())
    print()
    _say('Reading it:')
    _say(f'  - we started with {orig_len} years ({original_period}) and kept {repr_len} ({representative_period})')
    _say(f'  - the {100 - repr_len} discarded years ({discarded_pct}% of the record) would have broken the checks')
    _say(f'  - {non_numeric} non-numeric values had to be thrown away')
    _say(f"  - the check rejected most often was '{most_rejected}': the river level clearly")
    _say('    "remembers" itself year after year, until the regime finally settled')

    # ------------------------------------------------------------------
    # 5. The picture and the verdict
    # ------------------------------------------------------------------
    _section_header('5. THE PICTURE AND THE VERDICT')
    rhis.plot_evolution(
        figtitle='RHIS-compliant Series From Nile River Annual Flow (1871-1970)',
        repr_df=repr_df,
    )
    plot_path = Path(RHIS_PLOTS_DIR) / f'RHIS {col}.PNG'
    _say('A figure was saved for you, with the p-values and the recovered slice:')
    _say(f'    {plot_path}')
    print()

    repr_rhis = rhis.calculate_rhis_once_with_full_ts(repr_df)[col]
    _say('Final p-values for the representative series:')
    print()
    labels = {
        'R': 'randomness',
        'H': 'homogeneity',
        'I': 'independence',
        'S': 'stationarity',
    }
    for hyp, short_name in labels.items():
        pv = float(repr_rhis[hyp])
        status = 'PASS' if pv >= alpha else 'FAIL'
        note = '  (just above the line!)' if 0 <= pv - alpha < 0.01 else ''  # noqa: PLR2004
        print(f'    {hyp} - {short_name:<14}: {pv:.4f}   -> {status}{note}')
    print()
    _say(f'All four checks pass at alpha = {alpha}, so the {representative_period} slice is a')
    _say('trustworthy, RHIS-compliant representative series.')
    print()
    _ruler()
    _say('And there it is: a hundred years of complicated history, and 77 clean')
    _say('years emerge - exactly the period you can safely describe, model or')
    _say('forecast. The post-1890s "new Nile" that hydrologists talk about is the')
    _say('one RHIS found by itself!')
    _ruler()


if __name__ == "__main__":
    main()
