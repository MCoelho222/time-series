from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis.core import Rhis

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
    # 3. The step-by-step p-values, as a dict
    # ------------------------------------------------------------------
    col = df.columns[0]
    evol = rhis.build_rhis_evol_dict(col)[col]
    evol_table = pd.DataFrame(evol, index=df.index)
    idxs = rhis.get_repr_idxs(col)
    start, end = idxs[col]
    break_year = df.index[start]

    _section_header('3. THE STEP-BY-STEP P-VALUES   (evolution dict)')
    _say('The package hands the whole evolution back as a plain dict: one list of')
    _say('p-values per hypothesis. Starting at 1871, then 1872, ... up to 1970,')
    _say('every stretch that ends at 1970 is tested. Each number is a p-value:')
    print()
    print('    near 0  -> that check FAILS for this stretch (below the threshold)')
    print('    near 1  -> the check PASSES')
    print('    NaN     -> a stretch too short to test')
    print()
    _say(f'The dict keys are {list(evol)}; the threshold is alpha = {alpha}. Below it,')
    _say('a check fails. "RHIS-min" is the row-wise minimum of the four - the single')
    _say('number that decides whether a stretch is RHIS-compliant.')
    print()
    print('  Every row is the stretch that starts in that year and ends in 1970:')
    print()
    print(evol_table.round(4).head(8).to_string())
    print()
    print('  ...')
    print()
    print(evol_table.round(4).tail(8).to_string())
    print()
    _say('In 1871 almost every check fails (H, I and S are essentially zero): the')
    _say('pre/post-changepoint mixture is not one world. The failures calm down')
    _say(f'as we move the starting year forward - until {break_year}, when all four checks')
    _say('start passing. That break is exactly what RHIS is looking for.')

    # ------------------------------------------------------------------
    # 4. The representative series, as indexes
    # ------------------------------------------------------------------
    _section_header('4. THE REPRESENTATIVE SERIES   (repr indexes dict)')
    _say('get_repr_idxs() returns the answer as a dict of (start, end) indexes -')
    _say('positional, and ready to feed straight into .iloc:')
    print()
    print(f'    {idxs}')
    print()
    kept = int(end - start)
    discarded = int(len(df) - kept)
    discarded_pct = round(100 * discarded / len(df), 1)
    repr_df = df.iloc[start:end]
    _say(f'For "flow" that is indexes {start}..{end}: the {repr_df.index[0]}-{repr_df.index[-1]}')
    _say(f'stretch ({kept} values), setting the {discarded} earlier years aside:')
    print()
    print('  Kept slice:')
    print(repr_df.head(5).to_string())
    print('  ...')
    print(repr_df.tail(5).to_string())
    print()
    _say(f'So we started with {len(df)} years and kept {kept}; the {discarded} discarded')
    _say(f'years ({discarded_pct}% of the record) would have broken the checks.')

    # ------------------------------------------------------------------
    # 5. The verdict
    # ------------------------------------------------------------------
    _section_header('5. THE VERDICT')
    repr_ts = rhis.numeric_orig_df[col].iloc[start:end].to_numpy()
    repr_rhis = Rhis.calculate_rhis(repr_ts)

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
    _say(f'All four checks pass at alpha = {alpha}, so the {repr_df.index[0]}-{repr_df.index[-1]} slice')
    _say('is a trustworthy, RHIS-compliant representative series.')
    print()
    _ruler()
    _say(f'And there it is: a hundred years of complicated history, and {kept} clean')
    _say('years emerge - exactly the period you can safely describe, model or')
    _say('forecast. The post-1890s "new Nile" that hydrologists talk about is the')
    _say('one RHIS found by itself!')
    _ruler()


if __name__ == "__main__":
    main()
