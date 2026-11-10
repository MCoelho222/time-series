import numpy as np
import pandas as pd
from rhis_ts import Rhis


def _slice_and_pad(orig_ts, idx: tuple[int, int]):
    nums_ts = orig_ts[idx[0]:idx[1]]
    nan_init = np.full(idx[0], np.nan)
    nan_fin = np.full(len(orig_ts) - idx[1], np.nan)
    full_ts = np.append(nan_init, nums_ts)
    full_ts = np.append(full_ts, nan_fin)

    return full_ts


def build_rhis_evol_df(rhis: Rhis):
    frames = {
        col: pd.DataFrame(evol, index=rhis.orig_df.index)
        for col, evol in rhis.pvalues.items()
    }

    return pd.concat(frames, axis=1, names=['series', 'hypothesis'])


def build_repr_df(rhis):
    series = {}
    for col in rhis.numeric_orig_df.columns:
        orig_ts = rhis.numeric_orig_df[col]
        idx = rhis.repr_idxs[col]
        series[col] = _slice_and_pad(orig_ts, idx)

    return pd.DataFrame(series, index=rhis.numeric_orig_df.index)

# ---------------------------------------------------------------------------
# RHIS-TS summaries and plots used by the example notebooks
# ---------------------------------------------------------------------------
HYPOTHESES = ['R', 'H', 'I', 'S']


def fmt_p(p):
    """Readable p-value: very small values are shown as '<0.001'."""
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def run_rhis(series, name, label, results=None, alpha=0.05, date_fmt='%Y-%m-%d'):
    """
    Run RHIS-TS on one series and print a plain-language summary.

    Missing values (NaN) are allowed: RHIS-TS skips them, so the values on
    each side of a gap are read as consecutive. Counts below refer to the
    values that are present. When `results` is a dict, a summary row is
    stored in it under `label`.
    """
    rhis = Rhis(series.to_frame(name))
    start, end = rhis.get_repr_idxs(alpha=alpha)[name]
    rhis.build_rhis_evol_dict()
    evol_df = build_rhis_evol_df(rhis)
    repr_df = build_repr_df(rhis)

    values = rhis.numeric_orig_df[name]
    index = values.index
    present = values.notna()
    n = int(present.sum())
    full = evol_df[name].dropna().iloc[0]  # p-values of the four tests on the full record

    print(f'{label}: {n:,} values, {index[present][0]:{date_fmt}} to {index[present][-1]:{date_fmt}}')
    print('p-values on the full record: ' + ', '.join(f'{h} = {fmt_p(full[h])}' for h in HYPOTHESES))
    if end > start:
        kept = int(present.iloc[start:end].sum())
        stretch = f'{index[start]:{date_fmt}} to {index[end - 1]:{date_fmt}}'
        print(f'RHIS-compliant stretch: {stretch} ({kept:,} values, {kept / n:.0%} of the record)')
    else:
        kept, stretch = 0, 'none'
        print('RHIS-compliant stretch: none (even the last 8 values fail at least one test)')

    if results is not None:
        results[label] = {
            'values': n,
            **{f'{h} (full record)': fmt_p(full[h]) for h in HYPOTHESES},
            'compliant stretch': stretch,
            'compliant values': kept,
            'share of record': f'{kept / n:.0%}',
        }
    return rhis, evol_df, repr_df


def plot_evolution(rhis, evol_df, repr_df, title, alpha=0.05, last=None):
    """Plot a series and its p-value evolution. `last` limits the plot to the last N values."""
    from plotting import plot_rhis_evolution  # local import: plotting registers fonts on import

    window = slice(-last, None) if last else slice(None)
    plot_rhis_evolution(
        rhis.numeric_orig_df.iloc[window],
        evol_df.iloc[window], alpha,
        show_repr=True,
        save=False,
        repr_df=repr_df.iloc[window],
        figtitle=title,
        figsize=(12, 6),
    )


def lp3_quantile(values, prob):
    """Log-Pearson type III quantile, fitted by the method of moments on log10 values."""
    from scipy import stats

    logs = np.log10(np.asarray(values, dtype=float))
    return 10 ** stats.pearson3.ppf(prob, stats.skew(logs, bias=False), loc=logs.mean(), scale=logs.std(ddof=1))


def lp3_estimate(values, prob, n_boot=2000, seed=42):
    """Log-Pearson type III quantile plus a 90% bootstrap range."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boot = [lp3_quantile(rng.choice(values, len(values)), prob) for _ in range(n_boot)]
    low, high = np.percentile(boot, [5, 95])
    return lp3_quantile(values, prob), low, high


# ---------------------------------------------------------------------------
# ANA (Brazil) HidroWeb: daily discharge download
# ---------------------------------------------------------------------------
ANA_SERVICE = 'http://telemetriaws1.ana.gov.br/ServiceANA.asmx/'


def download_ana_daily_flows(station, tries=15, timeout=300, wait=10):
    """
    Download the full daily discharge record of an ANA fluviometric station.

    Uses the public HidroWeb web service (`HidroSerieHistorica`, data type 3 =
    discharge). The service often answers with a gateway time-out, so the
    request is retried. ANA returns one row per month and consistency level
    (1 = raw, 2 = consistency-checked); when a day has both, the checked value
    is kept.

    Returns a DataFrame indexed by date with columns `flow` (m³/s) and
    `consistency_level`. Days without a value are not included.
    """
    import time
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    params = {'codEstacao': station, 'dataInicio': '', 'dataFim': '', 'tipoDados': '3', 'nivelConsistencia': ''}
    url = ANA_SERVICE + 'HidroSerieHistorica?' + urllib.parse.urlencode(params)
    body = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read()
            if body.lstrip().startswith(b'<?xml'):
                break
            print(f'Attempt {attempt}: unexpected answer from ANA, retrying...')
        except Exception as error:  # noqa: BLE001 - the service fails in many ways (time-outs, 504s)
            print(f'Attempt {attempt}: {error}, retrying...')
        body = None
        time.sleep(wait)
    if body is None:
        msg = f'Could not download station {station} from ANA after {tries} attempts.'
        raise RuntimeError(msg)

    rows = []
    for table in ET.fromstring(body).iter('SerieHistorica'):
        record = {child.tag: (child.text or '').strip() for child in table}
        month = pd.Timestamp(record['DataHora'][:10])
        level = int(record['NivelConsistencia'])
        for day in range(1, month.days_in_month + 1):
            value = record.get(f'Vazao{day:02d}', '')
            if value:
                rows.append((month + pd.Timedelta(days=day - 1), float(value), level))

    data = pd.DataFrame(rows, columns=['date', 'flow', 'consistency_level'])
    data = data.sort_values(['date', 'consistency_level']).groupby('date').tail(1)  # keep the checked value
    return data.set_index('date').sort_index()
