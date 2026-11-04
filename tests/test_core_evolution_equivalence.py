"""Equivalence of the incremental evolution against the per-slice reference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhis_ts.core import DEFAULT_ALPHA, MIN_SERIES_LENGTH, Rhis
from rhis_ts.utils import slices_to_evol

HYPOTHESES = ('R', 'H', 'I', 'S')
SHORT_SERIES_LENGTH = 80


def _brute_evolution(values: np.ndarray) -> dict[str, np.ndarray]:
    """Reproduce the old per-slice evolution via Rhis.calculate_rhis."""
    size = len(values)
    evolution = {key: np.full(size, np.nan) for key in HYPOTHESES}
    for i, sli in enumerate(slices_to_evol(values, MIN_SERIES_LENGTH)):
        result = Rhis.calculate_rhis(sli)
        for key in HYPOTHESES:
            evolution[key][i] = result[key]
    return evolution


def _brute_repr(rhis: Rhis, col, alpha: float) -> tuple[int, int]:
    """Reproduce the old representative-index scan, padded series included."""
    ts = rhis.numeric_orig_df[col].to_numpy()
    i = 0
    while i <= (len(ts) - MIN_SERIES_LENGTH):
        rhis_min = np.min(list(Rhis.calculate_rhis(ts[i:]).values()))
        if not np.isnan(rhis_min) and rhis_min >= alpha:
            return (i, rhis.series_lengths[col])
        i += 1
    return (0, 0)


def _make_case(kind: str, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == 'normal':
        return rng.normal(size=size)
    if kind == 'ties':
        return rng.integers(0, 4, size=size).astype(float)
    if kind == 'rounded':
        return np.round(rng.normal(size=size), 1)
    if kind == 'constant_tail':
        tail = int(rng.integers(1, size - MIN_SERIES_LENGTH + 1))
        values = rng.normal(size=size)
        values[size - tail:] = 7.0
        return values
    if kind == 'nan_holes':
        values = rng.normal(size=size)
        holes = rng.choice(size, size=int(rng.integers(1, 4)), replace=False)
        values[holes] = np.nan
        return values
    values = rng.normal(size=size)
    values[int(rng.integers(0, size))] = np.inf
    return values


@pytest.mark.parametrize('kind', ['normal', 'ties', 'rounded', 'constant_tail', 'nan_holes', 'inf'])
@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_build_evolution_matches_per_slice_reference(kind: str, seed: int) -> None:
    values = _make_case(kind, size=40, seed=seed)
    if np.count_nonzero(np.isfinite(values)) < MIN_SERIES_LENGTH:
        pytest.skip('not enough finite observations')

    rhis = Rhis(pd.DataFrame({'x': values}))
    evolution = rhis.build_rhis_evol_dict('x')['x']
    expected = _brute_evolution(values)

    for hyp in HYPOTHESES:
        np.testing.assert_allclose(
            np.asarray(evolution[hyp], dtype=float), expected[hyp], equal_nan=True,
        )

    expected_min = np.min([expected[key] for key in HYPOTHESES], axis=0)
    np.testing.assert_allclose(
        np.asarray(evolution['RHIS-min'], dtype=float), expected_min, equal_nan=True,
    )


def test_build_evolution_matches_reference_for_ragged_mapping() -> None:
    rng = np.random.default_rng(11)
    data = {
        'long': list(rng.normal(10.0, 1.0, 30)),
        'short': list(rng.normal(20.0, 2.0, 20)),
    }
    rhis = Rhis(data)

    evolution = rhis.build_rhis_evol_dict()

    for col, series in data.items():
        expected = _brute_evolution(np.asarray(series, dtype=float))
        for hyp in HYPOTHESES:
            np.testing.assert_allclose(
                np.asarray(evolution[col][hyp], dtype=float), expected[hyp],
                equal_nan=True,
            )


def test_get_repr_idxs_matches_reference_without_cache() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=60)
    values[rng.choice(60, size=2, replace=False)] = np.nan
    rhis = Rhis(pd.DataFrame({'x': values}))

    obtained = rhis.get_repr_idxs()['x']

    assert obtained == _brute_repr(rhis, 'x', DEFAULT_ALPHA)


def test_get_repr_idxs_matches_reference_with_cache() -> None:
    rng = np.random.default_rng(6)
    values = rng.normal(size=60)
    rhis = Rhis(pd.DataFrame({'x': values}))
    rhis.build_rhis_evol_dict('x')

    obtained = rhis.get_repr_idxs()['x']

    assert obtained == _brute_repr(rhis, 'x', DEFAULT_ALPHA)


def test_get_repr_idxs_matches_reference_for_ragged_mapping() -> None:
    rng = np.random.default_rng(7)
    data = {
        'alpha': list(rng.normal(10.0, 1.0, 30)),
        'beta': list(rng.normal(20.0, 2.0, 20)),
    }
    rhis = Rhis(data)

    obtained = rhis.get_repr_idxs()

    for col in data:
        assert obtained[col] == _brute_repr(rhis, col, DEFAULT_ALPHA)


def test_get_repr_idxs_uses_cached_evolution_when_available() -> None:
    rng = np.random.default_rng(8)
    rhis = Rhis(pd.DataFrame({'x': rng.normal(size=50)}))
    uncached = Rhis(pd.DataFrame(rhis.orig_df)).get_repr_idxs()['x']

    rhis.build_rhis_evol_dict('x')
    cached = rhis.get_repr_idxs()['x']

    assert cached == uncached


def test_get_repr_idxs_does_not_build_the_full_evolution() -> None:
    rng = np.random.default_rng(9)
    rhis = Rhis(pd.DataFrame({'x': rng.normal(size=SHORT_SERIES_LENGTH)}))

    start, end = rhis.get_repr_idxs()['x']

    assert end == SHORT_SERIES_LENGTH
    assert start == _brute_repr(rhis, 'x', DEFAULT_ALPHA)[0]
    # The lazy scan must not populate the evolution cache.
    assert rhis.pvalues['x']['RHIS-min'] == []
