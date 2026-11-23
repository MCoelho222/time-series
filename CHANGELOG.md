# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A test that checks the incremental p-value evolution stays at least eight
  times as fast as the per-slice reference.

### Changed

- Faster p-values. Normal-approximation p-values call `scipy.special.ndtr`
  directly instead of `scipy.stats.norm`, whose per-call argument checking
  dominated the run time. Results are bit-identical. `build_rhis_evol_dict` is
  about 6x faster on 600–900-value series.
- `get_repr_idxs` checks the constant-time tests (Mann-Kendall, then
  Wallis-Moore) first and skips Mann-Whitney and Wald-Wolfowitz on windows
  that already fail. Together with the change above it is about 13x faster on
  600–900-value series, with the same results. A window with an untestable
  (NaN) p-value is now always treated as non-compliant, as it already was when
  the p-value evolution was cached.

## [0.2.0] - 2026-11-04

### Added

- `Rhis` accepts a single `pandas.Series`, treated as `series.to_frame()`
  (keyed by the Series name, or `0` when unnamed).

### Changed

- Much faster p-value evolution. The series is ranked once and the test
  statistics are updated incrementally as the window shrinks, instead of
  re-running every test from scratch on each window. On a ~1,200-value monthly
  series, `get_repr_idxs` drops from over a minute to under a second.
- `get_repr_idxs` stops at the longest compliant stretch instead of building the
  full p-value evolution, so it no longer fills `Rhis.pvalues` as a side effect.
  Call `build_rhis_evol_dict` when you need the p-values. If they were already
  computed, `get_repr_idxs` reuses them.
- A window that cannot be tested (fewer than 8 values, or all values equal)
  stops the search. A series whose most recent values are all equal now returns
  `(0, 0)` instead of a stretch.

## [0.1.0] - 2026-10-15

### Added

- `Rhis` class for representativeness verification of time series.
- Randomness (Wallis-Moore), homogeneity (Mann-Whitney), independence
  (Wald-Wolfowitz) and stationarity (Mann-Kendall) tests.
- `Rhis.build_rhis_evol_dict` for the p-value evolution of every window ending
  at the most recent observation.
- `Rhis.get_repr_idxs` for the positional indexes of the longest RHIS-compliant
  stretch.
- `rhis-ts` console command with a guided tour on the River Nile dataset.

[Unreleased]: https://github.com/MCoelho222/time-series/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/MCoelho222/time-series/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MCoelho222/time-series/releases/tag/v0.1.0
