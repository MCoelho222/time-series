# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/MCoelho222/time-series/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MCoelho222/time-series/releases/tag/v0.1.0
