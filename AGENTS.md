# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. `CLAUDE.md` imports this file, so Claude Code reads the same instructions. Human contributors should also read `CONTRIBUTING.md`.

## Project

`rhis-ts` (PyPI) is a Python ≥3.10 library that finds the longest recent stretch of a time series passing four non-parametric tests: **R**andomness (Wallis-Moore), **H**omogeneity (Mann-Whitney), **I**ndependence (Wald-Wolfowitz) and **S**tationarity (Mann-Kendall). Public API is a single class, `rhis_ts.Rhis`, with `build_rhis_evol_dict()` and `get_repr_idxs()`. The `rhis-ts` console script (`src/rhis_ts/__main__.py`) runs a guided demo on the bundled Nile dataset.

## Commands

```bash
pip install -e .[dev]                                   # dev install (deps come from requirements/*.txt)
python -m pytest                                        # all tests
python -m pytest tests/test_mann_kendall.py::test_name  # single test
python -m pytest --cov --cov-report=term-missing        # coverage (fail_under = 90, covdefaults plugin)
python -m ruff check src tests                          # lint
python -m mypy src tests                                # strict typing (disallow_untyped_defs on src)
tox                                                     # full matrix py310–py314 + lint + type (uses tox-uv)
tox -e py313 -- -k evolution                            # one env, args forwarded to pytest
python -m build && python -m twine check dist/*         # build/check distribution
```

If `python -m build` fails with `No module named build.__main__`, a stale `build/` directory is shadowing the module: `rm -rf build dist` (or use `uv build`).

## Code conventions enforced by tooling

- Every module must start with `from __future__ import annotations` (ruff isort `required-imports`).
- Single quotes, line length 128, numpy-style docstrings. Error messages are assigned to a `msg` variable before `raise` (ruff `EM`), and are typically also `logger.debug(msg)`-ed via loguru.
- Type-only imports go under `if TYPE_CHECKING:` (ruff `TCH`).

## Architecture

**Two computation paths that must agree.** Each test in `hypothesis/` exposes a public function (`mann_kendall`, `mann_whitney`, `wald_wolfowitz`, `wallis_moore`) that tests one series from scratch, plus a private kernel (`_mann_kendall_from_s`, `_mann_whitney_from_ranks`, `_wald_wolfowitz_from_array`, `_wallis_moore_from_runs`) that computes the p-value from precomputed statistics. `hypothesis/evolution.py::SuffixEvolution` is the fast path: it ranks the series once and then, for each suffix, drops the first value and updates ranks, tie counts, inversions (Fenwick tree, used for Mann-Kendall's S) and up/down run counts incrementally, feeding the private kernels. `Rhis.calculate_rhis` is the slow per-slice reference. `tests/test_core_evolution_equivalence.py` asserts the two match. When changing a test's statistic, update both the public function and the incremental state in `SuffixEvolution._slide`/`_current_pvalues`.

**Input normalization (`core.py`).** `_normalize_input` accepts a DataFrame (row MultiIndex rejected; column MultiIndex allowed, keys become tuples), a Series (converted with `to_frame()`, so it follows the DataFrame path; unnamed → key `0`), or a mapping of name → sequence (positional; ragged series NaN-padded). `series_lengths` keeps each original length so returned `(end)` indexes are relative to the user's series, not the padded frame. Values are coerced with `pd.to_numeric(errors='coerce')`; columns with fewer than `MIN_SERIES_LENGTH = 8` finite values are dropped.

**NaN semantics.** NaN is the package-wide signal for "untestable" (suffix shorter than 8, all values equal, or Wald-Wolfowitz raising `ValueError` on degenerate variance). Non-finite entries are removed before testing; several raw start positions that differ only by leading NaNs map onto the same cleaned suffix (`np.searchsorted` over `finite_positions` in `_evol_dict_from_ts` / `_raw_started_pvalues`). In `get_repr_idxs`, a NaN RHIS-min does not count as compliant; `(0, 0)` means no passing stretch.

**Evolution layout.** Entry `i` of every list in the evolution dict is the p-value for the suffix starting at raw position `i`; lists are as long as the original series, with trailing NaNs. `RHIS-min` is the row-wise minimum of R/H/I/S. Results are cached per series in `Rhis.pvalues`; `get_repr_idxs` reuses the cache if present, otherwise iterates `SuffixEvolution` lazily and stops at the first suffix with RHIS-min ≥ alpha (the longest compliant one).

## Rules for changes

- **Run the checks before proposing a change:** `python -m pytest`, `python -m ruff check src tests` and `python -m mypy src tests`. CI runs the same checks on Python 3.10–3.14 for every pull request; a PR with failing checks will not be merged.
- **Add tests with every behavior change**, next to the existing ones in `tests/`. Coverage must stay at or above 90%.
- **Keep the two computation paths in sync** (see Architecture). If you change how a test statistic or p-value is computed, `tests/test_core_evolution_equivalence.py` must still pass without loosening its tolerances.
- **Record user-visible changes** in `CHANGELOG.md` under `## [Unreleased]` (sections `Added`, `Changed`, `Fixed`, `Removed`).
- **Do not change `__version__`, tags, releases or the publish workflow.** Releases are made by the maintainer (see Release).
- **Keep the public API stable.** `Rhis`, `build_rhis_evol_dict()` and `get_repr_idxs()` are used by others; discuss breaking changes in an issue first.
- **Do not commit** notebook outputs that were not asked for, local data, build artifacts (`build/`, `dist/`), or anything in git-ignored folders (`posts/`, `deprecated/`, `rhis_plots/`, `hypothesis_testing_plots/`).
- **Changes to a statistical test need scientific support.** Any change that can alter a statistic or p-value (formula, ties, variance or continuity correction, approximation limits, minimum length) must cite peer-reviewed papers, textbooks or established statistical references (with equation, table or page where possible), include a test that reproduces a published or independently computed result, and update the README References and `src/rhis_ts/docs/hypothesis_tests/`. Never change test statistics based only on making tests pass. See `CONTRIBUTING.md`.
- **Statistical claims need evidence.** When docs or notebooks state a result (a stretch, a p-value, a year), compute it with the current code rather than copying it from older text.
- Keep PRs focused: one topic per PR, with a description of what changed and how it was checked.

## Where things are

- `src/rhis_ts/core.py`: `Rhis` and input normalization.
- `src/rhis_ts/hypothesis/`: the four tests and `evolution.py` (incremental suffix evolution).
- `src/rhis_ts/docs/hypothesis_tests/`: explanations of each test; update them when a test changes.
- `src/rhis_ts/data/nile_river_annual_flow.csv`: bundled dataset used by the `rhis-ts` demo and tests.
- `example_notebooks/`: example analyses (`usgs.ipynb`) with `helpers.py` and the `plotting` helpers.

## Release

Version lives only in `src/rhis_ts/__init__.py` (`__version__`); add a matching `CHANGELOG.md` section. Publishing a GitHub Release tagged `v<__version__>` triggers `.github/workflows/publish.yml` (PyPI Trusted Publishing; it fails if tag and version disagree). See `RELEASING.md`.
