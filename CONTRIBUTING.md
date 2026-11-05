# Contributing to RHIS-TS

Thanks for your interest in improving `rhis-ts`! Bug reports, examples, documentation fixes and code are all welcome.

## Ways to contribute

- **Report a bug** or **suggest a feature** by [opening an issue](https://github.com/MCoelho222/time-series/issues/new/choose). The templates ask for what we need to reproduce or understand it.
- **Ask a question** about using the package or interpreting a result: open an issue too.
- **Improve the docs or examples**: the README, the test explanations in `src/rhis_ts/docs/hypothesis_tests/`, or the notebooks in `example_notebooks/`.
- **Fix a bug or add a feature**: for anything beyond a small fix, please open an issue first so we can agree on the approach before you invest time.

## Setting up

You need Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/time-series.git   # your fork
cd time-series
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Checks

Run these before opening a pull request. CI runs the same checks on Python 3.10–3.14.

```bash
python -m pytest                    # tests (coverage must stay at or above 90%)
python -m ruff check src tests      # lint
python -m mypy src tests            # type checks
```

To run the full matrix locally, use `tox` (it needs `tox` and `tox-uv` installed).

## Making a change

1. Fork the repository and create a branch from `master` (for example `fix/nan-handling` or `docs/readme-example`).
2. Make your change, with tests for any change in behavior.
3. Add an entry to `CHANGELOG.md` under `## [Unreleased]` if users will notice the change.
4. Run the checks above.
5. Open a pull request and fill in the template.

### Code style

The linter enforces most of it:

- Every module starts with `from __future__ import annotations`.
- Single quotes, lines up to 128 characters, [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
- Assign error messages to a `msg` variable before raising them.
- Type-only imports go under `if TYPE_CHECKING:`.

### Changing a statistical test

The package computes p-values in two ways that must give the same results: each test's public function (one series at a time) and the fast incremental path in `src/rhis_ts/hypothesis/evolution.py`. If you change how a statistic is computed, update both. `tests/test_core_evolution_equivalence.py` checks that they agree. [`AGENTS.md`](AGENTS.md) describes the architecture in more detail.

## Using AI coding agents

AI-assisted contributions are welcome. [`AGENTS.md`](AGENTS.md) holds the instructions agents need (commands, architecture, rules), and `CLAUDE.md` points Claude Code to it. You remain responsible for what you submit: please review the changes yourself, make sure the checks pass, and confirm that any numbers or results in docs come from running the code.

## Releases

Releases are made by the maintainer. Please don't change the version number in your pull request. See [`RELEASING.md`](RELEASING.md) if you're curious about the process.

## Code of conduct

This project follows a [code of conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.
