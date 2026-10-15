from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

from rhis_ts.__main__ import load_example_data, main

if TYPE_CHECKING:
    import pytest

NILE_FIRST_YEAR = 1871
NILE_LAST_YEAR = 1970
NILE_LENGTH = 100


def test_example_data_is_shipped() -> None:
    """
    The River Nile CSV must travel inside the installed package: without it
    the 'rhis-ts' console script fails with FileNotFoundError.
    """
    data_file = files('rhis_ts').joinpath('data', 'nile_river_annual_flow.csv')

    assert data_file.is_file()


def test_typed_marker_and_docs_are_shipped() -> None:
    """
    The PEP 561 marker and the hypothesis-test notes are declared as package
    data, so they must be present in the installed distribution as well.
    """
    package_root = files('rhis_ts')

    assert package_root.joinpath('py.typed').is_file()
    assert package_root.joinpath('docs', 'hypothesis_tests', 'mann_kendall.md').is_file()


def test_load_example_data_shape() -> None:
    df = load_example_data()

    assert list(df.columns) == ['flow']
    assert df.index[0] == NILE_FIRST_YEAR
    assert df.index[-1] == NILE_LAST_YEAR
    assert len(df) == NILE_LENGTH


def test_console_entry_point_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """
    The guided tour shipped as the 'rhis-ts' entry point must run to
    completion and report the representative stretch.
    """
    main()

    out = capsys.readouterr().out
    assert 'RHIS-TS' in out
    assert 'THE VERDICT' in out
