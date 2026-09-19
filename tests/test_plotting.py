from __future__ import annotations

import pytest

from rhis.core import Rhis
from rhis.exceptions import RhisEvolNotCalledError
from rhis.plotting import PLOTS_DIR, RHIS_PLOTS_DIR, plot_test
from tests.test_core_basics import _make_df


def test_plot_raises_if_evol_not_run() -> None:
    rhis = Rhis(_make_df())

    with pytest.raises(RhisEvolNotCalledError):
        rhis.plot()


def test_plot_raises_if_rhis_df_not_initialized() -> None:
    rhis = Rhis(_make_df())
    rhis.evol()
    rhis.rhis_df = None

    with pytest.raises(RuntimeError, match='has not been initialized'):
        rhis.plot()


def test_add_rhis_compliant_raises_if_rhis_df_not_initialized() -> None:
    rhis = Rhis(_make_df())
    rhis.evol()
    rhis.rhis_df = None

    with pytest.raises(RuntimeError, match='has not been initialized'):
        rhis.add_rhis_compliant_to_df()


def test_plot_saves_one_figure_per_series(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rhis = Rhis(_make_df(n_rows=60, n_cols=2))
    rhis.evol()
    rhis.add_rhis_compliant_to_df()
    rhis.plot()

    plots_dir = tmp_path / RHIS_PLOTS_DIR
    assert plots_dir.is_dir()
    saved = sorted(path.name for path in plots_dir.glob('*.PNG'))
    assert saved == ['RHIS series_0.PNG', 'RHIS series_1.PNG']


def test_plot_without_repr_column(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rhis = Rhis(_make_df())
    rhis.evol()
    rhis.plot()

    assert (tmp_path / RHIS_PLOTS_DIR / 'RHIS series_0.PNG').is_file()


def test_plot_show_repr_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rhis = Rhis(_make_df())
    rhis.evol()
    rhis.add_rhis_compliant_to_df()
    rhis.plot(show_repr=False)

    assert (tmp_path / RHIS_PLOTS_DIR / 'RHIS series_0.PNG').is_file()


def test_plot_test_creates_figure(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    plot_test([1.0, 2.0, 3.0, 4.0], 0.05)

    assert (tmp_path / PLOTS_DIR / 'hypothesis_test.PNG').is_file()


def test_plot_test_with_ranks(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    plot_test([1.0, 2.0, 3.0, 4.0], 0.05, ranks_p_value=0.13, filename='with_ranks')

    assert (tmp_path / PLOTS_DIR / 'with_ranks.PNG').is_file()
