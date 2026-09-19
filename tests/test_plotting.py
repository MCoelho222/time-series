from __future__ import annotations

import pandas as pd
import pytest

from rhis.core import Rhis
from rhis.exceptions import RhisEvolNotCalledError
from rhis.plotting import PLOTS_DIR, RHIS_PLOTS_DIR, _period_label, _year_label, plot_test
from tests.test_core_basics import _make_df


def test_plot_evolution_raises_if_build_rhis_evol_df_not_run() -> None:
    rhis = Rhis(_make_df())

    with pytest.raises(RhisEvolNotCalledError):
        rhis.plot_evolution()


def test_plot_evolution_raises_if_rhis_df_not_initialized() -> None:
    rhis = Rhis(_make_df())
    rhis.build_rhis_evol_df()
    rhis.rhis_df = None

    with pytest.raises(RuntimeError, match='has not been initialized'):
        rhis.plot_evolution()


def test_plot_evolution_saves_one_figure_per_series(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rhis = Rhis(_make_df(n_rows=60, n_cols=2))
    rhis.build_rhis_evol_df()
    rhis.plot_evolution(repr_df=rhis.build_rhis_compliant_df())

    plots_dir = tmp_path / RHIS_PLOTS_DIR
    assert plots_dir.is_dir()
    saved = sorted(path.name for path in plots_dir.glob('*.PNG'))
    assert saved == ['RHIS series_0.PNG', 'RHIS series_1.PNG']


def test_plot_evolution_without_repr_df(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rhis = Rhis(_make_df())
    rhis.build_rhis_evol_df()
    rhis.plot_evolution()

    assert (tmp_path / RHIS_PLOTS_DIR / 'RHIS series_0.PNG').is_file()


def test_plot_evolution_show_repr_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rhis = Rhis(_make_df())
    rhis.build_rhis_evol_df()
    rhis.plot_evolution(show_repr=False, repr_df=rhis.build_rhis_compliant_df())

    assert (tmp_path / RHIS_PLOTS_DIR / 'RHIS series_0.PNG').is_file()


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(pd.Timestamp('1871-07-05'), '1871'), (1871, '1871'), (1871.0, '1871'), ('1970', '1970')],
)
def test_year_label(value, expected) -> None:
    assert _year_label(value) == expected


def test_period_label_from_years() -> None:
    assert _period_label(1871, 1970) == '1871-1970'


def test_period_label_from_timestamps() -> None:
    assert _period_label(pd.Timestamp('1871-01-01'), pd.Timestamp('1970-12-31')) == '1871-1970'


def test_plot_test_creates_figure(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    plot_test([1.0, 2.0, 3.0, 4.0], 0.05)

    assert (tmp_path / PLOTS_DIR / 'hypothesis_test.PNG').is_file()


def test_plot_test_with_ranks(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    plot_test([1.0, 2.0, 3.0, 4.0], 0.05, ranks_p_value=0.13, filename='with_ranks')

    assert (tmp_path / PLOTS_DIR / 'with_ranks.PNG').is_file()
