"""A person can mark a 2D run best, and clear a mark, in the Run History panel,
and the batch then uses (or stops using) that run's numbers.

Defect covered: nothing in the widget could mark a 2D run best, so the batch
could never be given a person's choice; and a mark on another sample that
disagreed (which stops the batch) could only be cleared from a terminal.
"""

from types import SimpleNamespace

import pytest

from test_encr_batch_takes_only_approved_runs import (  # noqa: F401  (fixture)
    SAMPLE_A, SAMPLE_B, _coloc, be, tracker)
from test_widget_reload_does_not_draw_sigma_as_fold_change import _Table


def test_history_panel_marks_a_run_best_and_the_batch_uses_it(tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    run = _coloc(tracker, SAMPLE_A, value=0.25)
    fake = SimpleNamespace(run_history_table=_Table(run), tracker=tracker,
                           _refresh_run_history=lambda: None)
    handler = getattr(mw.BrainSliceWidget, "_mark_selected_run_best", None)
    assert handler is not None, "the run-history panel has no way to mark a run best"
    handler(fake)
    assert tracker.get_run(run)["marked_best"] == "True"
    assert be.get_tuned_parameters()["threshold"] == 0.25
    assert "will use" in shown[-1][2]


def test_history_panel_says_when_the_batch_will_not_use_a_marked_run(tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    run = _coloc(tracker, SAMPLE_A, method="fold_change", value=2.0)
    fake = SimpleNamespace(run_history_table=_Table(run), tracker=tracker,
                           _refresh_run_history=lambda: None)
    handler = getattr(mw.BrainSliceWidget, "_mark_selected_run_best", None)
    assert handler is not None, "the run-history panel has no way to mark a run best"
    handler(fake)
    assert "will NOT use" in shown[-1][2]


# --- a disagreement can be settled in the GUI ---------------------------------

def test_unmarking_one_of_two_disagreeing_runs_settles_the_conflict(tracker, monkeypatch):
    mw = pytest.importorskip("mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget")
    keep = _coloc(tracker, SAMPLE_A, value=0.0)
    drop = _coloc(tracker, SAMPLE_B, value=1.0)
    tracker.mark_as_best(keep)
    tracker.mark_as_best(drop)
    with pytest.raises(be.TunedParameterConflict):
        be.get_tuned_parameters()
    # Marking another run of a THIRD sample does not clear either mark ...
    tracker.mark_as_best(_coloc(tracker, "PRJ01_09_S9_RZ", value=0.0))
    with pytest.raises(be.TunedParameterConflict):
        be.get_tuned_parameters()
    # ... the panel's Unmark Best button on the unwanted run does.
    shown = []
    monkeypatch.setattr(mw, "QMessageBox", SimpleNamespace(
        warning=lambda *a: shown.append(a), information=lambda *a: shown.append(a)))
    handler = getattr(mw.BrainSliceWidget, "_unmark_selected_run_best", None)
    assert handler is not None, "the run-history panel has no way to clear a best mark"
    handler(SimpleNamespace(run_history_table=_Table(f"★ {drop}"), tracker=tracker,
                            _refresh_run_history=lambda: None))
    assert tracker.get_run(drop)["marked_best"] == ""
    assert be.get_tuned_parameters()["threshold"] == 0.0
