"""A run the closing panel marked failed stays failed when its result comes in late.

Closing the Setup & Tuning panel while a detection or classification run is
still going marks the run failed ("napari was closed before the run
finished"). With napari still open, the run's thread can finish afterwards;
its late result must not set the row to 'completed', because the row would
then say completed while its notes said it had been cut off. A run's end is
recorded once: the late result is kept in the notes and the row stays failed.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mousebrain.plugin_3d.tuning_widget")
pytest.importorskip("napari")

import mousebrain.plugin_3d.tuning_widget as tw  # noqa: E402
from tuning_widget_stand_in import BRAIN_B, _row, _tracker, _widget  # noqa: E402


class _Stub:
    """Any Qt control the handlers touch: every call does nothing."""
    def __getattr__(self, name):
        return lambda *a, **k: ""


def _close_panel(w):
    event = SimpleNamespace(accept=lambda: None, ignore=lambda: None)
    tw.TuningWidget.closeEvent(w, event)


def test_a_detection_finishing_after_the_panel_closed_stays_failed(tmp_path, monkeypatch):
    t = _tracker(tmp_path, [])
    run_dir = tmp_path / "optimization_runs" / BRAIN_B / "test_20000101_000000"
    run_dir.mkdir(parents=True)
    exp_id = t.log_detection(brain=BRAIN_B, output_path=str(run_dir), status="started")
    w = _widget(tmp_path, monkeypatch, t)
    w._runs_in_progress = {exp_id}
    w.test_detection_btn = _Stub()
    w._refresh_prefilter_runs = lambda: None
    w._generate_detection_qc_image = lambda: None

    _close_panel(w)
    assert _row(t, exp_id)["status"] == "failed"

    run = {"brain": w.current_brain, "exp_id": exp_id, "output_path": str(run_dir),
           "z_offset": 0}
    w._on_detection_finished(True, "done", 0, [], run=run)
    row = _row(t, exp_id)
    assert row["status"] == "failed", "a late result turned a failed run into completed"
    assert "closed" in row["notes"]
    assert "arrived after" in row["notes"] and "det_cells_found=0" in row["notes"]


def test_a_classification_finishing_after_the_panel_closed_stays_failed(tmp_path, monkeypatch):
    t = _tracker(tmp_path, [])
    exp_id = t.log_classification(brain=BRAIN_B, model_path="default",
                                  candidates_path=str(tmp_path / "c.xml"),
                                  output_path=str(tmp_path / "class_run"))
    w = _widget(tmp_path, monkeypatch, t)
    w._runs_in_progress = {exp_id}
    w.classify_btn = _Stub()
    w.classify_status = _Stub()
    w.model_combo = _Stub()

    _close_panel(w)
    assert _row(t, exp_id)["status"] == "failed"

    # The run belongs to a brain no longer selected, so the handler records
    # it and draws nothing.
    run = {"brain": Path(tmp_path / "M" / "other_brain"), "exp_id": exp_id,
           "output_path": str(tmp_path / "class_run")}
    w._on_classification_finished(True, "done", 5, 7, exp_id, run=run)
    row = _row(t, exp_id)
    assert row["status"] == "failed", "a late result turned a failed run into completed"
    assert row["class_cells_found"] == ""
    assert "class_cells_found=5" in row["notes"]


def test_a_run_still_going_is_recorded_as_usual(tmp_path):
    t = _tracker(tmp_path, [])
    exp_id = t.log_detection(brain=BRAIN_B, status="started")
    assert t.update_status(exp_id, status="completed", det_cells_found=9,
                           expect_status="started")
    row = _row(t, exp_id)
    assert (row["status"], row["det_cells_found"]) == ("completed", "9")


def test_a_failure_reported_after_the_close_adds_nothing(tmp_path):
    t = _tracker(tmp_path, [])
    exp_id = t.log_detection(brain=BRAIN_B, status="started")
    t.fail_unfinished([exp_id], "napari was closed before the run finished")
    notes = _row(t, exp_id)["notes"]
    assert not t.update_status(exp_id, status="failed", expect_status="started")
    row = _row(t, exp_id)
    assert (row["status"], row["notes"]) == ("failed", notes)
