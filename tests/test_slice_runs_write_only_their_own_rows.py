"""Each 2D run's result is written into that run's own tracker row.

Detection, colocalization and particle analysis in the napari window must not
share one "last run id": a run started while another is still going would
re-point it, and the first run's result would be written into the second
run's row (a detection's nucleus count in a colocalization row marked
'completed' with no colocalization result, the detection row left
'started'). So each kind of run keeps its own row id, and the window runs one
analysis at a time: a second one is refused, with a message, until the first
one's result has been recorded.
"""
import pandas as pd

from slice_widget_helpers import StubWorker, labels_with_one_nucleus, make_widget

SAMPLE = "PRJ_01_02_S07"


def _rows(w):
    return {r["run_type"]: r for r in w.tracker.search(sample_id=SAMPLE, limit=None)}


def test_a_colocalization_clicked_during_detection_cannot_take_the_detections_result(
        tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE)
    w.nuclei_labels = labels_with_one_nucleus()  # an earlier detection's nuclei

    w._run_detection()                 # its worker is now going
    w._run_colocalization()            # clicked before detection reports
    w._on_detect_finished(True, "Detected 4 nuclei", 4, labels_with_one_nucleus())

    rows = _rows(w)
    assert rows["detection"]["status"] == "completed"
    assert rows["detection"]["det_nuclei_found"] == "4"
    coloc = rows.get("colocalization")
    assert coloc is None or coloc["det_nuclei_found"] == "", \
        "the detection's count was written into the colocalization's row"
    assert "still running" in w.status_label.text() or coloc is None


def test_a_detection_clicked_during_colocalization_cannot_take_the_colocalizations_result(
        tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE)
    w.nuclei_labels = labels_with_one_nucleus()
    monkeypatch.setattr(
        "mousebrain.plugin_2d.sliceatlas.core.config.get_sample_dir",
        lambda stem: tmp_path / "subject")
    w._visualize_colocalization = lambda m: None
    w._update_diagnostic_plot = lambda: None
    w._refresh_run_history = lambda: None

    w._run_colocalization()            # its worker is now going
    w._run_detection()                 # clicked before colocalization reports
    summary = {"positive_cells": 1, "negative_cells": 2, "positive_fraction": 1 / 3,
               "background_used": 5.0, "mean_fold_change": 1.5}
    w._on_coloc_finished(True, "done", pd.DataFrame({"label": [1, 2, 3]}), summary, None)

    rows = _rows(w)
    assert rows["colocalization"]["status"] == "completed"
    assert rows["colocalization"]["coloc_positive_cells"] == "1"
    det = rows.get("detection")
    assert det is None or det["coloc_positive_cells"] == "", \
        "the colocalization's counts were written into the detection's row"


def test_a_second_analysis_is_refused_with_a_message_until_the_first_reports(
        tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE)
    w._run_detection()
    started = len(StubWorker.made)

    w._run_detection()
    assert len(StubWorker.made) == started, "a second detection was started"
    assert "Detection not started: detection is still running" in w.status_label.text()

    w._on_detect_finished(True, "done", 1, labels_with_one_nucleus())
    w._run_colocalization()
    assert len(StubWorker.made) == started + 1, "the window stayed busy after detection reported"


def test_a_detection_that_could_not_start_leaves_the_window_free(tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE)

    class Refused(StubWorker):
        def start(self):
            raise RuntimeError("no thread available")
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import workers
    monkeypatch.setattr(workers, "DetectionWorker", Refused)

    w._run_detection()
    assert _rows(w)["detection"]["status"] == "failed"
    # Nothing is left going, so the next click starts.
    w.nuclei_labels = labels_with_one_nucleus()
    made = len(StubWorker.made)
    w._run_colocalization()
    assert len(StubWorker.made) == made + 1
