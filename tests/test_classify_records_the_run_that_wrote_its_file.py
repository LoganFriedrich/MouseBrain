"""The Setup & Tuning Classify button records, as the classification's parent,
the detection run that wrote the file it classifies -- never just the run that
happens to be chosen.

The button classifies the file step 5 would classify. It used to record the run
chosen with Load Best / Load Selected as the parent whatever that run's own file
was, so a classification of a newer run's file was filed under an older run.
Now the chosen run is the parent only when its own file is the one classified;
otherwise the provenance record beside the file names the parent, and with no
record the parent is left empty.
"""
import json
from types import SimpleNamespace

import pytest

BRAIN = "101_PROJ_01_02_2p5x_z5"


def _markers(n):
    return "<CellCounter_Marker_File><Marker_Data><Marker_Type>%s</Marker_Type>" \
           "</Marker_Data></CellCounter_Marker_File>" % ("<Marker><MarkerX>1</MarkerX></Marker>" * n)


class _Stop(Exception):
    pass


def _brain(tmp_path):
    brain = tmp_path / "mouse_a" / BRAIN
    (brain / "4_Cell_Candidates").mkdir(parents=True)
    reg = brain / "3_Registered_Atlas"
    reg.mkdir()
    (reg / "brainreg.json").write_text("{}", encoding="utf-8")
    ch0 = brain / "2_Cropped_For_Registration" / "ch0"
    ch0.mkdir(parents=True)
    (ch0 / "Z0000.tif").write_bytes(b"x")
    (brain / "4_Cell_Candidates" / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    return brain


def _record(xml, exp_id):
    from mousebrain.candidates_io import provenance_path
    provenance_path(xml).write_text(json.dumps({"exp_id": exp_id}), encoding="utf-8")


def _classify(tw, monkeypatch, tmp_path, brain, tracker, chosen):
    """Press Classify and return the parent the tracker was asked to record."""
    import brainglobe_utils.IO.cells as cells_io
    import mousebrain.brain_context as brain_context
    monkeypatch.setattr(tw, "DATA_SUMMARY_DIR", tmp_path / "summary")
    monkeypatch.setattr(cells_io, "get_cells", lambda _p: [])
    # A signal layer loaded from the registered images (classification
    # refuses any other folder's layers).
    layer = SimpleNamespace(data=object(), metadata={
        "source_brain": str(brain),
        "source_folder": str(brain / "2_Cropped_For_Registration")})
    monkeypatch.setattr(brain_context, "pick_image_layers", lambda _l, _b: (layer, None, []))
    warnings = []
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    asked = {}

    def log_classification(**kw):
        asked.update(kw)
        raise _Stop  # the parent is chosen; nothing further is needed

    monkeypatch.setattr(tracker, "log_classification", log_classification)
    spin = SimpleNamespace(value=lambda: 1)
    fake = SimpleNamespace(
        current_brain=brain, tracker=tracker, metadata=None, viewer=SimpleNamespace(layers=[]),
        model_combo=SimpleNamespace(currentData=lambda: None, currentIndex=lambda: 0),
        classify_cube_size=spin, classify_batch_size=spin,
        _prefilter_source_exp_id=chosen)
    with pytest.raises(_Stop):
        tw.TuningWidget.run_classification(fake)
    assert not warnings
    assert asked["candidates_path"] == str(brain / "4_Cell_Candidates" / "detected_cells.xml")
    return asked["parent_experiment"]


def _tracker(tmp_path):
    from mousebrain.tracker import ExperimentTracker
    return ExperimentTracker(csv_path=tmp_path / "calibration_runs.csv")


def test_a_chosen_run_with_another_file_is_not_the_parent(tmp_path, monkeypatch):
    """The chosen run kept its own file elsewhere; the file classified was
    written by another run, which its provenance record names."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain = _brain(tmp_path)
    tracker = _tracker(tmp_path)
    kept = tmp_path / "kept_run" / "earlier_candidates.xml"
    kept.parent.mkdir()
    kept.write_text(_markers(7), encoding="utf-8")
    chosen = tracker.log_detection(brain=BRAIN, output_path=str(kept), status="completed")
    writer = tracker.log_detection(
        brain=BRAIN, output_path=str(brain / "4_Cell_Candidates" / "detected_cells.xml"),
        status="completed")
    _record(brain / "4_Cell_Candidates" / "detected_cells.xml", writer)
    assert _classify(tw, monkeypatch, tmp_path, brain, tracker, chosen) == writer


def test_no_record_and_another_file_leaves_the_parent_empty(tmp_path, monkeypatch):
    import mousebrain.plugin_3d.tuning_widget as tw
    brain = _brain(tmp_path)
    tracker = _tracker(tmp_path)
    kept = tmp_path / "kept_run" / "earlier_candidates.xml"
    kept.parent.mkdir()
    kept.write_text(_markers(7), encoding="utf-8")
    chosen = tracker.log_detection(brain=BRAIN, output_path=str(kept), status="completed")
    assert _classify(tw, monkeypatch, tmp_path, brain, tracker, chosen) in (None, "")


def test_a_chosen_run_whose_file_is_classified_is_the_parent(tmp_path, monkeypatch):
    import mousebrain.plugin_3d.tuning_widget as tw
    brain = _brain(tmp_path)
    tracker = _tracker(tmp_path)
    xml = brain / "4_Cell_Candidates" / "detected_cells.xml"
    chosen = tracker.log_detection(brain=BRAIN, output_path=str(xml), status="completed")
    _record(xml, chosen)
    assert _classify(tw, monkeypatch, tmp_path, brain, tracker, chosen) == chosen
