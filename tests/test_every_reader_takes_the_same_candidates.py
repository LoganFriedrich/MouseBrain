"""Every reader of 'the candidates' takes the file step 5 would classify.

Step 5 has one rule for which candidates file it classifies
(pipeline_status.find_candidates: the registered-space candidates folders, the
standard name first). Four other readers kept their own: the batch's detection
check took the newest .xml at the top of the folder, the pre-filter a legacy
Detected_*.xml first and then any subfolder (an archived run, or a trial run in
a space the atlas was never fitted to), and the tuning widget's pre-filter and
classification buttons each a rule of their own. So a check could pass a brain
on one file while the next step used another, and the widget could filter or
classify candidates the pipeline would never use.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


def _markers(n):
    return "<CellCounter_Marker_File><Marker_Data><Marker_Type>%s</Marker_Type>" \
           "</Marker_Data></CellCounter_Marker_File>" % ("<Marker><MarkerX>1</MarkerX></Marker>" * n)


def _brain(tmp_path, registered_on=None):
    brain = tmp_path / "mouse_a" / "101_PROJ_01_02_2p5x_z5"
    (brain / "4_Cell_Candidates").mkdir(parents=True)
    reg = brain / "3_Registered_Atlas"
    reg.mkdir()
    (reg / "brainreg.json").write_text("{}", encoding="utf-8")
    if registered_on:
        (reg / "registered_from.json").write_text(
            json.dumps({"source": registered_on}), encoding="utf-8")
    return brain


@pytest.fixture
def batch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("batch_process_cand", PIPELINE / "batch_process.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PIPELINE_STATUS_CSV", tmp_path / "pipeline_status.csv")
    return mod


def test_the_batch_checks_the_file_step_5_will_classify(batch, tmp_path):
    brain = _brain(tmp_path)
    det = brain / "4_Cell_Candidates"
    (det / "detected_cells.xml").write_text(_markers(250), encoding="utf-8")
    # A stray file written later (a copy, an export) is not the candidates.
    (det / "notes_export.xml").write_text(_markers(3), encoding="utf-8")
    status, msg = batch.qc_check_detection(brain)
    assert status == batch.QCStatus.PASS, msg
    assert "250" in msg


def test_the_prefilter_takes_the_standard_file_not_a_legacy_name(tmp_path):
    from mousebrain.prefilter import find_latest_candidates
    brain = _brain(tmp_path)
    det = brain / "4_Cell_Candidates"
    (det / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    (det / "detected_old_run.xml").write_text(_markers(5), encoding="utf-8")
    assert find_latest_candidates(brain) == det / "detected_cells.xml"


def test_the_prefilter_help_describes_the_step_5_file_not_the_latest():
    """The command-line help must say which file is used without --candidates.

    It used to promise "latest in 4_Cell_Candidates", while the code takes the
    file step 5 classifies (the standard name first, not the newest file). An
    operator reading the help would expect a newer run to be filtered.
    """
    import subprocess
    import sys
    src = Path(__file__).resolve().parents[1] / "src"
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['prefilter','--help']; "
         "from mousebrain import prefilter; prefilter.main()"],
        capture_output=True, text=True, cwd=str(src),
        env={**__import__("os").environ, "PYTHONPATH": str(src)},
    ).stdout
    flat = " ".join(out.split())
    assert "--candidates" in flat
    assert "latest" not in flat.lower()
    assert "step 5 classifies" in flat


def test_the_prefilter_never_takes_a_run_outside_registered_space(tmp_path):
    from mousebrain.prefilter import find_latest_candidates
    brain = _brain(tmp_path, registered_on="cropped")
    trial = brain / "4_Cell_Candidates" / "from_1_Extracted_Full"
    trial.mkdir()
    (trial / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    archived = brain / "4_Cell_Candidates" / "_archive" / "20200101_000000"
    archived.mkdir(parents=True)
    (archived / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    assert find_latest_candidates(brain) is None


class _Stop(Exception):
    pass


class _Label:
    def __init__(self):
        self.text = None

    def setText(self, text):
        self.text = text
        raise _Stop  # the reader has chosen its file; nothing further is needed


def test_the_widget_prefilter_takes_the_file_step_5_will_classify(tmp_path, monkeypatch):
    """Registered on the full extraction: step 5 classifies the run kept in
    from_1_Extracted_Full, so that is what the pre-filter must filter."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain = _brain(tmp_path, registered_on="full")
    (brain / "1_Extracted_Full" / "ch0").mkdir(parents=True)
    (brain / "1_Extracted_Full" / "ch0" / "Z0000.tif").write_bytes(b"x")
    run = brain / "4_Cell_Candidates" / "from_1_Extracted_Full"
    run.mkdir()
    (run / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    warnings = []
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    fake = SimpleNamespace(current_brain=brain, _prefilter_source_exp_id=None, tracker=None,
                           prefilter_status=_Label())
    with pytest.raises(_Stop):
        tw.TuningWidget.run_prefilter(fake)
    assert not warnings
    assert "detected_cells.xml" in fake.prefilter_status.text


def test_the_widget_classifies_a_run_kept_in_registered_space(tmp_path, monkeypatch):
    """A brain registered on its full extraction: step 5 classifies the run in
    from_1_Extracted_Full, so the widget must find it too."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain = _brain(tmp_path, registered_on="full")
    (brain / "1_Extracted_Full" / "ch0").mkdir(parents=True)
    (brain / "1_Extracted_Full" / "ch0" / "Z0000.tif").write_bytes(b"x")
    run = brain / "4_Cell_Candidates" / "from_1_Extracted_Full"
    run.mkdir()
    (run / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    warnings = []
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    combo = SimpleNamespace(currentData=lambda: None, currentIndex=lambda: 1)
    fake = SimpleNamespace(current_brain=brain, model_combo=combo)
    tw.TuningWidget.run_classification(fake)
    # It got past the candidates and stopped at the model choice.
    assert warnings == ["Select a model first"]


def _tracker_with_run(tmp_path, output_path):
    from mousebrain.tracker import ExperimentTracker
    tracker = ExperimentTracker(csv_path=tmp_path / "calibration_runs.csv")
    exp_id = tracker.log_detection(brain="101_PROJ_01_02_2p5x_z5",
                                   output_path=str(output_path), status="completed")
    return tracker, exp_id


def test_a_chosen_run_whose_file_is_gone_prefilters_nothing(tmp_path, monkeypatch):
    """A run is chosen (Load Best / Load Selected) but its file cannot be found.
    The pre-filter is recorded with the chosen run as its parent, so filtering
    the brain's detected_cells.xml instead would record a false provenance: the
    button must stop and pre-filter nothing."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain = _brain(tmp_path)
    (brain / "4_Cell_Candidates" / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    tracker, exp_id = _tracker_with_run(tmp_path, tmp_path / "moved_away" / "detected_cells.xml")
    warnings = []
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: warnings.append(a[1:3]))
    fake = SimpleNamespace(current_brain=brain, _prefilter_source_exp_id=exp_id, tracker=tracker,
                           prefilter_status=_Label())
    tw.TuningWidget.run_prefilter(fake)  # returns without starting: no _Stop
    assert fake.prefilter_status.text is None
    assert len(warnings) == 1
    assert warnings[0][0] == "Selected Run Not Found"
    assert exp_id in warnings[0][1]


def test_a_chosen_run_prefilters_its_own_file(tmp_path, monkeypatch):
    """The chosen run's own candidates file is filtered, not the brain's
    current detected_cells.xml."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain = _brain(tmp_path)
    (brain / "4_Cell_Candidates" / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    kept = tmp_path / "kept_run" / "earlier_candidates.xml"
    kept.parent.mkdir()
    kept.write_text(_markers(7), encoding="utf-8")
    tracker, exp_id = _tracker_with_run(tmp_path, kept)
    warnings = []
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    fake = SimpleNamespace(current_brain=brain, _prefilter_source_exp_id=exp_id, tracker=tracker,
                           prefilter_status=_Label())
    with pytest.raises(_Stop):
        tw.TuningWidget.run_prefilter(fake)
    assert not warnings
    assert "earlier_candidates.xml" in fake.prefilter_status.text
