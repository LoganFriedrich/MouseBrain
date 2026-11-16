"""The Setup & Tuning pre-filter uses a chosen detection run only on its own
brain, and only a candidates file that is that run's own.

A run chosen with Load Best / Load Selected was never forgotten when another
brain was loaded, and nothing checked that it belonged to the loaded brain: the
button filtered brain A's candidates against brain B's atlas, wrote them into
brain B's folder and recorded them as brain B's. And a run whose tracker row
names a folder (old step-4 runs, runs that never finished) took whatever
detected_cells.xml sat in that folder, which can be a newer run's file.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

BRAIN_A = "101_PROJ_01_02_2p5x_z5"
BRAIN_B = "102_PROJ_01_03_2p5x_z5"


def _markers(n):
    return "<CellCounter_Marker_File><Marker_Data><Marker_Type>%s</Marker_Type>" \
           "</Marker_Data></CellCounter_Marker_File>" % ("<Marker><MarkerX>1</MarkerX></Marker>" * n)


def _brain(tmp_path, name):
    brain = tmp_path / "mouse_a" / name
    (brain / "4_Cell_Candidates").mkdir(parents=True)
    reg = brain / "3_Registered_Atlas"
    reg.mkdir()
    (reg / "brainreg.json").write_text("{}", encoding="utf-8")
    (brain / "4_Cell_Candidates" / "detected_cells.xml").write_text(_markers(5), encoding="utf-8")
    return brain


class _Stop(Exception):
    pass


class _Label:
    def __init__(self, text=None):
        self.text = text

    def setText(self, text):
        self.text = text
        raise _Stop  # the button has chosen its file and would start filtering

    def setStyleSheet(self, _style):
        pass


def _tracker(tmp_path):
    from mousebrain.tracker import ExperimentTracker
    return ExperimentTracker(csv_path=tmp_path / "calibration_runs.csv")


def _press(tw, monkeypatch, brain, tracker, exp_id):
    warnings = []
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: warnings.append(a[1:3]))
    fake = SimpleNamespace(current_brain=brain, _prefilter_source_exp_id=exp_id, tracker=tracker,
                           prefilter_status=_Label())
    tw.TuningWidget.run_prefilter(fake)  # returns without starting: no _Stop
    assert fake.prefilter_status.text is None, "the pre-filter started"
    return warnings


def test_a_run_of_another_brain_prefilters_nothing(tmp_path, monkeypatch):
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_a = _brain(tmp_path, BRAIN_A)
    brain_b = _brain(tmp_path, BRAIN_B)
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed",
                                   output_path=str(brain_a / "4_Cell_Candidates" / "detected_cells.xml"))
    warnings = _press(tw, monkeypatch, brain_b, tracker, exp_id)
    assert len(warnings) == 1
    title, text = warnings[0]
    assert title == "Run Belongs To Another Brain"
    assert BRAIN_A in text and BRAIN_B in text
    assert not list((brain_b / "4_Cell_Candidates").glob("prefilter*"))


def test_a_pipeline_run_named_mouse_slash_brain_is_this_brains_run(tmp_path, monkeypatch):
    """Step 4 records "<mouse folder>/<brain folder>"; that is the same brain."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_a = _brain(tmp_path, BRAIN_A)
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain="mouse_a/" + BRAIN_A, status="completed",
                                   output_path=str(brain_a / "4_Cell_Candidates" / "detected_cells.xml"))
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: pytest.fail(a[2]))
    fake = SimpleNamespace(current_brain=brain_a, _prefilter_source_exp_id=exp_id, tracker=tracker,
                           prefilter_status=_Label())
    with pytest.raises(_Stop):
        tw.TuningWidget.run_prefilter(fake)


def test_loading_another_brain_forgets_the_chosen_run(tmp_path):
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_b = _brain(tmp_path, BRAIN_B)
    label = SimpleNamespace(text="Loaded: some_run", setText=None, setStyleSheet=lambda s: None)
    label.setText = lambda t: setattr(label, "text", t)
    fake = SimpleNamespace(current_brain=brain_b, tracker=None,
                           _prefilter_source_exp_id="run_of_a", _prefilter_source_brain=BRAIN_A,
                           prefilter_runs_list=SimpleNamespace(clear=lambda: None),
                           prefilter_source_label=label)
    tw.TuningWidget._refresh_prefilter_runs(fake)
    assert fake._prefilter_source_exp_id is None
    assert label.text == "No detection loaded"


def test_refreshing_the_same_brain_keeps_the_chosen_run(tmp_path):
    """A detection finishing on the loaded brain refreshes the list too; the
    run chosen for this brain stays chosen."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_a = _brain(tmp_path, BRAIN_A)
    fake = SimpleNamespace(current_brain=brain_a, tracker=None,
                           _prefilter_source_exp_id="run_of_a", _prefilter_source_brain=BRAIN_A,
                           prefilter_runs_list=SimpleNamespace(clear=lambda: None),
                           prefilter_source_label=_Label("Loaded: run_of_a"))
    tw.TuningWidget._refresh_prefilter_runs(fake)
    assert fake._prefilter_source_exp_id == "run_of_a"


def test_a_run_recorded_as_the_shared_candidates_folder_prefilters_nothing(tmp_path, monkeypatch):
    """The run recorded only the folder step 4 writes run after run, and the
    file there carries no record of which run wrote it: it may be a newer run's."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_a = _brain(tmp_path, BRAIN_A)
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed",
                                   output_path=str(brain_a / "4_Cell_Candidates"))
    warnings = _press(tw, monkeypatch, brain_a, tracker, exp_id)
    assert [w[0] for w in warnings] == ["Selected Run Not Found"]


def test_a_file_whose_record_names_another_run_prefilters_nothing(tmp_path, monkeypatch):
    import mousebrain.plugin_3d.tuning_widget as tw
    from mousebrain.candidates_io import write_provenance
    brain_a = _brain(tmp_path, BRAIN_A)
    xml = brain_a / "4_Cell_Candidates" / "detected_cells.xml"
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed", output_path=str(xml))
    write_provenance(xml, "a_newer_run")
    warnings = _press(tw, monkeypatch, brain_a, tracker, exp_id)
    assert [w[0] for w in warnings] == ["Selected Run Not Found"]


def test_a_folder_whose_file_is_recorded_as_the_runs_own_is_used(tmp_path, monkeypatch):
    import mousebrain.plugin_3d.tuning_widget as tw
    from mousebrain.candidates_io import write_provenance
    brain_a = _brain(tmp_path, BRAIN_A)
    folder = brain_a / "4_Cell_Candidates"
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed", output_path=str(folder))
    write_provenance(folder / "detected_cells.xml", exp_id)
    assert tw._candidates_file_of_run(tracker.get_experiment(exp_id)) == folder / "detected_cells.xml"


def _test_run_folder(tmp_path):
    folder = tmp_path / "optimization_runs" / BRAIN_A / "test_run_1"
    folder.mkdir(parents=True)
    (folder / "detected_cells.xml").write_text(_markers(3), encoding="utf-8")
    return folder


def test_a_test_runs_own_folder_prefilters_nothing(tmp_path, monkeypatch):
    """A Setup & Tuning test run keeps its cells in a folder of its own,
    outside the brain. Nothing shows they are in the space the atlas was
    fitted to, so they are not compared with it."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_a = _brain(tmp_path, BRAIN_A)
    folder = _test_run_folder(tmp_path)
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed", output_path=str(folder),
                                   det_scope="full")
    warnings = _press(tw, monkeypatch, brain_a, tracker, exp_id)
    assert [w[0] for w in warnings] == ["Run Cannot Be Pre-filtered"]
    assert exp_id in warnings[0][1] and "Nothing was pre-filtered" in warnings[0][1]
    assert not list((brain_a / "4_Cell_Candidates").glob("prefilter*"))


def test_a_test_of_part_of_the_stack_prefilters_nothing(tmp_path, monkeypatch):
    """A test of part of the stack saves positions counted from the first
    plane tested; against the atlas every cell would be in the wrong plane."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_a = _brain(tmp_path, BRAIN_A)
    folder = _test_run_folder(tmp_path)
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed", output_path=str(folder),
                                   det_scope="partial", det_z_start=200, det_z_end=260)
    warnings = _press(tw, monkeypatch, brain_a, tracker, exp_id)
    assert [w[0] for w in warnings] == ["Run Cannot Be Pre-filtered"]
    assert "part of the stack" in warnings[0][1]
    assert not list((brain_a / "4_Cell_Candidates").glob("prefilter*"))


def test_a_partial_run_is_refused_wherever_its_file_is(tmp_path, monkeypatch):
    """Even a tracker row that names the brain's own candidates file: a run
    recorded as covering part of the stack is never compared with the atlas."""
    import mousebrain.plugin_3d.tuning_widget as tw
    brain_a = _brain(tmp_path, BRAIN_A)
    tracker = _tracker(tmp_path)
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed", det_scope="partial",
                                   output_path=str(brain_a / "4_Cell_Candidates" / "detected_cells.xml"))
    warnings = _press(tw, monkeypatch, brain_a, tracker, exp_id)
    assert [w[0] for w in warnings] == ["Run Cannot Be Pre-filtered"]


def test_the_run_that_wrote_the_brains_candidates_is_used(tmp_path, monkeypatch):
    """The pipeline run whose file is the one step 5 classifies starts the
    pre-filter on that file."""
    import mousebrain.plugin_3d.tuning_widget as tw
    from mousebrain.pipeline_status import find_candidates
    brain_a = _brain(tmp_path, BRAIN_A)
    tracker = _tracker(tmp_path)
    xml = brain_a / "4_Cell_Candidates" / "detected_cells.xml"
    exp_id = tracker.log_detection(brain=BRAIN_A, status="completed", output_path=str(xml),
                                   det_scope="full")
    path, why_not = tw._prefilter_file_of_run(tracker.get_experiment(exp_id), brain_a)
    assert (path, why_not) == (xml, "")
    assert path == find_candidates(brain_a)
    monkeypatch.setattr(tw.QMessageBox, "warning", lambda *a, **k: pytest.fail(a[2]))
    fake = SimpleNamespace(current_brain=brain_a, _prefilter_source_exp_id=exp_id, tracker=tracker,
                           prefilter_status=_Label())
    with pytest.raises(_Stop):
        tw.TuningWidget.run_prefilter(fake)
