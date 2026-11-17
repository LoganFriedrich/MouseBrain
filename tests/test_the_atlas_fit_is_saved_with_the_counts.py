"""Run Quantification saves and logs the atlas fit the region counts are made with.

Before this, the Align tab held the atlas position, orientation and transform
only in memory and the tracker's registration logger had no caller: once the
window closed, nothing said which fit a set of region counts came from.

The record must never contradict the counts it sits beside:
- before the counts can change, any earlier fit file is replaced by an
  'in progress' record, so a window closed (or a crash) between the export
  and the finished record never leaves the old fit beside new counts; if the
  earlier file cannot be replaced, nothing is counted;
- the finished fit file is written only after the counts were exported with
  that fit;
- a count that fails without touching the count files leaves the folder (old
  counts + old fit file) untouched and marks its tracker row failed; one that
  rewrote some of them replaces the fit file with an 'incomplete' record;
- the export itself is all-or-nothing;
- a fit file that cannot be written, or a tracker that cannot be written,
  never leaves an older fit file beside new counts;
- counts exported WITHOUT an Align-tab fit replace any earlier fit file with
  one saying no fit was used;
- with no tracker available, the person is warned.

These tests drive the real BrainSliceWidget._run_quantification and
_on_quant_finished and the real AlignmentWidget.get_current_registration on
stand-in objects (no window), with a stand-in for the background worker.
Every name the fix added is looked up with a fallback, so on code without
the fix these tests fail on the behaviour they check, not in their setup.
"""
import csv
import json
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mousebrain.plugin_2d.sliceatlas.core import config as slice_config
from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget, workers
from mousebrain.plugin_2d.sliceatlas.napari_plugin.alignment_widget import AlignmentWidget
from mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget import BrainSliceWidget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

SAMPLE = "SUBJ_X_S03"


class _Signal:
    def connect(self, _slot):
        pass


class _FakeQuantWorker:
    """Stands in for the counting thread; remembers what it was handed."""
    last = None

    def __init__(self, cell_measurements, atlas_labels, atlas_manager=None,
                 output_dir=None, sample_id="sample"):
        self.atlas_labels = atlas_labels
        self.output_dir = output_dir
        self.progress = _Signal()
        self.finished = _Signal()
        _FakeQuantWorker.last = self

    def start(self):
        pass

    def isRunning(self):
        # No thread: each test reports the result itself (see _report).
        return False


class _Check:
    def __init__(self, on):
        self._on = on

    def isChecked(self):
        return self._on


class _Table:
    def __init__(self):
        self.n = 0

    def setRowCount(self, n):
        self.n = n

    def rowCount(self):
        return self.n

    def insertRow(self, _r):
        self.n += 1

    def setItem(self, *_a):
        pass


class _Box:
    """Records every message window instead of opening one."""
    def __init__(self):
        self.shown = []

    def warning(self, _parent, title, text, *a):
        self.shown.append(("warning", title, text))

    def information(self, _parent, title, text, *a):
        self.shown.append(("information", title, text))


def _alignment_widget(loaded=True):
    atlas = SimpleNamespace(resolution=(10.0, 10.0, 10.0))
    aw = SimpleNamespace(
        atlas_loaded=loaded,
        atlas_manager=SimpleNamespace(brain_atlas_name="test_atlas_10um", brain_atlas=atlas),
        current_orientation="coronal",
        current_atlas_position=640,
        transform_offset_x=12,
        transform_offset_y=-7,
        transform_rotation=3.5,
        transform_scale=1.25,
        warped_annotation=None,
        _warp_shown_for=None,
        parent_widget=SimpleNamespace(red_channel=np.zeros((8, 8), dtype=np.uint16)),
    )
    aw.get_registered_atlas_labels = lambda: np.ones((8, 8), dtype=np.int32)
    for name in ("get_current_registration", "_current_image_key", "_key_is_current", "_elastix_warp_on_screen"):
        if hasattr(AlignmentWidget, name):
            setattr(aw, name, types.MethodType(getattr(AlignmentWidget, name), aw))
    return aw


def _widget(tmp_path, export, tracker="csv", atlas_loaded=True):
    w = SimpleNamespace(
        cell_measurements=object(),
        status_label=SimpleNamespace(setText=lambda _t: None),
        quant_btn=SimpleNamespace(setEnabled=lambda _b: None),
        sample_id_edit=SimpleNamespace(text=lambda: SAMPLE),
        export_csv_check=_Check(export),
        current_file=tmp_path / "raw" / f"{SAMPLE}.nd2",
        alignment_widget=_alignment_widget(atlas_loaded),
        tracker=SliceTracker(csv_path=tmp_path / "runs.csv") if tracker == "csv" else None,
        _tracker_error=None if tracker == "csv" else "no tracker file configured",
        atlas_labels=None,
        red_channel=np.zeros((8, 8), dtype=np.uint16),
        results_table=_Table(),
        region_counts=None,
        _on_quant_progress=lambda *_a: None,
        _analysis_running=None, _loading_file=None,
        quant_worker=None,
    )
    w._get_current_slice = lambda arr: arr
    for name in ("_refuse_if_analysis_running", "_analysis_in_progress"):
        if hasattr(BrainSliceWidget, name):
            setattr(w, name, types.MethodType(getattr(BrainSliceWidget, name), w))
    w._finish_atlas_fit_record = types.MethodType(
        getattr(BrainSliceWidget, "_finish_atlas_fit_record", lambda self, s: None), w)
    w._on_quant_finished = types.MethodType(BrainSliceWidget._on_quant_finished, w)
    return w


_SUMMARY = {"total_cells": 3, "positive_cells": 1, "negative_cells": 2,
            "positive_fraction": 1 / 3, "regions_with_cells": 1, "top_region": "R1"}


def _report(w, success):
    """The worker reporting back, as its finished signal would."""
    if success:
        BrainSliceWidget._on_quant_finished(w, True, "done", object(), object(), _SUMMARY)
    else:
        BrainSliceWidget._on_quant_finished(w, False, "export failed", None, None, None)


@pytest.fixture
def msg_box():
    return _Box()


@pytest.fixture
def quant_env(tmp_path, monkeypatch, msg_box):
    box = msg_box
    monkeypatch.setattr(workers, "QuantificationWorker", _FakeQuantWorker)
    monkeypatch.setattr(slice_config, "get_sample_dir", lambda stem: tmp_path / "subject")
    monkeypatch.setattr(main_widget, "QMessageBox", box)
    monkeypatch.setattr(main_widget, "QTableWidgetItem", lambda *a: None)
    return tmp_path


def _fit_file(root):
    return root / "subject" / slice_config.SampleDirs.QUANTIFIED / f"{SAMPLE}_atlas_fit.json"


def _read_fit(root):
    """The saved fit record, failing as an assertion when there is none."""
    path = _fit_file(root)
    assert path.exists(), f"no atlas fit file beside the counts: {path.name}"
    return json.loads(path.read_text(encoding="utf-8"))


def _registration_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["run_type"] == "registration"]


def test_the_fit_is_saved_beside_the_exported_counts(quant_env):
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)

    fit = _read_fit(quant_env)
    assert fit["sample_id"] == SAMPLE
    assert fit["atlas_fit_used"] is True
    assert fit["atlas"] == "test_atlas_10um"
    assert fit["orientation"] == "coronal"
    assert fit["atlas_position"] == 640
    assert fit["transform"]["offset_x"] == 12
    assert fit["transform"]["offset_y"] == -7
    assert fit["transform"]["rotation"] == 3.5
    assert fit["transform"]["scale"] == 1.25
    assert fit["elastix_warp_on_screen"] is False


def test_the_fit_is_logged_in_the_tracker(quant_env):
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)

    rows = _registration_rows(quant_env / "runs.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == SAMPLE
    assert row["status"] == "completed"
    assert row["reg_atlas"] == "test_atlas_10um"
    assert row["reg_orientation"] == "coronal"
    assert row["reg_method"] == "align_tab_affine"
    assert float(row["reg_ap_position"]) == 6400.0     # slice 640 x 10 um
    assert row["reg_transform_path"].endswith(f"{SAMPLE}_atlas_fit.json")
    assert json.loads(row["notes"])["transform"]["scale"] == 1.25


def _says_complete(notes_text):
    """True if a row's notes claim, anywhere, that the counts are complete."""
    return "complete" in notes_text.lower()


def test_a_failed_counts_tracker_row_does_not_say_the_counts_are_complete(quant_env, msg_box):
    # The row's notes are written when the count starts and never rewritten,
    # so they must not carry a verdict on the counts: a failed run's row said
    # "counts_status": "complete" beside status = failed.
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=False)

    rows = _registration_rows(quant_env / "runs.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert not _says_complete(rows[0]["notes"])
    # The fit itself is still there to be read.
    assert json.loads(rows[0]["notes"])["transform"]["scale"] == 1.25


def test_a_count_that_never_reported_back_does_not_say_the_counts_are_complete(quant_env):
    # Window closed or crashed while counting: the row stays 'started'.
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)

    rows = _registration_rows(quant_env / "runs.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "started"
    assert not _says_complete(rows[0]["notes"])
    assert json.loads(rows[0]["notes"])["atlas_position"] == 640


def test_an_in_progress_or_incomplete_fit_file_holds_no_nested_claim_of_complete(quant_env, msg_box):
    # First run of a sample (no earlier fit file): while counting, the file
    # says in_progress and nothing inside it says complete.
    w = _widget(quant_env, export=True)
    quant_dir = _fit_file(quant_env).parent
    BrainSliceWidget._run_quantification(w)
    rec = _read_fit(quant_env)
    assert rec["counts_status"] == "in_progress"
    assert "counts_status" not in rec["run_record"]
    assert rec["run_record"]["transform"]["scale"] == 1.25

    # The run rewrites a count file and then fails: the file says incomplete
    # and the failed run's copy inside it carries no verdict either.
    (quant_dir / f"{SAMPLE}_cells.csv").write_text("label\n1\n")
    _report(w, success=False)
    rec = _read_fit(quant_env)
    assert rec["counts_status"] == "incomplete"
    assert "counts_status" not in rec["failed_run_record"]
    assert rec["failed_run_record"]["transform"]["scale"] == 1.25


def test_before_the_counts_exist_no_file_claims_a_fit(quant_env):
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)

    path = _fit_file(quant_env)
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["counts_status"] == "in_progress"
        assert rec["atlas_fit_used"] is None
        assert "transform" not in rec
    rows = _registration_rows(quant_env / "runs.csv")
    assert [r["status"] for r in rows] == ["started"]


def _export_counts(root):
    """What the counting thread does: write the three count files."""
    import pandas as pd
    from mousebrain.plugin_2d.sliceatlas.core.quantification import RegionQuantifier
    cells = pd.DataFrame({"label": [1, 2], "is_positive": [True, False],
                          "region_id": [5, 5], "region_name": ["R", "R"]})
    regions = pd.DataFrame({"region_id": [5], "region_name": ["R"], "total_cells": [2],
                            "positive_cells": [1], "negative_cells": [1],
                            "positive_fraction": [0.5]})
    RegionQuantifier().export_results(cells, regions, _fit_file(root).parent, SAMPLE)


def test_counts_exported_but_never_reported_leave_no_file_claiming_the_old_fit(quant_env):
    # Run 1 finishes normally: its fit file says scale 1.25, complete.
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _export_counts(quant_env)
    _report(w, success=True)
    assert _read_fit(quant_env)["transform"]["scale"] == 1.25

    # Run 2 uses another fit. The counting thread exports the new counts, then
    # the window is closed (or napari crashes) before its report is handled:
    # finishing never runs.
    w.alignment_widget.transform_scale = 2.0
    BrainSliceWidget._run_quantification(w)
    _export_counts(quant_env)

    path = _fit_file(quant_env)
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec.get("counts_status") != "complete", \
            "a fit file still claims a complete fit beside counts it did not make"
        assert "transform" not in rec
        assert rec.get("atlas_fit_used") is not True


def test_an_earlier_fit_file_that_cannot_be_replaced_stops_the_count(quant_env, monkeypatch, msg_box):
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)
    before = _fit_file(quant_env).read_bytes()

    # Another program holds the fit file: it can be neither replaced nor removed.
    from mousebrain.plugin_2d.sliceatlas.core import atlas_fit
    import pathlib

    def _locked_write(*_a, **_k):
        raise PermissionError("fit file is open in another program")

    real_unlink = pathlib.Path.unlink

    def _locked_unlink(self, *a, **k):
        if self.name.endswith("_atlas_fit.json"):
            raise PermissionError("fit file is open in another program")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(atlas_fit, "_write_json_atomically", _locked_write)
    monkeypatch.setattr(pathlib.Path, "unlink", _locked_unlink)
    _FakeQuantWorker.last = None
    w.alignment_widget.transform_scale = 2.0
    BrainSliceWidget._run_quantification(w)

    assert _FakeQuantWorker.last is None, "counting started beside a fit file it could not replace"
    assert _fit_file(quant_env).read_bytes() == before
    assert any(t == "Quantification not run" for _, t, _x in msg_box.shown)
    assert [r["status"] for r in _registration_rows(quant_env / "runs.csv")] == ["completed"]


def test_quantify_sample_replaces_a_fit_file_the_window_wrote(quant_env):
    import pandas as pd
    from mousebrain.plugin_2d.sliceatlas.core.quantification import quantify_sample

    # The window counted this sample with an Align-tab fit ...
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)
    assert _read_fit(quant_env)["atlas_fit_used"] is True

    # ... then a script recounts it into the same folder with its own labels.
    cells = pd.DataFrame({"label": [1, 2], "centroid_y": [1.0, 2.0],
                          "centroid_x": [1.0, 2.0], "is_positive": [True, False]})
    quantify_sample(cells, np.full((8, 8), 3, dtype=np.int32),
                    output_dir=_fit_file(quant_env).parent, sample_id=SAMPLE)

    rec = _read_fit(quant_env)
    assert rec["atlas_fit_used"] is False
    assert rec["labels_source"] == "supplied_by_caller"
    assert "transform" not in rec


def test_a_failed_count_leaves_the_previous_fit_file_and_marks_the_row_failed(quant_env):
    # Run 1 succeeds with one fit ...
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)
    assert _fit_file(quant_env).exists(), "run 1 saved no fit file"
    before = _fit_file(quant_env).read_bytes()

    # ... run 2 uses a different fit and the worker fails: the folder still
    # holds run 1's counts, so it must still hold run 1's fit.
    w.alignment_widget.transform_scale = 2.0
    w.alignment_widget.current_atlas_position = 700
    BrainSliceWidget._run_quantification(w)
    _report(w, success=False)

    assert _fit_file(quant_env).read_bytes() == before
    rows = _registration_rows(quant_env / "runs.csv")
    assert [r["status"] for r in rows] == ["completed", "failed"]
    assert json.loads(rows[1]["notes"])["atlas_position"] == 700


def test_counts_without_an_atlas_fit_replace_an_earlier_fit_file(quant_env):
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)
    assert _read_fit(quant_env)["atlas"] == "test_atlas_10um"

    # Reopened later with no atlas loaded: the counts are re-exported with the
    # all-zero labels, so the old fit file must not stay beside them.
    w2 = _widget(quant_env, export=True, atlas_loaded=False)
    BrainSliceWidget._run_quantification(w2)
    assert not _FakeQuantWorker.last.atlas_labels.any()
    _report(w2, success=True)

    rec = _read_fit(quant_env)
    assert rec.get("atlas_fit_used") is False
    assert rec["labels_source"] == "none_all_cells_unassigned"
    assert "transform" not in rec


def test_counts_with_stored_labels_say_so(quant_env):
    w = _widget(quant_env, export=True, atlas_loaded=False)
    w.atlas_labels = np.full((8, 8), 5, dtype=np.int32)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)

    rec = _read_fit(quant_env)
    assert rec["atlas_fit_used"] is False
    assert rec["labels_source"] == "stored_atlas_labels"


def test_without_export_the_fit_is_still_logged(quant_env):
    w = _widget(quant_env, export=False)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)

    assert not (quant_env / "subject").exists()
    rows = _registration_rows(quant_env / "runs.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert json.loads(rows[0]["notes"])["atlas_position"] == 640


def test_no_tracker_and_no_export_warns_the_person(quant_env, msg_box):
    w = _widget(quant_env, export=False, tracker=None)
    BrainSliceWidget._run_quantification(w)

    warnings = [s for s in msg_box.shown if s[0] == "warning"]
    assert len(warnings) == 1
    assert "not be logged" in warnings[0][2]
    assert "not be saved anywhere" in warnings[0][2]


def test_a_warp_on_screen_is_recorded_as_not_counted(quant_env):
    w = _widget(quant_env, export=True)
    w.alignment_widget.warped_annotation = np.zeros((4, 4), dtype=np.int32)
    key = getattr(w.alignment_widget, "_current_image_key", lambda: None)
    w.alignment_widget._warp_shown_for = key()
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)
    rows = _registration_rows(quant_env / "runs.csv")
    assert len(rows) == 1
    assert json.loads(rows[0]["notes"]).get("elastix_warp_on_screen") is True


def test_a_crash_while_writing_leaves_no_file_claiming_the_old_fit(quant_env, monkeypatch, msg_box):
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)
    assert _fit_file(quant_env).exists(), "run 1 saved no fit file"

    def _dies_halfway(obj, f, **kw):
        f.write('{"sample_id": ')
        raise OSError("disk went away")

    monkeypatch.setattr(json, "dump", _dies_halfway)
    w.alignment_widget.transform_scale = 3.0
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)

    # The new counts were made at scale 3.0; the old file said 1.25. It must
    # not stay beside them, and no half-written file may be left either.
    assert not _fit_file(quant_env).exists()
    assert list(_fit_file(quant_env).parent.glob("*.tmp")) == []
    assert any(s[1] == "Atlas fit not saved" for s in msg_box.shown)


def test_a_tracker_that_cannot_be_written_still_gets_the_fit_file_right(quant_env, monkeypatch, msg_box):
    # Run 1: everything works, the fit file says scale 1.25.
    w = _widget(quant_env, export=True)
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)
    assert _read_fit(quant_env)["transform"]["scale"] == 1.25

    # Run 2: the tracker CSV is locked (open in a spreadsheet program), the
    # fit is different, and the counts succeed.
    def _locked(*_a, **_kw):
        raise PermissionError("tracker CSV is open in another program")

    monkeypatch.setattr(w.tracker, "log_registration", _locked)
    w.alignment_widget.transform_scale = 2.0
    w.alignment_widget.current_atlas_position = 900
    BrainSliceWidget._run_quantification(w)
    _report(w, success=True)

    rec = _read_fit(quant_env)
    assert rec["transform"]["scale"] == 2.0
    assert rec["atlas_position"] == 900
    warnings = [s for s in msg_box.shown if s[0] == "warning"]
    assert any(t == "Atlas fit not logged" and "still be saved" in x for _, t, x in warnings)


def test_a_failed_count_that_rewrote_some_count_files_says_the_folder_is_incomplete(quant_env, msg_box):
    # Run 1 exports counts and its fit file.
    w = _widget(quant_env, export=True)
    quant_dir = _fit_file(quant_env).parent
    BrainSliceWidget._run_quantification(w)
    (quant_dir / f"{SAMPLE}_cells.csv").write_text("label\n1\n")
    (quant_dir / f"{SAMPLE}_regions.csv").write_text("region_id\n1\n")
    _report(w, success=True)

    # Run 2, a different fit, rewrites cells.csv and then fails (as an export
    # that died part-way would).
    w.alignment_widget.transform_scale = 2.0
    BrainSliceWidget._run_quantification(w)
    (quant_dir / f"{SAMPLE}_cells.csv").write_text("label\n1\n2\n3\n")
    _report(w, success=False)

    rec = _read_fit(quant_env)
    assert rec["counts_status"] == "incomplete"
    assert rec["atlas_fit_used"] is None
    assert "transform" not in rec
    assert any(s[1] == "Counts incomplete" for s in msg_box.shown)


def test_editing_the_sample_id_during_a_count_never_touches_another_samples_fit(quant_env, monkeypatch, msg_box):
    # Another sample in the same results folder has its own, correct fit file.
    w = _widget(quant_env, export=True)
    quant_dir = _fit_file(quant_env).parent
    quant_dir.mkdir(parents=True, exist_ok=True)
    other = quant_dir / "SUBJ_X_S04_atlas_fit.json"
    other.write_text('{"sample_id": "SUBJ_X_S04"}', encoding="utf-8")
    # This sample has a fit file from an earlier run.
    _fit_file(quant_env).write_text('{"sample_id": "%s"}' % SAMPLE, encoding="utf-8")

    # The fit cannot be captured this time, so finishing must remove the stale
    # fit file of the sample the counts were exported for. (Where the fit
    # module does not exist nothing captures a fit at all, which must end the
    # same way.)
    try:
        from mousebrain.plugin_2d.sliceatlas.core import atlas_fit
    except ImportError:
        atlas_fit = None

    def _cannot_capture(*_a, **_k):
        raise RuntimeError("fit unreadable")

    if atlas_fit is not None:
        monkeypatch.setattr(atlas_fit, "begin_fit_record", _cannot_capture)
    BrainSliceWidget._run_quantification(w)
    # While the count runs, the person types another sample's name.
    w.sample_id_edit = SimpleNamespace(text=lambda: "SUBJ_X_S04")
    _report(w, success=True)

    assert other.exists(), "another sample's fit file was deleted"
    assert not _fit_file(quant_env).exists(), "the stale fit file stayed beside the new counts"


def test_a_failed_count_whose_fit_file_cannot_be_written_does_not_say_the_counts_were_made(quant_env, monkeypatch, msg_box):
    w = _widget(quant_env, export=True)
    quant_dir = _fit_file(quant_env).parent
    BrainSliceWidget._run_quantification(w)
    (quant_dir / f"{SAMPLE}_cells.csv").write_text("label\n1\n")
    _report(w, success=True)

    # The next count rewrites a count file, then fails; the 'incomplete'
    # record cannot be written either.
    BrainSliceWidget._run_quantification(w)
    (quant_dir / f"{SAMPLE}_cells.csv").write_text("label\n1\n2\n")

    def _dies(obj, f, **kw):
        raise OSError("disk went away")

    monkeypatch.setattr(json, "dump", _dies)
    _report(w, success=False)

    texts = [x for _, t, x in msg_box.shown if t == "Atlas fit not saved"]
    assert texts, "no warning about the fit file"
    assert not any("counts were made" in x for x in texts)
    assert any(x.startswith("The count failed") for x in texts)
