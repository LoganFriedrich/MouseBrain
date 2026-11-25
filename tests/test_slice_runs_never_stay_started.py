"""No 2D tracker row stays 'started', and a recorded result is never overwritten.

A row left at 'started' forever cannot be told apart from a run still going.
So the widget does not log rows before its input checks and abandon them
there, and errors in its result handlers must not end with the row still
'started'. The batch must not set the DETECTION row to 'error' on any later
exception (that would overwrite a detection that had completed), and must
mark the colocalization row it is really about. 'error' and 'failed' are one
state.
"""
import contextlib
import csv
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from example_method import PROCESS_SINGLE_VALUES
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr
from mousebrain.plugin_2d.sliceatlas.core import (
    colocalization, config as slice_config, detection, io, visualization)
from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget
from mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget import BrainSliceWidget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

SAMPLE = "PRJ_01_01_S04"


def _statuses(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["run_type"]: r["status"] for r in csv.DictReader(f)}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# The tracker itself
# --------------------------------------------------------------------------

def test_a_run_that_raises_is_marked_failed_with_the_reason(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    run_id = t.log_detection(sample_id=SAMPLE)
    with pytest.raises(KeyboardInterrupt):
        with t.finish_or_fail(run_id):
            raise KeyboardInterrupt()
    row = t.get_run(run_id)
    assert row["status"] == "failed" and "KeyboardInterrupt" in row["notes"]


def test_a_run_that_ends_without_a_result_is_failed_and_a_finished_one_is_not(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    forgot = t.log_detection(sample_id=SAMPLE)
    done = t.log_detection(sample_id=SAMPLE)
    with t.finish_or_fail(forgot):
        pass
    with t.finish_or_fail(done):
        t.update_status(done, status="completed", det_nuclei_found=7)
    assert t.get_run(forgot)["status"] == "failed"
    assert t.get_run(done)["status"] == "completed"


def test_failing_an_unfinished_run_never_touches_a_completed_one(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    run_id = t.log_detection(sample_id=SAMPLE)
    t.update_status(run_id, status="completed", det_nuclei_found=7)
    assert t.fail_if_unfinished(run_id, "a later step crashed") is False
    row = t.get_run(run_id)
    assert (row["status"], row["det_nuclei_found"], row["notes"]) == ("completed", "7", "")


def test_error_is_recorded_as_failed(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    run_id = t.log_detection(sample_id=SAMPLE)
    t.update_status(run_id, status="error")
    assert t.get_run(run_id)["status"] == "failed"


# --------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------

# process_single has no built-in nucleus size range; any range will do here,
# since the detector is replaced.
# Every value process_single needs (it has no built-in values).
SIZE_RANGE = dict(PROCESS_SINGLE_VALUES)


@pytest.fixture
def batch_env(tmp_path, monkeypatch):
    """process_single with the image reading and detection replaced, and the
    tracker pointed at a temporary CSV."""
    csv_path = tmp_path / "runs.csv"
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda: SliceTracker(csv_path=csv_path))
    monkeypatch.setattr(io, "load_image",
                        lambda p: (np.zeros((2, 8, 8)), {"voxel_size_um": {"x": 1.0}}))
    monkeypatch.setattr(io, "extract_channels",
                        lambda data, red_idx=0, green_idx=1: (np.zeros((8, 8)), np.zeros((8, 8))))
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:4, 2:4] = 1
    monkeypatch.setattr(detection, "detect_with_log_augmentation",
                        lambda red, pixel_um=1.0, **k: (labels, {"filtered_count": 1}))
    monkeypatch.setattr(batch_encr, "DATA_DIR", tmp_path / "data", raising=False)
    nd2 = tmp_path / f"{SAMPLE}.nd2"
    nd2.write_bytes(b"")
    return SimpleNamespace(csv=csv_path, nd2=nd2, out=tmp_path / "out")


def test_a_colocalization_crash_fails_the_colocalization_run_not_the_finished_detection(batch_env, monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("colocalization crashed")
    monkeypatch.setattr(colocalization, "ColocalizationAnalyzer", Boom)
    result = batch_encr.process_single(batch_env.nd2, batch_env.out, **SIZE_RANGE)
    assert result["status"] == "error"
    assert _statuses(batch_env.csv) == {"detection": "completed", "colocalization": "failed"}
    coloc = [r for r in _rows(batch_env.csv) if r["run_type"] == "colocalization"][0]
    assert "colocalization crashed" in coloc["notes"]


class _FakeAnalyzer:
    """Colocalization that succeeds with one negative cell."""
    def __init__(self, *a, **k):
        pass

    def estimate_background(self, *a, **k):
        return 1.0

    def estimate_tissue_mask(self, labels, *a, **k):
        return labels > 0

    def measure_cytoplasm_intensities(self, *a, **k):
        return pd.DataFrame({"label": [1], "mean_intensity": [0.5]})

    def classify_positive_negative(self, measurements, *a, **k):
        return measurements.assign(is_positive=False)

    def get_summary_statistics(self, classified):
        return {"positive_cells": 0, "negative_cells": 1, "positive_fraction": 0.0,
                "mean_fold_change": 0.5, "median_fold_change": 0.5,
                "background_used": 1.0}


def test_a_colocalization_whose_qc_pictures_cannot_be_saved_is_failed_not_completed(batch_env, monkeypatch):
    # The colocalization's outputs are its CSV and QC pictures. A run
    # recorded 'completed' while they were never written looked finished in
    # the run history with nothing to reload.
    monkeypatch.setattr(colocalization, "ColocalizationAnalyzer", _FakeAnalyzer)

    def unsaveable(**k):
        raise OSError("no space left for the QC pictures")
    monkeypatch.setattr(visualization, "save_all_qc_figures", unsaveable)
    result = batch_encr.process_single(batch_env.nd2, batch_env.out, **SIZE_RANGE)
    assert result["status"] == "error"
    assert _statuses(batch_env.csv) == {"detection": "completed", "colocalization": "failed"}
    coloc = [r for r in _rows(batch_env.csv) if r["run_type"] == "colocalization"][0]
    assert "no space left for the QC pictures" in coloc["notes"]


def test_a_colocalization_whose_outputs_are_saved_is_completed(batch_env, monkeypatch):
    monkeypatch.setattr(colocalization, "ColocalizationAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(visualization, "save_all_qc_figures", lambda **k: None)
    result = batch_encr.process_single(batch_env.nd2, batch_env.out, **SIZE_RANGE)
    assert result["status"] == "completed"
    assert _statuses(batch_env.csv) == {"detection": "completed", "colocalization": "completed"}


def test_ctrl_c_during_detection_leaves_no_run_started(batch_env, monkeypatch):
    def interrupted(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(detection, "detect_with_log_augmentation", interrupted)
    with pytest.raises(KeyboardInterrupt):
        batch_encr.process_single(batch_env.nd2, batch_env.out, **SIZE_RANGE)
    assert _statuses(batch_env.csv) == {"detection": "failed"}


# --------------------------------------------------------------------------
# The napari widget
# --------------------------------------------------------------------------

class _Box:
    def __init__(self):
        self.shown = []

    def warning(self, _parent, title, text, *a):
        self.shown.append((title, text))


def _widget(tmp_path):
    w = SimpleNamespace(
        tracker=SliceTracker(csv_path=tmp_path / "runs.csv"),
        status_label=SimpleNamespace(setText=lambda t: setattr(w, "status", t)),
        coloc_btn=SimpleNamespace(setEnabled=lambda b: setattr(w, "coloc_btn_enabled", b)),
        quant_btn=SimpleNamespace(setEnabled=lambda _b: None),
        coloc_result_label=SimpleNamespace(setText=lambda _t: None),
        sample_id_edit=SimpleNamespace(text=lambda: SAMPLE),
        current_file=tmp_path / f"{SAMPLE}.nd2",
        coloc_mode_combo=SimpleNamespace(currentText=lambda: "Single"),
        nuclei_labels=np.ones((8, 8), dtype=np.int32),
        green_channel=np.ones((8, 8)),
        red_channel=np.ones((8, 8)),
        last_run_id=None,
        _coloc_run_id=None,
        _labels_run_id=None,
        _detection_run_id=None,
        _analysis_running=None, _loading_file=None,
        quant_worker=None,
        session_run_ids=[],
        _unwritable_run_ids=set(),
        _unrecorded_results={},
        _unrecorded_retry_timer=None,
        _tracker_error=None,
        status="",
    )
    for name in ("bg_method_combo", "thresh_method_combo", "bg_method_combo_ch2"):
        setattr(w, name, SimpleNamespace(currentText=lambda: "gmm"))
    for name in ("bg_percentile_spin", "thresh_value_spin", "sigma_threshold_spin",
                 "bg_dilation_spin", "area_fraction_spin", "bg_block_size_spin",
                 "soma_dilation_spin", "thresh_value_spin_ch2", "bg_dilation_spin_ch2",
                 "soma_dilation_spin_ch2"):
        setattr(w, name, SimpleNamespace(value=lambda: 1))
    w.bg_local_check = SimpleNamespace(isChecked=lambda: False)
    for name in ("_fail_run_if_unfinished", "_result_not_recorded", "_saving_to_tracker",
                 "_schedule_unrecorded_retry", "_retry_unrecorded_results",
                 "_open_tracker", "_run_colocalization", "_run_dual_colocalization",
                 "_on_dual_coloc_finished", "_on_coloc_finished", "_coloc_could_not_start",
                 "_refuse_if_analysis_running", "_analysis_in_progress",
                 "_set_new_nuclei"):
        if hasattr(BrainSliceWidget, name):
            setattr(w, name, types.MethodType(getattr(BrainSliceWidget, name), w))
    return w


def test_a_colocalization_refused_by_its_input_check_leaves_no_started_row(tmp_path, monkeypatch):
    monkeypatch.setattr(main_widget, "QMessageBox", _Box())
    w = _widget(tmp_path)
    w._get_current_slice = lambda arr: None if arr is w.green_channel else arr
    BrainSliceWidget._run_colocalization(w)
    assert "started" not in _statuses(tmp_path / "runs.csv").values()


def test_a_dual_colocalization_whose_csv_cannot_be_saved_does_not_stay_started(tmp_path, monkeypatch):
    w = _widget(tmp_path)
    w._coloc_run_id = w.last_run_id = w.tracker.log_colocalization(sample_id=SAMPLE, signal_channel="dual")
    monkeypatch.setattr(slice_config, "get_sample_dir", lambda stem: tmp_path / "subject")
    w._visualize_dual_colocalization = lambda m: None

    class Unsaveable:
        def to_csv(self, *a, **k):
            raise OSError("disk full")
    summary = {"total_nuclei": 1, "n_dual": 0, "n_neither": 1}
    try:
        BrainSliceWidget._on_dual_coloc_finished(w, True, "done", Unsaveable(), summary, None)
    except OSError:
        pass
    row = w.tracker.get_run(w.last_run_id)
    assert row["status"] == "failed" and "disk full" in row["notes"]
    # The person is told it is the CSV that was not saved -- not the tracker.
    assert "measurements CSV not saved (disk full)" in w.status
    assert "run marked failed" in w.status and "tracker (" not in w.status


@contextlib.contextmanager
def _held_open_like_a_spreadsheet(path):
    """Hold `path` open the way a spreadsheet program does: others may read
    it, nobody may write, rename over or delete it."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NORMAL = 0x80000000, 1, 3, 0x80
    handle = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ, None,
                             OPEN_EXISTING, NORMAL, None)
    assert handle not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    try:
        yield
    finally:
        k32.CloseHandle(handle)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing rules")
def test_a_result_the_tracker_file_refuses_is_reported_quickly_and_the_row_is_said_to_be_started(
        tmp_path, monkeypatch):
    # The window writes on its own thread: a write that waits on a tracker
    # held open by a spreadsheet freezes it. It must give up within the
    # window's short budget, not try a second write that waits again, and
    # say that the row still reads 'started'.
    csv_path = tmp_path / "runs.csv"
    monkeypatch.setattr(tracker_pkg, "SliceTracker",
                        lambda **kw: SliceTracker(csv_path=csv_path, **kw))
    w = _widget(tmp_path)
    if hasattr(w, "_open_tracker"):
        w._open_tracker()  # the tracker exactly as the window opens it
    w._coloc_run_id = w.last_run_id = w.tracker.log_colocalization(sample_id=SAMPLE, signal_channel="dual")
    monkeypatch.setattr(slice_config, "get_sample_dir", lambda stem: tmp_path / "subject")
    w._visualize_dual_colocalization = lambda m: None
    writes = []  # whole-file rewrites (this code makes every status change one)
    rewrite = getattr(w.tracker, "_rewrite", None)
    if rewrite is not None:
        w.tracker._rewrite = lambda mutate: (writes.append(1), rewrite(mutate))[1]
    summary = {"total_nuclei": 1, "n_dual": 0, "n_neither": 1}

    started = time.monotonic()
    with _held_open_like_a_spreadsheet(csv_path):
        try:
            BrainSliceWidget._on_dual_coloc_finished(
                w, True, "done", pd.DataFrame({"label": [1]}), summary, None)
        except OSError:
            pass
    elapsed = time.monotonic() - started

    assert "not saved to the tracker" in w.status
    assert "still says 'started'" in w.status
    assert w.tracker.get_run(w.last_run_id)["status"] == "started"
    assert len(writes) <= 1, "a second write would freeze the window again"
    assert elapsed < (main_widget.GUI_TRACKER_LOCK_WAIT_SECONDS
                      + main_widget.GUI_TRACKER_REPLACE_RETRY_SECONDS)
    # Later attempts this session (an error path, a 'finally') do not wait again.
    assert BrainSliceWidget._fail_run_if_unfinished(w, w.last_run_id, "later") is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing rules")
def test_an_inset_detection_the_tracker_file_refuses_is_still_shown_and_said_not_saved(
        tmp_path, monkeypatch):
    # The inset path wrote its result inside the same try as building the
    # layers: a tracker held open by a spreadsheet skipped the layers (the
    # result was not on screen), and the error handler then tried a second
    # locked write. It must behave like the other result handlers.
    from mousebrain.plugin_2d.sliceatlas.core import inset_detection

    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:4, 2:4] = 1
    props = pd.DataFrame({"area": [4], "from_inset": [True],
                          "centroid_y_base": [3.0], "centroid_x_base": [3.0]})

    class FakePipeline:
        def __init__(self, *a, **k):
            pass

        def run_full_detection(self, **k):
            return {"merged_labels": labels, "merged_properties": props, "total_cells": 1}

    monkeypatch.setattr(inset_detection, "InsetDetectionPipeline", FakePipeline)
    monkeypatch.setattr(detection, "NucleiDetector", lambda **k: object())

    csv_path = tmp_path / "runs.csv"
    monkeypatch.setattr(tracker_pkg, "SliceTracker",
                        lambda **kw: SliceTracker(csv_path=csv_path, **kw))
    w = _widget(tmp_path)
    if hasattr(w, "_open_tracker"):
        w._open_tracker()  # the tracker exactly as the window opens it
    added = []
    w.viewer = SimpleNamespace(
        layers=[],
        add_labels=lambda data, name=None, **k: (added.append(name),
                                                  SimpleNamespace(contour=0))[1],
        add_points=lambda *a, name=None, **k: added.append(name))
    w.detect_btn = SimpleNamespace(setEnabled=lambda _b: None)
    w.detect_result_label = SimpleNamespace(setText=lambda _t: None)
    w._detection_run_id = w.last_run_id = w.tracker.log_detection(sample_id=SAMPLE, model="2D_versatile_fluo",
                                            prob_thresh=0.5, nms_thresh=0.4,
                                            min_area=1, max_area=100, status="started")
    writes = []
    rewrite = getattr(w.tracker, "_rewrite", None)
    if rewrite is not None:
        w.tracker._rewrite = lambda mutate: (writes.append(1), rewrite(mutate))[1]
    params = {"model": "2D_versatile_fluo", "prob_thresh": 0.5, "nms_thresh": 0.4,
              "filter_size": True, "min_area": 1, "max_area": 100}
    inset_settings = {"inset_manager": SimpleNamespace(insets=[object()]),
                      "detect_in_base": True}

    with _held_open_like_a_spreadsheet(csv_path):
        BrainSliceWidget._run_inset_detection(w, params, inset_settings)

    assert "Nuclei (1)" in added, "the result must be on screen"
    assert "Detected 1 nuclei" in w.status
    assert "not saved to the tracker" in w.status
    assert "still says 'started'" in w.status
    assert w.tracker.get_run(w.last_run_id)["status"] == "started"
    assert len(writes) <= 1, "a second write would freeze the window again"


class _AppendRefused(SliceTracker):
    """A tracker whose file refuses a new row (open in another program)."""

    def log_colocalization(self, *a, **k):
        raise PermissionError("the tracker file is open in another program")


def test_a_colocalization_whose_row_the_tracker_refuses_does_not_start_and_frees_the_button(
        tmp_path, monkeypatch):
    # The error must not escape the button's handler and leave the Colocalize
    # button disabled for the rest of the session.
    box = _Box()
    monkeypatch.setattr(main_widget, "QMessageBox", box)
    started = []
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import workers
    monkeypatch.setattr(workers, "ColocalizationWorker",
                        lambda *a, **k: started.append(1))
    w = _widget(tmp_path)
    w.tracker = _AppendRefused(csv_path=tmp_path / "runs.csv")
    w._get_current_slice = lambda arr: arr
    w.coloc_btn_enabled = None

    BrainSliceWidget._run_colocalization(w)

    assert w.coloc_btn_enabled is True
    assert "Colocalization not started" in w.status
    assert "open in another program" in w.status
    assert box.shown and box.shown[0][0] == "Colocalization not started"
    assert not started, "an unlogged run must not start"


def test_a_colocalization_result_that_cannot_be_handled_is_failed_not_left_started(
        tmp_path, monkeypatch):
    # A summary missing a value raised inside the result handler before the
    # tracker was updated: the error left the Qt slot, the row stayed
    # 'started' forever and nothing on screen said so.
    w = _widget(tmp_path)
    w._coloc_run_id = w.last_run_id = w.tracker.log_colocalization(sample_id=SAMPLE, signal_channel="green")
    w.coloc_btn_enabled = None
    summary = {"positive_cells": 1, "negative_cells": 0}  # no 'background_used'

    BrainSliceWidget._on_coloc_finished(w, True, "done", pd.DataFrame({"label": [1]}),
                                        summary, None)

    row = w.tracker.get_run(w.last_run_id)
    assert row["status"] == "failed" and "background_used" in row["notes"]
    assert w.status.startswith("Colocalization error")
    assert w.coloc_btn_enabled is True


def test_a_dual_colocalization_result_that_cannot_be_handled_is_failed_not_left_started(
        tmp_path, monkeypatch):
    w = _widget(tmp_path)
    w._coloc_run_id = w.last_run_id = w.tracker.log_colocalization(sample_id=SAMPLE, signal_channel="dual")
    w.current_file = None  # no CSV to save; the error comes from drawing the result

    def cannot_draw(_m):
        raise ValueError("layer could not be drawn")
    w._visualize_dual_colocalization = cannot_draw
    w.tracker.update_status = lambda *a, **k: True  # the 'completed' write never lands
    summary = {"total_nuclei": 1, "n_dual": 0, "n_neither": 1}

    BrainSliceWidget._on_dual_coloc_finished(w, True, "done", pd.DataFrame({"label": [1]}),
                                             summary, None)

    row = SliceTracker(csv_path=tmp_path / "runs.csv").get_run(w.last_run_id)
    assert row["status"] == "failed" and "layer could not be drawn" in row["notes"]
    assert w.status.startswith("Dual colocalization error")
