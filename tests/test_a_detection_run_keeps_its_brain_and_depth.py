"""A Setup & Tuning detection run keeps its own record from start to finish,
and its file holds positions counted from the first plane of the whole stack.

* The finish handler took the run's folder, tracker id and Z offset from the
  widget's fields; selecting another brain while the run went resets those,
  so the cells were not saved and the tracker row stayed 'started'.
* A test of part of the stack saved positions counted from the first plane
  TESTED; the offset was added only to the points drawn right after the run,
  so every reload of the run put its cells too shallow.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

napari = pytest.importorskip("napari")
from brainglobe_utils.cells.cells import Cell  # noqa: E402

import mousebrain.plugin_3d.tuning_widget as tw  # noqa: E402
from mousebrain.brain_context import reset_per_brain_state, tag_layer  # noqa: E402
from tuning_widget_stand_in import BRAIN_A, BRAIN_B, _tracker, _widget  # noqa: E402

Z_START, Z_END = 3, 7


class _Value:
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v


class _Check:
    def __init__(self, v):
        self.v = v

    def isChecked(self):
        return self.v


class _Btn:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, v):
        self.enabled = v

    def setText(self, _t):
        pass


class _Sig:
    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)

    def emit(self, *a):
        for fn in self.slots:
            fn(*a)


class _FakeDetection:
    """Stands in for DetectionWorker: started, never run; finished on demand."""
    made = []

    def __init__(self, signal, background, voxels, params):
        self.signal = signal
        self.progress, self.finished = _Sig(), _Sig()
        self.running = False
        _FakeDetection.made.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running


def _start_run(tmp_path, monkeypatch):
    monkeypatch.setattr(tw, "DATA_SUMMARY_DIR", tmp_path / "summary")
    monkeypatch.setattr(tw, "DetectionWorker", _FakeDetection)
    _FakeDetection.made = []
    tracker = _tracker(tmp_path, [])
    w = _widget(tmp_path, monkeypatch, tracker, brain=BRAIN_A)
    brain_a = w.current_brain
    for name in ("Signal", "Background"):
        layer = napari.layers.Image(np.zeros((10, 4, 4), np.uint16), name=f"{BRAIN_A} - {name}")
        tag_layer(layer, brain_a, brain_a / "2_Cropped_For_Registration")
        w.viewer.layers.append(layer)
    w.viewer.dims = SimpleNamespace(current_step=(5, 0, 0))
    w.metadata = {"voxel_size_um": {"x": 4, "y": 4, "z": 5}}
    w.scope_current, w.scope_range, w.scope_all = _Check(False), _Check(True), _Check(False)
    w.test_z_start, w.test_z_end = _Value(Z_START), _Value(Z_END)
    for name, v in (("ball_xy", 6), ("ball_z", 15), ("ball_overlap_fraction", 0.6),
                    ("soma_diameter", 16), ("soma_spread_factor", 1.4), ("threshold", 10),
                    ("log_sigma_size", 0.2), ("max_cluster_size", 100000), ("n_free_cpus", 2)):
        setattr(w, name, _Value(v))
    w.test_detection_btn = _Btn()
    w.brain_combo = _Btn()
    w.last_run_id = None
    w._runs_in_progress = set()
    w._refresh_prefilter_runs = lambda: None
    w._generate_detection_qc_image = lambda: None
    w.CELL_COLORS = tw.TuningWidget.CELL_COLORS
    w.run_test_detection()
    assert len(_FakeDetection.made) == 1, w.msgs.seen
    return w, tracker, _FakeDetection.made[0]


def _finish(worker, cells):
    worker.running = False
    worker.finished.emit(True, "done", len(cells), cells)


def test_a_run_finishing_after_another_brain_is_selected_is_still_saved(tmp_path, monkeypatch):
    w, tracker, worker = _start_run(tmp_path, monkeypatch)
    exp_id = w.last_run_id
    held = w.brain_combo.enabled is False  # the list is held while the run goes

    # Another brain is selected while the run goes (on_brain_changed).
    w.current_brain = w.current_brain.parent / BRAIN_B
    reset_per_brain_state(w)

    _finish(worker, [Cell([1, 2, 1], Cell.UNKNOWN), Cell([2, 3, 1], Cell.UNKNOWN)])
    row = tracker.get_experiment(exp_id)
    assert row["status"] == "completed"
    assert row["output_path"].endswith("detected_cells.xml")
    assert row["brain"] == BRAIN_A
    assert held and w.brain_combo.enabled is True
    # nothing of brain A's run drawn over brain B
    assert not [l for l in w.viewer.layers if isinstance(l, napari.layers.Points)]


def test_a_partial_run_is_saved_and_reloaded_at_its_depth(tmp_path, monkeypatch):
    w, tracker, worker = _start_run(tmp_path, monkeypatch)
    exp_id = w.last_run_id
    _finish(worker, [Cell([1, 2, 1], Cell.UNKNOWN), Cell([2, 3, 2], Cell.UNKNOWN)])

    # what was drawn right after the run: slice Z + Z_START
    drawn = [l for l in w.viewer.layers if isinstance(l, napari.layers.Points)][-1]
    assert sorted(drawn.data[:, 0].tolist()) == [Z_START + 1, Z_START + 2]

    # the file holds the same positions (whole-stack Z)
    from brainglobe_utils.IO.cells import get_cells
    row = tracker.get_experiment(exp_id)
    saved = sorted(c.z for c in get_cells(row["output_path"]))
    assert saved == [Z_START + 1, Z_START + 2]

    # and loading the run again puts its cells at the same depth (not twice)
    w._load_historical_run(exp_id)
    reloaded = w.viewer.layers[-1]
    assert sorted(reloaded.data[:, 0].tolist()) == [Z_START + 1, Z_START + 2]


@pytest.mark.parametrize("row", [
    {"det_scope": "partial", "det_z_start": "40", "det_z_end": "90"},
    # logged before det_scope existed: the range is in the notes
    {"det_scope": "", "notes": "Z40-90"},
], ids=["det_scope", "notes_only"])
def test_a_partial_run_saved_the_old_way_is_shifted_when_loaded(tmp_path, monkeypatch, row):
    """A file written before run files held whole-stack Z has no marker in its
    folder; its cells are moved down by the run's first plane when loaded."""
    run_dir = tmp_path / "optimization_runs" / BRAIN_B / "test_old"
    run_dir.mkdir(parents=True)
    from brainglobe_utils.IO.cells import save_cells
    save_cells([Cell([1, 2, 1], Cell.UNKNOWN), Cell([2, 3, 2], Cell.UNKNOWN)],
               str(run_dir / "detected_cells.xml"))
    tracker = _tracker(tmp_path, [dict({"exp_id": "old_partial", "brain": BRAIN_B,
                                        "output_path": str(run_dir / "detected_cells.xml")},
                                       **row)])
    w = _widget(tmp_path, monkeypatch, tracker, brain=BRAIN_B)
    w.CELL_COLORS = tw.TuningWidget.CELL_COLORS
    w._load_historical_run("old_partial")
    assert sorted(w.viewer.layers[-1].data[:, 0].tolist()) == [41, 42]
    w._load_detection_results(best=False)
    assert sorted(w.viewer.layers[-1].data[:, 0].tolist()) == [41, 42]
