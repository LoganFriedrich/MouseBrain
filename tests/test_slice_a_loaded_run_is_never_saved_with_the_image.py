"""A run loaded from the Run History is shown, never saved with the image,
and the saved settings name the colocalization the saved measurements came
from.

A loaded run brings back its measurements without its nuclei. Saved with the
image, it became a measurements.csv with no nuclei.npz beside it; every later
opening refused it ("has no saved nuclei") and kept it, so no later
colocalization of that image could be saved again. The settings also named
whatever run the last Colocalize click had made -- on another image, or a run
that failed -- instead of the run the measurements came from.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from test_slice_saved_analysis_is_one_state import _open
from slice_widget_helpers import finish_load

SAMPLE = "PRJ_01_01_S1"


def _nuclei():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:4, 2:4] = 1
    return labels


def _measurements(label):
    return pd.DataFrame({"label": [label], "centroid_y": [2.0], "centroid_x": [2.0],
                         "is_positive": [True], "fold_change": [3.0]})


class _Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text

    def row(self):
        return 0


def _load_history_run(w, tmp_path, label=7):
    """Load Selected Run on a completed colocalization run of this sample."""
    csv_path = tmp_path / f"history_{label}.csv"
    _measurements(label).to_csv(csv_path, index=False)
    run_id = w.tracker.log_colocalization(sample_id=SAMPLE)
    w.tracker.update_status(run_id, status="completed", measurements_path=str(csv_path),
                            coloc_background_value=10.0)
    w.run_history_table = SimpleNamespace(selectedItems=lambda: [_Item(run_id)],
                                          item=lambda r, c: _Item(run_id))
    w._visualize_colocalization = lambda m: None
    w._update_diagnostic_plot = lambda: None
    w._load_selected_run()
    assert list(w.cell_measurements["label"]) == [label]      # it is on screen
    return run_id


def _folder(image):
    return image.parent / f"{SAMPLE}_analysis"


def _settings(image):
    return json.loads((_folder(image) / "settings.json").read_text())


def test_a_loaded_run_on_an_image_with_no_nuclei_writes_no_measurements(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    _load_history_run(w, tmp_path)
    w._pa_results = pd.DataFrame({"label": [1], "area": [10]})   # something to save

    w._leave_current_image()
    assert not (_folder(image) / "measurements.csv").exists()
    assert _settings(image)["coloc_run_id"] is None


def test_a_loaded_run_never_replaces_the_images_own_measurements(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    w.nuclei_labels = _nuclei()
    w._labels_run_id = "det-own"
    w.cell_measurements = _measurements(1)
    w._coloc_run_id = w._measurements_run_id = "coloc-own"
    w._save_analysis_state()

    w2, _ = _open(tmp_path, monkeypatch)
    assert list(w2.cell_measurements["label"]) == [1]
    _load_history_run(w2, tmp_path, label=7)
    w2._leave_current_image()

    saved = pd.read_csv(_folder(image) / "measurements.csv")
    assert list(saved["label"]) == [1]
    assert _settings(image)["coloc_run_id"] == "coloc-own"
    assert (_folder(image) / "nuclei.npz").exists()


def test_the_saved_run_id_is_the_run_the_measurements_came_from(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    w.nuclei_labels = _nuclei()
    w.cell_measurements = _measurements(1)
    w._coloc_run_id = w._measurements_run_id = "coloc-1"
    w._save_analysis_state()

    w2, _ = _open(tmp_path, monkeypatch)
    # A new Colocalize click logs its row and then fails: the measurements
    # on screen are still those of coloc-1.
    w2._coloc_run_id = "coloc-2"
    w2._save_analysis_state()
    assert _settings(image)["coloc_run_id"] == "coloc-1"


def test_the_next_image_does_not_name_the_last_images_colocalization(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    w.nuclei_labels = _nuclei()
    w.cell_measurements = _measurements(1)
    w._coloc_run_id = w._measurements_run_id = "coloc-of-S1"

    other = image.parent / "PRJ_01_01_S2.tif"
    other.write_bytes(b"")
    w._open_image(other, peek=False)
    finish_load(w)
    assert w.current_file == other
    # A detection on the new image, no colocalization yet.
    w.nuclei_labels = _nuclei()
    w._save_analysis_state()

    settings = json.loads((other.parent / "PRJ_01_01_S2_analysis" / "settings.json").read_text())
    assert settings["coloc_run_id"] is None
