"""No new result is saved beside a settings.json that does not describe it.

An image's saved analysis is not restored onto the image loaded with
another rotation, z-projection or channel choice, and its files are kept.
But a detection or colocalization on the differently loaded pixels then
wrote NEW files (nuclei.npz, measurements.csv) into the same folder, beside
the OLD settings.json. Opened again, the image was loaded with the old
settings, nothing looked amiss, and the new nuclei were restored onto
pixels they were not made on whenever the image kept its shape (a channel
or z-projection change, a 180-degree turn, 90 degrees on a square image).
"""
import json

import numpy as np
import pandas as pd

from slice_widget_helpers import finish_load, image_files, make_loading_widget

SAVED_WITH = {"rotation": "None", "z_projection": "max",
              "red_channel_idx": 0, "green_channel_idx": 1}
TURNED = dict(SAVED_WITH, rotation="180")


def _widget(tmp_path, monkeypatch):
    w = make_loading_widget(tmp_path, monkeypatch)
    w._pa_get_scale = lambda: (1.0, 1.0)
    w._pa_populate_table = lambda df: None
    w._pa_find_layer = lambda name: None
    w._pa_draw_classification_overlay = lambda *a: None
    w._pa_register_click_callback = lambda layer: None
    return w


def _saved_then_reloaded_turned(tmp_path, monkeypatch):
    """Image A saved with one Load setting, then reloaded (Load button)
    turned 180 degrees: the restore is refused."""
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = _widget(tmp_path, monkeypatch)
    w._load_settings_now = lambda: dict(SAVED_WITH)
    w._open_image(a, peek=False)
    finish_load(w)
    w._pa_results = pd.DataFrame({"label": [1], "area": [10]})
    w._pa_labels = np.ones((8, 8), dtype=np.int32)
    w._save_analysis_state()

    w._load_settings_now = lambda: dict(TURNED)
    w._open_image(a, peek=False)
    finish_load(w)
    assert "NOT restored" in w.status_label.text()
    folder = a.parent / "SAMPLE_A_analysis"
    return w, a, b, folder


def test_new_nuclei_are_not_saved_beside_the_old_settings(tmp_path, monkeypatch):
    w, a, b, folder = _saved_then_reloaded_turned(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    # A detection and a colocalization on the turned pixels, then a save.
    w.nuclei_labels = np.full((8, 8), 7, dtype=np.int32)
    w.cell_measurements = pd.DataFrame({"label": [7], "is_positive": [True]})
    problem = w._save_analysis_state()

    assert not (folder / "nuclei.npz").exists(), \
        "nuclei made on the turned image saved beside settings saying not turned"
    assert not (folder / "measurements.csv").exists()
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert problem and "nuclei.npz" in problem
    assert "NOT saved" in w.not_saved_label.text()


def test_leaving_such_an_image_asks_and_keeps_its_results_by_default(tmp_path, monkeypatch):
    w, a, b, folder = _saved_then_reloaded_turned(tmp_path, monkeypatch)
    w.nuclei_labels = np.full((8, 8), 7, dtype=np.int32)
    asked = []
    w._ask_leave_unsaved = lambda problem: asked.append(problem) or False

    assert w._open_image(b, peek=False) is False
    assert asked and "nuclei.npz" in asked[0]
    assert w.current_file == a and w.nuclei_labels is not None
    assert not (folder / "nuclei.npz").exists()


def test_the_saved_analysis_still_comes_back_as_it_was_made(tmp_path, monkeypatch):
    w, a, b, folder = _saved_then_reloaded_turned(tmp_path, monkeypatch)
    w.nuclei_labels = np.full((8, 8), 7, dtype=np.int32)
    w._ask_leave_unsaved = lambda problem: True     # leave, losing the nuclei
    w._open_image(b, peek=False)
    finish_load(w)

    # Opened again with the settings it was saved with.
    w._load_settings_now = lambda: dict(SAVED_WITH)
    w._open_image(a, peek=False)
    finish_load(w)

    assert w._pa_results is not None and len(w._pa_results) == 1
    assert w.nuclei_labels is None, "nuclei of the turned image restored"
    saved = json.loads((folder / "settings.json").read_text())
    assert saved["loaded_with"]["rotation"] == "None"


def test_an_unreadable_settings_file_gets_advice_that_fits(tmp_path, monkeypatch):
    """Reloading another way cannot help when settings.json cannot be read:
    the message says the results can be exported and a person has to repair
    or move the file."""
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = _widget(tmp_path, monkeypatch)
    w._load_settings_now = lambda: dict(SAVED_WITH)
    w._open_image(a, peek=False)
    finish_load(w)
    w._pa_results = pd.DataFrame({"label": [1], "area": [10]})
    w._pa_labels = np.ones((8, 8), dtype=np.int32)
    w._open_image(b, peek=False)        # leaving A saves it
    finish_load(w)
    folder = a.parent / "SAMPLE_A_analysis"
    (folder / "settings.json").write_text("{ not json")

    w._open_image(a, peek=False)
    finish_load(w)
    w.nuclei_labels = np.full((8, 8), 7, dtype=np.int32)
    problem = w._save_analysis_state()

    assert problem and "nuclei.npz" in problem
    assert "cannot be read" in problem and "Export" in problem
    assert "repair" in problem and "move it aside" in problem
    assert "earlier analysis that could not be restored" not in problem
    assert not (folder / "nuclei.npz").exists()
