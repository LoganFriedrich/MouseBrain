"""A saved analysis is restored only onto pixels loaded the way it was made.

The restore used to apply the saved rotation, z-projection and channel
numbers AFTER the image had loaded with whatever the controls held, so
labels saved on a turned image were drawn on an unturned one (a square
image keeps its shape, so nothing showed it), and the restored settings
silently carried to the next image's load. Now the saved load settings are
set before the image loads, and a restore onto pixels loaded another way
(a reload with channels changed by hand) is refused and its files kept.
"""
import json

import numpy as np
import pandas as pd

from slice_widget_helpers import StubLoader, finish_load, make_loading_widget


class FakeCombo:
    """A QComboBox holding a fixed list of choices."""

    def __init__(self, items, current=0):
        self.items, self.index = list(items), current

    def currentText(self):
        return self.items[self.index]

    def findText(self, text):
        return self.items.index(text) if text in self.items else -1

    def setCurrentIndex(self, index):
        self.index = index


class FakeSpin:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value


ROTATIONS = ["None", "90 CCW", "180"]
PROJECTIONS = ["Max", "Mean"]


def _widget(tmp_path, monkeypatch):
    w = make_loading_widget(tmp_path, monkeypatch)
    w.rotation_combo = FakeCombo(ROTATIONS, 0)
    w.z_projection_combo = FakeCombo(PROJECTIONS, 0)
    w.red_channel_spin = FakeSpin(0)
    w.green_channel_spin = FakeSpin(1)
    w._pa_get_scale = lambda: (1.0, 1.0)
    w._pa_populate_table = lambda df: None
    w._pa_find_layer = lambda name: None
    w._peek_and_configure = lambda path: None   # the file holds no metadata
    return w


def _saved_analysis(tmp_path, rotation="180", red=2):
    """An image whose saved analysis was made with the image turned 180
    degrees and the nuclear channel set to 2."""
    image = tmp_path / "imgs" / "PRJ_01_01_S1.tif"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"")
    folder = image.parent / "PRJ_01_01_S1_analysis"
    folder.mkdir()
    loaded_with = {"rotation": rotation, "z_projection": "Mean",
                   "red_channel_idx": red, "green_channel_idx": 1}
    (folder / "settings.json").write_text(json.dumps(
        dict(loaded_with, loaded_with=loaded_with)))
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[0:2, 0:2] = 1
    np.savez_compressed(folder / "labels.npz", labels=labels)
    pd.DataFrame({"label": [1], "area": [4]}).to_csv(folder / "particles.csv", index=False)
    return image, folder


def test_the_saved_load_settings_are_set_before_the_image_loads(tmp_path, monkeypatch):
    image, _ = _saved_analysis(tmp_path)
    w = _widget(tmp_path, monkeypatch)
    w._open_image(image)

    # The load was started with the saved analysis's channels ...
    loader = StubLoader.made[-1]
    assert loader.kwargs["red_idx"] == 2
    # ... and turned the way the analysis was made.
    finish_load(w)
    assert w._loaded_with["rotation"] == "180"
    assert w._pa_labels is not None


def test_a_restore_onto_pixels_loaded_another_way_is_refused_and_kept(tmp_path, monkeypatch):
    image, folder = _saved_analysis(tmp_path)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    w = _widget(tmp_path, monkeypatch)
    # A reload with the Load button keeps what the controls hold (no
    # rotation, nuclear channel 0): not what the analysis was made on.
    w._open_image(image, peek=False)
    finish_load(w)

    assert w._pa_labels is None and w._pa_results is None
    status = w.status_label.text()
    assert "NOT restored" in status and "rotation 180" in status

    # The controls were not changed behind the person's back ...
    assert w.rotation_combo.currentText() == "None"
    # ... and leaving the image keeps the saved analysis as it was.
    w._save_analysis_state()
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_labels_of_another_size_are_not_drawn_on_the_image(tmp_path, monkeypatch):
    image, folder = _saved_analysis(tmp_path, rotation="None", red=0)
    w = _widget(tmp_path, monkeypatch)
    w._open_image(image)
    finish_load(w, shape=(8, 12))     # the file now holds another image size

    assert w._pa_labels is None
    assert "labels.npz" in w.status_label.text()


def test_a_load_tab_changed_to_the_saved_settings_is_said_on_screen(tmp_path, monkeypatch):
    image, _ = _saved_analysis(tmp_path)
    w = _widget(tmp_path, monkeypatch)
    w._open_image(image)
    finish_load(w)

    status = w.status_label.text()
    assert "Load tab set" in status and "rotation 180" in status
    assert "kept for the next image" in status
