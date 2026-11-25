"""A save made while the next image is loading never removes or empties the
saved analysis of the image being left.

Opening image B saves image A's results and clears them from memory, but
the window keeps naming A until B has loaded. A save in between -- Save
State, or the auto-save at the end of a particle analysis whose tracker
write lets a waiting Load click through -- must not see A's folder as the
window's own with nothing in memory and delete A's particles, labels, nuclei
and measurements and empty its ROIs while the window says 'State saved'.
"""
import json

import numpy as np
import pandas as pd

from slice_widget_helpers import Control, finish_load, image_files, make_loading_widget


class FakeShapes:
    def __init__(self, name, shapes=()):
        self.name, self.visible, self.mode = name, True, None
        self.data = [np.asarray(s, dtype=float) for s in shapes]


SQUARE = [[1, 1], [1, 5], [5, 5], [5, 1]]


def _widget(tmp_path, monkeypatch):
    w = make_loading_widget(tmp_path, monkeypatch)
    w._pa_get_scale = lambda: (1.0, 1.0)
    w._pa_populate_table = lambda df: None
    w._pa_find_layer = lambda name: None
    w._pa_draw_classification_overlay = lambda *a: None
    w._pa_scroll_to_results = lambda: None
    w._pa_register_click_callback = lambda layer: None
    w.pa_bg_manual_spin = Control(value=0)
    return w


def _a_with_saved_analysis(w, a):
    """A opened, analyzed and saved: the window holds A's saved analysis."""
    w._open_image(a, peek=False)
    finish_load(w)
    w._pa_results = pd.DataFrame({"label": [1, 2], "area": [10, 12]})
    w._pa_labels = np.ones((8, 8), dtype=np.int32)
    w.roi_shapes_layer = FakeShapes("ROIs", [SQUARE])
    w.viewer.layers.append(w.roi_shapes_layer)
    w._roi_names = ["ROI_A"]
    w._save_analysis_state()
    saved = a.parent / f"{a.stem}_analysis"
    assert (saved / "particles.csv").exists()
    return saved


def _a_is_intact(saved):
    assert (saved / "particles.csv").exists(), "A's particles.csv deleted"
    assert (saved / "labels.npz").exists(), "A's labels.npz deleted"
    rois = json.loads((saved / "rois.json").read_text())["rois"]
    assert len(rois) == 1, "A's ROIs emptied"


def test_save_state_while_the_next_image_loads_keeps_the_saved_analysis(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = _widget(tmp_path, monkeypatch)
    saved = _a_with_saved_analysis(w, a)

    w._open_image(b, peek=False)        # B is still loading
    assert w._loading_file is not None
    w._manual_save_state()               # Save State clicked meanwhile

    _a_is_intact(saved)
    assert "State saved" not in w.status_label.text()


def test_save_state_is_off_while_an_image_loads(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = _widget(tmp_path, monkeypatch)
    _a_with_saved_analysis(w, a)

    w._open_image(b, peek=False)
    assert not w._img_save_state_btn.isEnabled()
    assert not w._roi_save_state_btn.isEnabled()
    finish_load(w)
    assert w._img_save_state_btn.isEnabled()
    assert w._roi_save_state_btn.isEnabled()


def test_any_save_between_leaving_and_loading_removes_nothing(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = _widget(tmp_path, monkeypatch)
    saved = _a_with_saved_analysis(w, a)

    w._open_image(b, peek=False)
    w._save_analysis_state()             # e.g. an analysis's auto-save

    _a_is_intact(saved)


def test_a_load_click_during_a_particle_analysis_waits_for_it(tmp_path, monkeypatch):
    a, b = image_files(tmp_path / "imgs", ["SAMPLE_A.tif", "SAMPLE_B.tif"])
    w = _widget(tmp_path, monkeypatch)
    saved = _a_with_saved_analysis(w, a)

    image = np.zeros((40, 40))
    image[5:12, 5:12] = 100.0
    w.channels = [image, image * 2]
    w.pa_det_combo = Control(text="ch0", index=0)
    w.pa_meas_combo = Control(text="ch1", index=1)
    w._pa_thresh_spin = Control(value=50)
    w.pa_min_area = Control(value=4)
    w.pa_max_area = Control(value=1000)
    w.pa_min_circ = Control(value=0.0)
    w.pa_watershed_check = Control(checked=False)
    w.pa_pos_pct_spin = Control(value=50.0)
    w.pa_binary_toggle = Control()
    w.pa_show_signal_mask = Control()
    w._pa_binary_view = False

    # The tracker write lets the window answer clicks; a Load click for B
    # was waiting.
    opened = []

    def answer_clicks():
        opened.append(w._open_image(b, peek=False))
    w._saving_to_tracker = answer_clicks

    w._pa_run_analysis()

    assert opened == [False], "B was opened in the middle of the analysis"
    assert w._loading_file is None
    _a_is_intact(saved)
    # The analysis's own result was saved for A, the image it ran on.
    particles = pd.read_csv(saved / "particles.csv")
    assert len(particles) == 1
    assert w._analysis_running is None   # B can be opened now
