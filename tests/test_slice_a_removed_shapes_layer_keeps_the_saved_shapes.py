"""Shapes a person drew are not emptied on disk because their layer vanished
from napari's layer list.

When the ROIs layer of a restored image was removed from the layer list (the
list's delete button, another plugin), leaving the image overwrote the saved
rois.json with an empty list: the save could not tell a removed layer from
ROIs a person deleted, and the polygons were kept nowhere. Now a removed
layer leaves the file as it is until shapes are drawn again or the Clear
button is clicked.
"""
import json

import numpy as np

from test_slice_saved_analysis_is_one_state import SQUARE, _draw_roi, _open


def _saved_rois(image):
    folder = image.parent / "PRJ_01_01_S1_analysis"
    return json.loads((folder / "rois.json").read_text())["rois"]


def _reopened_with_rois(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    _draw_roi(w)
    w._save_analysis_state()
    w2, _ = _open(tmp_path, monkeypatch)
    assert w2.roi_shapes_layer is not None and len(w2.roi_shapes_layer.data) == 1
    # The real ROI drawing button, not the test's stand-in.
    del w2._add_roi_layer
    return w2, image


def _remove_layer(w, layer):
    w.viewer.layers.remove(layer)


def test_a_removed_rois_layer_leaves_the_saved_rois_as_they_are(tmp_path, monkeypatch):
    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    _remove_layer(w, w.roi_shapes_layer)

    w._save_analysis_state()
    assert len(_saved_rois(image)) == 1

    # Leaving the image (the save that used to empty the file) keeps them too.
    w._leave_current_image()
    assert len(_saved_rois(image)) == 1


def test_a_new_empty_rois_layer_after_removal_still_keeps_them(tmp_path, monkeypatch):
    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    _remove_layer(w, w.roi_shapes_layer)
    w._add_roi_layer()           # Add ROI clicked, nothing drawn yet
    assert w.roi_shapes_layer is not None and len(w.roi_shapes_layer.data) == 0

    w._save_analysis_state()
    assert len(_saved_rois(image)) == 1


def test_shapes_drawn_after_removal_are_saved(tmp_path, monkeypatch):
    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    _remove_layer(w, w.roi_shapes_layer)
    w._add_roi_layer()
    w.roi_shapes_layer.data = [np.asarray(SQUARE, dtype=float),
                               np.asarray(SQUARE, dtype=float) + 1]

    w._save_analysis_state()
    assert len(_saved_rois(image)) == 2


def test_clear_rois_after_removal_saves_none_drawn(tmp_path, monkeypatch):
    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    _remove_layer(w, w.roi_shapes_layer)
    w._clear_rois()

    w._save_analysis_state()
    assert _saved_rois(image) == []


def test_rois_deleted_inside_the_layer_are_still_saved_as_none_drawn(tmp_path, monkeypatch):
    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    w.roi_shapes_layer.data = []

    w._save_analysis_state()
    assert _saved_rois(image) == []


# Once shapes drawn on the new layer are saved, the file follows that layer
# again: deleting those shapes is saved as none drawn. (The removal used to be
# remembered until a Clear button or another image, so the deleted shapes
# stayed in the file and came back, and were counted, on the next open.)

def test_rois_drawn_after_removal_then_deleted_are_saved_as_none_drawn(tmp_path, monkeypatch):
    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    _remove_layer(w, w.roi_shapes_layer)
    w._add_roi_layer()
    w.roi_shapes_layer.data = [np.asarray(SQUARE, dtype=float) + 2]
    w._save_analysis_state()
    assert len(_saved_rois(image)) == 1

    w.roi_shapes_layer.data = []      # every shape deleted inside the layer
    w._save_analysis_state()
    assert _saved_rois(image) == []


def test_new_rois_deleted_before_any_save_still_keep_the_file(tmp_path, monkeypatch):
    # The safe side: nothing new reached the file, so it still holds the
    # shapes from before the removal (Clear saves it as none drawn).
    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    _remove_layer(w, w.roi_shapes_layer)
    w._add_roi_layer()
    w.roi_shapes_layer.data = [np.asarray(SQUARE, dtype=float) + 2]
    w.roi_shapes_layer.data = []
    w._save_analysis_state()
    assert len(_saved_rois(image)) == 1


def test_background_rois_drawn_after_removal_then_deleted_are_saved_as_none_drawn(
        tmp_path, monkeypatch):
    from test_slice_saved_analysis_is_one_state import FakeShapes

    w, image = _reopened_with_rois(tmp_path, monkeypatch)
    folder = image.parent / "PRJ_01_01_S1_analysis"

    def bg_layer():
        w._pa_bg_shapes_layer = FakeShapes("Background ROIs")
        w.viewer.layers.append(w._pa_bg_shapes_layer)
    w._pa_setup_bg_shapes_layer = bg_layer

    def saved_bg():
        return json.loads((folder / "bg_rois.json").read_text())["regions"]

    w._pa_activate_bg_drawing()
    w._pa_bg_shapes_layer.data = [np.asarray(SQUARE, dtype=float)]
    w._save_analysis_state()
    assert len(saved_bg()) == 1

    _remove_layer(w, w._pa_bg_shapes_layer)
    w._save_analysis_state()
    assert len(saved_bg()) == 1          # removed layer: file kept

    w._pa_activate_bg_drawing()          # the drawing button makes a new layer
    w._pa_bg_shapes_layer.data = [np.asarray(SQUARE, dtype=float) + 3]
    w._save_analysis_state()
    assert len(saved_bg()) == 1

    w._pa_bg_shapes_layer.data = []
    w._save_analysis_state()
    assert saved_bg() == []
