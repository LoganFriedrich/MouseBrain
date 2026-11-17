"""An image's <image name>_analysis folder holds ONE state: what the window
holds for the image when it is saved.

Saving used to only ever add files. A re-analysis that found 0 particles, or
ROIs that were cleared, left the earlier particles.csv / rois.json in place,
and the next time the image was opened they came back as current. A saved
file the window could not bring back (unreadable) must still never be
overwritten or removed: its contents are not in memory.
"""
import json

import numpy as np
import pandas as pd

from slice_widget_helpers import Control, finish_load, make_loading_widget


class FakeShapes:
    """A napari Shapes layer: its drawn shapes are in .data."""

    def __init__(self, name, shapes=()):
        self.name, self.visible, self.mode = name, True, None
        self.data = [np.asarray(s, dtype=float) for s in shapes]

    def add_polygons(self, shapes):
        self.data.extend(np.asarray(s, dtype=float) for s in shapes)

    add_rectangles = add_polygons


SQUARE = [[1, 1], [1, 5], [5, 5], [5, 1]]


def _open(tmp_path, monkeypatch, name="PRJ_01_01_S1.tif"):
    """A window with `name` opened (and its saved state restored)."""
    image = tmp_path / "imgs" / name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"")
    w = make_loading_widget(tmp_path, monkeypatch)
    w._pa_get_scale = lambda: (1.0, 1.0)
    w._pa_populate_table = lambda df: None
    w._pa_find_layer = lambda name: None
    w._pa_draw_classification_overlay = lambda: None
    w.pa_bg_manual_spin = Control(value=0)

    def add_roi_layer():
        w.roi_shapes_layer = FakeShapes("ROIs")
        w.viewer.layers.append(w.roi_shapes_layer)
    w._add_roi_layer = add_roi_layer
    w._open_image(image, peek=False)
    finish_load(w)
    return w, image


def _draw_roi(w):
    w.roi_shapes_layer = FakeShapes("ROIs", [SQUARE])
    w.viewer.layers.append(w.roi_shapes_layer)
    w._roi_names = ["ROI_A"]


def test_a_re_analysis_with_no_particles_replaces_the_saved_particles(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    w._pa_results = pd.DataFrame({"label": [1, 2], "area": [10, 12]})
    w._pa_labels = np.ones((8, 8), dtype=np.int32)
    w._save_analysis_state()

    # Particle analysis again: this time nothing passes the filters.
    w._pa_results = pd.DataFrame({"label": [], "area": []})
    w._pa_labels = np.zeros((8, 8), dtype=np.int32)
    w._save_analysis_state()

    saved = pd.read_csv(image.parent / "PRJ_01_01_S1_analysis" / "particles.csv")
    assert len(saved) == 0

    # Opened again, the image shows the analysis that found nothing.
    w2, _ = _open(tmp_path, monkeypatch)
    assert w2._pa_results is not None and len(w2._pa_results) == 0


def test_cleared_rois_are_saved_as_none_drawn(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    _draw_roi(w)
    w._pa_results = pd.DataFrame({"label": [1], "area": [10]})
    w._save_analysis_state()
    rois_file = image.parent / "PRJ_01_01_S1_analysis" / "rois.json"
    assert len(json.loads(rois_file.read_text())["rois"]) == 1

    # The person deletes the ROI, then the state is saved again.
    w.roi_shapes_layer.data = []
    w._save_analysis_state()

    # The file is kept (drawn by a person) but says none are drawn.
    assert json.loads(rois_file.read_text())["rois"] == []


def test_a_result_that_is_gone_from_memory_is_gone_from_the_folder(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    w._pa_results = pd.DataFrame({"label": [1], "area": [10]})
    w._roi_detail_data = pd.DataFrame({"roi": ["ROI_A"], "total": [3]})
    w._save_analysis_state()
    detail = image.parent / "PRJ_01_01_S1_analysis" / "roi_detail.csv"
    assert detail.exists()

    w._roi_detail_data = None          # counts dropped
    w._save_analysis_state()
    assert not detail.exists()


def test_a_saved_file_that_could_not_be_restored_is_never_overwritten(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    _draw_roi(w)
    w._pa_results = pd.DataFrame({"label": [1], "area": [10]})
    w._pa_labels = np.ones((8, 8), dtype=np.int32)
    w._save_analysis_state()
    folder = image.parent / "PRJ_01_01_S1_analysis"
    # labels.npz is damaged on disk.
    (folder / "labels.npz").write_bytes(b"not a numpy file")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    w2, _ = _open(tmp_path, monkeypatch)
    # The ROIs came back; the labels and the particles that belong to them
    # did not, and the status bar says so.
    assert w2.roi_shapes_layer is not None and len(w2.roi_shapes_layer.data) == 1
    assert w2._pa_labels is None and w2._pa_results is None
    assert "labels.npz" in w2.status_label.text()

    # Leaving the image saves it: what was not restored stays as it was.
    w2._save_analysis_state()
    assert (folder / "labels.npz").read_bytes() == before["labels.npz"]
    assert (folder / "particles.csv").read_bytes() == before["particles.csv"]
    assert len(json.loads((folder / "rois.json").read_text())["rois"]) == 1


def test_nuclei_and_measurements_are_saved_and_restored(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    nuclei = np.zeros((8, 8), dtype=np.int32)
    nuclei[2:4, 2:4] = 1
    w.nuclei_labels = nuclei
    w._labels_run_id = "det-1"
    w.cell_measurements = pd.DataFrame({"label": [1], "is_positive": [True]})
    w._coloc_run_id = "coloc-1"
    w._save_analysis_state()

    w2, _ = _open(tmp_path, monkeypatch)
    assert w2.nuclei_labels is not None and (w2.nuclei_labels == nuclei).all()
    assert list(w2.cell_measurements["label"]) == [1]
    # A colocalization on the restored nuclei names their detection.
    assert w2._labels_run_id == "det-1"
    assert w2.coloc_btn.isEnabled() and w2.quant_btn.isEnabled()


def test_analyzed_marker_needs_a_saved_result(tmp_path, monkeypatch):
    image = tmp_path / "imgs" / "PRJ_01_01_S2.tif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")
    folder = image.parent / "PRJ_01_01_S2_analysis"
    folder.mkdir()
    (folder / "settings.json").write_text("{}")

    w = make_loading_widget(tmp_path, monkeypatch)
    w._nav_siblings, w._nav_selected_idx = [image], 0
    w._nav_update_label()
    assert "[analyzed" not in w._img_nav_label.text()

    pd.DataFrame({"label": [1]}).to_csv(folder / "measurements.csv", index=False)
    w._nav_update_label()
    assert "[analyzed" in w._img_nav_label.text()


def test_an_error_part_way_through_a_restore_leaves_the_folder_untouched(tmp_path, monkeypatch):
    w, image = _open(tmp_path, monkeypatch)
    _draw_roi(w)
    nuclei = np.zeros((8, 8), dtype=np.int32)
    nuclei[2:4, 2:4] = 1
    w.nuclei_labels = nuclei
    w.cell_measurements = pd.DataFrame({"label": [1], "is_positive": [True]})
    w._roi_detail_data = pd.DataFrame({"roi": ["ROI_A"], "total": [1]})
    w._save_analysis_state()
    folder = image.parent / "PRJ_01_01_S1_analysis"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    # Drawing the restored nuclei fails unexpectedly: nothing after it
    # (measurements, ROI counts, ROIs) is read back.
    from slice_widget_helpers import FakeViewer

    class FailingViewer(FakeViewer):
        def add_labels(self, data, name=None, **kwargs):
            if name and name.startswith("Nuclei"):
                raise RuntimeError("display failed")
            return super().add_labels(data, name, **kwargs)

    image2 = image
    w2 = make_loading_widget(tmp_path, monkeypatch)
    w2.viewer = FailingViewer()
    w2._pa_get_scale = lambda: (1.0, 1.0)
    w2._open_image(image2, peek=False)
    finish_load(w2)
    assert "only partly restored" in w2.status_label.text()

    # Leaving the image must not remove or empty what was never read back.
    w2._save_analysis_state()
    after = {p.name: p.read_bytes() for p in folder.iterdir()}
    for name in ("measurements.csv", "roi_detail.csv", "rois.json", "nuclei.npz"):
        assert after[name] == before[name], name
