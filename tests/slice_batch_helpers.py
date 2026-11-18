"""A stand-in BrainSlice window with a batch folder chosen, for tests that
drive the Batch section of the particle tab (Run Batch Analysis, the ROI
drawing that follows it, and Export Batch Results) without opening a window.

Images are never read from disk: load_image is replaced by a table of
invented images, so a test says exactly which image holds what (or fails to
load).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from slice_widget_helpers import Control, make_loading_widget


class FakeShapes:
    """A napari Shapes layer: its drawn shapes are in .data."""

    def __init__(self, name, shapes=()):
        self.name, self.visible, self.mode = name, True, None
        self.data = [np.asarray(s, dtype=float) for s in shapes]


def two_channel_image(n_particles):
    """Green and red channels, 40x40, with n_particles bright squares in the
    detection channel (red, index 1) and a bright measurement channel
    (green, index 0) so every particle is positive."""
    red = np.zeros((40, 40))
    for k in range(n_particles):
        r = 3 + 9 * k
        red[r:r + 5, 3:8] = 100.0
    green = np.full((40, 40), 50.0)
    return [green, red]


BATCH_SETTINGS = {
    'version': 2, 'threshold': 50.0, 'min_area': 4, 'max_area': 1000,
    'min_circularity': 0.0, 'watershed': False, 'bg_value': 10.0,
    'min_pct_above_bg': 50.0, 'rotation': 'None', 'z_projection': 'Max Intensity',
    'red_channel_idx': 1, 'green_channel_idx': 0,
    'detect_channel_idx': 1, 'measure_channel_idx': 0,
}


def make_batch_widget(tmp_path, monkeypatch, images, folder_name="batch_one",
                      settings=None):
    """A window whose Batch section has `folder_name` chosen.

    images: {file name: channels list, or None for a file that cannot be
    read, or an Exception instance raised when it is read}. Every file is
    created (empty) in tmp_path/folder_name. Returns (widget, folder)."""
    from mousebrain.plugin_2d.sliceatlas.core import io as core_io
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget

    folder = tmp_path / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, channels in images.items():
        (folder / name).write_bytes(b"")
        table[name] = channels

    def load_image(path, metadata_only=False, **kwargs):
        from pathlib import Path
        p = Path(path)
        entry = table.get(p.name, LOOKUP.get(p.name))
        if isinstance(entry, Exception):
            raise entry
        if entry is None:
            return None, None
        return [np.array(c) for c in entry], {}

    LOOKUP = {}
    monkeypatch.setattr(core_io, "load_image", load_image)

    w = make_loading_widget(tmp_path, monkeypatch)
    w.messages = []
    monkeypatch.setattr(main_widget, "QMessageBox", SimpleNamespace(
        warning=lambda *a, **k: w.messages.append(a[2] if len(a) > 2 else ""),
        critical=lambda *a, **k: w.messages.append(a[2] if len(a) > 2 else ""),
        information=lambda *a, **k: w.messages.append(a[2] if len(a) > 2 else ""),
        question=lambda *a, **k: 0,
        Yes=1, No=0))
    w._batch_folder = folder
    w._batch_files = sorted(folder / name for name in images)
    w._batch_results = {}
    w._batch_current_idx = -1
    w._batch_settings = None
    w._batch_roi_mode = False
    w._batch_progress = Control()
    w._batch_roi_nav = Control()
    w._pa_get_settings = lambda: dict(settings or BATCH_SETTINGS)
    w._pa_get_scale = lambda: (1.0, 1.0)
    w._pa_populate_table = lambda df: None
    w._pa_find_layer = lambda name: None
    w._pa_remove_layer = lambda name: None
    w.pa_bg_manual_spin = Control(value=0)
    w._roi_names = []

    def add_roi_layer():
        w.roi_shapes_layer = FakeShapes("ROIs")
        w.viewer.layers.append(w.roi_shapes_layer)
    w._add_roi_layer = add_roi_layer
    w._extra_images = LOOKUP
    return w, folder


def draw_roi_around_everything(w):
    """Draw one ROI over the whole batch image on screen."""
    w.roi_shapes_layer.data = [np.array([[0, 0], [0, 39], [39, 39], [39, 0]], float)]


def export_batch(w, monkeypatch, path):
    """Export Batch Results to `path` (the summary CSV)."""
    from mousebrain.plugin_2d.sliceatlas.napari_plugin import main_widget
    monkeypatch.setattr(main_widget, "QFileDialog", SimpleNamespace(
        getSaveFileName=lambda *a, **k: (str(path), "")))
    w._batch_export()
    return pd.read_csv(path)
