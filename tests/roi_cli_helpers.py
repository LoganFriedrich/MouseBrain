"""Shared fixtures for the brainslice-roi (run_roi) tests.

Everything is built in tmp_path with invented project codes and sample names;
no machine configuration is read. The registry root is pinned with
MOUSEBRAIN_REGISTRY_ROOT so nothing can resolve to a real pipeline folder.
"""

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pandas as pd

from mousebrain.plugin_2d.sliceatlas.cli import run_roi


PROJECT = "PRJA"
SAMPLE = "Q01_02_S05_ZONE"

# One polygon covering the top-left 50x50 corner of a 100x100 image.
SQUARE = [[0, 0], [0, 50], [50, 50], [50, 0]]


def write_measurements(path, cells=((10, 10, True), (20, 20, False), (80, 80, True))):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"centroid_y": y, "centroid_x": x, "is_positive": pos} for y, x, pos in cells]
    ).to_csv(path, index=False)
    return path


def write_rois(path, vertices=SQUARE, name="Left"):
    path = Path(path)
    path.write_text(json.dumps({
        "version": 1, "image": path.name, "image_shape": [100, 100],
        "created": "2000-01-01T00:00:00",
        "rois": [{"name": name, "vertices": vertices}],
    }))
    return path


def make_dataset(tmp_path, project=PROJECT, samples=(SAMPLE,)):
    """A dataset folder with one empty ND2 per sample; returns its path."""
    base = tmp_path / "data" / project
    sub = base / "images"
    sub.mkdir(parents=True)
    for s in samples:
        (sub / f"{s}.nd2").write_bytes(b"")
    return base


def nd2_of(base, sample=SAMPLE):
    return Path(base) / "images" / f"{sample}.nd2"


def rois_of(base, sample=SAMPLE):
    return nd2_of(base, sample).with_suffix(".rois.json")


def batch_args(base, **kw):
    a = dict(folder=str(base), region=None, skip_done=False, count_only=False,
             output=None, project=None, detection_dir=None)
    a.update(kw)
    return Namespace(**a)


def pin_registry(monkeypatch, tmp_path):
    root = tmp_path / "registry"
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(root))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    return root


class _FakeLayer:
    def __init__(self, shapes):
        self.data = list(shapes)

    @property
    def nshapes(self):
        return len(self.data)


class FakeEditor:
    """Stands in for napari: records which samples were opened and returns
    the polygons a person 'drew' (``drawn``) when the window closes.

    ``keep=True`` is a person who changes nothing and clicks Save & Next: the
    window closes holding exactly the polygons it was opened with."""

    def __init__(self, monkeypatch, drawn=(), keep=False):
        self.opened = []
        self.preloaded = []
        self.drawn = list(drawn)
        self.shown_shapes = []
        editor = self

        class Viewer:
            def __init__(self, title=""):
                editor.opened.append(title)

            def add_image(self, *a, **k):
                return None

            def add_shapes(self, data, **k):
                editor.shown_shapes.append(list(data))

        fake = types.ModuleType("napari")
        fake.Viewer = Viewer
        fake.run = lambda: None
        monkeypatch.setitem(sys.modules, "napari", fake)

        def _load(path, red_ch=None, green_ch=None):
            import numpy as np
            img = np.zeros((100, 100))
            return img, img, (100, 100), {}

        def _setup(viewer, existing_rois=None):
            editor.preloaded.append(existing_rois)
            if keep:
                rois = existing_rois or []
                left = [r["vertices"] for r in rois if "right" not in r.get("name", "").lower()]
                right = [r["vertices"] for r in rois if "right" in r.get("name", "").lower()]
                return _FakeLayer(left), _FakeLayer(right)
            return _FakeLayer(editor.drawn), _FakeLayer([])

        monkeypatch.setattr(run_roi, "_load_image_channels", _load)
        monkeypatch.setattr(run_roi, "_add_measurement_points", lambda v, m: None)
        monkeypatch.setattr(run_roi, "_setup_roi_editor", _setup)
