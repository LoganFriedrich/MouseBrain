"""Runs are never left unlogged in silence, and never logged to a stand-in.

When the 2D tracker could not be opened, the napari widget and the batch set
it to None and carried on with nothing logged and no warning a person would
see. And on an install that does not know where the lab's data lives, the
default tracker path sits under a temporary stand-in folder: SliceTracker()
created a tracker there, so runs looked logged and were lost.
"""
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mousebrain import config as cfg
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr
from mousebrain.plugin_2d.sliceatlas.core import config as slice_config, detection, io
from mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget import BrainSliceWidget
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker


def test_an_unconfigured_install_does_not_open_a_stand_in_tracker(tmp_path, monkeypatch):
    stand_in = tmp_path / "stand_in" / "calibration_runs.csv"
    monkeypatch.setattr(cfg, "ROOT_PATH", cfg._UNCONFIGURED_ROOT)
    monkeypatch.setattr(slice_config, "TRACKER_CSV", stand_in)
    assert cfg.is_configured() is False
    with pytest.raises(RuntimeError, match="not configured"):
        SliceTracker()
    assert not stand_in.exists()


def test_a_configured_install_and_an_explicit_path_still_open(tmp_path, monkeypatch):
    real = tmp_path / "2_Data_Summary" / "calibration_runs.csv"
    monkeypatch.setattr(cfg, "ROOT_PATH", tmp_path)
    monkeypatch.setattr(slice_config, "TRACKER_CSV", real)
    assert SliceTracker().csv_path == real and real.exists()
    monkeypatch.setattr(cfg, "ROOT_PATH", cfg._UNCONFIGURED_ROOT)
    assert SliceTracker(csv_path=tmp_path / "given.csv").csv_path == tmp_path / "given.csv"


def _refuse(**_budgets):
    raise RuntimeError("the tracker CSV is unreadable")


def test_the_window_shows_in_red_why_its_runs_are_not_logged(monkeypatch):
    # Built through the window's own constructor, as napari builds it. Before,
    # a tracker that could not be opened left one console line and nothing
    # in the window: every run then went unlogged with no warning in view.
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from unittest.mock import MagicMock
    from qtpy.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(tracker_pkg, "SliceTracker", _refuse)
    w = BrainSliceWidget(MagicMock())
    try:
        assert w.tracker is None
        shown = [lbl.text() for lbl in w.findChildren(QLabel)
                 if lbl.isVisibleTo(w) and "NOT be logged" in lbl.text()]
        assert len(shown) == 1 and "unreadable" in shown[0]
    finally:
        w.deleteLater()
        app.processEvents()


def test_the_warning_goes_away_when_the_tracker_is_open():
    w = SimpleNamespace(tracker=object(), _tracker_error=None)
    assert BrainSliceWidget._tracker_warning_text(w) == ""


def test_the_batch_says_when_its_runs_are_not_logged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tracker_pkg, "SliceTracker", _refuse)
    monkeypatch.setattr(io, "load_image",
                        lambda p: (np.zeros((2, 8, 8)), {"voxel_size_um": {"x": 1.0}}))
    monkeypatch.setattr(io, "extract_channels",
                        lambda data, red_idx=0, green_idx=1: (np.zeros((8, 8)), np.zeros((8, 8))))
    monkeypatch.setattr(detection, "detect_with_log_augmentation",
                        lambda red, pixel_um=1.0, **k: (np.zeros((8, 8), dtype=np.int32),
                                                         {"filtered_count": 0}))
    nd2 = tmp_path / "PRJ_01_01_S01.nd2"
    nd2.write_bytes(b"")
    result = batch_encr.process_single(nd2, tmp_path / "out", min_diameter_um=8.0, max_diameter_um=25.0)
    assert result["status"] == "no_nuclei"
    out = capsys.readouterr().out
    assert "NOT be logged" in out and "unreadable" in out
