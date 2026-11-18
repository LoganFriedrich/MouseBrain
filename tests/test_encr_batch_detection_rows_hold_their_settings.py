"""A 2D batch detection row records what produced its count, in its columns.

Defect covered: the batch's detection rows left det_backend and
det_preprocessing blank and kept the settings only as free text in the
notes, and left the before-filter count and the per-filter removals blank
although the detector reports them. A row could not say on its own which
detector, diameter range, pixel size and thresholds gave its count, nor how
the count was reached.
"""

import json

import numpy as np
import pytest

from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.core import visualization as vis_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

SAMPLE = "PRJ_01_02_S3_RA"

# Every value process_single needs. Test data only.
VALUES = dict(threshold=0, sigma_threshold=0, dilation=50, bg_percentile=10,
              soma_dilation=6, threshold_fraction=0.25, log_threshold=0.004,
              min_diameter_um=9.0, max_diameter_um=24.0)


def _image():
    """Round bright nuclei about 14 um across at 1 um/px."""
    yy, xx = np.mgrid[0:200, 0:200]
    red = np.full((200, 200), 10.0)
    green = np.full((200, 200), 10.0)
    for cy, cx in [(40, 40), (40, 120), (120, 40), (120, 120), (170, 170)]:
        red[(yy - cy) ** 2 + (xx - cx) ** 2 <= 7 ** 2] = 1000.0
    return red.astype(np.float32), green.astype(np.float32)


@pytest.fixture
def run(tmp_path, monkeypatch):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(vis_mod, "save_all_qc_figures", lambda **k: [])
    red, green = _image()
    monkeypatch.setattr(io_mod, "load_image", lambda path, **k: (
        np.stack([green, red]), {"voxel_size_um": {"x": 1.0}, "voxel_size_from_file": True}))
    monkeypatch.setattr(io_mod, "extract_channels",
                        lambda data, red_idx=0, green_idx=1, **k: (red, green))
    applied = {}
    real = det_mod.detect_with_log_augmentation

    def spy(*a, **k):
        labels, details = real(*a, **k)
        applied.update(details)
        return labels, details

    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", spy)
    nd2 = tmp_path / f"{SAMPLE}.nd2"
    nd2.write_bytes(b"")
    result = be.process_single(nd2, tmp_path / "out", **VALUES)
    rows = t.search(run_type="detection")
    assert len(rows) == 1
    return result, rows[0], applied


def test_the_row_names_the_detector_and_every_setting_it_ran_with(run):
    result, row, applied = run
    assert result["status"] == "completed", result
    assert row["det_backend"] == "threshold+log"
    settings = json.loads(row["det_preprocessing"])
    assert settings["min_diameter_um"] == 9.0 and settings["max_diameter_um"] == 24.0
    assert settings["threshold_fraction"] == 0.25
    assert settings["log_threshold"] == 0.004
    assert settings["pixel_um"] == 1.0
    assert settings["min_area_px"] == applied["min_area"]
    assert settings["max_area_px"] == applied["max_area"]


def test_the_row_says_how_the_count_was_reached(run):
    result, row, applied = run
    assert row["det_nuclei_found"] == str(applied["filtered_count"])
    assert row["det_raw_count"] == str(applied["raw_count"])
    assert row["det_removed_by_size"] == str(applied["removed_by_size"])
    assert row["det_removed_by_morphology"] == str(applied["removed_by_morphology"])
    # Not reported by this detector: left blank, never written as 0.
    assert row["det_removed_by_confidence"] == ""
