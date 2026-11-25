"""A tracker that cannot be written is one sample's error, not the batch's end.

When a sample's detection fails AFTER its tracker row was logged, the error
handler marks the row 'error' with protection of its own. If the tracker
cannot be rewritten (its CSV held open by another program), an unprotected
handler would raise the same exception again, leave process_single, and stop
the whole batch: no summary.csv, and no result for any later sample.
"""

import csv
import sys

import pytest

from example_method import write_example_approved_method
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

import numpy as np

PROJECT = "PRJ"
SAMPLES = ["PRJ_01_02_S3_RA", "PRJ_01_02_S4_RB"]


class _LockedTracker(SliceTracker):
    """Logs new rows but cannot rewrite the file to change one: every change
    to an existing row goes through _rewrite, which is refused here the way
    a file another program holds open refuses the swap."""

    def _rewrite(self, mutate):
        raise PermissionError("the tracker file is open in another program")


def test_batch_finishes_when_the_tracker_cannot_be_rewritten(tmp_path, monkeypatch, capsys):
    t = _LockedTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    # The batch has no built-in values: it needs an approved method to run.
    write_example_approved_method(tmp_path / "registry")
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    folder.mkdir(parents=True)
    for s in SAMPLES:
        (folder / f"{s}.nd2").write_bytes(b"")
    img = np.zeros((32, 32), dtype=np.float32)
    monkeypatch.setattr(io_mod, "load_image",
                        lambda path, **k: (np.stack([img, img]), {"voxel_size_um": {"x": 1.0}}))
    monkeypatch.setattr(io_mod, "extract_channels",
                        lambda data, red_idx=0, green_idx=1, **k: (img, img))

    def broken_detector(*a, **k):
        raise ValueError("detector failed")

    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", broken_detector)

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(out), "--ignore-tracker"])
    be.main()     # must return, not raise PermissionError

    with open(out / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["file"] for r in rows] == [f"{s}.nd2" for s in SAMPLES]
    assert {r["status"] for r in rows} == {"error"}
    assert all("detector failed" in r["error"] and "tracker row" in r["error"]
               for r in rows)
    printed = capsys.readouterr().out
    assert "BATCH COMPLETE" in printed
    for s in SAMPLES:       # the end-of-batch summary names each failed sample
        assert f"    {s}.nd2: [sample failure while detect] detector failed" in printed
