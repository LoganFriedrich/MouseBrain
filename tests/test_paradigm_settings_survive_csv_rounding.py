"""The settings a lab calibrated must come back out of the tracker.

The tracker is a CSV, so every value in it is text, and the text is not
necessarily the text that was written. A threshold logged as the integer 8
reads back as "8.0" as soon as anything has rewritten the file -- pandas does
it, Excel does it, and so does an ordinary later run. Parsing that with int()
raises ValueError.

That mattered more than it looks. The crash was not in some reporting corner:
it was in the one function that answers "what settings worked for this kind of
imaging?". So asking for a paradigm's proven settings raised, and the pipeline
step that asks fell through to a generic preset -- running expensive detection
with parameters nobody had validated, while reporting nothing wrong.
"""
import csv

import pytest

from mousebrain.tracker import ExperimentTracker, CSV_COLUMNS


def _tracker_with_one_best_run(tmp_path, values):
    """A tracker holding a single detection run marked best for its paradigm."""
    csv_path = tmp_path / "calibration_runs.csv"
    row = {col: "" for col in CSV_COLUMNS}
    row.update({
        "exp_id": "TEST001",
        "brain": "101_PROJ_01_02_2p5x_z5",
        "imaging_paradigm": "2p5x_z5",
        "exp_type": "detection",
        "paradigm_best": "True",
        "created_at": "2026-01-01T00:00:00",
    })
    row.update(values)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return ExperimentTracker(csv_path=csv_path)


@pytest.mark.parametrize("threshold_text", ["8", "8.0", " 8.0 "])
def test_threshold_is_returned_however_the_csv_spelled_it(tmp_path, threshold_text):
    tracker = _tracker_with_one_best_run(tmp_path, {
        "det_ball_xy": "10.0",
        "det_ball_z": "10.0",
        "det_soma_diameter": "10.0",
        "det_threshold": threshold_text,
        "det_preset": "custom",
    })
    settings = tracker.get_paradigm_detection_settings("2p5x_z5")
    assert settings is not None, "the calibrated run is there and must be found"
    assert settings["threshold"] == 8
    assert settings["ball_xy"] == 10
    assert settings["ball_z"] == 10
    assert settings["soma_diameter"] == 10


def test_an_uncalibrated_paradigm_says_so_rather_than_inventing_numbers(tmp_path):
    """None means "nobody calibrated this", which the caller must be able to see.

    If this returned defaults instead, a brain imaged a way nobody has tuned for
    would be detected with invented parameters and look exactly like a calibrated
    one in the tracker.
    """
    tracker = _tracker_with_one_best_run(tmp_path, {"det_threshold": "8.0"})
    assert tracker.get_paradigm_detection_settings("1p625x_z4") is None
