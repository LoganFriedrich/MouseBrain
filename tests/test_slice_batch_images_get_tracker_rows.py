"""Every image a batch analysis works on gets its own particle-analysis row
in the 2D calibration tracker: completed with what it found, or failed with
why. No row is left 'started'.

Run Batch Analysis used to leave no trace in the tracker at all: a batch's
results could not be found in the Run History, and nothing said which
settings had produced them.
"""
import csv

from slice_batch_helpers import BATCH_SETTINGS, make_batch_widget, two_channel_image


def _rows(w):
    with open(w.tracker.csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["run_type"] == "particle_analysis"]


def test_each_batch_image_has_a_finished_row(tmp_path, monkeypatch):
    images = {"A_01.tif": two_channel_image(2), "A_02.tif": None}
    w, folder = make_batch_widget(tmp_path, monkeypatch, images)
    w._batch_enter_roi_mode = lambda: None
    w._batch_run_analysis()

    rows = {r["sample_id"]: r for r in _rows(w)}
    assert set(rows) == {"A_01", "A_02"}
    done, failed = rows["A_01"], rows["A_02"]

    assert done["status"] == "completed"
    assert int(float(done["pa_particles_found"])) == 2
    assert int(float(done["pa_positive_particles"])) == 2
    assert float(done["pa_bg_value"]) == BATCH_SETTINGS["bg_value"]
    assert float(done["pa_threshold_value"]) == BATCH_SETTINGS["threshold"]
    assert int(float(done["pa_min_area"])) == BATCH_SETTINGS["min_area"]
    assert done["input_path"] == str(folder / "A_01.tif")
    assert "batch" in done["notes"]

    assert failed["status"] == "failed"
    assert "could not be read" in failed["notes"]
    assert not any(r["status"] == "started" for r in _rows(w))


def test_without_a_background_the_row_says_nothing_was_classified(tmp_path, monkeypatch):
    settings = dict(BATCH_SETTINGS, bg_value=0.0)
    w, folder = make_batch_widget(tmp_path, monkeypatch,
                                  {"A_01.tif": two_channel_image(1)}, settings=settings)
    w._batch_enter_roi_mode = lambda: None
    w._batch_run_analysis()
    (row,) = _rows(w)
    assert row["status"] == "completed"
    assert row["pa_bg_method"] == "none"
    assert row["pa_positive_particles"] == ""


def test_a_tracker_that_refuses_is_tried_once_and_the_window_says_so(tmp_path, monkeypatch):
    images = {f"A_0{k}.tif": two_channel_image(1) for k in range(1, 4)}
    w, folder = make_batch_widget(tmp_path, monkeypatch, images)
    w._batch_enter_roi_mode = lambda: None
    tries = []

    def refuse(**kwargs):
        tries.append(kwargs["sample_id"])
        raise PermissionError("calibration_runs.csv is open elsewhere")
    monkeypatch.setattr(w.tracker, "log_particle_analysis", refuse)
    w._batch_run_analysis()

    assert tries == ["A_01"]            # not once per image, each a frozen wait
    assert sorted(w._batch_results) == ["A_01.tif", "A_02.tif", "A_03.tif"]
    text = w._batch_status_label.text()
    assert "3 image(s) NOT logged" in text and "open elsewhere" in text
