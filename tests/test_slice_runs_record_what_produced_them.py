"""A 2D run's tracker row holds every setting it ran with and the counts it
reported, so the run can be reproduced and its result explained from the row.

For a detection that means more than StarDist's numbers (a threshold run's
probability and overlap are not 0.0, its thresholds are recorded) and the
counts its filters removed; a particle analysis says where its background
came from (drawn rectangles or the particles' median). update_status() must
not drop a result it does not know without a word.
"""
import json

import pytest

from slice_widget_helpers import (
    Control, labels_with_one_nucleus, make_particle_widget, make_widget)
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

SAMPLE = "PRJ_01_03_S02"


def _only_row(w, run_type):
    rows = [r for r in w.tracker.search(sample_id=SAMPLE, limit=None)
            if r["run_type"] == run_type]
    assert len(rows) == 1, rows
    return rows[0]


def test_a_threshold_detection_row_holds_every_setting_it_ran_with(tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE, backend="Threshold")
    w.thresh_detect_method_combo = Control(text="otsu")
    w.thresh_detect_percentile_spin = Control(value=97.5)
    w.thresh_opening_spin = Control(value=2)
    w.thresh_split_check = Control(checked=True)

    w._run_detection()

    row = _only_row(w, "detection")
    assert row["det_backend"] == "threshold"
    settings = json.loads(row["det_preprocessing"])
    assert settings["threshold_method"] == "otsu"
    assert settings["threshold_percentile"] == 97.5
    assert settings["opening_radius"] == 2
    assert settings["split_touching"] is True
    # Numbers this backend does not use are blank, not a made-up 0.
    assert row["det_prob_thresh"] == "" and row["det_nms_thresh"] == ""


def test_a_stardist_detection_row_holds_its_preprocessing(tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE, backend="StarDist")
    w.preproc_clahe_check = Control(checked=True)
    w.preproc_clahe_clip_spin = Control(value=0.03)
    w._run_detection()
    row = _only_row(w, "detection")
    settings = json.loads(row["det_preprocessing"])
    assert row["det_backend"] == "stardist"
    assert settings["clahe"] is True and settings["clahe_clip_limit"] == 0.03


def test_a_finished_detection_row_holds_what_each_filter_removed(tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, SAMPLE)
    w._run_detection()
    metrics = {"raw_count": 12, "filtered_count": 7, "removed_by_size": 3,
               "removed_by_border": 1, "removed_by_morphology": 1,
               "removed_by_confidence": 0, "backend": "stardist"}
    w._on_detect_finished(True, "Detected 7 nuclei", 7, labels_with_one_nucleus(), metrics)

    row = _only_row(w, "detection")
    assert (row["det_nuclei_found"], row["det_raw_count"], row["det_removed_by_size"],
            row["det_removed_by_border"], row["det_removed_by_morphology"],
            row["det_removed_by_confidence"]) == ("7", "12", "3", "1", "1", "0")


def test_a_result_under_an_unknown_name_is_refused_not_dropped(tmp_path):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    run_id = t.log_detection(sample_id=SAMPLE)
    with pytest.raises(ValueError, match="det_nucleus_found"):
        t.update_status(run_id, status="completed", det_nucleus_found=5)
    # Nothing at all was written: the row is not 'completed' without its count.
    assert t.get_run(run_id)["status"] == "started"


def test_a_particle_run_row_says_where_its_background_came_from_and_what_was_filtered(
        tmp_path, monkeypatch):
    w = make_particle_widget(tmp_path, monkeypatch, SAMPLE, background=0.0)
    w._pa_run_analysis()
    row = _only_row(w, "particle_analysis")
    assert row["status"] == "completed"
    assert row["pa_bg_method"] == "percentile_50"   # not 'manual_roi'
    assert (row["pa_particles_raw"], row["pa_particles_found"]) == ("2", "1")
    assert float(row["pa_mean_intensity"]) == pytest.approx(200.0)

    w2 = make_particle_widget(tmp_path / "second", monkeypatch, SAMPLE, background=20.0)
    w2._pa_run_analysis()
    assert _only_row(w2, "particle_analysis")["pa_bg_method"] == "manual_roi"
