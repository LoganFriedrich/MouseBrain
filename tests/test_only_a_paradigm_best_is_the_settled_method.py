"""Only the run a person marked Paradigm Best is the settled method of an imaging setup.

With no Paradigm Best (never marked, or archived by mistake) the tracker used to
fall back to the newest run marked Best for any ONE brain of the setup and hand
it to step 4 as the settled method -- so a single brain's tuning silently became
the detection of every brain imaged that way, logged as 'paradigm_best'. The same
lookup chose the classification model. Now no Paradigm Best means no settled
method, and step 4 stops and says how to settle one. The per-brain Best is still
shown where the widget only displays it.
"""
import csv

from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

SETUP = "2p5x_z5"
EIGHT = {"det_ball_xy": "6", "det_ball_z": "15", "det_soma_diameter": "16",
         "det_threshold": "10", "det_ball_overlap": "0.6", "det_soma_spread": "1.4",
         "det_log_sigma": "0.2", "det_max_cluster": "100000"}


def _tracker(tmp_path, rows):
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            full = {c: "" for c in CSV_COLUMNS}
            full.update(r)
            w.writerow(full)
    return ExperimentTracker(csv_path=path)


def _brain_best(exp_id, exp_type="detection", **extra):
    row = {"exp_id": exp_id, "brain": "7_PROJ_01_07_" + SETUP, "exp_type": exp_type,
           "imaging_paradigm": SETUP, "status": "completed", "marked_best": "True",
           "created_at": "2026-01-02T00:00:00"}
    row.update(extra)
    return row


def test_one_brains_best_is_not_the_detection_method_of_its_setup(tmp_path):
    t = _tracker(tmp_path, [_brain_best("det_one_brain", **EIGHT)])
    assert t.get_paradigm_detection_settings(SETUP) is None


def test_one_brains_best_classification_is_not_the_model_of_its_setup(tmp_path):
    t = _tracker(tmp_path, [_brain_best("cls_one_brain", exp_type="classification",
                                        class_model_path="models/one_brain.h5")])
    assert t.get_paradigm_model(SETUP) is None


def test_the_paradigm_best_is_the_method_even_beside_a_newer_brain_best(tmp_path):
    t = _tracker(tmp_path, [
        _brain_best("det_paradigm", paradigm_best="True",
                    created_at="2026-01-01T00:00:00", **EIGHT),
        _brain_best("det_newer_brain", created_at="2026-01-05T00:00:00",
                    **dict(EIGHT, det_threshold="3")),
    ])
    s = t.get_paradigm_detection_settings(SETUP)
    assert s["source_exp_id"] == "det_paradigm" and s["threshold"] == 10


def test_the_display_lookup_still_shows_a_brain_best(tmp_path):
    t = _tracker(tmp_path, [_brain_best("det_one_brain", **EIGHT)])
    assert t.get_best_for_paradigm(SETUP)["exp_id"] == "det_one_brain"
    assert t.get_best_for_paradigm(SETUP, strict=True) is None
