"""A brain has one Best, whichever way its runs name it.

The napari widget logs a run's brain as "<brain>", the pipeline scripts as
"<mouse>/<brain>". Marking a run Best (or Paradigm Best) cleared the brain's
previous Best only when the two names were the very same string, so a Best
logged in the other form stayed: the brain had two Bests, and "Load best"
(get_best_for_brain) returned the first in file order -- the older run.
"""
import csv

from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

BRAIN = "901_PROJ_09_01_2p5x_z5"
OTHER = "902_PROJ_09_02_2p5x_z5"


def _tracker(tmp_path, rows):
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"exp_type": "detection", "status": "completed",
                        "imaging_paradigm": "2p5x_z5",
                        "created_at": "2000-01-0%dT00:00:00" % (i + 1), **r})
    return ExperimentTracker(path)


def _bests(t):
    return sorted(r["exp_id"] for r in t._read_all() if r.get("marked_best") == "True")


def test_marking_a_script_run_best_clears_the_widget_runs_best(tmp_path):
    t = _tracker(tmp_path, [
        {"exp_id": "widget_old", "brain": BRAIN, "marked_best": "True"},
        {"exp_id": "script_new", "brain": "PROJ_09_01/" + BRAIN},
        {"exp_id": "other_brain", "brain": OTHER, "marked_best": "True"},
    ])
    assert t.mark_as_best("script_new")
    assert _bests(t) == ["other_brain", "script_new"]
    assert t.get_best_for_brain(BRAIN)["exp_id"] == "script_new"


def test_marking_a_widget_run_best_clears_the_script_runs_best(tmp_path):
    t = _tracker(tmp_path, [
        {"exp_id": "script_old", "brain": "PROJ_09_01\\" + BRAIN, "marked_best": "True"},
        {"exp_id": "widget_new", "brain": BRAIN},
    ])
    assert t.mark_as_best("widget_new")
    assert _bests(t) == ["widget_new"]
    assert t.get_best_for_brain(BRAIN)["exp_id"] == "widget_new"


def test_paradigm_best_also_leaves_its_brain_one_best(tmp_path):
    t = _tracker(tmp_path, [
        {"exp_id": "widget_old", "brain": BRAIN, "marked_best": "True"},
        {"exp_id": "script_new", "brain": "PROJ_09_01/" + BRAIN},
        {"exp_id": "other_brain", "brain": OTHER, "marked_best": "True"},
    ])
    assert t.mark_paradigm_best("script_new")
    assert _bests(t) == ["other_brain", "script_new"]
    assert t.get_best_for_brain(BRAIN)["exp_id"] == "script_new"


def test_a_brain_whose_name_only_contains_this_one_keeps_its_best(tmp_path):
    t = _tracker(tmp_path, [
        {"exp_id": "longer_name", "brain": "1" + BRAIN, "marked_best": "True"},
        {"exp_id": "this_brain", "brain": "PROJ_09_01/" + BRAIN},
    ])
    assert t.mark_as_best("this_brain")
    assert _bests(t) == ["longer_name", "this_brain"]
