"""Best and Paradigm Best marks stay consistent.

Paradigm Best is the settled detection method for every brain imaged the same
way (step 4 runs it). Three ways it went wrong: archiving a run cleared its
brain-level best but left it Paradigm Best; marking a Paradigm Best gave its
brain a second best without clearing the first; and the napari "load best"
loaded the newest run and called it the best.
"""
import csv
from pathlib import Path

from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

TW = (Path(__file__).resolve().parents[1] / "src" / "mousebrain" / "plugin_3d"
      / "tuning_widget.py")


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


def test_archiving_a_run_ends_its_paradigm_best(tmp_path):
    t = _tracker(tmp_path, [{"exp_id": "a", "brain": "B1", "marked_best": "True",
                             "paradigm_best": "True", "det_ball_xy": "6"}])
    t.archive_experiment("a", archived_by="b")
    row = t.get_experiment("a")
    assert row["paradigm_best"] == "" and row["marked_best"] == ""
    assert t.get_best_for_paradigm("2p5x_z5") is None


def test_an_archived_run_is_never_the_settled_method(tmp_path):
    # A row archived by hand (or by the old code) that still carries the mark.
    t = _tracker(tmp_path, [{"exp_id": "old", "brain": "B1", "paradigm_best": "True",
                             "archived_at": "2000-02-01T00:00:00"},
                            {"exp_id": "new", "brain": "B1", "marked_best": "True"}])
    assert t.get_best_for_paradigm("2p5x_z5")["exp_id"] == "new"


def test_paradigm_best_leaves_its_brain_one_best_and_other_brains_theirs(tmp_path):
    t = _tracker(tmp_path, [{"exp_id": "b1_old", "brain": "B1", "marked_best": "True"},
                            {"exp_id": "b2_best", "brain": "B2", "marked_best": "True",
                             "paradigm_best": "True"},
                            {"exp_id": "b1_new", "brain": "B1"}])
    assert t.mark_paradigm_best("b1_new")
    rows = {r["exp_id"]: r for r in t._read_all()}
    assert rows["b1_old"]["marked_best"] == ""          # same brain: cleared
    assert rows["b1_new"]["marked_best"] == "True"
    assert rows["b2_best"]["marked_best"] == "True"     # other brain: untouched
    assert rows["b2_best"]["paradigm_best"] == ""       # one paradigm best
    assert rows["b1_new"]["paradigm_best"] == "True"
    assert t.get_best_for_brain("B1")["exp_id"] == "b1_new"


def test_mark_as_best_moves_the_mark_within_the_brain_only(tmp_path):
    t = _tracker(tmp_path, [{"exp_id": "a", "brain": "B1", "marked_best": "True"},
                            {"exp_id": "c", "brain": "B2", "marked_best": "True"},
                            {"exp_id": "b", "brain": "B1"}])
    assert t.mark_as_best("b")
    rows = {r["exp_id"]: r for r in t._read_all()}
    assert (rows["a"]["marked_best"], rows["b"]["marked_best"], rows["c"]["marked_best"]) == \
        ("", "True", "True")


def test_best_for_brain_matches_the_whole_name_and_skips_archived_ratings(tmp_path):
    t = _tracker(tmp_path, [{"exp_id": "x", "brain": "M/B1", "rating": "5",
                             "archived_at": "2000-02-01"},
                            {"exp_id": "y", "brain": "M/B1", "rating": "3"},
                            {"exp_id": "z", "brain": "B10", "marked_best": "True"}])
    assert t.get_best_for_brain("B1")["exp_id"] == "y"


def test_load_best_asks_the_tracker_for_the_marked_best():
    text = TW.read_text(encoding="utf-8")
    body = text[text.index("    def _load_detection_results(self, best: bool = True)"):]
    body = body[:body.index("\n    def ", 10)]
    assert "get_best_for_brain(" in body
    assert 'label_suffix = "(best attempt)"' not in body
