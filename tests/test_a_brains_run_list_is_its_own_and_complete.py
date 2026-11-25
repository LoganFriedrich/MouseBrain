"""Asking the tracker for one brain's runs returns that brain's runs -- all of them.

tracker.search(brain=...) matches the whole brain name and is not capped at
the 50 newest rows. Otherwise a brain's run list in the tuning widget would
also show the runs of every brain whose name merely contains this one, and
once a brain had more than 50 runs its older ones -- its marked Best among
them -- would fall out of the list and out of the widget's "has this detection
been classified" lookups. A person typing part of a name into a search box
still gets the substring search, by asking for it (brain_contains).
"""
import csv

from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

BRAIN = "12_PROJ_01_02_1p625x_z4"
LONGER = "112_PROJ_01_02_1p625x_z4"   # contains BRAIN as a substring


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


def _run(exp_id, brain, day, minute, **extra):
    row = {"exp_id": exp_id, "brain": brain, "exp_type": "detection",
           "status": "completed",
           "created_at": "2026-01-%02dT00:%02d:00" % (day, minute)}
    row.update(extra)
    return row


def test_another_brain_whose_name_contains_this_one_is_not_listed(tmp_path):
    t = _tracker(tmp_path, [
        _run("det_mine", "MOUSE_A/" + BRAIN, 1, 0),
        _run("det_widget", BRAIN, 1, 1),
        _run("det_other", LONGER, 1, 2),
        _run("det_other2", "MOUSE_B/" + LONGER, 1, 3),
    ])
    got = {r["exp_id"] for r in t.search(brain=BRAIN, exp_type="detection")}
    assert got == {"det_mine", "det_widget"}


def test_a_brain_with_many_runs_keeps_its_oldest_run_and_its_best(tmp_path):
    rows = [_run("det_best", BRAIN, 1, 0, marked_best="True")]
    rows += [_run("det_%02d" % i, BRAIN, 2 + i // 60, i % 60) for i in range(60)]
    t = _tracker(tmp_path, rows)
    got = t.search(brain=BRAIN, exp_type="detection")
    assert len(got) == 61, "every run of the brain, not the 50 newest"
    assert "det_best" in {r["exp_id"] for r in got}


def test_a_page_is_still_available_when_asked_for(tmp_path):
    rows = [_run("det_%02d" % i, BRAIN, 2, i) for i in range(10)]
    t = _tracker(tmp_path, rows)
    page = t.search(brain=BRAIN, limit=3)
    assert [r["exp_id"] for r in page] == ["det_09", "det_08", "det_07"]


def test_a_typed_search_still_finds_part_of_a_name(tmp_path):
    t = _tracker(tmp_path, [
        _run("det_mine", BRAIN, 1, 0),
        _run("det_other", LONGER, 1, 1),
    ])
    got = {r["exp_id"] for r in t.search(brain_contains="PROJ_01_02")}
    assert got == {"det_mine", "det_other"}
