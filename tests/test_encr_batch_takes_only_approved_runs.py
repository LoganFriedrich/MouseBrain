"""The 2D batch takes parameters only from a run a person marked best, and only
when that run used the batch's own method and recorded the sigma it used.

Defects covered:
- an unmarked tracker still changed the batch's parameters: the newest
  completed run (usually the previous batch's own row) was used as if approved;
- the newest colocalization row of ANY method became the batch's sigma
  threshold, so a fold-change value could be used as a sigma.
"""

import pytest

from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker
try:
    from mousebrain.plugin_2d.sliceatlas.tracker.schema import COLOC_SIGMA_RECORDED_VERSION
except ImportError:   # lets the same tests show the old behavior on older code
    COLOC_SIGMA_RECORDED_VERSION = "1.1.0"

SAMPLE_A = "PRJ01_01_S1_RA"
SAMPLE_B = "PRJ01_02_S2_RB"


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    # get_tuned_parameters builds its own SliceTracker(); hand it this one.
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    return t


def _coloc(t, sample, *, method="background_mean", value=0.0, bg="gmm",
           channel="green", pct=10.0, version=COLOC_SIGMA_RECORDED_VERSION,
           status="completed", parent=None,
           notes="soma_dilation=6; bg_exclusion_dilation=50"):
    run_id = t.log_colocalization(
        sample_id=sample, signal_channel=channel, background_method=bg,
        background_percentile=pct, threshold_method=method, threshold_value=value,
        parent_run=parent, status="started", script_version=version, notes=notes)
    t.update_status(run_id, status=status)
    return run_id


def test_unmarked_tracker_changes_no_batch_parameter(tracker):
    det = tracker.log_detection(sample_id=SAMPLE_A, model="threshold+log", status="started")
    tracker.update_status(det, status="completed", det_nuclei_found=12)
    _coloc(tracker, SAMPLE_A, value=1.5, parent=det)
    assert be.get_tuned_parameters() == {}


def test_newest_fold_change_row_is_never_a_sigma(tracker):
    det = tracker.log_detection(sample_id=SAMPLE_A, status="started")
    tracker.update_status(det, status="completed")
    _coloc(tracker, SAMPLE_B, method="fold_change", value=2.0)
    assert be.get_tuned_parameters() == {}


def test_marked_best_background_mean_run_supplies_sigma_and_percentile(tracker):
    best = _coloc(tracker, SAMPLE_A, value=0.5, pct=15.0)
    tracker.mark_as_best(best)
    # A newer, unmarked row in other units must not win.
    _coloc(tracker, SAMPLE_B, method="fold_change", value=2.0)
    tuned = be.get_tuned_parameters()
    assert tuned["threshold"] == 0.5
    assert tuned["bg_percentile"] == 15.0
    assert tuned["source_runs"] == [best]


@pytest.mark.parametrize("kwargs", [
    {"method": "fold_change"},
    {"bg": "percentile"},
    {"bg": "gmm+local"},
    {"channel": "dual"},
    {"version": "1.0.0"},
    {"status": "failed"},
    {"notes": "soma_dilation=0; bg_exclusion_dilation=50"},   # nuclear intensity
    {"notes": ""},                                             # dilation unknown
])
def test_marked_best_run_with_another_method_is_refused(tracker, kwargs, capsys):
    run = _coloc(tracker, SAMPLE_A, value=2.0, **kwargs)
    tracker.mark_as_best(run)
    assert be.get_tuned_parameters() == {}
    assert f"marked-best run {run}" in capsys.readouterr().out


def test_marked_best_runs_that_disagree_stop_the_batch(tracker):
    tracker.mark_as_best(_coloc(tracker, SAMPLE_A, value=0.0))
    tracker.mark_as_best(_coloc(tracker, SAMPLE_B, value=1.0))
    with pytest.raises(Exception, match="different parameters"):
        be.get_tuned_parameters()


def test_marked_best_runs_that_agree_are_used_together(tracker):
    a = _coloc(tracker, SAMPLE_A, value=1.0)
    b = _coloc(tracker, SAMPLE_B, value=1.0)
    tracker.mark_as_best(a)
    tracker.mark_as_best(b)
    assert be.get_tuned_parameters()["source_runs"] == sorted([a, b])


def test_an_old_approval_is_not_lost_behind_newer_rows(tracker):
    best = _coloc(tracker, SAMPLE_A, value=0.5)
    tracker.mark_as_best(best)
    for i in range(60):   # search() kept only the 50 newest rows
        tracker.log_detection(sample_id=f"PRJ01_03_S{i}_RC", status="completed")
        _coloc(tracker, f"PRJ01_03_S{i}_RC", method="fold_change", value=2.0)
    # created_at has one-second resolution; make the marked row the oldest.
    rows = tracker._read_all()
    for r in rows:
        if r["run_id"] == best:
            r["created_at"] = "2000-01-01T00:00:00"
    import csv
    from mousebrain.plugin_2d.sliceatlas.tracker.schema import CSV_COLUMNS
    with open(tracker.csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in CSV_COLUMNS} for r in rows)
    assert be.get_tuned_parameters()["threshold"] == 0.5


# --- only this project's marked runs count -----------------------------------

OTHER_PROJECT_SAMPLE = "QQQ_01_01_S1_RA"


def test_marked_run_of_another_project_changes_nothing(tracker):
    tracker.mark_as_best(_coloc(tracker, OTHER_PROJECT_SAMPLE, value=0.5))
    assert be.get_tuned_parameters(project="PRJ", sample_ids={SAMPLE_A}) == {}


def test_marked_run_of_another_project_does_not_conflict_with_this_one(tracker):
    mine = _coloc(tracker, SAMPLE_A, value=1.0)
    tracker.mark_as_best(mine)
    tracker.mark_as_best(_coloc(tracker, OTHER_PROJECT_SAMPLE, value=0.5))
    tuned = be.get_tuned_parameters(project="PRJ", sample_ids={SAMPLE_A})
    assert tuned["threshold"] == 1.0 and tuned["source_runs"] == [mine]


def test_marked_run_filed_under_the_project_counts_without_its_file(tracker):
    run = _coloc(tracker, "PRJ_01_01_S9_RA", value=0.25)
    tracker.mark_as_best(run)
    assert be.get_tuned_parameters(project="PRJ", sample_ids=set())["threshold"] == 0.25
