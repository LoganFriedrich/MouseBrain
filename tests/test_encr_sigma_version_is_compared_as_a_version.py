"""Which colocalization rows count as recording a sigma threshold, by version.

- a row logged before rows recorded their sigma (script_version 1.0.0, when
  the widget logged its fold-change box as the threshold of a background_mean
  run) must not supply the batch's sigma threshold when marked best (shown by
  test_a_marked_row_from_before_sigma_was_recorded_supplies_nothing);
- the version marker must be compared as numbers, so a LATER version (1.2.0,
  1.10.0) still counts -- an exact-text comparison would refuse every newer
  run already marked best and draw recorded sigmas as fold-change thresholds.

Imports of the batch and schema are inside the tests so the file collects on
code that has neither.
"""

import pytest


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
    from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    # get_tuned_parameters builds its own SliceTracker(); hand it this one.
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    return t


SAMPLE = "PRJ01_01_S1_RA"


def _marked_row(t, version, value=0.7):
    det = t.log_detection(sample_id=SAMPLE, model="threshold+log", status="started")
    t.update_status(det, status="completed", det_nuclei_found=12)
    run = t.log_colocalization(
        sample_id=SAMPLE, signal_channel="green", background_method="gmm",
        background_percentile=10.0, threshold_method="background_mean",
        threshold_value=value, parent_run=det, status="started",
        script_version=version, notes="soma_dilation=6; bg_exclusion_dilation=50")
    t.update_status(run, status="completed")
    t.mark_as_best(run)
    return run


def _row(version):
    return {"run_type": "colocalization", "status": "completed",
            "coloc_signal_channel": "green",
            "coloc_threshold_method": "background_mean",
            "coloc_background_method": "gmm", "coloc_threshold_value": "0.7",
            "script_version": version,
            "notes": "soma_dilation=6; bg_exclusion_dilation=50"}


def test_a_marked_row_from_before_sigma_was_recorded_supplies_nothing(tracker):
    from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
    _marked_row(tracker, "1.0.0")
    assert "threshold" not in be.get_tuned_parameters()


def test_a_marked_later_version_row_supplies_the_batch_sigma(tracker):
    from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
    _marked_row(tracker, "1.2.0")
    assert be.get_tuned_parameters()["threshold"] == 0.7


def test_versions_compare_as_numbers_not_text():
    from mousebrain.plugin_2d.sliceatlas.tracker.schema import records_sigma
    assert records_sigma({"script_version": "1.1.0"})
    assert records_sigma({"script_version": "1.2.0"})
    assert records_sigma({"script_version": "1.10.0"})
    assert records_sigma({"script_version": "2.0"})
    assert not records_sigma({"script_version": "1.0.0"})
    assert not records_sigma({"script_version": ""})
    assert not records_sigma({"script_version": "abc"})


def test_a_later_version_row_is_accepted_and_never_drawn_as_fold_change():
    from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
    from mousebrain.plugin_2d.sliceatlas.tracker.schema import coloc_fold_change_threshold
    row = _row("1.2.0")
    assert be.batch_approval_problem(row) is None
    assert coloc_fold_change_threshold(row) is None
    assert be.batch_approval_problem(_row("1.0.0")) is not None
