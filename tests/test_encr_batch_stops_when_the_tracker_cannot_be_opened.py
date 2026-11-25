"""The 2D batch stops when it cannot open the tracker to look for a marked run.

When the tracker cannot be opened (its file held open by another program,
say), the batch must not print one warning and carry on as if no run were
marked best: it would run at the approved method's sigma, silently replacing
the sigma a person had chosen by marking a run, while its banner said "no run
of this project is marked best".

So it stops, as it does for disagreeing marked runs, unless the person types
both numbers a marked run could supply or runs with --ignore-tracker.
"""

import json
import sys

import pytest

from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"

# Test data only.
METHOD = {
    "detection": "threshold+log", "log_decision_tree": True,
    "colocalization": "background_mean", "background": "gmm",
    "size_filter_um": [9, 24], "threshold_fraction": 0.3, "log_threshold": 0.01,
    "sigma_threshold": 0.5, "soma_dilation": 5, "background_percentile": 12,
    "bg_exclusion_dilation": 40,
}


def _tracker_held_open(*a, **k):
    raise PermissionError("the tracker file is open in another program")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker_pkg, "SliceTracker", _tracker_held_open)
    root = tmp_path / "registry"
    root.mkdir()
    (root / "approved_method.json").write_text(json.dumps(METHOD), encoding="utf-8")
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(root))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    folder.mkdir(parents=True)
    (folder / f"{SAMPLE}.nd2").write_bytes(b"")
    calls = []

    def fake(nd2_path, output_dir, **kwargs):
        calls.append(kwargs)
        return {"file": nd2_path.name, "status": "error", "error": "not run in test"}

    monkeypatch.setattr(be, "process_single", fake)
    return tmp_path, calls


def _main(monkeypatch, tmp_path, *extra):
    """Run the batch; return its exit code (None when it ran to the end)."""
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out"), *extra])
    try:
        be.main()
    except SystemExit as e:
        return e.code
    return None


def test_a_tracker_that_cannot_be_opened_stops_the_batch(env, monkeypatch, capsys):
    tmp_path, calls = env
    code = _main(monkeypatch, tmp_path)
    out, err = capsys.readouterr()
    assert code == 2
    assert calls == []                   # no sample ran at a sigma nobody chose
    assert "tracker could not be opened" in err
    assert "open in another program" in err          # the reason is shown
    assert "--ignore-tracker" in err and "--bg-percentile" in err
    assert "no run of this project is marked best" not in out


def test_typing_only_one_number_still_stops(env, monkeypatch):
    tmp_path, calls = env
    assert _main(monkeypatch, tmp_path, "--threshold", "1.0") == 2
    assert calls == []


def test_typing_both_numbers_runs_and_says_the_tracker_was_not_read(
        env, monkeypatch, capsys):
    tmp_path, calls = env
    assert _main(monkeypatch, tmp_path, "--threshold", "1.0",
                 "--bg-percentile", "15") is None
    assert calls and calls[0]["sigma_threshold"] == 1.0
    out = capsys.readouterr().out
    assert "could not be opened" in out
    assert "no run of this project is marked best" not in out


def test_ignore_tracker_runs_without_opening_it(env, monkeypatch, capsys):
    tmp_path, calls = env
    assert _main(monkeypatch, tmp_path, "--ignore-tracker") is None
    assert calls and calls[0]["sigma_threshold"] == 0.5   # the approved value
