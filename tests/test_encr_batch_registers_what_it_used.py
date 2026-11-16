"""Registry entries of the 2D batch record the parameters the run used.

Defect covered: every entry recorded the approved method whatever the run
used, so a run with a typed --threshold hashed like an approved run and the
staleness check could never find it.
"""

import json

import pytest

from mousebrain.analysis_registry import AnalysisRegistry, DEFAULT_METHOD, get_approved_method
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be

from test_encr_batch_runs_and_records_its_parameters import (  # noqa: F401  (fixture)
    PROJECT, SAMPLE, _fake_loader, _mark_best_approval, _nd2, _run_main,
    _synthetic_image, env)


def _approve(tmp_path, monkeypatch, **overrides):
    """Write an approved-method file equal to what the batch runs by default."""
    method = dict(DEFAULT_METHOD)
    method["size_filter_um"] = [10, 25]
    method.update(overrides)
    f = tmp_path / "approved.json"
    f.write_text(json.dumps(method))
    monkeypatch.setenv("MOUSEBRAIN_APPROVED_METHOD", str(f))
    return method


def _stale_samples():
    reg = AnalysisRegistry(analysis_name="PRJ_Detection")
    return {e["sample"] for e in reg.check_staleness(get_approved_method())}, reg


def test_run_with_a_typed_threshold_is_reported_stale(env, monkeypatch):
    t, tmp_path = env
    _approve(tmp_path, monkeypatch)
    _fake_loader(monkeypatch, *_synthetic_image())
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker", "--threshold", "1.5"])
    stale, reg = _stale_samples()
    assert SAMPLE in stale
    assert reg.get_entry(SAMPLE)["method_params"]["sigma_threshold"] == 1.5


def test_run_with_the_approved_method_is_current(env, monkeypatch):
    t, tmp_path = env
    _approve(tmp_path, monkeypatch)
    _fake_loader(monkeypatch, *_synthetic_image())
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker"])
    stale, reg = _stale_samples()
    assert reg.get_entry(SAMPLE) is not None
    assert SAMPLE not in stale


def test_an_approved_key_the_batch_does_not_control_is_not_recorded(env, monkeypatch, capsys):
    # The approved method names a setting the batch has no control over; the
    # run did not use it, so its entry must not claim it did.
    t, tmp_path = env
    _approve(tmp_path, monkeypatch, other_detector="method_x")
    _fake_loader(monkeypatch, *_synthetic_image())
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker"])
    stale, reg = _stale_samples()
    assert "other_detector" not in reg.get_entry(SAMPLE)["method_params"]
    assert SAMPLE in stale
    assert "no setting for other_detector" in capsys.readouterr().out


def test_banner_says_a_marked_run_does_not_change_the_approved_method(
        env, monkeypatch, capsys):
    t, tmp_path = env
    _nd2(tmp_path)
    _mark_best_approval(t, sigma=1.0)
    _run_main(monkeypatch, ["--project", PROJECT, "--dry-run",
                            "--output", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "sigma_threshold: this run 1.0, approved 0" in out
    assert "does not change the approved method" in out
