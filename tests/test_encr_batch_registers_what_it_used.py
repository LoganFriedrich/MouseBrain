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
    PROJECT, SAMPLE, _fake_loader, _nd2, _run_main, _synthetic_image, env)


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
    reg = AnalysisRegistry(analysis_name="ENCR_Detection")
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


def test_recorded_size_range_is_the_one_detection_used(env, monkeypatch, capsys):
    t, tmp_path = env
    approved = _approve(tmp_path, monkeypatch, size_filter_um=[8, 25])
    _fake_loader(monkeypatch, *_synthetic_image())
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker"])
    stale, reg = _stale_samples()
    recorded = reg.get_entry(SAMPLE)["method_params"]["size_filter_um"]
    assert recorded == [10, 25] and recorded != approved["size_filter_um"]
    assert SAMPLE in stale
    assert "NOT the approved method" in capsys.readouterr().out
