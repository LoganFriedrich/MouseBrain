"""The 2D batch takes its numbers from, compares itself with, and is judged by
ONE approved method: the one the registry holds for its analysis.

Defect covered: the batch resolved its untyped numbers and printed its
"NOT the approved method" differences against the SHARED approved method,
while the registry judges ENCR_Detection's entries by that analysis's own
entry in a per-analysis approved-method file. With such an entry the batch ran
the shared numbers, listed the wrong differences, and registered entries the
manifest header called stale.
"""

import json

from mousebrain.analysis_registry import AnalysisRegistry, DEFAULT_METHOD

from test_encr_batch_runs_and_records_its_parameters import (  # noqa: F401  (fixture)
    PROJECT, SAMPLE, _capture_process_single, _fake_loader, _nd2, _run_main,
    _synthetic_image, env)


def _per_analysis_file(tmp_path, monkeypatch, own):
    """A per-analysis approved-method file: shared default = built-in method
    with the batch's size range; ENCR_Detection gets *own*."""
    shared = dict(DEFAULT_METHOD)
    shared["size_filter_um"] = [10, 25]
    f = tmp_path / "approved.json"
    f.write_text(json.dumps({"default": shared,
                             "analyses": {"ENCR_Detection": own}}))
    monkeypatch.setenv("MOUSEBRAIN_APPROVED_METHOD", str(f))
    return shared


def _own_method():
    own = dict(DEFAULT_METHOD)
    own.update({"size_filter_um": [10, 25], "sigma_threshold": 0.75,
                "soma_dilation": 4, "background_percentile": 20})
    return own


def test_untyped_numbers_come_from_the_analysis_own_approved_method(env, monkeypatch):
    t, tmp_path = env
    _per_analysis_file(tmp_path, monkeypatch, _own_method())
    _nd2(tmp_path)
    calls = _capture_process_single(monkeypatch)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker"])
    assert calls[0]["sigma_threshold"] == 0.75
    assert calls[0]["soma_dilation"] == 4
    assert calls[0]["bg_percentile"] == 20


def test_banner_compares_with_the_analysis_own_approved_method(env, monkeypatch, capsys):
    t, tmp_path = env
    _per_analysis_file(tmp_path, monkeypatch, _own_method())
    _nd2(tmp_path)
    # Running exactly the analysis's own method: no difference to report.
    _run_main(monkeypatch, ["--project", PROJECT, "--dry-run", "--ignore-tracker",
                            "--output", str(tmp_path / "out")])
    assert "NOT the approved method" not in capsys.readouterr().out
    # A typed value is compared with the analysis's own sigma (0.75), not the
    # shared one (0).
    _run_main(monkeypatch, ["--project", PROJECT, "--dry-run", "--ignore-tracker",
                            "--threshold", "1.5", "--output", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "sigma_threshold: this run 1.5, approved 0.75" in out


def test_entries_hash_like_the_registry_header(env, monkeypatch):
    t, tmp_path = env
    _per_analysis_file(tmp_path, monkeypatch, _own_method())
    _fake_loader(monkeypatch, *_synthetic_image())
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker"])
    reg = AnalysisRegistry(analysis_name="ENCR_Detection")
    data = json.loads(reg.registry_path.read_text(encoding="utf-8"))
    entry = data["entries"][SAMPLE]
    assert entry["is_current"]
    assert entry["method_hash"] == data["approved_method_hash"]
    assert reg.check_staleness(reg.approved_method()) == []


def test_batch_approved_method_is_the_registry_approved_method(env, monkeypatch):
    from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
    t, tmp_path = env
    _per_analysis_file(tmp_path, monkeypatch, _own_method())
    reg = AnalysisRegistry(analysis_name=be.ANALYSIS_NAME, follows_approved_method=True)
    assert be.batch_approved_method() == reg.approved_method() == _own_method()


def test_analysis_without_an_approved_method_is_never_called_stale(env, monkeypatch, capsys):
    t, tmp_path = env
    _per_analysis_file(tmp_path, monkeypatch, None)
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--dry-run", "--ignore-tracker",
                            "--threshold", "1.5", "--output", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "Approved:  none" in out
    assert "NOT the approved method" not in out
