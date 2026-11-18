"""The 2D batch fills and judges its parameters by the approved method the
registry holds for ITS analysis.

Defect covered: the batch always read the general approved method. A
registry that gives each analysis its own approved method judges the batch's
entries by that one, so the batch filled its unset options from, and
compared its banner with, a method the registry does not judge it by --
announcing "approved" numbers the registry then reported stale.
"""

import sys

import pytest

from example_method import EXAMPLE_METHOD
from mousebrain import analysis_registry as ar
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"


@pytest.fixture
def env(tmp_path, monkeypatch):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
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
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out")])
    return calls


def test_unset_options_come_from_the_analysis_own_approved_method(env, monkeypatch, capsys):
    asked = []

    def per_analysis(name, *a, **k):
        asked.append(name)
        method = dict(EXAMPLE_METHOD)
        method.update({"sigma_threshold": 1.5, "soma_dilation": 4,
                       "size_filter_um": [10, 25]})
        return method

    monkeypatch.setattr(ar, "approved_method_for", per_analysis, raising=False)
    be.main()
    assert asked and set(asked) == {"PRJ_Detection"}
    assert env[0]["sigma_threshold"] == 1.5 and env[0]["soma_dilation"] == 4
    # The run uses exactly that method, so the banner raises no stale warning.
    assert "NOT the approved method" not in capsys.readouterr().out


def test_an_analysis_without_an_approved_method_runs_only_on_typed_values(env, monkeypatch, capsys):
    monkeypatch.setattr(ar, "approved_method_for", lambda name, *a, **k: None,
                        raising=False)
    # No approved method and nothing typed: there are no built-in values, so
    # the batch stops and names every setting it needs.
    with pytest.raises(SystemExit) as exc:
        be.main()
    assert exc.value.code == 2 and env == []
    err = capsys.readouterr().err
    for option in ("--diameter", "--threshold", "--soma-dilation", "--dilation",
                   "--bg-percentile", "--threshold-fraction", "--log-threshold"):
        assert option in err
    # Everything typed: the batch runs with exactly the typed values.
    monkeypatch.setattr(sys, "argv", sys.argv + [
        "--diameter", "9", "24", "--threshold", "0.5", "--soma-dilation", "5",
        "--dilation", "40", "--bg-percentile", "12",
        "--threshold-fraction", "0.3", "--log-threshold", "0.01"])
    be.main()
    assert env[0]["sigma_threshold"] == 0.5 and env[0]["soma_dilation"] == 5
    assert env[0]["threshold_fraction"] == 0.3 and env[0]["log_threshold"] == 0.01
    assert (env[0]["min_diameter_um"], env[0]["max_diameter_um"]) == (9, 24)
    out = capsys.readouterr().out
    assert "NOT the approved method" not in out
    assert "[approved method]" not in out
    assert "Approved:  none for PRJ_Detection" in out
