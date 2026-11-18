"""The 2D batch and the registry hold no built-in lab values.

Defect covered: with no approved-method file configured, the registry fell
back to a parameter set compiled into the tool (nucleus diameter range,
threshold fraction, LoG threshold, sigma, dilations) and the slice batch ran
with it -- printing every value as "[approved method]" and registering its
entries as current against that built-in set. A setting nobody chose was
used and recorded as approved.

Now: no approved-method file means nothing is approved. The batch stops
before any sample and names every setting it needs and how to give it; a
value is labeled "[approved method]" only when it came from the
approved-method file; and the registry judges nothing against a built-in
method.
"""

import json
import sys

import pytest

from mousebrain import analysis_registry as ar
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"

# Test data only: an approved method a lab might write.
METHOD = {
    "detection": "threshold+log", "log_decision_tree": True,
    "colocalization": "background_mean", "background": "gmm",
    "size_filter_um": [9, 24], "threshold_fraction": 0.3, "log_threshold": 0.01,
    "sigma_threshold": 0.5, "soma_dilation": 5, "background_percentile": 12,
    "bg_exclusion_dilation": 40,
}

ALL_SETTINGS = ("size_filter_um", "threshold_fraction", "log_threshold",
                "sigma_threshold", "soma_dilation", "background_percentile",
                "bg_exclusion_dilation")
ALL_OPTIONS = ("--diameter", "--threshold-fraction", "--log-threshold",
               "--threshold", "--soma-dilation", "--bg-percentile", "--dilation")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """An empty registry root (no approved-method file), a tracker and one
    image, all in tmp_path. process_single is replaced by a recorder."""
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    (tmp_path / "registry").mkdir()
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
    return tmp_path, calls


def _main(monkeypatch, tmp_path, *extra):
    """Run the batch; return its exit code (None when it ran to the end)."""
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out"),
                                      "--ignore-tracker", *extra])
    try:
        be.main()
    except SystemExit as e:
        return e.code
    return None


def _approve(tmp_path, method):
    (tmp_path / "registry" / "approved_method.json").write_text(
        json.dumps(method), encoding="utf-8")


def test_with_no_approved_method_the_batch_stops_and_names_every_setting(
        env, monkeypatch, capsys):
    tmp_path, calls = env
    code = _main(monkeypatch, tmp_path)
    out, err = capsys.readouterr()
    assert code == 2
    assert calls == []                      # no sample was attempted
    assert not (tmp_path / "out").exists()  # nothing was written
    for name in ALL_SETTINGS + ALL_OPTIONS:
        assert name in err, name
    assert "none configured" in err         # where it looked: nowhere
    assert "approved_method.json" in err    # how to fix it
    assert "[approved method]" not in out


def test_a_setting_the_approved_method_leaves_out_is_named_not_filled_in(
        env, monkeypatch, capsys):
    tmp_path, calls = env
    _approve(tmp_path, {k: v for k, v in METHOD.items() if k != "log_threshold"})
    code = _main(monkeypatch, tmp_path)
    err = capsys.readouterr().err
    assert code == 2 and calls == []
    assert "log_threshold (or --log-threshold <value>)" in err
    assert "threshold_fraction (or" not in err   # only the missing one is listed


def test_values_from_the_approved_method_file_run_and_are_labeled_approved(
        env, monkeypatch, capsys):
    tmp_path, calls = env
    _approve(tmp_path, METHOD)
    assert _main(monkeypatch, tmp_path) is None
    run = calls[0]
    assert (run["min_diameter_um"], run["max_diameter_um"]) == (9, 24)
    assert run["threshold_fraction"] == 0.3 and run["log_threshold"] == 0.01
    assert run["sigma_threshold"] == 0.5 and run["soma_dilation"] == 5
    assert run["bg_percentile"] == 12 and run["dilation"] == 40
    out = capsys.readouterr().out
    assert "nucleus diameter 9-24 um [approved method]" in out
    assert "NOT the approved method" not in out


def test_typed_detector_settings_reach_the_detector_and_are_labeled_typed(
        env, monkeypatch, capsys):
    tmp_path, calls = env
    _approve(tmp_path, METHOD)
    code = _main(monkeypatch, tmp_path,
                 "--threshold-fraction", "0.25", "--log-threshold", "0.02")
    assert code is None
    assert calls[0]["threshold_fraction"] == 0.25
    assert calls[0]["log_threshold"] == 0.02
    out = capsys.readouterr().out
    assert "threshold_fraction=0.25 [command line]" in out
    assert "log_threshold=0.02 [command line]" in out


def test_the_registry_judges_nothing_against_a_built_in_method(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(root))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    src = tmp_path / "m.csv"
    src.write_text("label\n1\n", encoding="utf-8")
    reg = ar.AnalysisRegistry(analysis_name="PRJ_Detection", registry_root=root,
                              follows_approved_method=True)
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(src)}, results={"n_nuclei": 1},
                        method_params=dict(METHOD))
    header = json.loads(reg.registry_path.read_text(encoding="utf-8"))
    # No approved-method file: no approved method, so nothing is current or
    # stale "against" one, and the header says where it looked.
    assert header["approved_method"] is None
    assert header["approved_method_hash"] is None
    assert header["approved_method_source"] == "none configured"
    assert ar.get_approved_method(root) is None
    assert be.batch_approved_method("PRJ_Detection") == {}


def test_a_setting_the_approved_method_gives_as_null_is_named_not_used(
        env, monkeypatch, capsys):
    """A hand-edited approved_method.json may hold a key with no value. That
    key is unset, exactly like a missing one: the batch stops with the
    message naming it, instead of labeling None "[approved method]" and
    failing later inside the detector."""
    tmp_path, calls = env
    _approve(tmp_path, dict(METHOD, sigma_threshold=None, size_filter_um=None))
    code = _main(monkeypatch, tmp_path)
    out, err = capsys.readouterr()
    assert code == 2 and calls == []
    assert "sigma_threshold (or --threshold <value>)" in err
    assert "size_filter_um (or --diameter MIN MAX)" in err
    assert "threshold_fraction (or" not in err   # the given ones are not listed
    assert "[approved method]" not in out


def test_resolve_parameters_treats_a_null_approved_value_as_unset():
    values, sources = be.resolve_parameters(
        {}, {}, dict(METHOD, log_threshold=None))
    assert "log_threshold" not in values and "log_threshold" not in sources
    assert be.unset_parameters(values) == ["log_threshold"]
    assert sources["threshold_fraction"] == "approved method"
