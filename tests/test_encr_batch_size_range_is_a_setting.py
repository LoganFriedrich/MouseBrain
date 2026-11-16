"""The 2D batch's nucleus diameter range is a setting, not a value in the code.

Defect covered: the batch detected at a diameter range fixed in its code and
ignored the approved method's size_filter_um, so with an approved method
naming another range every entry was registered stale and a lab could not
make the batch run at its approved range. Now the range comes from
--diameter MIN MAX, else the approved method; with neither the batch stops
before any sample and says how to set it.
"""

import json

import pytest

from mousebrain.analysis_registry import AnalysisRegistry, DEFAULT_METHOD, get_approved_method
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod

from test_encr_batch_runs_and_records_its_parameters import (  # noqa: F401  (fixture)
    PROJECT, SAMPLE, _fake_loader, _nd2, _run_main, _synthetic_image, env)


def _write_approved(tmp_path, monkeypatch, data):
    f = tmp_path / "approved.json"
    f.write_text(json.dumps(data))
    monkeypatch.setenv("MOUSEBRAIN_APPROVED_METHOD", str(f))


def _spy_detector(monkeypatch):
    seen = []
    real = det_mod.detect_with_log_augmentation

    def spy(*a, **k):
        seen.append((k["min_diameter_um"], k["max_diameter_um"]))
        return real(*a, **k)

    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", spy)
    return seen


def _entry_and_stale():
    reg = AnalysisRegistry(analysis_name="PRJ_Detection")
    stale = {e["sample"] for e in reg.check_staleness(get_approved_method())}
    return reg.get_entry(SAMPLE), stale


@pytest.mark.parametrize("approved_range", [[8, 25], [12, 30]])
def test_the_approved_range_reaches_the_detector_and_the_entry_is_current(
        env, monkeypatch, capsys, approved_range):
    t, tmp_path = env
    _write_approved(tmp_path, monkeypatch,
                    dict(DEFAULT_METHOD, size_filter_um=approved_range))
    _fake_loader(monkeypatch, *_synthetic_image())
    seen = _spy_detector(monkeypatch)
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker"])
    lo, hi = approved_range
    assert seen == [(lo, hi)]
    entry, stale = _entry_and_stale()
    assert entry["method_params"]["size_filter_um"] == approved_range
    assert SAMPLE not in stale
    out = capsys.readouterr().out
    assert f"nucleus diameter {lo}-{hi} um [approved method]" in out
    assert "NOT the approved method" not in out
    det = t.search(run_type="detection")[0]
    assert f"diameter {lo}-{hi} um" in det["notes"]


def test_a_typed_range_wins_and_is_recorded_as_not_approved(env, monkeypatch, capsys):
    t, tmp_path = env
    _write_approved(tmp_path, monkeypatch, dict(DEFAULT_METHOD, size_filter_um=[8, 25]))
    _fake_loader(monkeypatch, *_synthetic_image())
    seen = _spy_detector(monkeypatch)
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker", "--diameter", "11", "27"])
    assert seen == [(11, 27)]
    entry, stale = _entry_and_stale()
    assert entry["method_params"]["size_filter_um"] == [11, 27]
    assert SAMPLE in stale
    out = capsys.readouterr().out
    assert "nucleus diameter 11.0-27.0 um [command line]" in out
    assert "size_filter_um: this run [11.0, 27.0], approved [8, 25]" in out


@pytest.mark.parametrize("data", [
    # a per-analysis entry for the batch that leaves the range out
    {"analyses": {"PRJ_Detection": {k: v for k, v in DEFAULT_METHOD.items()
                                     if k != "size_filter_um"}}},
    # the batch's analysis is told it has no approved method at all
    {"analyses": {"PRJ_Detection": None}},
    # a flat approved method without the range
    {k: v for k, v in DEFAULT_METHOD.items() if k != "size_filter_um"},
])
def test_no_range_anywhere_stops_before_any_sample(env, monkeypatch, capsys, data):
    t, tmp_path = env
    _write_approved(tmp_path, monkeypatch, data)
    seen = _spy_detector(monkeypatch)
    _nd2(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                                "--ignore-tracker"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "size_filter_um" in err and "--diameter" in err
    assert seen == []
    assert t.search(run_type="detection") == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", [[25, 8], [0, 25], [10], ["10", "25"], [True, 25]])
def test_an_unusable_approved_range_stops(env, monkeypatch, capsys, bad):
    t, tmp_path = env
    _write_approved(tmp_path, monkeypatch, dict(DEFAULT_METHOD, size_filter_um=bad))
    seen = _spy_detector(monkeypatch)
    _nd2(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                                "--ignore-tracker"])
    assert exc.value.code == 2
    assert "smallest first" in capsys.readouterr().err
    assert seen == []


def test_a_typed_range_runs_when_the_batch_has_no_approved_method(env, monkeypatch):
    t, tmp_path = env
    _write_approved(tmp_path, monkeypatch, {"analyses": {"PRJ_Detection": None}})
    _fake_loader(monkeypatch, *_synthetic_image())
    seen = _spy_detector(monkeypatch)
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--ignore-tracker", "--diameter", "9", "24"])
    assert seen == [(9, 24)]
