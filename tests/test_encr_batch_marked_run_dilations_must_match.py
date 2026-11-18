"""A marked-best run's sigma is used only at the dilations it was chosen at.

Defect covered: the batch took a marked run's sigma threshold but always ran
with the soma and background-exclusion dilations of the approved method (or
the command line). A run marked at soma dilation 3 handed its sigma to a batch
measuring at 6, and the banner credited the sigma to the marked run without a
word about the difference.
"""

import sys

import pytest

from example_method import write_example_approved_method
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import visualization as vis_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

try:
    from mousebrain.plugin_2d.sliceatlas.tracker.schema import COLOC_SIGMA_RECORDED_VERSION
except ImportError:   # lets the same tests show the old behavior on older code
    COLOC_SIGMA_RECORDED_VERSION = "1.1.0"

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"
OTHER = "PRJ_01_04_S5_RB"


@pytest.fixture
def env(tmp_path, monkeypatch):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    # The batch has no built-in values: it needs an approved method to run.
    write_example_approved_method(tmp_path / "registry")
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    monkeypatch.setattr(vis_mod, "save_all_qc_figures", lambda **k: [])
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    folder.mkdir(parents=True)
    (folder / f"{SAMPLE}.nd2").write_bytes(b"")
    calls = []

    def fake(nd2_path, output_dir, **kwargs):
        calls.append(kwargs)
        return {"file": nd2_path.name, "status": "error", "error": "not run in test"}

    monkeypatch.setattr(be, "process_single", fake)
    return t, tmp_path, calls


def _marked(t, sample, sigma=1.0, soma=6, excl=50):
    run = t.log_colocalization(
        sample_id=sample, signal_channel="green", background_method="gmm",
        background_percentile=10.0, threshold_method="background_mean",
        threshold_value=sigma, status="started",
        script_version=COLOC_SIGMA_RECORDED_VERSION,
        notes=f"soma_dilation={soma}; bg_exclusion_dilation={excl}")
    t.update_status(run, status="completed")
    t.mark_as_best(run)
    return run


def _main(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out"), *extra])
    return be.main()


def test_run_marked_at_another_soma_dilation_stops_the_batch(env, monkeypatch, capsys):
    t, tmp_path, calls = env
    run = _marked(t, SAMPLE, sigma=1.0, soma=3)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, tmp_path)
    assert exc.value.code == 2
    assert calls == []          # nothing was processed with a mismatched sigma
    err = capsys.readouterr().err
    assert run in err and "made at 3" in err and "would run at 6" in err
    assert "--soma-dilation 3" in err


def test_run_marked_at_another_exclusion_dilation_stops_the_batch(env, monkeypatch, capsys):
    t, tmp_path, calls = env
    _marked(t, SAMPLE, sigma=1.0, excl=30)
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path)
    assert calls == []
    assert "--dilation 30" in capsys.readouterr().err


def test_typing_the_marked_runs_dilation_runs_with_its_sigma(env, monkeypatch):
    t, tmp_path, calls = env
    _marked(t, SAMPLE, sigma=1.0, soma=3)
    _main(monkeypatch, tmp_path, "--soma-dilation", "3")
    assert calls[0]["sigma_threshold"] == 1.0 and calls[0]["soma_dilation"] == 3


def test_typing_another_dilation_runs_and_says_so(env, monkeypatch, capsys):
    t, tmp_path, calls = env
    _marked(t, SAMPLE, sigma=1.0, soma=3)
    _main(monkeypatch, tmp_path, "--soma-dilation", "6")
    assert calls[0]["soma_dilation"] == 6
    assert "the marked-best run was made at 3" in capsys.readouterr().out


def test_matching_dilations_run_without_a_question(env, monkeypatch):
    t, tmp_path, calls = env
    _marked(t, SAMPLE, sigma=1.0)
    _main(monkeypatch, tmp_path)
    assert calls[0]["sigma_threshold"] == 1.0 and calls[0]["soma_dilation"] == 6


def test_runs_with_one_sigma_at_different_dilations_disagree(env):
    t, _, _ = env
    _marked(t, SAMPLE, sigma=1.0, soma=6)
    _marked(t, OTHER, sigma=1.0, soma=3)
    with pytest.raises(be.TunedParameterConflict, match="soma dilation=3"):
        be.get_tuned_parameters(project=PROJECT)


def test_typing_both_numbers_runs_whatever_the_marked_runs_dilations(env, monkeypatch):
    # The marked run supplies nothing when both of its numbers are typed, so
    # the dilations it was made at say nothing about this batch.
    t, tmp_path, calls = env
    _marked(t, SAMPLE, sigma=0.5, soma=3, excl=30)
    _main(monkeypatch, tmp_path, "--threshold", "1.0", "--bg-percentile", "10")
    assert calls[0]["sigma_threshold"] == 1.0
    assert calls[0]["soma_dilation"] == 6 and calls[0]["dilation"] == 50
