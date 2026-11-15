"""The 2D batch runs with the parameters it states, and states the ones it runs.

Defects covered:
- a value typed on the command line that equalled the built-in default
  (e.g. --threshold 0) was replaced by a tracker value;
- the detector ran on its own default diameter range while the banner printed
  pixel areas that were never passed and the docstring said 8-25 um;
- the tracker's detection row recorded the tracker's own default areas
  (50/5000 px), never the areas the detector applied.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.core import visualization as vis_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

try:
    from mousebrain.plugin_2d.sliceatlas.tracker.schema import COLOC_SIGMA_RECORDED_VERSION
except ImportError:   # lets the same tests show the old behaviour on older code
    COLOC_SIGMA_RECORDED_VERSION = "1.1.0"

PROJECT = "PRJ"
SAMPLE = "X01_02_S3_RA"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tracker, registry root and data folder that all live in tmp_path."""
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    monkeypatch.setattr(vis_mod, "save_all_qc_figures", lambda **k: [])
    return t, tmp_path


def _synthetic_image(n_cells=9):
    """Nuclear channel with round bright nuclei ~15 um across at 1 um/px."""
    yy, xx = np.mgrid[0:240, 0:240]
    red = np.full((240, 240), 10.0)
    green = np.full((240, 240), 10.0)
    centres = [(40 + 70 * (i // 3), 40 + 70 * (i % 3)) for i in range(n_cells)]
    for k, (cy, cx) in enumerate(centres):
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= 7 ** 2
        red[disk] = 1000.0
        ring = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 10 ** 2)
        if k % 2 == 0:
            green[ring] = 500.0
    return red.astype(np.float32), green.astype(np.float32)


def _fake_loader(monkeypatch, red, green):
    monkeypatch.setattr(io_mod, "load_image",
                        lambda path, **k: (np.stack([green, red]), {"voxel_size_um": {"x": 1.0}}))
    monkeypatch.setattr(io_mod, "extract_channels",
                        lambda data, red_idx=0, green_idx=1, **k: (red, green))


def _nd2(tmp_path, name=SAMPLE):
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"{name}.nd2"
    f.write_bytes(b"")
    return f


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["batch_encr"] + argv)
    return be.main()


def _mark_best_approval(t, sigma):
    """A marked-best detection run with a linked, marked-best coloc run.

    Both kinds are marked so the old and the new lookup each find an approval.
    """
    det = t.log_detection(sample_id=SAMPLE, model="threshold+log", status="started")
    t.update_status(det, status="completed")
    t.mark_as_best(det)
    col = t.log_colocalization(
        sample_id=SAMPLE, signal_channel="green", background_method="gmm",
        background_percentile=10.0, threshold_method="background_mean",
        threshold_value=sigma, parent_run=det, status="started",
        script_version=COLOC_SIGMA_RECORDED_VERSION,
        notes="soma_dilation=6; bg_exclusion_dilation=50")
    t.update_status(col, status="completed")
    t.mark_as_best(col)


def _capture_process_single(monkeypatch):
    calls = []

    def fake(nd2_path, output_dir, **kwargs):
        calls.append(kwargs)
        return {"file": nd2_path.name, "status": "error", "error": "not run in test"}

    monkeypatch.setattr(be, "process_single", fake)
    return calls


# --- a typed value always wins ------------------------------------------------

def test_typed_threshold_equal_to_default_is_not_overridden(env, monkeypatch):
    t, tmp_path = env
    _nd2(tmp_path)
    _mark_best_approval(t, sigma=1.0)
    calls = _capture_process_single(monkeypatch)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out"),
                            "--threshold", "0"])
    assert calls and calls[0]["sigma_threshold"] == 0


def test_untyped_threshold_takes_the_marked_best_run(env, monkeypatch):
    t, tmp_path = env
    _nd2(tmp_path)
    _mark_best_approval(t, sigma=1.0)
    calls = _capture_process_single(monkeypatch)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out")])
    assert calls[0]["sigma_threshold"] == 1.0


def test_resolve_parameters_precedence():
    approved = {"sigma_threshold": 0.5, "soma_dilation": 4, "bg_exclusion_dilation": 40,
                "background_percentile": 5, "threshold_fraction": 0.3, "log_threshold": 0.01}
    tuned = {"threshold": 1.25, "bg_percentile": 20.0, "source_runs": ["r1"]}
    values, sources = be.resolve_parameters(
        {"threshold": 0.0, "soma_dilation": None, "dilation": None, "bg_percentile": None},
        tuned, approved)
    assert values["sigma_threshold"] == 0.0 and sources["sigma_threshold"] == "command line"
    assert values["background_percentile"] == 20.0 and "r1" in sources["background_percentile"]
    assert values["soma_dilation"] == 4 and sources["soma_dilation"] == "approved method"
    assert values["threshold_fraction"] == 0.3
    values, sources = be.resolve_parameters(
        {"threshold": None, "soma_dilation": None, "dilation": None, "bg_percentile": None},
        {}, {})
    assert values == be.BUILTIN_PARAMETERS
    assert set(sources.values()) == {"built-in default"}


# --- the stated detection parameters are the ones used --------------------------

def test_banner_states_the_diameter_range_detection_uses(env, monkeypatch, capsys):
    t, tmp_path = env
    _nd2(tmp_path)
    _run_main(monkeypatch, ["--project", PROJECT, "--dry-run", "--ignore-tracker",
                            "--output", str(tmp_path / "out")])
    out = capsys.readouterr().out
    lo, hi = getattr(be, "BATCH_NUCLEUS_DIAMETER_UM", (10, 25))   # what the detector has always used
    assert f"nucleus diameter {lo}-{hi} um" in out
    assert "px" not in out.split("Detection:")[1].split("\n")[0].replace("pixel", "")


def test_pixel_area_options_that_did_nothing_are_refused(env, monkeypatch):
    t, tmp_path = env
    _nd2(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["--project", PROJECT, "--dry-run", "--ignore-tracker",
                                "--output", str(tmp_path / "out"), "--min-area", "3"])
    assert exc.value.code == 2


def test_tracker_row_records_the_areas_the_detector_applied(env, monkeypatch):
    t, tmp_path = env
    red, green = _synthetic_image()
    _fake_loader(monkeypatch, red, green)
    from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod
    applied = {}
    real = det_mod.detect_with_log_augmentation

    def spy(*a, **k):
        labels, details = real(*a, **k)
        applied.update(details)
        return labels, details

    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", spy)
    result = be.process_single(_nd2(tmp_path), tmp_path / "out")
    assert result["status"] == "completed", result
    lo, hi = getattr(be, "BATCH_NUCLEUS_DIAMETER_UM", (10, 25))   # what the detector has always used
    assert (applied["min_diameter_um"], applied["max_diameter_um"]) == (lo, hi)
    det_rows = t.search(run_type="detection")
    assert len(det_rows) == 1
    assert int(float(det_rows[0]["det_min_area"])) == applied["min_area"]
    assert int(float(det_rows[0]["det_max_area"])) == applied["max_area"]
    assert f"diameter {lo}-{hi} um" in det_rows[0]["notes"]
    coloc = t.search(run_type="colocalization")[0]
    assert coloc["script_version"] == COLOC_SIGMA_RECORDED_VERSION
    assert "soma_dilation=6" in coloc["notes"]


def _mark_coloc(t, sample, sigma):
    col = t.log_colocalization(
        sample_id=sample, signal_channel="green", background_method="gmm",
        background_percentile=10.0, threshold_method="background_mean",
        threshold_value=sigma, status="started",
        script_version=COLOC_SIGMA_RECORDED_VERSION,
        notes="soma_dilation=6; bg_exclusion_dilation=50")
    t.update_status(col, status="completed")
    t.mark_as_best(col)


def test_disagreeing_marked_runs_stop_the_batch_unless_both_values_are_typed(
        env, monkeypatch, capsys):
    t, tmp_path = env
    _nd2(tmp_path)
    _nd2(tmp_path, "X01_03_S4_RB")
    _mark_coloc(t, SAMPLE, 0.0)
    _mark_coloc(t, "X01_03_S4_RB", 1.0)
    calls = _capture_process_single(monkeypatch)
    base = ["--project", PROJECT, "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, base + ["--threshold", "0.5"])
    assert exc.value.code == 2 and not calls
    # The message names a remedy the GUI offers.
    assert "Unmark Best" in capsys.readouterr().err
    _run_main(monkeypatch, base + ["--threshold", "0.5", "--bg-percentile", "10"])
    assert calls[0]["sigma_threshold"] == 0.5


def test_a_marked_run_of_another_project_does_not_set_this_batch(env, monkeypatch):
    t, tmp_path = env
    _nd2(tmp_path)
    # Marked (detection and colocalization) for a slice of ANOTHER project.
    det = t.log_detection(sample_id="QQQ_01_01_S1_RA", model="threshold+log",
                          status="started")
    t.update_status(det, status="completed")
    t.mark_as_best(det)
    _mark_coloc(t, "QQQ_01_01_S1_RA", 0.5)
    calls = _capture_process_single(monkeypatch)
    _run_main(monkeypatch, ["--project", PROJECT, "--output", str(tmp_path / "out")])
    assert calls[0]["sigma_threshold"] == 0      # the approved method's, not 0.5


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


# --- the QC figures draw the cutoff the cells were split at --------------------

def test_qc_figures_draw_the_background_mean_cutoff_as_a_fold_change(env, monkeypatch):
    t, tmp_path = env
    _fake_loader(monkeypatch, *_synthetic_image())
    drawn = []
    monkeypatch.setattr(vis_mod, "save_all_qc_figures",
                        lambda **k: drawn.append(k) or [])
    from mousebrain.plugin_2d.sliceatlas.core import colocalization as coloc_mod
    seen = {}
    real = coloc_mod.ColocalizationAnalyzer.classify_positive_negative

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        seen.update(self.adaptive_diagnostics)
        return out

    monkeypatch.setattr(coloc_mod.ColocalizationAnalyzer, "classify_positive_negative", spy)
    result = be.process_single(_nd2(tmp_path), tmp_path / "out",
                               threshold=1.5, sigma_threshold=1.5)
    assert result["status"] == "completed", result
    expected = seen["positive_cutoff"] / seen["background_mean"]
    assert drawn and all(abs(k["threshold"] - expected) < 1e-3 for k in drawn)
    assert all(k["threshold"] != 1.5 for k in drawn)
