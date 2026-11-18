"""Step 6 never switches to the less accurate counting method on its own, and
the counts say which method placed the cells.

When the BrainGlobe transform could not run, step 6 quietly counted with a
direct lookup in the registered atlas image instead, and the counts it wrote
looked like any others. Identifiers are invented.
"""
import contextlib
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mousebrain.config import record_registered_space
from mousebrain.registration_approval import FINGERPRINT_FILES, approve

STEP6 = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "6_count_regions.py"


@pytest.fixture
def step6(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("count_step_method", STEP6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("REGION_COUNTS_CSV", "REGION_COUNTS_ARCHIVE_CSV",
                 "ELIFE_COUNTS_CSV", "ELIFE_COUNTS_ARCHIVE_CSV"):
        monkeypatch.setattr(mod, name, tmp_path / "summary" / (name.lower() + ".csv"))
    monkeypatch.setattr(mod, "SUMMARY_DATA_DIR", tmp_path / "summary")
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "Registry"))
    return mod


def _broken_transform(*a, **k):
    raise RuntimeError("deformation field could not be read")


def test_a_failed_transform_counts_nothing_unless_asked(step6, tmp_path, monkeypatch):
    direct = []
    monkeypatch.setattr(step6, "run_counts_brainglobe_transform", _broken_transform)
    monkeypatch.setattr(step6, "run_counts_direct_lookup",
                        lambda *a, **k: direct.append(1) or (True, 0.1, 5, tmp_path / "x.csv"))
    result = step6.run_brainglobe_counts(tmp_path / "cells.xml", tmp_path / "reg", tmp_path / "out")
    assert direct == []
    assert result[0] is False


def test_the_direct_lookup_runs_only_when_asked_and_says_so(step6, tmp_path, monkeypatch):
    monkeypatch.setattr(step6, "run_counts_brainglobe_transform", _broken_transform)
    monkeypatch.setattr(step6, "run_counts_direct_lookup",
                        lambda *a, **k: (True, 0.1, 5, tmp_path / "x.csv"))
    result = step6.run_brainglobe_counts(tmp_path / "cells.xml", tmp_path / "reg",
                                         tmp_path / "out", allow_direct_lookup=True)
    assert result[0] is True
    assert result[-1] == "direct_atlas_lookup"


class _Tracker:
    def log_counts(self, **k):
        return "count_1"

    @contextlib.contextmanager
    def finish_or_fail(self, exp_id):
        yield

    def update_status(self, *a, **k):
        pass

    def fail_unfinished(self, *a, **k):
        pass

    def _read_all(self):
        return []


def _brain(root, name):
    b = root / ("mouse_" + name) / name
    crop = b / "2_Cropped_For_Registration"
    (crop / "ch0").mkdir(parents=True)
    (crop / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    (crop / "ch0" / "Z0000.tif").write_bytes(b"x")
    reg = b / "3_Registered_Atlas"
    reg.mkdir(parents=True)
    (reg / "brainreg.json").write_text("{}")
    record_registered_space(reg, "cropped", crop)
    # Step 6 counts only a finished registration a person approved.
    for n in FINGERPRINT_FILES:
        (reg / n).write_bytes(b"x")
    approve(reg, approved_by="tester")
    (b / "5_Classified_Cells").mkdir()
    (b / "5_Classified_Cells" / "cells.xml").write_text("<Marker></Marker>")
    return b


@pytest.mark.parametrize("allow, method", [(False, "brainglobe_transform"),
                                           (True, "direct_atlas_lookup")])
def test_the_counts_record_which_method_placed_the_cells(step6, tmp_path, monkeypatch,
                                                         allow, method):
    root = tmp_path / "1_Brains"
    b = _brain(root, "811_PROJ_08_11_2p5x_z5")

    def counted(cells_path, registration_path, output_path, output_csv, *a, **k):
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(output_csv).write_text("region_acronym,cell_count\nAAA,2\n", encoding="utf-8")
        return True, 0.1, 2, Path(output_csv)

    if allow:
        monkeypatch.setattr(step6, "run_counts_brainglobe_transform", _broken_transform)
        monkeypatch.setattr(step6, "run_counts_direct_lookup",
                            lambda c, r, out_csv, *a, **k: counted(c, r, None, out_csv))
    else:
        monkeypatch.setattr(step6, "run_counts_brainglobe_transform", counted)
    monkeypatch.setattr(step6, "ExperimentTracker", _Tracker)
    monkeypatch.setattr(step6, "generate_comparison_report", lambda **k: None)
    monkeypatch.setattr(step6, "generate_cross_brain_graphs", lambda *a, **k: None)
    argv = ["6", "--brain", b.name, "--root", str(root)] + (["--allow-direct-lookup"] if allow else [])
    monkeypatch.setattr(sys, "argv", argv)
    step6.main()
    with open(step6.REGION_COUNTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["count_method"] == method
    manifest = json.loads((tmp_path / "Registry" / "exports" / step6.REGISTRY_ANALYSIS
                           / "registry.json").read_text())
    assert manifest["entries"][b.name]["method_params"]["count_method"] == method
