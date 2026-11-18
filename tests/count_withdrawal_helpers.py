"""Shared set-up for the tests of withdrawn region counts: production tables
and an analysis registry in a temporary folder, a brain with a finished
(approved) registration, and step 6 loaded with its files pointed there.
Identifiers are invented."""
import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import mousebrain.config as config
from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.config import record_registered_space
from mousebrain.registration_approval import FINGERPRINT_FILES, approve
from mousebrain.tracker import ExperimentTracker

ROOT = Path(__file__).resolve().parents[1]
STEP6 = ROOT / "scripts" / "pipeline" / "6_count_regions.py"
NAME = "861_PROJ_08_61_2p5x_z5"
OTHER = "862_PROJ_08_62_2p5x_z5"
KEY = "mouse_861/" + NAME
OTHER_KEY = "mouse_862/" + OTHER
ANALYSIS = "3D_Region_Counts"


def write_table(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def brains_in(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r["brain"] for r in csv.DictReader(f)]


def make_brain(root, name=NAME, approved=True):
    """A brain with a finished registration (approved unless asked not to)
    and a production classification. Its images are older than the
    registration, as they are after a normal registration."""
    b = root / ("mouse_" + name.split("_")[0]) / name
    crop = b / "2_Cropped_For_Registration"
    (crop / "ch0").mkdir(parents=True)
    (crop / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    (crop / "ch0" / "Z0000.tif").write_bytes(b"x")
    old = time.time() - 600
    os.utime(crop / "ch0" / "Z0000.tif", (old, old))
    reg = b / "3_Registered_Atlas"
    reg.mkdir(parents=True)
    (reg / "brainreg.json").write_text("{}")
    record_registered_space(reg, "cropped", crop)
    for n in FINGERPRINT_FILES:
        (reg / n).write_bytes(b"x")
    if approved:
        approve(reg, approved_by="tester")
    (b / "5_Classified_Cells").mkdir()
    (b / "5_Classified_Cells" / "cells.xml").write_text("<Marker></Marker>")
    return b


def make_lab(tmp_path, monkeypatch):
    """Production tables holding this brain and another, and a registry
    holding a current region-count entry for each."""
    s = tmp_path / "2_Data_Summary"
    s.mkdir()
    monkeypatch.setattr(config, "DATA_SUMMARY_DIR", s)
    registry_root = tmp_path / "Registry"
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(registry_root))
    write_table(s / "region_counts.csv", [
        {"brain": KEY, "total_cells": "5", "region_AAA": "5"},
        {"brain": OTHER_KEY, "total_cells": "3", "region_AAA": "3"}])
    write_table(s / "elife_counts.csv", [
        {"brain": KEY, "total_cells": "5", "group_A": "5"},
        {"brain": OTHER_KEY, "total_cells": "3", "group_A": "3"}])
    (s / ("mouse_861_%s_counts.csv" % NAME)).write_text("region_acronym,cell_count\nAAA,5\n")
    reg = AnalysisRegistry(analysis_name=ANALYSIS, registry_root=registry_root,
                           follows_approved_method=False)
    for sample, n in ((NAME, 5), (OTHER, 3)):
        f = tmp_path / ("%s_counts.csv" % sample)
        f.write_text("region_acronym,cell_count\nAAA,%d\n" % n)
        reg.register_output(sample=sample, category="region_counts",
                            files={"region_counts": str(f)},
                            results={"total_cells": n}, method_params={"atlas": "atlas_x"})
    return {"summary": s, "registry": registry_root, "root": tmp_path / "1_Brains"}


def put_in_production(tmp_path, monkeypatch, brain_folder, mouse="mouse_x"):
    """A production table (in a temporary 2_Data_Summary that the
    configuration points at) holding a row for this brain: a counted brain
    is one with its own count files AND a row in region_counts.csv."""
    s = tmp_path / "2_Data_Summary"
    monkeypatch.setattr(config, "DATA_SUMMARY_DIR", s)
    write_table(s / "region_counts.csv",
                [{"brain": "%s/%s" % (mouse, brain_folder), "total_cells": "1"}])
    return s


def entry_current(registry_root, sample):
    data = json.loads((registry_root / "exports" / ANALYSIS / "registry.json")
                      .read_text(encoding="utf-8"))
    return any(e.get("sample") == sample and e.get("is_current", True)
               for e in data["entries"].values())


def load_step6(lab, tmp_path, monkeypatch, module_name):
    """Step 6 with its production files in the lab folder, its tracker in
    tmp_path, and the counting itself replaced by one that writes a
    one-region result (``mod.counted`` lists what it counted;
    ``mod.during_count(brain)`` runs while it counts)."""
    spec = importlib.util.spec_from_file_location(module_name, STEP6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = lab["summary"]
    monkeypatch.setattr(mod, "REGION_COUNTS_CSV", s / "region_counts.csv")
    monkeypatch.setattr(mod, "REGION_COUNTS_ARCHIVE_CSV", s / "region_counts_archive.csv")
    monkeypatch.setattr(mod, "ELIFE_COUNTS_CSV", s / "elife_counts.csv")
    monkeypatch.setattr(mod, "ELIFE_COUNTS_ARCHIVE_CSV", s / "elife_counts_archive.csv")
    monkeypatch.setattr(mod, "SUMMARY_DATA_DIR", s)
    tracker_csv = tmp_path / "tracker" / "calibration_runs.csv"
    monkeypatch.setattr(mod, "ExperimentTracker", lambda: ExperimentTracker(tracker_csv))
    monkeypatch.setattr(mod, "generate_comparison_report", lambda **k: None)
    monkeypatch.setattr(mod, "generate_cross_brain_graphs", lambda *a, **k: None)
    mod.counted = []
    mod.during_count = None

    def counts(cells_path, registration_path, output_path, *a, **k):
        mod.counted.append(Path(registration_path).parent.name)
        if mod.during_count:
            mod.during_count(Path(registration_path).parent)
        out = Path(output_path) / "cell_counts_by_region.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("region_acronym,cell_count\nAAA,7\n", encoding="utf-8")
        return True, 0.1, 7, out, "brainglobe_transform"
    monkeypatch.setattr(mod, "run_brainglobe_counts", counts)
    mod.tracker_rows = lambda: [r for r in ExperimentTracker(tracker_csv)._read_all()
                                if r.get("exp_type") == "counts"]
    return mod
