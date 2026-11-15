"""Step 6 writes the production count files correctly, with provenance.

The archive append wrote each old row in the sorted union of all columns but
never rewrote the header, so every archived row after the first had its values
under the wrong names. The det_* and voxel columns were never filled, the
subject column was built from "<mouse>/<brain>", and nothing was entered in
the analysis registry.
"""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from mousebrain.region_counts_io import (append_archive_row, detection_provenance,
                                         write_rows_atomic)

STEP6 = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "6_count_regions.py"
BRAIN = "101_PROJ_01_02/101_PROJ_01_02_2p5x_z5"  # mouse folder / brain folder


@pytest.fixture
def step6(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("count_step", STEP6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("REGION_COUNTS_CSV", "REGION_COUNTS_ARCHIVE_CSV",
                 "ELIFE_COUNTS_CSV", "ELIFE_COUNTS_ARCHIVE_CSV"):
        monkeypatch.setattr(mod, name, tmp_path / (name.lower() + ".csv"))
    return mod


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        return header, [dict(zip(header, row)) if len(row) == len(header) else row for row in r]


def test_every_archived_row_keeps_its_values_under_their_names(step6, tmp_path):
    counts = [{"AAA": 1, "BBB": 2}, {"BBB": 3, "CCC": 4}, {"AAA": 5, "DDD": 6}, {"EEE": 7}]
    for c in counts:
        step6.update_region_counts_csv(BRAIN, c, {"atlas": "atlas_x"})
    header, rows = _read(tmp_path / "region_counts_archive_csv.csv")
    assert all(isinstance(r, dict) for r in rows), "a row does not match the header"
    assert len(rows) == 3
    for archived, c in zip(rows, counts[:3]):
        assert archived["total_cells"] == str(sum(c.values()))
        for region, n in c.items():
            assert archived["region_" + region] == str(n)


def test_the_subject_comes_from_the_brain_folder_name(step6, tmp_path):
    step6.update_region_counts_csv(BRAIN, {"AAA": 1}, {"atlas": "atlas_x"})
    _h, rows = _read(tmp_path / "region_counts_csv.csv")
    assert rows[0]["brain"] == BRAIN  # the key an integrator matches on is unchanged
    assert rows[0]["subject"] == "PROJ_01_02"
    assert rows[0]["brain_id"] == "101"


def test_provenance_columns_are_filled_from_the_detection_params(step6, tmp_path):
    step6.update_region_counts_csv(BRAIN, {"AAA": 1}, {
        "atlas": "atlas_x", "preset": "paradigm_best", "ball_xy": "6", "ball_z": "15",
        "soma_diameter": "16", "threshold": "10", "voxel_xy": "4.0", "voxel_z": "4.0"})
    _h, rows = _read(tmp_path / "region_counts_csv.csv")
    assert (rows[0]["det_ball_xy"], rows[0]["det_threshold"], rows[0]["voxel_z"]) == ("6", "10", "4.0")


def test_an_archive_already_out_of_line_is_kept_aside_not_extended(tmp_path):
    bad = tmp_path / "archive.csv"
    bad.write_text("brain,total_cells\nb1,1,EXTRA,MORE\n", encoding="utf-8")
    append_archive_row(bad, {"brain": "b2", "total_cells": "2"})
    kept = list(tmp_path.glob("archive.malformed-*.csv"))
    assert len(kept) == 1 and "EXTRA" in kept[0].read_text(encoding="utf-8")
    assert _read(bad)[1] == [{"brain": "b2", "total_cells": "2"}]


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    write_rows_atomic(tmp_path / "x.csv", ["a"], [{"a": 1}])
    assert [p.name for p in tmp_path.iterdir()] == ["x.csv"]


def test_provenance_follows_classification_prefilter_detection(tmp_path):
    brain = tmp_path / "101_PROJ_01_02_2p5x_z5"
    cells = brain / "5_Classified_Cells" / "cells.xml"
    cells.parent.mkdir(parents=True)
    cells.write_text("<x/>")
    det_dir = brain / "4_Cell_Candidates"
    pre = det_dir / "prefiltered" / "interior_candidates.xml"
    rows = [
        {"exp_id": "det_old", "exp_type": "detection", "status": "completed", "brain": BRAIN,
         "created_at": "2000-01-01T00:00:00", "output_path": str(det_dir), "det_ball_xy": "5"},
        {"exp_id": "det_ok", "exp_type": "detection", "status": "completed", "brain": BRAIN,
         "created_at": "2000-01-02T00:00:00", "output_path": str(det_dir), "det_ball_xy": "6",
         "det_threshold": "10", "voxel_xy": "4.0"},
        {"exp_id": "det_after", "exp_type": "detection", "status": "completed", "brain": BRAIN,
         "created_at": "2000-01-04T00:00:00", "output_path": str(det_dir), "det_ball_xy": "9"},
        {"exp_id": "pre_1", "exp_type": "prefilter", "status": "completed", "brain": BRAIN,
         "created_at": "2000-01-03T00:00:00", "input_path": str(det_dir / "detected_cells.xml"),
         "output_path": str(pre)},
        {"exp_id": "cla_1", "exp_type": "classification", "status": "completed", "brain": BRAIN,
         "created_at": "2000-01-03T00:00:01", "input_path": str(pre),
         "output_path": str(cells.parent), "class_model_path": "model.h5"},
    ]
    prov = detection_provenance(rows, brain.name, cells)
    assert prov["exp_id"] == "det_ok"
    assert (prov["ball_xy"], prov["threshold"], prov["voxel_xy"]) == ("6", "10", "4.0")
    assert prov["classification_exp_id"] == "cla_1" and prov["model"] == "model.h5"


def test_provenance_is_none_rather_than_a_guess(tmp_path):
    cells = tmp_path / "5_Classified_Cells" / "cells.xml"
    cells.parent.mkdir()
    cells.write_text("<x/>")
    assert detection_provenance([], "B", cells) is None


def test_counts_are_entered_in_the_analysis_registry(step6, tmp_path, monkeypatch):
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "Registry"))
    brain = tmp_path / "101_PROJ_01_02_2p5x_z5"
    out = brain / "6_Region_Analysis" / "cell_counts_by_region.csv"
    out.parent.mkdir(parents=True)
    out.write_text("region,count\nAAA,1\n")
    step6.register_counts(brain, out, 1, {"atlas": "atlas_x", "ball_xy": "6"},
                          {"exp_id": "det_ok", "classification_exp_id": "cla_1", "model": "m"},
                          brain / "5_Classified_Cells" / "cells.xml", brain / "3_Registered_Atlas")
    manifest = json.loads((tmp_path / "Registry" / "exports" / step6.REGISTRY_ANALYSIS
                           / "registry.json").read_text())
    entry = manifest["entries"][brain.name]
    assert entry["method_params"]["detection_exp_id"] == "det_ok"
    assert entry["results"]["total_cells"] == 1


def test_step6_traces_provenance_and_registers():
    text = STEP6.read_text(encoding="utf-8")
    assert "detection_provenance(" in text and "register_counts(pipeline_folder" in text
