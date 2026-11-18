"""A brain's counts leave production -- the count tables AND the analysis
registry -- whenever the registration or classification they were made from
no longer stands for the brain.

What went wrong: no withdrawal ever touched the brain's 3D_Region_Counts
entry in the analysis registry, so it stayed current and an integrator kept
mirroring it; and rejecting a registration, or a re-crop that made it out of
date, withdrew nothing at all. Identifiers are invented.
"""
import os
import time

import pytest

import mousebrain.config as config
from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.classification_runs import promote_to_production
from mousebrain.registration_approval import reject, withdraw

from count_withdrawal_helpers import (KEY, NAME, OTHER, OTHER_KEY, ANALYSIS, brains_in,
                                      entry_current, make_brain, make_lab)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    return make_lab(tmp_path, monkeypatch)


def _assert_withdrawn(lab):
    s = lab["summary"]
    assert brains_in(s / "region_counts.csv") == [OTHER_KEY]
    assert brains_in(s / "elife_counts.csv") == [OTHER_KEY]
    assert brains_in(s / "region_counts_archive.csv") == [KEY]
    assert not (s / ("mouse_861_%s_counts.csv" % NAME)).exists()
    assert not entry_current(lab["registry"], NAME), "registry entry still current"
    assert entry_current(lab["registry"], OTHER), "another brain's entry was touched"


def test_rejecting_a_registration_withdraws_its_counts_and_registry_entry(lab):
    b = make_brain(lab["root"])
    reject(b / "3_Registered_Atlas", rejected_by="tester")
    _assert_withdrawn(lab)


def test_taking_an_approval_back_withdraws_the_counts(lab):
    b = make_brain(lab["root"])
    withdraw(b / "3_Registered_Atlas", "test")
    _assert_withdrawn(lab)


def test_a_recrop_that_makes_the_registration_out_of_date_withdraws_its_counts(lab):
    from mousebrain.plugin_3d.crop_subprocess import new_partial, swap_in
    b = make_brain(lab["root"])
    crop = b / "2_Cropped_For_Registration"
    partial = new_partial(crop)
    (partial / "ch0").mkdir()
    (partial / "ch0" / "Z0000.tif").write_bytes(b"new")
    later = time.time() + 600  # clearly newer than the registration
    os.utime(partial / "ch0" / "Z0000.tif", (later, later))
    (partial / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    swap_in(crop, partial)
    _assert_withdrawn(lab)


def test_a_recrop_of_a_folder_the_registration_was_not_fitted_to_keeps_the_counts(lab):
    from mousebrain.plugin_3d.crop_subprocess import new_partial, swap_in
    b = make_brain(lab["root"])
    manual = b / "2_Cropped_For_Registration_Manual"
    partial = new_partial(manual)
    (partial / "ch0").mkdir()
    (partial / "ch0" / "Z0000.tif").write_bytes(b"new")
    (partial / "metadata.json").write_text("{}")
    swap_in(manual, partial)
    assert brains_in(lab["summary"] / "region_counts.csv") == [KEY, OTHER_KEY]
    assert entry_current(lab["registry"], NAME)


def test_replacing_the_classification_also_withdraws_the_registry_entry(lab, tmp_path):
    b = make_brain(lab["root"])
    run = tmp_path / "optimization_runs" / NAME / "class_20000101_000000"
    run.mkdir(parents=True)
    (run / "cells.xml").write_text("new")
    promote_to_production(run, b, exp_id="cla_9")
    _assert_withdrawn(lab)


def test_a_brain_already_out_of_the_tables_still_leaves_the_registry(lab):
    from mousebrain.region_counts_io import withdraw_counts
    s = lab["summary"]
    for table in ("region_counts.csv", "elife_counts.csv"):
        rows = [{"brain": OTHER_KEY, "total_cells": "3"}]
        from count_withdrawal_helpers import write_table
        write_table(s / table, rows)
    (s / ("mouse_861_%s_counts.csv" % NAME)).unlink()
    withdraw_counts(NAME, "test")
    assert not entry_current(lab["registry"], NAME)


def test_a_withdrawn_registry_entry_is_entered_again_by_the_next_count(lab):
    from mousebrain.region_counts_io import withdraw_counts
    withdraw_counts(NAME, "test")
    assert not entry_current(lab["registry"], NAME)
    f = lab["summary"].parent / "again.csv"
    f.write_text("region_acronym,cell_count\nAAA,6\n")
    AnalysisRegistry(analysis_name=ANALYSIS, registry_root=lab["registry"],
                     follows_approved_method=False).register_output(
        sample=NAME, category="region_counts", files={"region_counts": str(f)},
        results={"total_cells": 6}, method_params={"atlas": "atlas_x"})
    assert entry_current(lab["registry"], NAME)


def test_a_withdrawal_with_nothing_to_withdraw_creates_nothing(tmp_path, monkeypatch):
    from mousebrain.region_counts_io import withdraw_counts
    monkeypatch.setattr(config, "DATA_SUMMARY_DIR", tmp_path / "no_summary")
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "no_registry"))
    assert withdraw_counts(NAME, "test") == []
    assert list(tmp_path.iterdir()) == []


def test_registering_again_retires_the_registry_entry_too(lab):
    """Step 3 moves the old registration into _archive before registering
    again: the counts made on it leave the tables AND the registry."""
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "3_register_to_atlas.py"
    spec = importlib.util.spec_from_file_location("register_step_registry", path)
    step3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step3)
    b = make_brain(lab["root"])
    assert step3.archive_existing_registration(b)
    _assert_withdrawn(lab)
