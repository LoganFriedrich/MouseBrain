"""A registration approval is a person's approval of ONE registration.

Step 4 waits for a person to approve the registration before spending hours of
detection on it. The gate holds only if: re-registering archives the old
approval (dot-files included) instead of leaving it beside the new,
unreviewed registration; the overnight batch never writes the approval itself;
and an approval says which registration it is for. These tests pin that on
synthetic brain folders.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space
from mousebrain.pipeline_status import brain_status

from count_withdrawal_helpers import put_in_production

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "pipeline"
UTILS = ROOT / "scripts" / "utils"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_time(path, t):
    os.utime(path, (t, t))


def _register(reg, t, size=10):
    """Write a fake registration's output files, all stamped at time t."""
    reg.mkdir(parents=True, exist_ok=True)
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_bytes(b"r" * size)
        _set_time(reg / name, t)


def _brain(tmp_path, t0=1_700_000_000):
    b = tmp_path / "M1" / "101_PROJ_01_01_2p5x_z5"
    crop = b / "2_Cropped_For_Registration"
    (crop / "ch0").mkdir(parents=True)
    (crop / "ch0" / "Z0000.tif").write_bytes(b"")
    _set_time(crop / "ch0" / "Z0000.tif", t0)
    (crop / "metadata.json").write_text("{}")
    (b / "1_Extracted_Full").mkdir()
    (b / "1_Extracted_Full" / "metadata.json").write_text("{}")
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", crop)
    _register(reg, t0 + 100)
    return b, reg


def _readers():
    out = {"pipeline_status": lambda b: brain_status(b)["approved"],
           "RUN_PIPELINE": lambda b: _load("rp", PIPELINE / "RUN_PIPELINE.py")
           .get_brain_status(b)["approved"],
           "util_approve_registration": lambda b: _load("ua", UTILS / "util_approve_registration.py")
           .check_approval_status(b)[0],
           "batch_process": lambda b: _load("bp", PIPELINE / "batch_process.py")
           .qc_check_registration(b)[1].startswith("OK"),
           "registration_approval": lambda b: ra.is_approved(b / "3_Registered_Atlas")}
    try:
        from mousebrain.plugin_3d.pipeline_widget import PipelineDashboard
        out["dashboard"] = lambda b: PipelineDashboard.get_brain_status(None, b)["approved"]
    except Exception:
        pass
    return out


def test_an_approval_covers_the_registration_it_was_given_for(tmp_path):
    b, reg = _brain(tmp_path)
    ra.approve(reg, approved_by="tester")
    for name, approved in _readers().items():
        assert approved(b), name


def test_re_registering_needs_a_new_approval(tmp_path):
    b, reg = _brain(tmp_path)
    ra.approve(reg, approved_by="tester")
    _register(reg, 1_700_000_000 + 5000, size=11)  # brainreg wrote a new one in place
    assert ra.approval_state(reg)[0] == ra.OTHER_REGISTRATION
    for name, approved in _readers().items():
        assert not approved(b), name


def test_an_approval_that_does_not_name_its_registration_is_not_an_approval(tmp_path):
    b, reg = _brain(tmp_path)
    for legacy in ("", "Approved: 2000-01-01 00:00:00\n"):
        (reg / ra.APPROVAL_MARKER).write_text(legacy)
        assert ra.approval_state(reg)[0] == ra.UNRECORDED
        for name, approved in _readers().items():
            assert not approved(b), name


def test_nothing_can_be_approved_before_it_is_registered(tmp_path):
    reg = tmp_path / "3_Registered_Atlas"
    reg.mkdir()
    with pytest.raises(ValueError):
        ra.approve(reg)
    assert not (reg / ra.APPROVAL_MARKER).exists()


def test_archiving_a_registration_takes_its_approval_with_it(tmp_path):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    step3 = _load("register_step", PIPELINE / "3_register_to_atlas.py")
    b, reg = _brain(tmp_path)
    ra.approve(reg, approved_by="tester")
    assert step3.archive_existing_registration(b)
    assert not (reg / ra.APPROVAL_MARKER).exists()
    archived = list((reg / "_archive").glob("*/" + ra.APPROVAL_MARKER))
    assert len(archived) == 1
    assert (archived[0].parent / "brainreg.json").exists()


def test_a_new_registration_run_retires_a_leftover_approval(tmp_path):
    b, reg = _brain(tmp_path)
    ra.approve(reg, approved_by="tester")
    moved = ra.retire_for_new_registration(reg)
    assert moved is not None and moved.exists()
    assert ra.approval_state(reg)[0] == ra.UNAPPROVED


def test_rejecting_keeps_the_old_approval_in_the_archive(tmp_path):
    b, reg = _brain(tmp_path)
    ra.approve(reg, approved_by="tester")
    moved = ra.withdraw(reg)
    assert moved.exists() and not (reg / ra.APPROVAL_MARKER).exists()


def test_the_overnight_batch_never_approves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # it opens its log file in the working directory
    ob = _load("overnight", PIPELINE / "overnight_batch.py")
    b, reg = _brain(tmp_path / "data")

    class Done:
        returncode = 0
    monkeypatch.setattr(ob.subprocess, "run", lambda *a, **k: Done())
    ok, _msg = ob.run_registration(b.name, b)
    assert ok
    assert not (reg / ra.APPROVAL_MARKER).exists()

    result = ob.process_brain(b.name, b, skip_complete=True)
    assert result["status"] == "awaiting_approval"
    assert "detection" not in result["steps"]


def test_no_approve_path_passes_an_argument_the_script_does_not_take():
    for path in (PIPELINE / "RUN_PIPELINE.py", PIPELINE / "batch_process.py",
                 ROOT / "src" / "mousebrain" / "plugin_3d" / "pipeline_widget.py"):
        text = path.read_text(encoding="utf-8")
        assert "'util_approve_registration.py', ['--brain', brain_name, '--yes']" not in text, path


# ------------------------------------------------------------- stale outputs
def _downstream(b, t):
    det = b / "4_Cell_Candidates"
    det.mkdir(exist_ok=True)
    (det / "detected_cells.xml").write_text("<x/>")
    (b / "5_Classified_Cells").mkdir(exist_ok=True)
    (b / "5_Classified_Cells" / "cells.xml").write_text("<x/>")
    (b / "6_Region_Analysis").mkdir(exist_ok=True)
    (b / "6_Region_Analysis" / "cell_counts_by_region.csv").write_text("a\n")
    for i, f in enumerate((det / "detected_cells.xml", b / "5_Classified_Cells" / "cells.xml",
                           b / "6_Region_Analysis" / "cell_counts_by_region.csv")):
        _set_time(f, t + 10 * i)


def test_counts_made_on_an_earlier_registration_are_stale(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path)
    _downstream(b, 1_700_000_000 + 200)
    put_in_production(tmp_path, monkeypatch, b.name)
    ra.approve(reg, approved_by="tester")
    assert brain_status(b)["step"] == "done"

    _register(reg, 1_700_000_000 + 9000, size=12)  # re-registered, same images
    ra.approve(reg, approved_by="tester")
    s = brain_status(b)
    assert "count" in s["stale"] and not s["counted"]
    # Detection and classification are positions in the images, which did not
    # change, so re-fitting the atlas does not make them stale.
    assert "detect" not in s["stale"] and "classify" not in s["stale"]
    assert s["step"] == "count"


def test_outputs_older_than_the_images_are_stale_all_the_way_down(tmp_path):
    b, reg = _brain(tmp_path)
    _downstream(b, 1_700_000_000 + 200)
    ra.approve(reg, approved_by="tester")
    tif = b / "2_Cropped_For_Registration" / "ch0" / "Z0000.tif"
    _set_time(tif, 1_700_000_000 + 8000)  # re-cropped
    _register(reg, 1_700_000_000 + 9000, size=12)
    ra.approve(reg, approved_by="tester")
    s = brain_status(b)
    assert set(s["stale"]) == {"detect", "classify", "count"}
    assert s["step"] == "detect"
