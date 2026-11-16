"""A person's "no" to a registration is recorded and kept.

Rejecting used to only move an existing approval away -- and for a
registration never approved it wrote nothing at all. The next reader showed
the registration as "waiting for approval", and anyone could approve what had
just been turned down. Now the "no" of every path (the approval script, the
RUN_PIPELINE menu, the dashboard, Setup & Tuning) is recorded, every reader
puts the brain back at "register", and step 3 registers it again. Identifiers
are invented.
"""
import builtins
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space
from mousebrain.pipeline_status import brain_status

ROOT = Path(__file__).resolve().parents[1]
MOUSE = "M9"
BRAIN = "903_PROJ_09_03_2p5x_z5"
T0 = 1_700_000_000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _brain(root):
    b = root / MOUSE / BRAIN
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / sub / "ch0").mkdir(parents=True)
        tif = b / sub / "ch0" / "Z0000.tif"
        tif.write_bytes(b"x")
        os.utime(tif, (T0, T0))
        (b / sub / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", b / "2_Cropped_For_Registration")
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_bytes(b"r")
        os.utime(reg / name, (T0 + 100, T0 + 100))
    return b


def _say(monkeypatch, *answers):
    queue = list(answers)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: queue.pop(0))


def _approval_script(monkeypatch, b):
    util = _load("approve_util_reject", "scripts/utils/util_approve_registration.py")
    monkeypatch.setattr(util, "find_brain", lambda *a, **k: b)
    monkeypatch.setattr(util, "view_qc_images", lambda *a, **k: True)
    monkeypatch.setattr(sys, "argv", ["util_approve_registration.py", "--brain", BRAIN])
    return util


def _is_rejected(b):
    s = brain_status(b)
    return (s["step"] == "register" and not s["registered"] and not s["approved"]
            and "rejected" in s["register_problem"])


def test_no_in_the_approval_script_is_recorded(tmp_path, monkeypatch):
    b = _brain(tmp_path)
    util = _approval_script(monkeypatch, b)
    _say(monkeypatch, "no")
    with pytest.raises(SystemExit):
        util.main()
    assert _is_rejected(b)
    # Nobody approves it by pressing an ordinary Approve button afterwards.
    with pytest.raises(ValueError):
        ra.approve(b / "3_Registered_Atlas", approved_by="someone else")
    assert not ra.is_approved(b / "3_Registered_Atlas")


def test_no_in_run_pipeline_is_recorded(tmp_path, monkeypatch):
    b = _brain(tmp_path)
    rp = _load("run_pipeline_reject", "scripts/pipeline/RUN_PIPELINE.py")
    answers = [0, 1]  # "Run this step (approve)", then "No, reject it (recorded)"
    monkeypatch.setattr(rp, "ask", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(rp, "run_script", lambda *a, **k: None)
    monkeypatch.setattr(rp, "clear", lambda: None)
    _say(monkeypatch, "", "", "")
    assert brain_status(b)["step"] == "approve"
    rp.show_brain_status({"name": BRAIN, "full_name": MOUSE + "/" + BRAIN, "path": b,
                          "mouse": MOUSE})
    assert _is_rejected(b)


def test_a_rejection_overruled_in_the_approval_script_is_archived(tmp_path, monkeypatch):
    b = _brain(tmp_path)
    reg = b / "3_Registered_Atlas"
    ra.reject(reg, rejected_by="tester")
    util = _approval_script(monkeypatch, b)
    _say(monkeypatch, "yes")  # "approve it anyway, overruling the rejection?"
    try:
        util.main()
    except SystemExit as e:
        assert e.code in (0, None)
    assert ra.is_approved(reg)
    assert not (reg / ra.REJECTION_MARKER).exists()
    assert list((reg / "_archive").glob("*/" + ra.REJECTION_MARKER))


def test_step_3_registers_a_rejected_registration_again(tmp_path, monkeypatch, capsys):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    step3 = _load("register_rejected", "scripts/pipeline/3_register_to_atlas.py")
    b = _brain(tmp_path / "1_Brains")
    reg = b / "3_Registered_Atlas"
    ra.reject(reg, rejected_by="tester")
    assert step3.check_registration_status(b)[0] == "rejected"
    registered = []

    def fake_brainreg(pipeline_folder, *a, **k):
        assert not (reg / ra.REJECTION_MARKER).exists()
        registered.append(Path(pipeline_folder).name)
        for name in ra.FINGERPRINT_FILES:
            (reg / name).write_bytes(b"new")
    monkeypatch.setattr(step3, "run_brainreg", fake_brainreg)
    monkeypatch.setattr(sys, "argv", ["3", str(tmp_path / "1_Brains"), "--batch", "--brain", BRAIN])
    assert step3.main() == 0, capsys.readouterr().out
    assert registered == [BRAIN]
    # The rejection went into _archive WITH the registration it rejected.
    rejected = list((reg / "_archive").glob("*_rejected/" + ra.REJECTION_MARKER))
    assert len(rejected) == 1 and (rejected[0].parent / "brainreg.json").exists()
    assert ra.approval_state(reg)[0] == ra.UNAPPROVED


def test_counts_made_on_a_rejected_registration_are_stale(tmp_path):
    b = _brain(tmp_path)
    (b / "6_Region_Analysis").mkdir()
    (b / "6_Region_Analysis" / "cell_counts_by_region.csv").write_text("a\n")
    ra.reject(b / "3_Registered_Atlas", rejected_by="tester")
    s = brain_status(b)
    assert not s["counted"] and "count" in s["stale"]
