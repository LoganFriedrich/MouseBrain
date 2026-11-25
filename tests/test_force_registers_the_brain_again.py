"""overnight_batch --force registers the brain again -- or says it did not.

--force passes --reprocess to step 3. Step 3 otherwise only takes brains that
are not registered: it would print "already registered" and end normally, the
batch would log "Registration complete" and, the approval still being valid,
redo detection, classification and counting on the unchanged registration.
--reprocess registers one named brain again and archives the old registration
and its approval; the brain then waits for a new approval.
Identifiers are invented.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "pipeline"
MOUSE = "M9"
BRAIN = "904_PROJ_09_04_2p5x_z5"
T0 = 1_700_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _registered_brain(root, name=BRAIN):
    b = root / MOUSE / name
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / sub / "ch0").mkdir(parents=True)
        tif = b / sub / "ch0" / "Z0000.tif"
        tif.write_bytes(b"x")
        os.utime(tif, (T0, T0))
        (b / sub / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", b / "2_Cropped_For_Registration")
    for n in ra.FINGERPRINT_FILES:
        (reg / n).write_bytes(b"old")
        os.utime(reg / n, (T0 + 100, T0 + 100))
    ra.approve(reg, approved_by="tester")
    return b


def _step3():
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    return _load("register_reprocess", PIPELINE / "3_register_to_atlas.py")


def test_step_3_registers_a_registered_brain_again_when_asked(tmp_path, monkeypatch, capsys):
    step3 = _step3()
    root = tmp_path / "1_Brains"
    b = _registered_brain(root)
    reg = b / "3_Registered_Atlas"
    ran = []

    def fake_brainreg(pipeline_folder, *a, **k):
        ran.append(Path(pipeline_folder).name)
        for n in ra.FINGERPRINT_FILES:
            (reg / n).write_bytes(b"new registration")
    monkeypatch.setattr(step3, "run_brainreg", fake_brainreg)
    monkeypatch.setattr(sys, "argv", ["3", str(root), "--batch", "--brain", BRAIN, "--reprocess"])
    assert step3.main() == 0, capsys.readouterr().out
    assert ran == [BRAIN]
    archived = [p.parent for p in (reg / "_archive").glob("*/" + ra.APPROVAL_MARKER)]
    assert len(archived) == 1 and (archived[0] / "brainreg.json").read_bytes() == b"old"
    assert ra.approval_state(reg)[0] == ra.UNAPPROVED


def test_reprocess_never_takes_more_than_one_brain(tmp_path, monkeypatch, capsys):
    step3 = _step3()
    root = tmp_path / "1_Brains"
    _registered_brain(root)
    _registered_brain(root, "9" + BRAIN)  # its name contains the other's ... and vice versa
    ran = []
    monkeypatch.setattr(step3, "run_brainreg", lambda p, *a, **k: ran.append(p))
    monkeypatch.setattr(sys, "argv", ["3", str(root), "--batch", "--brain", "PROJ_09_04",
                                      "--reprocess"])
    assert step3.main() == 1
    assert ran == []


class _Done:
    returncode = 0


def test_force_reregisters_and_then_waits_for_approval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # it opens its log file in the working directory
    ob = _load("overnight_force", PIPELINE / "overnight_batch.py")
    b = _registered_brain(tmp_path / "data")
    reg = b / "3_Registered_Atlas"
    commands = []

    def fake_step3(cmd, **k):
        commands.append(cmd)
        if "--reprocess" in cmd:  # what step 3 --reprocess does
            step3_archive = reg / "_archive" / "x"
            step3_archive.mkdir(parents=True)
            os.replace(reg / ra.APPROVAL_MARKER, step3_archive / ra.APPROVAL_MARKER)
            for n in ra.FINGERPRINT_FILES:
                (reg / n).write_bytes(b"new registration")
        return _Done()
    monkeypatch.setattr(ob.subprocess, "run", fake_step3)
    steps_after = []
    for name in ("run_detection", "run_classification", "run_counting"):
        monkeypatch.setattr(ob, name, lambda *a, _n=name, **k: steps_after.append(_n) or (True, ""))

    result = ob.process_brain(BRAIN, b, skip_complete=False)
    assert any("--reprocess" in c for c in commands)
    assert result["steps"]["registration"] == "success"
    assert result["status"] == "awaiting_approval"
    assert steps_after == []


def test_force_does_not_claim_a_registration_that_did_not_happen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ob = _load("overnight_force2", PIPELINE / "overnight_batch.py")
    b = _registered_brain(tmp_path / "data")
    monkeypatch.setattr(ob.subprocess, "run", lambda *a, **k: _Done())  # step 3 did nothing
    monkeypatch.setattr(ob, "RETRY_DELAY", 0)
    steps_after = []
    for name in ("run_detection", "run_classification", "run_counting"):
        monkeypatch.setattr(ob, name, lambda *a, _n=name, **k: steps_after.append(_n) or (True, ""))
    result = ob.process_brain(BRAIN, b, skip_complete=False)
    assert result["steps"]["registration"] == "failed"
    assert steps_after == []


class _Stop(Exception):
    pass


def _overnight_main(tmp_path, monkeypatch, argv, name):
    """Run overnight_batch.main() on one registered brain under tmp_path.
    Returns (exit code or _Stop, brains handed to process_brain, step-3 calls)."""
    monkeypatch.chdir(tmp_path)
    ob = _load(name, PIPELINE / "overnight_batch.py")
    _registered_brain(tmp_path / "data")
    monkeypatch.setattr(ob, "BRAINS_ROOT", tmp_path / "data")
    calls, processed = [], []
    monkeypatch.setattr(ob.subprocess, "run", lambda cmd, **k: calls.append(cmd) or _Done())

    def fake_process(brain_name, *a, **k):
        processed.append(brain_name)
        raise _Stop()
    monkeypatch.setattr(ob, "process_brain", fake_process)
    monkeypatch.setattr(sys, "argv", ["overnight_batch.py"] + argv)
    try:
        code = ob.main()
    except _Stop:
        code = _Stop
    return code, processed, calls


def test_force_without_a_brain_is_refused(tmp_path, monkeypatch, caplog):
    # One forgotten --brain would register every brain again and set aside
    # every approval: refused, nothing run, and the count is said.
    code, processed, calls = _overnight_main(tmp_path, monkeypatch, ["--force"], "ob_force_all")
    assert code not in (0, _Stop)
    assert processed == [] and calls == []
    assert "would register 1 registered brain(s) again" in caplog.text


def test_force_with_all_brains_runs(tmp_path, monkeypatch):
    code, processed, _ = _overnight_main(tmp_path, monkeypatch, ["--force", "--all-brains"],
                                         "ob_force_all2")
    assert code is _Stop and processed == [BRAIN]


def test_force_with_one_brain_runs(tmp_path, monkeypatch):
    code, processed, _ = _overnight_main(tmp_path, monkeypatch, ["--force", "--brain", BRAIN],
                                         "ob_force_one")
    assert code is _Stop and processed == [BRAIN]
