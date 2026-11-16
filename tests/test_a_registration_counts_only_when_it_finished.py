"""A registration counts only when it finished.

brainreg writes brainreg.json BEFORE it registers anything, so a run that
failed or was stopped left that file behind -- and every reader took it as a
finished registration: step 3 skipped the brain as registered, the dashboards
moved it on to "approve", a person could approve it, and the overnight batch
(which checked boundaries.tiff) called it a failure on every retry of every
night. These tests pin the fixed behaviour on synthetic brain folders.
Identifiers are invented.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.pipeline_status import brain_status

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "pipeline"
MOUSE = "M9"
BRAIN = "901_PROJ_09_01_2p5x_z5"
T0 = 1_700_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _step3():
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    return _load("register_step_complete", PIPELINE / "3_register_to_atlas.py")


def _brain(root):
    b = root / MOUSE / BRAIN
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / sub / "ch0").mkdir(parents=True)
        tif = b / sub / "ch0" / "Z0000.tif"
        tif.write_bytes(b"x")
        os.utime(tif, (T0, T0))
        (b / sub / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    return b


def _started_only(b):
    """What a failed or killed brainreg leaves: brainreg.json and nothing else."""
    reg = b / "3_Registered_Atlas"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "brainreg.json").write_text("{}")
    return reg


def _finished(reg):
    reg.mkdir(parents=True, exist_ok=True)
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_bytes(b"r")
        os.utime(reg / name, (T0 + 100, T0 + 100))


def test_a_registration_that_only_started_is_not_registered(tmp_path):
    b = _brain(tmp_path)
    reg = _started_only(b)
    s = brain_status(b)
    assert not s["registered"]
    assert s["step"] == "register"
    assert "boundaries.tiff" in s["register_problem"]
    assert ra.approval_state(reg)[0] == ra.INCOMPLETE


def test_a_registration_that_only_started_cannot_be_approved(tmp_path):
    reg = _started_only(_brain(tmp_path))
    with pytest.raises(ValueError):
        ra.approve(reg, approved_by="tester")
    assert not (reg / ra.APPROVAL_MARKER).exists()


def test_a_finished_registration_is_registered(tmp_path):
    b = _brain(tmp_path)
    _finished(b / "3_Registered_Atlas")
    s = brain_status(b)
    assert s["registered"] and s["step"] == "approve" and s["register_problem"] == ""


def test_step_3_takes_an_unfinished_registration_up_again(tmp_path, monkeypatch, capsys):
    step3 = _step3()
    b = _brain(tmp_path / "1_Brains")
    _started_only(b)
    assert step3.check_registration_status(b)[0] == "incomplete"
    seen = []

    def fake_brainreg(pipeline_folder, *a, **k):
        # brainreg is given an empty folder: the leftovers went to _archive first
        reg = Path(pipeline_folder) / "3_Registered_Atlas"
        seen.append(sorted(p.name for p in reg.iterdir() if p.name != "_archive"))
        _finished(reg)
    monkeypatch.setattr(step3, "run_brainreg", fake_brainreg)
    monkeypatch.setattr(sys, "argv", ["3", str(tmp_path / "1_Brains"), "--batch", "--brain", BRAIN])
    assert step3.main() == 0, capsys.readouterr().out
    assert seen == [[]]
    leftovers = list((b / "3_Registered_Atlas" / "_archive").glob("*_incomplete/brainreg.json"))
    assert len(leftovers) == 1
    assert step3.check_registration_status(b)[0] == "complete"


def test_a_failed_registration_run_leaves_no_registration_behind(tmp_path, monkeypatch, capsys):
    step3 = _step3()
    b = _brain(tmp_path / "1_Brains")

    def failing_brainreg(pipeline_folder, *a, **k):
        _started_only(Path(pipeline_folder))
        raise RuntimeError("brainreg failed with return code 1")
    monkeypatch.setattr(step3, "run_brainreg", failing_brainreg)
    monkeypatch.setattr(sys, "argv", ["3", str(tmp_path / "1_Brains"), "--batch", "--brain", BRAIN])
    assert step3.main() == 1
    reg = b / "3_Registered_Atlas"
    assert not (reg / "brainreg.json").exists()
    assert len(list((reg / "_archive").glob("*_failed/brainreg.json"))) == 1
    assert step3.check_registration_status(b)[0] == "needs_registration"
    assert brain_status(b)["step"] == "register"


def test_the_overnight_batch_registers_an_unfinished_registration_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # it opens its log file in the working directory
    ob = _load("overnight_complete", PIPELINE / "overnight_batch.py")
    b = _brain(tmp_path / "data")
    _started_only(b)
    called = []

    def fake_registration(name, pipeline_dir, **k):
        called.append(name)
        return False, "registration failed"
    monkeypatch.setattr(ob, "run_registration", fake_registration)
    monkeypatch.setattr(ob, "RETRY_DELAY", 0)
    result = ob.process_brain(BRAIN, b, skip_complete=True)
    assert called, "the unfinished registration was skipped as complete"
    assert result["steps"]["registration"] == "failed"


def test_the_overnight_batch_and_step_3_agree_on_a_finished_registration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ob = _load("overnight_complete2", PIPELINE / "overnight_batch.py")
    b = _brain(tmp_path / "data")
    reg = b / "3_Registered_Atlas"
    _finished(reg)
    (reg / "deformation_field_0.tiff").unlink()  # a file the batch did not check

    class Done:
        returncode = 0
    monkeypatch.setattr(ob.subprocess, "run", lambda *a, **k: Done())
    ok, msg = ob.run_registration(BRAIN, b)
    assert not ok and "deformation_field_0.tiff" in msg
