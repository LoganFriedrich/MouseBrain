"""A registration that is running right now is left alone.

brainreg writes brainreg.json first and then fits for a long time, so a
running registration has the same files as one that stopped half-way. Step 3
therefore holds .registration_running while it registers; while that marker is
live nothing touches the brain (otherwise --batch would archive the
registration from under the running brainreg and start a second brainreg into
the same folder), and a marker whose run is gone counts as did-not-finish.
Identifiers are invented.
"""
import importlib.util
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.pipeline_status import brain_status

ROOT = Path(__file__).resolve().parents[1]
MOUSE = "M9"
BRAIN = "905_PROJ_09_05_2p5x_z5"
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
    reg.mkdir(parents=True)
    (reg / "brainreg.json").write_text("{}")  # what brainreg writes first
    return b


def _dead_pid():
    """The id of a process that has ended."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _marker(reg, **fields):
    now = datetime.now()
    record = {"host": socket.gethostname(), "pid": os.getpid(),
              "started_at": now.isoformat(timespec="seconds"),
              "started_epoch": now.timestamp(), "token": "t"}
    record.update(fields)
    (reg / ra.RUNNING_MARKER).write_text(json.dumps(record))


def _step3(monkeypatch, tmp_path, *extra):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    step3 = _load("register_running", "scripts/pipeline/3_register_to_atlas.py")
    calls = []

    def fake_brainreg(pipeline_folder, *a, **k):
        calls.append(Path(pipeline_folder).name)
        reg = Path(pipeline_folder) / "3_Registered_Atlas"
        assert (reg / ra.RUNNING_MARKER).exists()  # held while brainreg runs
        for name in ra.FINGERPRINT_FILES:
            (reg / name).write_bytes(b"new")
    monkeypatch.setattr(step3, "run_brainreg", fake_brainreg)
    monkeypatch.setattr(sys, "argv", ["3", str(tmp_path / "1_Brains"), "--batch",
                                      "--brain", BRAIN, *extra])
    return step3, calls


def test_a_running_registration_is_shown_as_running(tmp_path):
    b = _brain(tmp_path)
    reg = b / "3_Registered_Atlas"
    claim = ra.claim_registration(reg)
    state, text = ra.approval_state(reg)
    assert state == ra.RUNNING and "being registered right now" in text
    s = brain_status(b)
    assert s["step"] == "register" and not s["registered"]
    assert "being registered right now" in s["register_problem"]
    with pytest.raises(ValueError):
        ra.approve(reg, approved_by="tester")
    with pytest.raises(ValueError):
        ra.reject(reg, rejected_by="tester")
    ra.release_registration(reg, claim)
    assert not (reg / ra.RUNNING_MARKER).exists()
    assert ra.approval_state(reg)[0] == ra.INCOMPLETE


@pytest.mark.parametrize("extra", [(), ("--reprocess",)])
def test_step_3_neither_archives_nor_registers_a_running_brain(tmp_path, monkeypatch, capsys,
                                                               extra):
    b = _brain(tmp_path / "1_Brains")
    reg = b / "3_Registered_Atlas"
    _marker(reg)  # this very process: alive
    step3, calls = _step3(monkeypatch, tmp_path, *extra)
    assert step3.check_registration_status(b)[0] == "running"
    assert step3.main() == 1
    out = capsys.readouterr().out
    assert calls == []
    assert (reg / "brainreg.json").exists() and not (reg / "_archive").exists()
    assert (reg / ra.RUNNING_MARKER).exists()  # the other run's marker is untouched
    assert "being registered right now" in out


def test_a_marker_whose_process_is_gone_is_an_unfinished_registration(tmp_path, monkeypatch,
                                                                     capsys):
    b = _brain(tmp_path / "1_Brains")
    reg = b / "3_Registered_Atlas"
    _marker(reg, pid=_dead_pid())
    assert ra.approval_state(reg)[0] == ra.INCOMPLETE
    step3, calls = _step3(monkeypatch, tmp_path)
    assert step3.check_registration_status(b)[0] == "incomplete"
    assert step3.main() == 0, capsys.readouterr().out
    assert calls == [BRAIN]
    assert list((reg / "_archive").glob("*_incomplete/brainreg.json"))
    assert not (reg / ra.RUNNING_MARKER).exists()
    assert ra.approval_state(reg)[0] == ra.UNAPPROVED


def test_another_computers_marker_is_live_until_it_is_a_day_old(tmp_path):
    reg = _brain(tmp_path) / "3_Registered_Atlas"
    _marker(reg, host="other-computer", pid=1)
    assert ra.approval_state(reg)[0] == ra.RUNNING
    old = datetime.now() - timedelta(hours=ra.RUNNING_MAX_HOURS_OTHER_HOST + 1)
    _marker(reg, host="other-computer", pid=1, started_epoch=old.timestamp())
    assert ra.approval_state(reg)[0] == ra.INCOMPLETE


def test_a_second_claim_is_refused_and_release_removes_only_its_own(tmp_path):
    reg = _brain(tmp_path) / "3_Registered_Atlas"
    first = ra.claim_registration(reg)
    with pytest.raises(ra.RegistrationRunning):
        ra.claim_registration(reg)
    ra.release_registration(reg, {"token": "someone else's"})
    assert (reg / ra.RUNNING_MARKER).exists()
    ra.release_registration(reg, first)
    assert not (reg / ra.RUNNING_MARKER).exists()


def test_the_marker_is_removed_when_brainreg_fails(tmp_path, monkeypatch, capsys):
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    step3 = _load("register_running_fail", "scripts/pipeline/3_register_to_atlas.py")
    b = _brain(tmp_path / "1_Brains")
    (b / "3_Registered_Atlas" / "brainreg.json").unlink()

    def failing(pipeline_folder, *a, **k):
        raise RuntimeError("brainreg failed")
    monkeypatch.setattr(step3, "run_brainreg", failing)
    monkeypatch.setattr(sys, "argv", ["3", str(tmp_path / "1_Brains"), "--batch", "--brain", BRAIN])
    assert step3.main() == 1
    assert not (b / "3_Registered_Atlas" / ra.RUNNING_MARKER).exists()


def test_a_failure_archiving_the_previous_registration_is_not_called_a_failed_run(
        tmp_path, monkeypatch, capsys):
    """Archiving the OLD registration and running brainreg were in one try:
    an error while archiving the old one archived the rest of it as the
    "failed run's output" -- splitting a perhaps-approved registration over two
    folders, one labeled failed."""
    pytest.importorskip("brainreg", reason="3_register_to_atlas checks for brainreg on import")
    step3 = _load("register_archive_fail", "scripts/pipeline/3_register_to_atlas.py")
    b = _brain(tmp_path / "1_Brains")
    reg = b / "3_Registered_Atlas"
    for name in ra.FINGERPRINT_FILES:
        (reg / name).write_bytes(b"old")
    labels, ran = [], []

    def archive(pipeline_folder, label=""):
        labels.append(label)
        raise PermissionError("a file is in use")
    monkeypatch.setattr(step3, "archive_existing_registration", archive)
    monkeypatch.setattr(step3, "run_brainreg", lambda *a, **k: ran.append(1))
    monkeypatch.setattr(sys, "argv", ["3", str(tmp_path / "1_Brains"), "--batch", "--brain",
                                      BRAIN, "--reprocess"])
    assert step3.main() == 1
    out = capsys.readouterr().out
    assert ran == [] and labels == [""], labels
    assert "failed run's output" not in out
    assert "PREVIOUS registration" in out
    assert not (reg / ra.RUNNING_MARKER).exists()


def test_brainreg_still_running_after_step_3_was_killed_keeps_the_brain_running(tmp_path,
                                                                                 monkeypatch,
                                                                                 capsys):
    """A batch's time limit or a closed dashboard kills step 3 but not the
    brainreg it started, which goes on writing into the folder. The marker
    names brainreg's process too, and stays live while it runs: a retry
    neither archives the folder nor starts a second brainreg into it."""
    b = _brain(tmp_path / "1_Brains")
    reg = b / "3_Registered_Atlas"
    brainreg = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _marker(reg, pid=_dead_pid(), child_pid=brainreg.pid,
                child_started_epoch=datetime.now().timestamp())
        assert ra.approval_state(reg)[0] == ra.RUNNING
        assert brain_status(b)["step"] == "register" and not brain_status(b)["registered"]
        step3, calls = _step3(monkeypatch, tmp_path)
        assert step3.check_registration_status(b)[0] == "running"
        assert step3.main() == 1
        assert calls == [] and not (reg / "_archive").exists()
        assert "brainreg: process %d" % brainreg.pid in capsys.readouterr().out
    finally:
        brainreg.kill()
        brainreg.wait()
    # Both processes gone: the run did not finish.
    assert ra.approval_state(reg)[0] == ra.INCOMPLETE


def test_step_3_records_brainreg_in_its_own_marker_only(tmp_path):
    reg = _brain(tmp_path) / "3_Registered_Atlas"
    claim = ra.claim_registration(reg)
    ra.record_registration_child(reg, claim, 4321)
    on_disk = json.loads((reg / ra.RUNNING_MARKER).read_text())
    assert on_disk["child_pid"] == 4321 and on_disk["token"] == claim["token"]
    # Its own run sees no foreign marker; anyone else sees it running.
    assert ra.running_registration(reg, own=claim) is None
    assert ra.running_registration(reg)["live"]
    ra.release_registration(reg, claim)
    _marker(reg, token="another run")
    ra.record_registration_child(reg, claim, 1234)
    assert "child_pid" not in json.loads((reg / ra.RUNNING_MARKER).read_text())
