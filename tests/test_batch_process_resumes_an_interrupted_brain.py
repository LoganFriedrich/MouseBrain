"""batch_process --resume finds a brain whose batch was interrupted.

The status file was written only when a brain finished, failed or stopped for
review, so a batch killed mid-step left nothing marked PENDING and --resume
(which looks for PENDING rows) never found the brain.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
NAME = "MOUSE_A/101_PROJ_01_02_2p5x_z5"


@pytest.fixture
def bp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("batch_process", PIPELINE / "batch_process.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PIPELINE_STATUS_CSV", tmp_path / "pipeline_status.csv")
    return mod


def _brain(tmp_path):
    b = tmp_path / "MOUSE_A" / "101_PROJ_01_02_2p5x_z5"
    b.mkdir(parents=True)
    return b


def test_an_interrupted_step_leaves_the_brain_pending_at_that_step(bp, tmp_path, monkeypatch):
    brain = _brain(tmp_path)

    def killed(script, args, timeout=7200):
        raise KeyboardInterrupt  # the batch is stopped while step 2 runs
    monkeypatch.setattr(bp, "run_step", killed)
    with pytest.raises(KeyboardInterrupt):
        bp.process_brain(brain, NAME)
    row = bp.load_pipeline_status()[NAME]
    assert row["status"] == bp.QCStatus.PENDING
    assert row["current_step"] == "extract"
    assert row["brain_path"] == str(brain)


def test_resume_picks_up_the_interrupted_brain(bp, tmp_path, monkeypatch):
    brain = _brain(tmp_path)
    bp.mark_step_started(NAME, brain, "register")
    seen = []
    monkeypatch.setattr(bp, "process_brain",
                        lambda path, name, **_: seen.append((name, path)) or {"status": bp.QCStatus.PASS,
                                                                         "brain_name": name})
    monkeypatch.setattr(sys, "argv", ["batch_process.py", "--resume"])
    bp.main()
    assert seen == [(NAME, brain)]


def test_a_new_attempt_clears_the_previous_verdict(bp, tmp_path, monkeypatch):
    brain = _brain(tmp_path)
    bp.update_brain_status(NAME, {"status": bp.QCStatus.FAIL, "failed_step": "extract",
                                  "error": "old", "brain_path": str(brain)})
    monkeypatch.setattr(bp, "run_step", lambda *a, **k: (False, "new failure"))
    bp.process_brain(brain, NAME)
    row = bp.load_pipeline_status()[NAME]
    assert row["error"] == "" and row["qc_extract"].endswith("new failure")
