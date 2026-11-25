"""A registration that produced nothing stops the batch at 'register'.

The batch must handle a registration that left no brainreg.json or
deformation field, not only "needs approval". Otherwise it falls through to
detection, which stops on the missing approval, and the row says the brain
failed at 'detect' -- three steps from the cause.
"""
import importlib.util
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
NAME = "MOUSE_A/101_PROJ_01_02_2p5x_z5"


@pytest.fixture
def bp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("batch_process_regfail", PIPELINE / "batch_process.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PIPELINE_STATUS_CSV", tmp_path / "pipeline_status.csv")
    return mod


def test_a_registration_without_its_files_is_a_registration_failure(bp, tmp_path, monkeypatch):
    brain = tmp_path / "MOUSE_A" / "101_PROJ_01_02_2p5x_z5"
    brain.mkdir(parents=True)
    ran = []
    monkeypatch.setattr(bp, "run_step", lambda script, args, timeout=7200: ran.append(script) or (True, "OK"))
    monkeypatch.setattr(bp, "qc_check_extraction", lambda p: (bp.QCStatus.PASS, "OK"))
    monkeypatch.setattr(bp, "qc_check_crop", lambda p: (bp.QCStatus.PASS, "OK"))
    # step 3 "succeeded" but wrote nothing: the real registration check fails it
    result = bp.process_brain(brain, NAME)
    assert result["status"] == bp.QCStatus.FAIL
    assert result["failed_step"] == "register"
    assert "4_detect_cells.py" not in ran
    row = bp.load_pipeline_status()[NAME]
    assert row["failed_step"] == "register"
    assert row["qc_register"].startswith("FAIL")
