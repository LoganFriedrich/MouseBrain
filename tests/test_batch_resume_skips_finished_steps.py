"""Resuming a brain does not redo the steps it already finished -- and only
resuming skips anything.

Every resumed attempt ran every step from extraction on. Extraction and
registration skip a finished brain themselves, detection does not: resuming a
brain stopped at counting re-ran hours of detection (archiving the finished
candidates) and classification.

What must still run: the step the batch was stopped IN (it may have written
its first files and not the rest), a step that failed, every step after
either, and -- on a run started with --input -- every step, as before.

The tests drive batch_process's own command line (main), so they check what an
operator gets from ``--resume`` and ``--input``.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
NAME = "MOUSE_A/101_PROJ_01_02_2p5x_z5"
ALL_STEPS = ["2_extract_and_analyze.py", "3_register_to_atlas.py", "4_detect_cells.py",
             "5_classify_cells.py", "6_count_regions.py"]


@pytest.fixture
def bp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("batch_process_resume", PIPELINE / "batch_process.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PIPELINE_STATUS_CSV", tmp_path / "pipeline_status.csv")
    for check in ("qc_check_extraction", "qc_check_crop", "qc_check_registration",
                  "qc_check_detection", "qc_check_classification", "qc_check_counting"):
        monkeypatch.setattr(mod, check, lambda p: (mod.QCStatus.PASS, "OK"))
    return mod


def _write(path: Path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _classified_brain(tmp_path):
    """Extracted, registered, detected and classified; stopped before counting."""
    brain = tmp_path / "MOUSE_A" / "101_PROJ_01_02_2p5x_z5"
    _write(brain / "1_Extracted_Full" / "metadata.json", "{}")
    _write(brain / "3_Registered_Atlas" / "brainreg.json", "{}")
    _write(brain / "4_Cell_Candidates" / "detected_cells.xml", "<Marker/>")
    _write(brain / "5_Classified_Cells" / "cells.xml", "<Marker/>")
    return brain


def _recorder(bp, monkeypatch, brain, effects=None):
    ran = []

    def fake_run_step(script, args, timeout=7200):
        ran.append(script)
        if effects and script in effects:
            effects[script](brain)
        return True, "OK"
    monkeypatch.setattr(bp, "run_step", fake_run_step)
    return ran


def _row(bp, brain, status, **fields):
    bp.update_brain_status(NAME, dict(status=status, brain_path=str(brain), **fields))


def _main(bp, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["batch_process.py", *argv])
    bp.main()


def test_resuming_a_brain_that_finished_classifying_runs_only_counting(bp, tmp_path, monkeypatch):
    """Stopped after classification: the next step, counting, never started."""
    brain = _classified_brain(tmp_path)
    _row(bp, brain, bp.QCStatus.PENDING, current_step="count")
    # The row is PENDING at 'count', so counting itself runs again; the four
    # earlier steps, done and current, do not.
    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--resume")
    assert ran == ["6_count_regions.py"]
    row = bp.load_pipeline_status()[NAME]
    assert row["status"] == bp.QCStatus.PASS
    assert row["skipped_steps"] == "extract,register,detect,classify"


def test_a_count_stopped_part_way_is_run_again(bp, tmp_path, monkeypatch):
    """Step 6 writes the per-brain CSV first and the summary CSVs and registry
    entry after it. A batch stopped in between leaves a CSV in
    6_Region_Analysis that the file rules call 'counted'; it must not be taken
    for a finished count."""
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "all_points_information.csv", "a\n1\n")
    _row(bp, brain, bp.QCStatus.PENDING, current_step="count")
    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--resume")
    assert "6_count_regions.py" in ran


@pytest.mark.parametrize("stopped_in,expected", [
    ("detect", ["4_detect_cells.py", "5_classify_cells.py", "6_count_regions.py"]),
    ("classify", ["5_classify_cells.py", "6_count_regions.py"]),
])
def test_the_step_a_batch_was_stopped_in_and_the_later_ones_run_again(
        bp, tmp_path, monkeypatch, stopped_in, expected):
    """A detection killed after writing its candidates but before recording
    them, or a classification killed between its two files, leaves files the
    file rules call done."""
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "counts.csv", "a\n1\n")
    _row(bp, brain, bp.QCStatus.PENDING, current_step=stopped_in)
    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--resume")
    assert ran == expected


def test_a_failed_step_and_the_steps_after_it_run_again(bp, tmp_path, monkeypatch):
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "counts.csv", "a\n1\n")
    _row(bp, brain, bp.QCStatus.FAIL, failed_step="classify")
    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--resume", "--retry-failed")
    assert ran == ["5_classify_cells.py", "6_count_regions.py"]


def test_a_step_run_again_makes_the_later_ones_run_too(bp, tmp_path, monkeypatch):
    """Resumed at counting, but the candidates are stale (older than the
    images they are positions in): detection is redone, and the new candidates
    make the existing classification stale, so it runs too."""
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "counts.csv", "a\n1\n")
    _write(brain / "2_Cropped_For_Registration" / "metadata.json", "{}")
    _write(brain / "2_Cropped_For_Registration" / "ch0" / "Z0000.tif", "x")
    old = time.time() - 3600
    for f in ("4_Cell_Candidates/detected_cells.xml", "5_Classified_Cells/cells.xml",
              "6_Region_Analysis/counts.csv"):
        os.utime(brain / f, (old, old))
    _row(bp, brain, bp.QCStatus.PENDING, current_step="count")

    def new_candidates(b):
        _write(b / "4_Cell_Candidates" / "detected_cells.xml", "<Marker/>")

    ran = _recorder(bp, monkeypatch, brain, {"4_detect_cells.py": new_candidates})
    _main(bp, monkeypatch, "--resume")
    assert ran == ["4_detect_cells.py", "5_classify_cells.py", "6_count_regions.py"]


def test_processing_a_brain_with_input_runs_every_step(bp, tmp_path, monkeypatch):
    """--input is how a brain is processed again on purpose (for example with
    a newly settled detection method, which the file rules cannot see): it
    skips nothing, finished outputs or not."""
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "counts.csv", "a\n1\n")
    ims = tmp_path / "incoming" / (brain.name + ".ims")
    _write(ims, "")
    monkeypatch.setattr(bp, "setup_brain_from_ims", lambda p: (brain, "ready"))
    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--input", str(ims))
    assert ran == ALL_STEPS
    assert not bp.load_pipeline_status()[NAME].get("skipped_steps")


def test_a_step_that_raised_is_run_again_on_retry(bp, tmp_path, monkeypatch):
    """A step that raises (rather than reporting failure) used to leave the
    row FAIL with no failed step, so --retry-failed skipped the step that
    raised whenever its files looked finished."""
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "counts.csv", "a\n1\n")
    _row(bp, brain, bp.QCStatus.PENDING, current_step="classify")

    def raising_run_step(script, args, timeout=7200):
        if script == "5_classify_cells.py":
            raise RuntimeError("classifier crashed")
        return True, "OK"
    monkeypatch.setattr(bp, "run_step", raising_run_step)
    _main(bp, monkeypatch, "--resume")
    row = bp.load_pipeline_status()[NAME]
    assert row["status"] == bp.QCStatus.FAIL
    assert row["failed_step"] == "classify"

    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--resume", "--retry-failed")
    assert ran == ["5_classify_cells.py", "6_count_regions.py"]


def test_a_crash_recorded_without_its_step_is_run_again_from_the_step_it_was_in(
        bp, tmp_path, monkeypatch):
    """Batches that recorded a crash wrote only FAIL and the error; the failed
    step stayed empty while current_step still named the step that crashed.
    Status files keep such rows. A count that crashed after writing its
    per-brain CSV must be run again, not taken for finished and filed PASS."""
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "all_points_information.csv", "a\n1\n")
    _write(brain / "6_Region_Analysis" / "counts.csv", "a\n1\n")
    _row(bp, brain, bp.QCStatus.FAIL, failed_step="", current_step="count",
         error="crashed part-way")
    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--resume", "--retry-failed")
    assert ran == ["6_count_regions.py"]
    row = bp.load_pipeline_status()[NAME]
    assert "count" not in row["skipped_steps"].split(",")


def test_a_crash_recorded_without_any_step_runs_every_step(bp, tmp_path, monkeypatch):
    """Nothing says which step was unfinished: every step runs again."""
    brain = _classified_brain(tmp_path)
    _write(brain / "6_Region_Analysis" / "counts.csv", "a\n1\n")
    _row(bp, brain, bp.QCStatus.FAIL, failed_step="", current_step="", error="crashed")
    ran = _recorder(bp, monkeypatch, brain)
    _main(bp, monkeypatch, "--resume", "--retry-failed")
    assert ran == ALL_STEPS
