"""brainreg.json alone is never taken for a finished registration.

brainreg writes brainreg.json BEFORE it fits anything, so that file is left
behind by a registration that failed, was stopped, or is running right now.
Step 3, the status readers and step 4 already use the one rule
(registration_approval). Three places still took the bare file as a finished
registration:

* util_backfill_registrations.py wrote a calibration-run row with status
  'completed' ("Backfilled from existing registration") for it -- also for a
  rejected registration, or one whose images were re-cropped since;
* Setup & Tuning's "Run Region Counts" started step 6 on it (and on any
  registration nobody had approved);
* util_optimize_crop.py scored a trial whose registration stopped part-way as
  a successful one.

Identifiers are invented; every folder is built in tmp_path.
"""
import csv
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from mousebrain import registration_approval as ra
from mousebrain.config import record_registered_space
from mousebrain.tracker import ExperimentTracker

ROOT = Path(__file__).resolve().parents[1]
MOUSE = "M7"
T0 = 1_700_000_000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _brain(root, name):
    b = root / MOUSE / name
    for sub in ("1_Extracted_Full", "2_Cropped_For_Registration"):
        (b / sub / "ch0").mkdir(parents=True)
        (b / sub / "ch0" / "Z0000.tif").write_bytes(b"x")
        os.utime(b / sub / "ch0" / "Z0000.tif", (T0, T0))
        (b / sub / "metadata.json").write_text('{"voxel_size_um": {"x": 4, "y": 4, "z": 4}}')
    reg = b / "3_Registered_Atlas"
    record_registered_space(reg, "cropped", b / "2_Cropped_For_Registration")
    return b, reg


def _started_only(reg):
    """What a failed, stopped or still running brainreg leaves."""
    (reg / "brainreg.json").write_text(json.dumps(
        {"atlas": "allen_mouse_10um", "orientation": "iar",
         "voxel_sizes": ["5", "2.5", "2.5"]}))


def _finished(reg):
    _started_only(reg)
    for name in ra.FINGERPRINT_FILES:
        if name != "brainreg.json":
            (reg / name).write_bytes(b"r")
    for name in ra.FINGERPRINT_FILES:
        os.utime(reg / name, (T0 + 100, T0 + 100))


# ------------------------------------------------------------------ backfill
def _backfill(monkeypatch, tmp_path, brains):
    util = _load("backfill_registrations_test", "scripts/utils/util_backfill_registrations.py")
    csv_path = tmp_path / "summary" / "calibration_runs.csv"
    csv_path.parent.mkdir()
    monkeypatch.setattr(util, "ExperimentTracker", lambda: ExperimentTracker(csv_path=csv_path))
    monkeypatch.setattr(util, "BRAINS_ROOT", tmp_path)
    monkeypatch.setattr(util, "iter_brains", lambda root: [(b.parent, b) for b in brains])
    util.backfill_registrations(dry_run=False)
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["brain"]: r for r in csv.DictReader(f) if r.get("exp_type") == "registration"}


def test_backfill_does_not_record_a_registration_that_only_started(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path, "701_PROJ_07_01_2p5x_z5")
    _started_only(reg)
    rows = _backfill(monkeypatch, tmp_path, [b])
    assert b.name not in rows, "a registration that never finished was recorded as %r" % (
        rows.get(b.name, {}).get("status"))


def test_backfill_does_not_record_a_rejected_or_out_of_date_registration(tmp_path, monkeypatch):
    rejected, reg1 = _brain(tmp_path, "702_PROJ_07_02_2p5x_z5")
    _finished(reg1)
    ra.reject(reg1, rejected_by="tester")
    recropped, reg2 = _brain(tmp_path, "703_PROJ_07_03_2p5x_z5")
    _finished(reg2)
    plane = recropped / "2_Cropped_For_Registration" / "ch0" / "Z0000.tif"
    os.utime(plane, (T0 + 5000, T0 + 5000))     # re-cropped after it was registered
    assert ra.approval_state(reg2)[0] == ra.OUT_OF_DATE
    rows = _backfill(monkeypatch, tmp_path, [rejected, recropped])
    assert rows == {}


def test_backfill_records_a_finished_registration(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path, "704_PROJ_07_04_2p5x_z5")
    _finished(reg)
    rows = _backfill(monkeypatch, tmp_path, [b])
    assert rows[b.name]["status"] == "completed"


# ------------------------------------------------- Setup & Tuning: counting
class _Any:
    def __getattr__(self, name):
        return lambda *a, **k: None


def _count_button(monkeypatch, b):
    tw = pytest.importorskip("mousebrain.plugin_3d.tuning_widget")
    warnings, started = [], []

    class _Msg:
        @staticmethod
        def warning(parent, title, text, *a, **k):
            warnings.append(title)
    monkeypatch.setattr(tw, "QMessageBox", _Msg)

    class _Worker:
        def __init__(self, cmd, brain, cwd=None):
            started.append(cmd)
            self.progress = self.finished = _Any()

        def isRunning(self):
            return False

        def start(self):
            pass
    monkeypatch.setattr(tw, "RegionCountWorker", _Worker)

    class _W:
        def __getattr__(self, name):
            fn = getattr(tw.TuningWidget, name)
            if isinstance(fn, types.FunctionType):
                return types.MethodType(fn, self)
            return fn
    w = _W()
    w.current_brain = b
    w.region_count_status = _Any()
    w.run_regions_btn = _Any()
    w._update_counting_source_label = lambda: None
    w._hold_brain_list = lambda: None
    (b / "5_Classified_Cells").mkdir(exist_ok=True)
    (b / "5_Classified_Cells" / "cells.xml").write_text("<Marker></Marker>")
    w.run_region_counting()
    return started, warnings


def test_run_region_counts_refuses_a_registration_that_only_started(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path, "705_PROJ_07_05_2p5x_z5")
    _started_only(reg)
    started, warnings = _count_button(monkeypatch, b)
    assert started == [], "step 6 was started on a registration that never finished"
    assert warnings


def test_run_region_counts_refuses_a_registration_nobody_approved(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path, "706_PROJ_07_06_2p5x_z5")
    _finished(reg)
    assert ra.approval_state(reg)[0] == ra.UNAPPROVED
    started, warnings = _count_button(monkeypatch, b)
    assert started == [], "step 6 was started on a registration nobody approved"
    assert warnings


def test_run_region_counts_refuses_a_rejected_registration(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path, "708_PROJ_07_08_2p5x_z5")
    _finished(reg)
    ra.reject(reg, rejected_by="tester")
    started, _w = _count_button(monkeypatch, b)
    assert started == []


def test_run_region_counts_runs_on_an_approved_registration(tmp_path, monkeypatch):
    b, reg = _brain(tmp_path, "707_PROJ_07_07_2p5x_z5")
    _finished(reg)
    ra.approve(reg, approved_by="tester")
    started, warnings = _count_button(monkeypatch, b)
    assert len(started) == 1 and warnings == []


# ------------------------------------------------------- crop optimization
def _optimizer(monkeypatch, writes):
    opt = _load("optimize_crop_test", "scripts/utils/util_optimize_crop.py")

    def fake_brainreg(cmd, *a, **k):
        out = Path(cmd[2])
        out.mkdir(parents=True, exist_ok=True)
        for name in writes:
            (out / name).write_text("{}")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(opt.subprocess, "run", fake_brainreg)
    return opt


def test_crop_optimization_does_not_score_an_unfinished_trial_as_a_success(tmp_path, monkeypatch):
    opt = _optimizer(monkeypatch, ["brainreg.json"])
    (tmp_path / "data" / "ch0").mkdir(parents=True)
    ok, _d = opt.run_registration(tmp_path / "data", tmp_path / "registration",
                                  {"voxel_size_um": {"x": 2.5, "z": 5}})
    assert ok is False


def test_crop_optimization_scores_a_finished_trial_as_a_success(tmp_path, monkeypatch):
    opt = _optimizer(monkeypatch, ra.FINGERPRINT_FILES)
    (tmp_path / "data" / "ch0").mkdir(parents=True)
    ok, _d = opt.run_registration(tmp_path / "data", tmp_path / "registration",
                                  {"voxel_size_um": {"x": 2.5, "z": 5}})
    assert ok is True
