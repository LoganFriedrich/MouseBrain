"""The production count tables are changed under one lock, and archived rows
keep their values under their own column names.

regenerate_derived_counts appended old eLife rows to the archive in each
row's own sorted-key order under a header written once, so a row with other
columns had its values under the wrong names. It rewrote elife_counts.csv in
place, and neither it nor step 6 held any lock, so two runs at once could drop
a row. Identifiers are invented.
"""
import csv
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGEN = ROOT / "scripts" / "pipeline" / "regenerate_derived_counts.py"
STEP6 = ROOT / "scripts" / "pipeline" / "6_count_regions.py"
BRAIN = "mouse_801/801_PROJ_08_01_2p5x_z5"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def regen(monkeypatch):
    fake = types.ModuleType("mousebrain.region_mapping")
    fake.aggregate_to_elife = lambda counts: {"Group A": {"count": sum(counts.values())}}
    fake.ELIFE_GROUPS = ["Group A"]
    monkeypatch.setitem(sys.modules, "mousebrain.region_mapping", fake)
    return _load(REGEN, "regen_locked")


@pytest.fixture
def step6(tmp_path, monkeypatch):
    mod = _load(STEP6, "count_step_locked")
    for name in ("REGION_COUNTS_CSV", "REGION_COUNTS_ARCHIVE_CSV",
                 "ELIFE_COUNTS_CSV", "ELIFE_COUNTS_ARCHIVE_CSV"):
        monkeypatch.setattr(mod, name, tmp_path / (name.lower() + ".csv"))
    return mod


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        return [dict(zip(header, row)) if len(row) == len(header) else row for row in r]


def test_regenerate_archives_old_rows_under_their_own_column_names(regen, tmp_path):
    _write(tmp_path / "region_counts.csv", ["brain", "total_cells", "region_ab"],
           [[BRAIN, "3", "3"]])
    # An archive started earlier, with fewer columns than the rows now archived.
    _write(tmp_path / "elife_counts_archive.csv", ["archived_at", "brain"],
           [["2000-01-01T00:00:00", "older_brain"]])
    _write(tmp_path / "elife_counts.csv", ["brain", "group_Group_A", "run_date"],
           [["old_brain", "7", "2000-01-02"]])
    regen.regenerate(tmp_path, log=lambda *a: None)
    archived = _rows(tmp_path / "elife_counts_archive.csv")
    assert all(isinstance(r, dict) for r in archived), "an archived row does not match the header"
    old = [r for r in archived if r["brain"] == "old_brain"]
    assert len(old) == 1
    assert old[0]["group_Group_A"] == "7" and old[0]["run_date"] == "2000-01-02"
    assert archived[0]["brain"] == "older_brain"


def _held_lock(region_csv):
    """A lock file as a live holder (another run) leaves it."""
    lock = Path(str(region_csv) + ".lock")
    lock.write_text("another-machine pid 1 at now token abc\n", encoding="utf-8")
    return lock


def _runs_only_after_the_lock_is_released(lock, target, written):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    time.sleep(1.5)
    assert not written(), "the table was changed while another run held the lock"
    lock.unlink()
    t.join(30)
    assert not t.is_alive()
    assert written()


def test_regenerate_waits_for_a_run_holding_the_count_tables(regen, tmp_path):
    region_csv = tmp_path / "region_counts.csv"
    _write(region_csv, ["brain", "total_cells", "region_ab"], [[BRAIN, "3", "3"]])
    lock = _held_lock(region_csv)
    _runs_only_after_the_lock_is_released(
        lock, lambda: regen.regenerate(tmp_path, log=lambda *a: None),
        lambda: (tmp_path / "elife_counts.csv").exists())


def test_step6_waits_for_a_run_holding_the_count_tables(step6, tmp_path):
    lock = _held_lock(step6.REGION_COUNTS_CSV)
    _runs_only_after_the_lock_is_released(
        lock, lambda: step6.update_region_counts_csv(BRAIN, {"AAA": 1}, {"atlas": "atlas_x"}),
        lambda: step6.REGION_COUNTS_CSV.exists())
