"""A 2D tracker saved by a spreadsheet program as "CSV UTF-8" keeps its runs.

Such a file starts with a byte-order mark (BOM). Read as plain UTF-8 the first
column name would become BOM+'run_id': the next open of the tracker would add
a second, empty 'run_id' column, and no existing run could be found by its id
again -- status updates, mark-as-best and failing an unfinished run would all
miss every old row. And a file already damaged that way (two 'run_id' columns,
the right-hand one empty) must be repaired by the next rewrite, never have its
ids blanked by it.
"""
import csv

import pytest

from mousebrain.plugin_2d.sliceatlas.tracker.schema import CSV_COLUMNS
from mousebrain.plugin_2d.sliceatlas.tracker.slice_tracker import SliceTracker

SAMPLE = "PRJ_01_02_S07"
BOM = "﻿"


def _header(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _saved_with_a_bom(tmp_path):
    """A tracker with one run, then saved again the way a spreadsheet
    program saves "CSV UTF-8": the same text with a BOM in front."""
    path = tmp_path / "runs.csv"
    t = SliceTracker(csv_path=path)
    run_id = t.log_detection(sample_id=SAMPLE, status="started")
    text = path.read_text(encoding="utf-8")
    path.write_text(BOM + text, encoding="utf-8", newline="")
    return path, run_id


def test_an_existing_run_is_still_found_after_the_file_was_saved_with_a_bom(tmp_path):
    path, run_id = _saved_with_a_bom(tmp_path)

    t = SliceTracker(csv_path=path)

    assert t.get_run(run_id) is not None
    assert _header(path).count("run_id") + _header(path).count(BOM + "run_id") == 1


def test_a_bom_saved_file_still_takes_status_updates_and_best_marks(tmp_path):
    path, run_id = _saved_with_a_bom(tmp_path)
    t = SliceTracker(csv_path=path)

    assert t.update_status(run_id, status="completed", det_nuclei_found=12) is True
    assert t.mark_as_best(run_id) is True

    row = t.get_run(run_id)
    assert (row["status"], row["det_nuclei_found"], row["marked_best"]) == (
        "completed", "12", "True")
    # The rewrite writes no mark, and every column name once.
    assert not path.read_text(encoding="utf-8").startswith(BOM)
    assert len(set(_header(path))) == len(_header(path))


def test_a_bom_saved_file_takes_a_new_row_under_the_right_columns(tmp_path):
    path, old_id = _saved_with_a_bom(tmp_path)
    t = SliceTracker(csv_path=path)

    new_id = t.log_detection(sample_id=SAMPLE, status="started")

    assert t.get_run(old_id) is not None
    assert t.get_run(new_id)["sample_id"] == SAMPLE


def _damaged_by_an_older_version(tmp_path):
    """The state an older version left a BOM-saved file in: the first column
    named BOM+'run_id' (holding the ids), a second 'run_id' added at the end,
    empty."""
    path = tmp_path / "runs.csv"
    cols = [BOM + "run_id"] + [c for c in CSV_COLUMNS if c != "run_id"] + ["run_id"]
    rows = []
    for rid, status in (("det_20000101_000000_aaaaaa", "completed"),
                        ("det_20000101_000001_bbbbbb", "started")):
        row = {c: "" for c in cols}
        row[BOM + "run_id"] = rid
        row.update(run_type="detection", sample_id=SAMPLE, status=status)
        rows.append([row[c] for c in cols])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Written with the BOM as a literal character in the first name, so
        # reading as utf-8-sig sees plain 'run_id' twice.
        w.writerow(cols)
        w.writerows(rows)
    return path


def test_a_file_an_older_version_damaged_is_repaired_not_blanked(tmp_path):
    path = _damaged_by_an_older_version(tmp_path)

    t = SliceTracker(csv_path=path)

    header = _header(path)
    assert header.count("run_id") == 1 and len(set(header)) == len(header)
    assert (t.get_run("det_20000101_000000_aaaaaa") or {}).get("status") == "completed"
    assert t.fail_if_unfinished("det_20000101_000001_bbbbbb", "never finished") is True
    assert t.get_run("det_20000101_000001_bbbbbb")["status"] == "failed"


def test_two_different_values_under_one_repeated_column_are_refused_not_guessed(tmp_path):
    path = tmp_path / "runs.csv"
    cols = list(CSV_COLUMNS) + ["run_id"]
    row = {c: "" for c in CSV_COLUMNS}
    row.update(run_id="det_20000101_000000_aaaaaa", run_type="detection", sample_id=SAMPLE)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow([row[c] for c in CSV_COLUMNS] + ["det_20000101_000009_cccccc"])
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="Repair it first"):
        SliceTracker(csv_path=path)

    assert path.read_bytes() == before
