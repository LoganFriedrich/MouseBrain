"""A brain is "counted" only while its row is in the production table.

A withdrawal (a rejection, an approval taken back, a re-crop or
re-registration, a replaced classification) takes the brain's row out of
region_counts.csv and leaves the brain's own 6_Region_Analysis files. The
status readers judged "counted" by those files. So when a person rejected a
registration and then overruled the rejection, approving the SAME
registration again, every reader showed the brain as Complete, and
batch_process --resume and the overnight batch skipped it. The brain had no
counts in production, and nothing said so.

Identifiers are invented.
"""
import importlib.util
import os
import time
from pathlib import Path

from mousebrain.pipeline_status import brain_status
from mousebrain.registration_approval import approve, reject

from count_withdrawal_helpers import (KEY, NAME, brains_in, load_step6, make_brain,
                                      make_lab)

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _counted_brain(lab):
    """The brain of make_lab, with an approved registration, a production
    classification and its own, current count files."""
    b = make_brain(lab["root"])
    # What the steps before registration and step 4 leave behind, so the
    # status readers reach the count step.
    (b / "1_Extracted_Full").mkdir()
    (b / "1_Extracted_Full" / "metadata.json").write_text("{}")
    (b / "4_Cell_Candidates").mkdir()
    (b / "4_Cell_Candidates" / "detected_cells.xml").write_text("<x/>")
    out = b / "6_Region_Analysis" / "cell_counts_by_region.csv"
    out.parent.mkdir(parents=True)
    out.write_text("region_acronym,cell_count\nAAA,5\n")
    later = time.time() + 60
    os.utime(out, (later, later))
    return b


def test_count_files_without_a_production_row_are_not_counted(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, monkeypatch)
    b = _counted_brain(lab)
    assert brain_status(b)["step"] == "done"

    # The row leaves the table (as a withdrawal takes it), the files stay.
    s = lab["summary"] / "region_counts.csv"
    s.write_text("brain,total_cells\n", encoding="utf-8")
    status = brain_status(b)
    assert status["step"] == "count"
    assert not status["counted"]
    assert "not in the production table" in status["stale"]["count"]


def test_overruling_a_rejection_does_not_make_the_brain_complete(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, monkeypatch)
    b = _counted_brain(lab)
    reg = b / "3_Registered_Atlas"
    assert brain_status(b)["step"] == "done"

    reject(reg, rejected_by="tester")
    approve(reg, approved_by="tester", allow_rejected=True)  # the rejection is overruled

    assert KEY not in brains_in(lab["summary"] / "region_counts.csv")
    status = brain_status(b)
    assert status["approved"]
    assert status["step"] == "count", status["step_name"]
    assert "count it again" in status["stale"]["count"]

    # batch_process --resume asks the same question before skipping a step.
    bp = _load("batch_process_production_row", PIPELINE / "batch_process.py")
    assert not bp.step_already_done(b, "count")

    # Counting it again (step 6 --all) puts the row back and the brain is done.
    step6 = load_step6(lab, tmp_path, monkeypatch, "count_regions_overruled")
    monkeypatch.setattr("sys.argv", ["6_count_regions.py", "--all", "--root", str(lab["root"])])
    assert step6.main() == 0
    assert step6.counted == [NAME]
    assert KEY in brains_in(lab["summary"] / "region_counts.csv")
    assert brain_status(b)["step"] == "done"


def test_an_unreadable_table_is_not_reported_as_a_withdrawal(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, monkeypatch)
    b = _counted_brain(lab)
    table = lab["summary"] / "region_counts.csv"
    os.replace(table, lab["summary"] / "region_counts.csv.aside")
    table.mkdir()  # there, but cannot be read as a table

    status = brain_status(b)
    assert status["step"] == "count"
    assert "could not be read" in status["stale"]["count"]
    assert "withdrawn" not in status["stale"]["count"]


def test_count_all_stops_when_the_table_cannot_be_read(tmp_path, monkeypatch, capsys):
    lab = make_lab(tmp_path, monkeypatch)
    _counted_brain(lab)
    step6 = load_step6(lab, tmp_path, monkeypatch, "count_regions_unreadable")
    table = lab["summary"] / "region_counts.csv"
    os.replace(table, lab["summary"] / "region_counts.csv.aside")
    table.mkdir()

    monkeypatch.setattr("sys.argv", ["6_count_regions.py", "--all", "--root", str(lab["root"])])
    assert step6.main() == 1
    assert step6.counted == []
    assert "could not be read" in capsys.readouterr().out
