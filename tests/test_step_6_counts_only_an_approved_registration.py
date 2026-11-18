"""Step 6 counts a brain only while its registration is approved, and --all
counts every approved brain that has no row in the production table.

What went wrong: step 6 counted any brain with a cells.xml, so a rejected,
unfinished, out of date or unapproved registration was written straight back
into region_counts.csv, elife_counts.csv and the analysis registry -- even
right after its counts had been withdrawn for that reason. And --all judged
"already counted" by the brain's own 6_Region_Analysis files, which a
withdrawal leaves in place, so it never counted a withdrawn brain again and
said "No brains need counting". Identifiers are invented.
"""
import os
import sys
import time

import pytest

from mousebrain.region_counts_io import withdraw_counts
from mousebrain.registration_approval import reject, withdraw

from count_withdrawal_helpers import (KEY, NAME, brains_in, entry_current, load_step6,
                                      make_brain, make_lab)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    return make_lab(tmp_path, monkeypatch)


@pytest.fixture
def step6(lab, tmp_path, monkeypatch):
    return load_step6(lab, tmp_path, monkeypatch, "count_step_approved_only")


def _run(step6, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["6"] + list(argv))
    return step6.main()


def _unfinished(reg):
    (reg / "boundaries.tiff").unlink()


def _rejected(reg):
    reject(reg, rejected_by="tester")


def _unapproved(reg):
    withdraw(reg, "test")


def _out_of_date(reg):
    tif = reg.parent / "2_Cropped_For_Registration" / "ch0" / "Z0000.tif"
    later = time.time() + 600
    os.utime(tif, (later, later))


@pytest.mark.parametrize("make", [_unfinished, _rejected, _unapproved, _out_of_date],
                         ids=["incomplete", "rejected", "unapproved", "out_of_date"])
def test_step_6_refuses_a_registration_that_is_not_approved(step6, lab, monkeypatch, make):
    b = make_brain(lab["root"])
    make(b / "3_Registered_Atlas")
    # Whatever withdrew them, the brain's counts are out of production; step
    # 6 must not put counts of this registration back.
    withdraw_counts(NAME, "test")
    code = _run(step6, monkeypatch, "--brain", NAME, "--root", str(lab["root"]))
    assert step6.counted == [], "a registration that is not approved was counted"
    assert KEY not in brains_in(lab["summary"] / "region_counts.csv")
    assert not entry_current(lab["registry"], NAME)
    assert step6.tracker_rows() == [], "a refused brain left a run behind"
    assert code == 1, "a refused brain must end step 6 with a failure exit code"


def test_all_counts_a_brain_whose_counts_were_withdrawn(step6, lab, monkeypatch):
    b = make_brain(lab["root"])
    # Counted before: its own 6_Region_Analysis files are there ...
    (b / "6_Region_Analysis").mkdir()
    (b / "6_Region_Analysis" / "cell_counts_by_region.csv").write_text(
        "region_acronym,cell_count\n")
    # ... and its production row was withdrawn.
    withdraw_counts(NAME, "test")
    code = _run(step6, monkeypatch, "--all", "--root", str(lab["root"]))
    assert step6.counted == [NAME]
    assert KEY in brains_in(lab["summary"] / "region_counts.csv")
    assert entry_current(lab["registry"], NAME)
    assert code == 0


def test_all_leaves_a_brain_that_still_has_its_row(step6, lab, monkeypatch):
    make_brain(lab["root"])
    _run(step6, monkeypatch, "--all", "--root", str(lab["root"]))
    assert step6.counted == []


def test_all_lists_an_unapproved_brain_and_does_not_count_it(step6, lab, monkeypatch, capsys):
    make_brain(lab["root"], approved=False)
    withdraw_counts(NAME, "test")
    code = _run(step6, monkeypatch, "--all", "--root", str(lab["root"]))
    assert step6.counted == []
    out = capsys.readouterr().out
    assert NAME in out and "not approved" in out
    assert code == 0, "a brain waiting for approval is not a failed count"


def test_an_approved_brain_is_counted_into_tables_and_registry(step6, lab, monkeypatch):
    make_brain(lab["root"])
    withdraw_counts(NAME, "test")
    code = _run(step6, monkeypatch, "--brain", NAME, "--root", str(lab["root"]))
    assert code == 0
    assert KEY in brains_in(lab["summary"] / "region_counts.csv")
    assert entry_current(lab["registry"], NAME)
    assert [r["status"] for r in step6.tracker_rows()] == ["completed"]
