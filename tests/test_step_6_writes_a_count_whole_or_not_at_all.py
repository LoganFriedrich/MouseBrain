"""Step 6 puts a count into production whole -- tables and analysis registry
-- and only while what it counted is still the brain's.

What went wrong:
  * a count whose cells.xml was replaced (step 5, a promotion) or whose
    registration was rejected while it ran was written AFTER the withdrawal
    those events make, so the brain was back in production with counts of
    what had just been replaced, and their files looked current;
  * a count that could not be entered in the analysis registry was called
    finished: rows in the tables, nothing in the registry, the tracker row
    'completed', and the problem only in the console.
Identifiers are invented.
"""
import sys

import pytest

from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.region_counts_io import withdraw_counts
from mousebrain.registration_approval import reject

from count_withdrawal_helpers import (KEY, NAME, brains_in, entry_current, load_step6,
                                      make_brain, make_lab)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    return make_lab(tmp_path, monkeypatch)


@pytest.fixture
def step6(lab, tmp_path, monkeypatch):
    return load_step6(lab, tmp_path, monkeypatch, "count_step_whole")


def _run(step6, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["6"] + list(argv))
    return step6.main()


def _replace_cells(brain):
    (brain / "5_Classified_Cells" / "cells.xml").write_text(
        "<Marker></Marker><Marker></Marker>")


def _reject_meanwhile(brain):
    reject(brain / "3_Registered_Atlas", rejected_by="tester")


@pytest.mark.parametrize("meanwhile", [_replace_cells, _reject_meanwhile],
                         ids=["classification_replaced", "registration_rejected"])
def test_a_count_whose_inputs_changed_while_it_ran_is_not_written(step6, lab, monkeypatch,
                                                                  meanwhile):
    make_brain(lab["root"])
    withdraw_counts(NAME, "test")
    step6.during_count = meanwhile
    code = _run(step6, monkeypatch, "--brain", NAME, "--root", str(lab["root"]))
    assert step6.counted == [NAME]
    assert KEY not in brains_in(lab["summary"] / "region_counts.csv")
    assert KEY not in brains_in(lab["summary"] / "elife_counts.csv")
    assert not entry_current(lab["registry"], NAME)
    rows = step6.tracker_rows()
    assert [r["status"] for r in rows] == ["failed"]
    assert "not written to production" in rows[0]["notes"]
    assert code == 1


def test_a_count_that_cannot_be_registered_is_not_left_in_production(step6, lab, monkeypatch):
    make_brain(lab["root"])
    withdraw_counts(NAME, "test")

    def refuse(self, *a, **k):
        raise RuntimeError("registry lock held past the wait")
    monkeypatch.setattr(AnalysisRegistry, "register_output", refuse)
    code = _run(step6, monkeypatch, "--brain", NAME, "--root", str(lab["root"]))
    assert step6.counted == [NAME]
    # Nothing half-written stays in the tables an integrator pulls ...
    assert KEY not in brains_in(lab["summary"] / "region_counts.csv")
    assert KEY not in brains_in(lab["summary"] / "elife_counts.csv")
    # ... and the run says why, where a person looks.
    rows = step6.tracker_rows()
    assert [r["status"] for r in rows] == ["failed"]
    assert "registry lock held past the wait" in rows[0]["notes"]
    assert code == 1


def test_a_count_that_cannot_be_registered_also_retires_the_earlier_entry(step6, lab,
                                                                          monkeypatch):
    """The brain's earlier registry entry describes the counts this run
    replaced in the tables; it must not stay current beside nothing."""
    make_brain(lab["root"])

    def refuse(self, *a, **k):
        raise RuntimeError("registry manifest set aside")
    monkeypatch.setattr(AnalysisRegistry, "register_output", refuse)
    _run(step6, monkeypatch, "--brain", NAME, "--root", str(lab["root"]))
    assert KEY not in brains_in(lab["summary"] / "region_counts.csv")
    assert not entry_current(lab["registry"], NAME)
