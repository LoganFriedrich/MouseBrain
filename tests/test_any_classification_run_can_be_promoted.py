"""Any completed classification run of a brain can be promoted to production,
and the Results tab says what region counting will count.

* "Promote to production" is not limited to the last run made in the window
  since the brain was selected: after a restart or a brain change an earlier
  run can be promoted without making it again.
* The Results tab's classification choice must drive what is promoted, and
  the tab names the production 5_Classified_Cells/cells.xml and the run it came
  from, because counting always counts that production file.

A run chosen in the run list or in the Results tab is promoted (after the same
confirmation). Promoting must not harm production: a run whose folder IS the
production folder, or another brain's run, is refused.
"""
import json
from types import SimpleNamespace

import pytest

napari = pytest.importorskip("napari")
import mousebrain.region_counts_io as region_counts_io  # noqa: E402
from tuning_widget_stand_in import BRAIN_A, BRAIN_B, _tracker, _widget  # noqa: E402


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, _s):
        pass


class _RunList:
    """The 'Classification Runs (from Tracker)' list with one run selected."""
    def __init__(self, exp_id):
        self.exp_id = exp_id

    def selectedItems(self):
        return [SimpleNamespace(data=lambda _role: self.exp_id)]


def _setup(tmp_path, monkeypatch, run_brain=BRAIN_A, output_path=None):
    withdrawn = []
    monkeypatch.setattr(region_counts_io, "withdraw_counts_or_warn",
                        lambda *a, **k: withdrawn.append(a) or [])
    brain = tmp_path / "M" / BRAIN_A
    prod = brain / "5_Classified_Cells"
    prod.mkdir(parents=True)
    (prod / "cells.xml").write_text("production", encoding="utf-8")
    run = tmp_path / "summary" / "optimization_runs" / run_brain / "class_20000101_000000"
    run.mkdir(parents=True)
    (run / "cells.xml").write_text("earlier run", encoding="utf-8")
    tracker = _tracker(tmp_path, [{
        "exp_id": "cla_early", "exp_type": "classification", "brain": run_brain,
        "status": "completed", "output_path": str(output_path or run)}])
    w = _widget(tmp_path, monkeypatch, tracker, brain=BRAIN_A)
    w.class_selection_status = _Label()
    w._last_class_output = None  # nothing made in this window: a restart
    w._last_class_exp_id = None
    return w, tracker, prod, run


def test_an_earlier_run_selected_in_the_run_list_is_promoted(tmp_path, monkeypatch):
    w, tracker, prod, _run = _setup(tmp_path, monkeypatch)
    w.class_compare_runs_list = _RunList("cla_early")
    # The run list's promote button; before, the only promote button took the
    # last run made in this window.
    promote = getattr(w, "_promote_selected_class_run", None) or w._promote_last_classification
    promote()
    assert (prod / "cells.xml").read_text(encoding="utf-8") == "earlier run"
    assert json.loads((prod / "promoted_from.json").read_text())["exp_id"] == "cla_early"


def test_choosing_a_run_in_the_results_tab_promotes_it_and_says_so(tmp_path, monkeypatch):
    w, tracker, prod, run = _setup(tmp_path, monkeypatch)
    entry = {"run": tracker.get_experiment("cla_early"), "cells_xml": run / "cells.xml",
             "date": "2000-01-01 00:00", "cells": "1", "source": "Tracker"}
    w._select_classification_entry(entry)
    assert [m[1] for m in w.msgs.seen if m[0] == "question"] == ["Promote to production"]
    # what counting counts is now that run, and the tab names it
    assert (prod / "cells.xml").read_text(encoding="utf-8") == "earlier run"
    label = w.class_selection_status.text
    assert "5_Classified_Cells/cells.xml" in label and "cla_early" in label
    assert not label.startswith("Selected")


def test_the_production_folder_is_never_promoted_onto_itself(tmp_path, monkeypatch):
    """A step 5 run's row names the production cells.xml: promoting it would
    archive production and then find nothing to copy."""
    w, tracker, prod, _run = _setup(
        tmp_path, monkeypatch, output_path=tmp_path / "M" / BRAIN_A / "5_Classified_Cells" / "cells.xml")
    w._select_specific_classification(tracker.get_experiment("cla_early"))
    assert (prod / "cells.xml").read_text(encoding="utf-8") == "production"
    assert not (prod / "_archive").exists()


def test_another_brains_run_is_not_promoted(tmp_path, monkeypatch):
    w, tracker, prod, _run = _setup(tmp_path, monkeypatch, run_brain=BRAIN_B)
    w._select_specific_classification(tracker.get_experiment("cla_early"))
    assert (prod / "cells.xml").read_text(encoding="utf-8") == "production"
    assert "Different brain" in [m[1] for m in w.msgs.seen]


def test_promoting_production_onto_itself_raises_and_keeps_it(tmp_path, monkeypatch):
    """The library refuses too, for every caller: before, it archived
    production and then had nothing to copy back."""
    from mousebrain.classification_runs import promote_to_production
    monkeypatch.setattr(region_counts_io, "withdraw_counts_or_warn", lambda *a, **k: [])
    brain = tmp_path / BRAIN_A
    prod = brain / "5_Classified_Cells"
    prod.mkdir(parents=True)
    (prod / "cells.xml").write_text("production", encoding="utf-8")
    with pytest.raises(ValueError):
        promote_to_production(prod, brain, exp_id="cla_step5")
    assert (prod / "cells.xml").read_text(encoding="utf-8") == "production"


def test_a_pipeline_row_naming_the_mouse_folder_is_this_brains(tmp_path, monkeypatch):
    """Pipeline steps record the brain as '<mouse folder>/<brain folder>'."""
    w, tracker, prod, run = _setup(tmp_path, monkeypatch, run_brain=BRAIN_A)
    row = dict(tracker.get_experiment("cla_early"), brain="M/" + BRAIN_A)
    w._select_specific_classification(row)
    assert (prod / "cells.xml").read_text(encoding="utf-8") == "earlier run"
