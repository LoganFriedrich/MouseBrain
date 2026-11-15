"""A napari classification run never changes the brain's production classification.

5_Classified_Cells/cells.xml is what step 6 counts. The Setup & Tuning widget
used to write every tuning run straight into it, replacing step 5's output with
no copy kept. Runs now go to their own folder; production changes only when a
person promotes a run, and the files it replaces are archived.
"""
import json
from pathlib import Path

import pytest

from mousebrain.classification_runs import (PROMOTED_TAG, gui_run_folder,
                                            promote_to_production)
from mousebrain.tracker import CSV_COLUMNS, ExperimentTracker

TW = (Path(__file__).resolve().parents[1] / "src" / "mousebrain" / "plugin_3d"
      / "tuning_widget.py")


def _method(name):
    text = TW.read_text(encoding="utf-8")
    body = text[text.index("    def %s(self" % name):]
    return body[:body.index("\n    def ", 10)]


def test_the_widget_classifies_into_its_own_run_folder():
    body = _method("run_classification")
    assert "gui_run_folder(" in body
    assert 'output_path = self.current_brain / "5_Classified_Cells"' not in body


def test_a_run_folder_sits_with_the_gui_detection_runs(tmp_path):
    folder = gui_run_folder(tmp_path, "BRAIN_A")
    assert folder.parent == tmp_path / "optimization_runs" / "BRAIN_A"
    assert folder.name.startswith("class_")


def _run(tmp_path, text):
    run = tmp_path / "run"
    run.mkdir()
    (run / "cells.xml").write_text(text)
    (run / "rejected.xml").write_text("rej-" + text)
    return run


def test_promoting_archives_production_then_copies_the_run(tmp_path):
    brain = tmp_path / "BRAIN_A"
    prod = brain / "5_Classified_Cells"
    prod.mkdir(parents=True)
    (prod / "cells.xml").write_text("step5")
    (prod / "rejected.xml").write_text("step5-rej")
    run = _run(tmp_path, "gui")

    record = promote_to_production(run, brain, exp_id="cla_1")

    assert (prod / "cells.xml").read_text() == "gui"
    assert (prod / "rejected.xml").read_text() == "rej-gui"
    archived = list((prod / "_archive").glob("*/cells.xml"))
    assert len(archived) == 1 and archived[0].read_text() == "step5"
    saved = json.loads((prod / "promoted_from.json").read_text())
    assert saved["exp_id"] == "cla_1" and saved["source"] == str(run)
    assert record["previous_production_archived_to"]


def test_promoting_tags_the_tracker_row(tmp_path):
    import csv
    path = tmp_path / "calibration_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerow({"exp_id": "cla_1", "exp_type": "classification", "status": "completed"})
    t = ExperimentTracker(path)
    promote_to_production(_run(tmp_path, "gui"), tmp_path / "BRAIN_A", exp_id="cla_1", tracker=t)
    assert PROMOTED_TAG in t.get_experiment("cla_1")["tags"]


def test_a_run_without_cells_cannot_be_promoted(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        promote_to_production(tmp_path / "empty", tmp_path / "BRAIN_A")
    assert not (tmp_path / "BRAIN_A" / "5_Classified_Cells").exists()
