"""A promoted classification records its method and whether it is the
Paradigm Best one.

A person may promote any classification run to production; that is never
refused. But step 5 writes production only with the Paradigm Best method, so a
production made any other way has to be visible: in promoted_from.json beside
cells.xml and on the run's tracker row. Before, neither said which method made
the files. Identifiers are invented.
"""
import json
from pathlib import Path

from mousebrain.classification_runs import (NOT_PARADIGM_BEST_TAG, PROMOTED_TAG,
                                            compare_with_paradigm_best, promote_to_production)
from mousebrain.tracker import ExperimentTracker

NAME = "881_PROJ_08_81_2p5x_z5"


def _setup(tmp_path, settled_model="settled.h5", run_model="settled.h5", cube=50):
    t = ExperimentTracker(tmp_path / "calibration_runs.csv")
    best = t.log_selective_classification(brain=NAME, model_path=settled_model,
                                          tracing_type="descending", status="completed",
                                          cube_size=50, batch_size=32)
    assert t.mark_paradigm_best(best)
    run = t.log_selective_classification(brain=NAME, model_path=run_model,
                                         tracing_type="descending", status="completed",
                                         cube_size=cube, batch_size=32)
    folder = tmp_path / "optimization_runs" / NAME / "class_20000101_000000"
    folder.mkdir(parents=True)
    (folder / "cells.xml").write_text("<Marker></Marker>")
    return t, best, run, folder, tmp_path / NAME


def test_a_promotion_of_another_model_is_recorded_as_not_the_settled_method(tmp_path):
    t, best, run, folder, brain = _setup(tmp_path, run_model="other.h5")
    promote_to_production(folder, brain, exp_id=run, tracker=t)
    record = json.loads((brain / "5_Classified_Cells" / "promoted_from.json").read_text())
    m = record["method"]
    assert m["is_paradigm_best_method"] is False and m["paradigm_best_exp_id"] == best
    assert m["method"] == "selective" and m["model_path"] == "other.h5"
    assert any("model other.h5" in d for d in m["differences"])
    row = t.get_experiment(run)
    assert set(row["tags"].split(",")) == {PROMOTED_TAG, NOT_PARADIGM_BEST_TAG}
    assert "NOT the Paradigm Best method" in row["notes"]


def test_a_promotion_of_the_settled_method_says_so(tmp_path):
    t, best, run, folder, brain = _setup(tmp_path)
    promote_to_production(folder, brain, exp_id=run, tracker=t)
    record = json.loads((brain / "5_Classified_Cells" / "promoted_from.json").read_text())
    assert record["method"]["is_paradigm_best_method"] is True
    row = t.get_experiment(run)
    assert row["tags"] == PROMOTED_TAG
    assert "the Paradigm Best method" in row["notes"]


def test_a_different_setting_is_named(tmp_path):
    t, best, run, folder, brain = _setup(tmp_path, cube=40)
    c = compare_with_paradigm_best(t, run, NAME)
    assert c["is_paradigm_best_method"] is False
    assert c["differences"] == ["cube_size 40, settled 50"]


def test_no_paradigm_best_is_recorded_as_not_the_settled_method(tmp_path):
    t = ExperimentTracker(tmp_path / "calibration_runs.csv")
    run = t.log_classification(brain=NAME, model_path="m.h5", status="completed")
    c = compare_with_paradigm_best(t, run, NAME)
    assert c["is_paradigm_best_method"] is False and c["method"] == "full"
    assert "no classification is marked Paradigm Best" in c["differences"][0]


def test_without_a_tracker_it_says_it_cannot_tell(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "cells.xml").write_text("x")
    promote_to_production(folder, tmp_path / NAME, exp_id=None)
    record = json.loads((tmp_path / NAME / "5_Classified_Cells" / "promoted_from.json").read_text())
    assert record["method"]["is_paradigm_best_method"] is None
