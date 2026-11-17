"""Counts of a classification promoted from Setup & Tuning can be traced to
the detection run behind them.

A promoted run wrote into its own run folder, so its tracker row never named
5_Classified_Cells and the provenance search found no classification at all:
the det_* columns of its counts stayed empty. The production record beside
cells.xml now names the run. Identifiers are invented.
"""
from mousebrain.classification_runs import promote_to_production
from mousebrain.region_counts_io import detection_provenance

BRAIN = "mouse_841/841_PROJ_08_41_2p5x_z5"


def test_the_promoted_runs_detection_settings_are_found(tmp_path):
    brain = tmp_path / "841_PROJ_08_41_2p5x_z5"
    det_dir = brain / "4_Cell_Candidates"
    run = tmp_path / "optimization_runs" / brain.name / "class_20000103_000000"
    run.mkdir(parents=True)
    (run / "cells.xml").write_text("<Marker></Marker>")
    rows = [
        {"exp_id": "det_7", "exp_type": "detection", "status": "completed", "brain": BRAIN,
         "created_at": "2000-01-02T00:00:00", "output_path": str(det_dir / "detected_cells.xml"),
         "det_ball_xy": "6", "det_threshold": "10"},
        {"exp_id": "cla_gui", "exp_type": "classification", "status": "completed", "brain": BRAIN,
         "created_at": "2000-01-03T00:00:00", "output_path": str(run),
         "input_path": str(det_dir / "detected_cells.xml"), "parent_experiment": "det_7",
         "class_model_path": "model_b.h5"},
    ]
    promote_to_production(run, brain, exp_id="cla_gui")
    prov = detection_provenance(rows, brain.name, brain / "5_Classified_Cells" / "cells.xml")
    assert prov is not None
    assert prov["exp_id"] == "det_7" and prov["ball_xy"] == "6"
    assert prov["classification_exp_id"] == "cla_gui" and prov["model"] == "model_b.h5"


def test_a_record_older_than_cells_xml_is_not_trusted(tmp_path):
    """Something that writes cells.xml without its record (an older install of
    step 5, a file put back by hand) leaves a record naming a run that did not
    make the file. That run's detection settings must not be given to the
    counts; the time-guarded search names the run that wrote the file."""
    import os
    brain = tmp_path / "842_PROJ_08_42_2p5x_z5"
    det_dir = brain / "4_Cell_Candidates"
    run = tmp_path / "optimization_runs" / brain.name / "class_20000103_000000"
    run.mkdir(parents=True)
    (run / "cells.xml").write_text("<Marker></Marker>")
    prod = brain / "5_Classified_Cells"
    b = "mouse_842/842_PROJ_08_42_2p5x_z5"
    rows = [
        {"exp_id": "det_old", "exp_type": "detection", "status": "completed", "brain": b,
         "created_at": "2000-01-02T00:00:00", "output_path": str(det_dir / "a.xml"),
         "det_ball_xy": "6"},
        {"exp_id": "cla_gui", "exp_type": "classification", "status": "completed", "brain": b,
         "created_at": "2000-01-03T00:00:00", "output_path": str(run),
         "input_path": str(det_dir / "a.xml"), "parent_experiment": "det_old"},
        {"exp_id": "det_new", "exp_type": "detection", "status": "completed", "brain": b,
         "created_at": "2000-01-04T00:00:00", "output_path": str(det_dir / "b.xml"),
         "det_ball_xy": "9"},
        {"exp_id": "cla_step5", "exp_type": "classification", "status": "completed", "brain": b,
         "created_at": "2000-01-05T00:00:00", "output_path": str(prod),
         "input_path": str(det_dir / "b.xml"), "parent_experiment": "det_new"},
    ]
    promote_to_production(run, brain, exp_id="cla_gui")
    # Step 5 (without the record) re-classifies later and writes cells.xml.
    rec = prod / "promoted_from.json"
    t = rec.stat().st_mtime
    os.utime(rec, (t - 3600, t - 3600))
    (prod / "cells.xml").write_text("<Marker>new</Marker>")
    prov = detection_provenance(rows, brain.name, prod / "cells.xml")
    assert prov is not None
    assert prov["classification_exp_id"] == "cla_step5" and prov["exp_id"] == "det_new"
