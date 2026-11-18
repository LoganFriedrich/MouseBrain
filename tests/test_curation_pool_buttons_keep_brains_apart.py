"""The other doors into the training data keep brains apart and keep answers.

"Merge Selected -> Shared Pool", "Merge Selected into Training Layers" (XML
batches), "Suspicious -> Non-Cells Training", "Start Training Data Fresh" and
the training sources table all read or write the same per-brain answers and
cubes as curation. What went wrong before: merging to the pool put every
brain's cubes in one folder under names without a brain, so a second brain's
cube at the same coordinates was skipped as "already exists"; XML batches of
other brains were loaded and then cut from this brain's image; suspicious
points went only into a layer, never into the answers.
"""
from pathlib import Path

import numpy as np
import tifffile
import yaml
from qtpy.QtWidgets import QApplication, QTableWidget

import mousebrain.plugin_3d.tuning_widget as tw
from curation_helpers import (BRAIN_A, BRAIN_B, _Label, all_saved, candidates,
                              curation_widget, image, select_brain)

_APP = QApplication.instance() or QApplication([])

P = (15, 30, 30)
Q = (14, 29, 28)


def _cube(folder, key, value):
    folder.mkdir(parents=True, exist_ok=True)
    z, y, x = key
    tifffile.imwrite(str(folder / f"pCellz{z}y{y}x{x}Ch0.tif"),
                     np.full((20, 50, 50), value, dtype=np.uint16))


def _sources_widget(tmp_path, monkeypatch, brain=BRAIN_A):
    w = curation_widget(tmp_path, monkeypatch, brain)
    brains = tmp_path / "brains"
    monkeypatch.setattr(tw, "BRAINS_ROOT", brains)
    w.training_sources_table = QTableWidget(0, 4)
    w.shared_pool_label = _Label()
    w.training_summary_label = _Label()
    return w, brains


def test_merge_to_pool_keeps_two_brains_cubes_at_the_same_point(tmp_path, monkeypatch):
    w, brains = _sources_widget(tmp_path, monkeypatch)
    _cube(brains / "M1" / BRAIN_A / "training_data" / "cells", P, 3)
    _cube(brains / "M2" / BRAIN_B / "training_data" / "cells", P, 7)
    w._refresh_training_sources()
    for row in range(w.training_sources_table.rowCount()):   # tick every source
        w.training_sources_table.cellWidget(row, 0).findChildren(tw.QCheckBox)[0].setChecked(True)
    w._merge_to_shared_pool()
    pool = w.models / "2p5x_z5" / "training_data"
    values = sorted(int(tifffile.imread(str(f)).max()) for f in pool.rglob("pCellz15y30x30Ch0.tif"))
    assert values == [3, 7]
    # and training on the pool reads both
    paths = w._get_selected_yaml_paths()
    dirs = [Path(e["cube_dir"]) for yp in paths
            for e in yaml.safe_load(yp.read_text())["data"]]
    assert {int(tifffile.imread(str(d / "pCellz15y30x30Ch0.tif")).max())
            for d in dirs if (d / "pCellz15y30x30Ch0.tif").exists()} == {3, 7}


def test_export_keeps_cubes_merged_in_and_removes_only_contradicted_ones(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    merged = w.training_data_dir / "cells"
    _cube(merged, (12, 33, 31), 5)                  # copied in by a merge, no answer
    _cube(w.training_data_dir / "non_cells", P, 5)  # an old export: P was a non-cell
    candidates(w, [P], BRAIN_A)
    w._on_candidate_confirmed(0)                    # ... now P is a cell
    image(w, BRAIN_A)
    w.export_training_data()
    assert (merged / "pCellz12y33x31Ch0.tif").exists()
    assert not (w.training_data_dir / "non_cells" / "pCellz15y30x30Ch0.tif").exists()
    assert (merged / "pCellz15y30x30Ch0.tif").exists()


def _batch(folder, kind, points):
    """An answers file as older versions wrote it (CellCounter markers)."""
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"curated_{kind}.xml"
    markers = "".join(f"<Marker><MarkerX>{x}</MarkerX><MarkerY>{y}</MarkerY>"
                      f"<MarkerZ>{z}</MarkerZ></Marker>" for z, y, x in points)
    f.write_text("<CellCounter_Marker_File><Marker_Data><Marker_Type><Type>1</Type>"
                 + markers + "</Marker_Type></Marker_Data></CellCounter_Marker_File>",
                 encoding="utf-8")
    return f


def test_xml_batch_merge_takes_only_this_brains_batches_and_keeps_answers(tmp_path, monkeypatch):
    w, brains = _sources_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, [P], BRAIN_A)
    w._on_candidate_confirmed(0)                    # a person: P is a cell
    _batch(brains / "M1" / BRAIN_A / "training_data", "non_cells", [P, Q])   # older answers of A
    _batch(brains / "M2" / BRAIN_B / "training_data", "cells", [(10, 30, 30)])  # B's
    w.xml_batches_table = QTableWidget(0, 5)
    w.xml_batch_summary_label = _Label()
    w._discover_xml_batches()                       # every row starts ticked
    w._merge_xml_batches()
    assert all_saved(w.models, "cells") == [P]      # the person's answer stands, B left out
    assert all_saved(w.models, "non_cells") == [Q]


def test_suspicious_points_are_saved_without_overwriting_answers(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, [P], BRAIN_A)
    w._on_candidate_confirmed(0)
    w._prefilter_result = {"suspicious_coords": [list(P), list(Q)]}
    w._send_suspicious_to_training()
    assert all_saved(w.models, "cells") == [P]
    assert all_saved(w.models, "non_cells") == [Q]


def test_start_fresh_clears_every_brains_cubes_but_keeps_the_answers(tmp_path, monkeypatch):
    w, _ = _sources_widget(tmp_path, monkeypatch)
    w.create_training_layers()
    candidates(w, [P], BRAIN_A)
    w._on_candidate_confirmed(0)
    image(w, BRAIN_A)
    w.export_training_data()
    select_brain(w, tmp_path, BRAIN_A)
    w._start_training_fresh()
    pool = w.models / "2p5x_z5" / "training_data"
    assert list(pool.rglob("pCell*.tif")) == []
    assert all_saved(w.models, "cells") == [P]
