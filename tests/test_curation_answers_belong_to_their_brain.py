"""Curation answers are kept per brain, and cubes are cut only from that brain's image.

Answers of every brain with the same imaging setup must not be appended to one
pooled pair of files with no brain recorded and loaded into the training
layers of whichever brain is selected next. And "Create Training Dataset" must
not cut a cube for EVERY pooled coordinate out of whatever image layer is
called 'Signal': that would cut another brain's coordinates from this brain's
pixels and write them under names without a brain over this brain's own
cubes.
"""
from pathlib import Path

import tifffile

from curation_helpers import (BRAIN_A, BRAIN_B, all_saved, candidates,
                              curation_widget, image, layer_points, select_brain)

PA = (15, 30, 30)   # answered on brain A
PB = (14, 29, 28)   # answered on brain B


def _cubes(models):
    return sorted(p.name for p in Path(models).rglob("pCell*.tif"))


def _answer_on_both(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, [PA], BRAIN_A)
    w._on_candidate_confirmed(0)
    select_brain(w, tmp_path, BRAIN_B)
    w.create_training_layers()
    candidates(w, [PB], BRAIN_B)
    w._on_candidate_confirmed(0)
    return w


def test_another_brains_answers_are_not_shown_as_this_brains(tmp_path, monkeypatch):
    w = _answer_on_both(tmp_path, monkeypatch)
    assert layer_points(w, "train_cells") == [PB]
    # brain A's answer is still saved -- in its own place
    assert sorted(all_saved(w.models, "cells")) == sorted([PA, PB])


def test_export_cuts_only_this_brains_points_from_its_own_image(tmp_path, monkeypatch):
    w = _answer_on_both(tmp_path, monkeypatch)
    image(w, BRAIN_B, "Signal", value=7)
    w.export_training_data()
    cubes = _cubes(w.models)
    assert "pCellz14y29x28Ch0.tif" in cubes
    assert "pCellz15y30x30Ch0.tif" not in cubes     # A's point, never cut from B
    cube = next(Path(w.models).rglob("pCellz14y29x28Ch0.tif"))
    assert BRAIN_B in str(cube)                      # the cube says whose it is
    assert int(tifffile.imread(str(cube)).max()) == 7


def test_export_refuses_another_brains_image(tmp_path, monkeypatch):
    w = _answer_on_both(tmp_path, monkeypatch)
    image(w, BRAIN_A, "Signal", value=3)             # only brain A's image is loaded
    w.export_training_data()
    assert _cubes(w.models) == []
    assert w.msgs.seen and w.msgs.seen[-1][0] == "warning"


def test_training_on_the_pool_sees_each_cube_cut_from_its_own_brain(tmp_path, monkeypatch):
    """Two brains exported one after the other: the pool's training.yml (what
    training reads) leads to A's cube cut from A's image and B's from B's."""
    import yaml
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, [PA], BRAIN_A)
    w._on_candidate_confirmed(0)
    image(w, BRAIN_A, "Signal", value=3)
    w.export_training_data()
    select_brain(w, tmp_path, BRAIN_B)
    w.viewer.layers.clear()
    w.create_training_layers()
    candidates(w, [PB], BRAIN_B)
    w._on_candidate_confirmed(0)
    image(w, BRAIN_B, "Signal", value=7)
    w.export_training_data()
    pool = w.models / "2p5x_z5" / "training_data"
    entries = yaml.safe_load((pool / "training.yml").read_text())["data"]
    seen = {}
    for e in entries:
        for cube in Path(e["cube_dir"]).glob("pCell*Ch0.tif"):
            seen.setdefault(cube.name, []).append(int(tifffile.imread(str(cube)).max()))
    assert seen == {"pCellz15y30x30Ch0.tif": [3], "pCellz14y29x28Ch0.tif": [7]}
