"""Points a person puts into the training layers by hand are answers too.

What went wrong: 'Train: Cells' / 'Train: Non-Cells' became a view of the
saved answers -- replaced after every C / X -- and "Create Training Dataset"
cut cubes from the saved answers only. A point a person placed in one of
those layers with napari's add-points mode was in no file: it vanished at the
next answer and was never cut, without a word. The Visualize button's layers
(cubes found on disk) had the same role as the answer layers, so they were
taken as answer layers and overwritten.
"""
from pathlib import Path

import numpy as np

from curation_helpers import (BRAIN_A, BRAIN_B, all_saved, candidates,
                              curation_widget, image, layer_points)
from test_curation_answers_are_saved_after_a_brain_change import _change_brain

PA = (15, 30, 30)    # a candidate answered with C / X
PH = (16, 31, 29)    # a point placed by hand


def _place_by_hand(w, role, point):
    """What napari's add-points mode does: append a (float) point to the layer."""
    layer = w._get_layer_by_type(role)
    layer.data = np.vstack([layer.data, np.asarray([point], dtype=float) + 0.4])
    return layer


def test_a_point_placed_by_hand_is_saved_and_cut_on_export(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    _place_by_hand(w, "train_cells", PH)
    image(w, BRAIN_A, "Signal", value=5)
    w.export_training_data()
    assert all_saved(w.models, "cells") == [PH]
    assert any(p.name == "pCellz16y31x29Ch0.tif" for p in Path(w.models).rglob("pCell*.tif"))


def test_a_point_placed_by_hand_stays_after_the_next_answer(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    _place_by_hand(w, "train_non_cells", PH)
    candidates(w, [PA], BRAIN_A)
    w._on_candidate_confirmed(0)
    assert layer_points(w, "train_non_cells") == [PH]
    assert all_saved(w.models, "non_cells") == [PH]
    assert all_saved(w.models, "cells") == [PA]


def test_a_point_placed_against_its_saved_answer_is_not_saved_and_the_person_is_told(
        tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, [PA], BRAIN_A)
    w._on_candidate_confirmed(0)                     # PA saved as a cell
    _place_by_hand(w, "train_non_cells", PA)         # ... then placed as a non-cell
    w.create_training_layers()
    assert all_saved(w.models, "cells") == [PA]      # the answer is not flipped silently
    assert all_saved(w.models, "non_cells") == []
    assert any(t == "Points Not Saved" and "z=15 y=30 x=30" in text
               for kind, t, text in w.msgs.seen)


def test_points_placed_by_hand_are_saved_for_the_brain_they_were_placed_on(
        tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    _place_by_hand(w, "train_cells", PH)
    _change_brain(w, tmp_path, BRAIN_B)
    saved = sorted(Path(w.models).rglob("curated_cells.xml"))
    assert [f.parent.name for f in saved] == [BRAIN_A]
    assert all_saved(w.models, "cells") == [PH]
    assert layer_points(w, "train_cells") == []


def test_a_layer_recorded_as_another_brains_is_refused_and_left_as_it_is(
        tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    layer = _place_by_hand(w, "train_cells", PH)
    layer.metadata["source_brain"] = str(w.current_brain.parent / BRAIN_B)
    image(w, BRAIN_A, "Signal", value=5)
    w.export_training_data()
    assert all_saved(w.models, "cells") == []
    assert list(Path(w.models).rglob("pCell*.tif")) == []
    assert layer_points(w, "train_cells") == [PH]    # not wiped either
    assert any(t == "Another Brain's Points" for kind, t, text in w.msgs.seen)


def test_the_visualize_layers_are_never_taken_as_answers(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    cubes = w.current_brain / "training_data" / "cells"
    cubes.mkdir(parents=True)
    (cubes / "z5_y6_x7.tif").write_bytes(b"")
    w.visualize_training_data()
    view = w.viewer.layers[-1]                       # the layer Visualize added
    w.create_training_layers()
    candidates(w, [PA], BRAIN_A)
    w._on_candidate_confirmed(0)
    assert w._get_layer_by_type("train_cells") is not view
    assert [tuple(int(v) for v in p) for p in view.data] == [(5, 6, 7)]
    assert all_saved(w.models, "cells") == [PA]


def test_an_export_after_points_were_deleted_from_a_layer_stops_and_says_so(
        tmp_path, monkeypatch):
    """Older versions cut from the layer, so a deleted point was left out; the
    saved answers must not be cut behind the person's back instead."""
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, [PA], BRAIN_A)
    w._on_candidate_confirmed(0)
    w._get_layer_by_type("train_cells").data = np.empty((0, 3))   # deleted by hand
    image(w, BRAIN_A, "Signal", value=5)
    w.export_training_data()
    assert list(Path(w.models).rglob("pCell*.tif")) == []
    assert all_saved(w.models, "cells") == [PA]                   # the answer is kept
    assert any(t == "Saved Answers Missing From the Layers" for kind, t, text in w.msgs.seen)
    w.create_training_layers()                                    # redraw, then export
    w.export_training_data()
    assert any(p.name == "pCellz15y30x30Ch0.tif" for p in Path(w.models).rglob("pCell*.tif"))
