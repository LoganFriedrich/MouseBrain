"""A curation answer is kept once, with the label the person gave last.

What went wrong: every C (cell) / X (not a cell) was appended to the training
files, so an answer the person changed stayed in BOTH curated_cells.xml and
curated_non_cells.xml -- the classifier was trained on the same point as a
cell and as a non-cell -- and an answer given twice was written twice.
"""
import pytest

from curation_helpers import (BRAIN_A, all_saved, candidates, curation_widget,
                              layer_points)

P = (15, 30, 30)


def test_a_changed_answer_leaves_the_other_file(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch)
    w.create_training_layers()
    candidates(w, [P], BRAIN_A)
    w._on_candidate_confirmed(0)
    w._on_candidate_rejected(0)          # the person changed their mind
    assert all_saved(w.models, "cells") == []
    assert all_saved(w.models, "non_cells") == [P]
    assert layer_points(w, "train_cells") == []
    assert layer_points(w, "train_non_cells") == [P]


def test_changing_back_moves_it_back(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch)
    w.create_training_layers()
    candidates(w, [P], BRAIN_A)
    w._on_candidate_rejected(0)
    w._on_candidate_confirmed(0)
    assert all_saved(w.models, "cells") == [P]
    assert all_saved(w.models, "non_cells") == []


def test_a_repeated_answer_is_saved_once(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch)
    w.create_training_layers()
    candidates(w, [P], BRAIN_A)
    w._on_candidate_confirmed(0)
    w._on_candidate_confirmed(0)
    assert all_saved(w.models, "cells") == [P]
    assert layer_points(w, "train_cells") == [P]


def test_points_added_by_hand_move_too(tmp_path, monkeypatch):
    """'Add selected to cells / non-cells' obeys the same rule."""
    w = curation_widget(tmp_path, monkeypatch)
    w.create_training_layers()
    layer = candidates(w, [P], BRAIN_A)
    w.viewer.layers.append(layer)
    w.viewer.layers.selection = [layer]
    w.add_confirmed_to_training()
    w.add_rejected_to_training()
    assert all_saved(w.models, "cells") == []
    assert all_saved(w.models, "non_cells") == [P]


def test_an_unreadable_answers_file_is_never_replaced(tmp_path, monkeypatch):
    """A damaged file stops the save; it used to be replaced by a new, nearly empty one."""
    w = curation_widget(tmp_path, monkeypatch)
    w.create_training_layers()
    candidates(w, [P], BRAIN_A)
    damaged = w._cells_xml
    damaged.write_text("<CellCounter_Marker_File><Marker_Data>", encoding="utf-8")
    w._on_candidate_confirmed(0)
    assert damaged.read_text(encoding="utf-8") == "<CellCounter_Marker_File><Marker_Data>"
    assert any("NOT saved" in text for _, _, text in w.msgs.seen)
