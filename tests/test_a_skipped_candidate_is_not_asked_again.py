"""A candidate a person skipped (S / Space) is not asked again when the review resumes.

What went wrong: answered candidates were restored after a restart of napari
(or choosing the layer again), but skips were kept only in memory, so every
skipped candidate came up as unreviewed again. Skips are now saved in the
brain's own folder, apart from the answers: a skip is never training data,
never overrides an answer, and ends when the candidate is answered.
"""
from types import SimpleNamespace

from qtpy.QtWidgets import QApplication

from curation_helpers import BRAIN_A, BRAIN_B, all_saved, candidates, curation_widget, select_brain

_APP = QApplication.instance() or QApplication([])

P = [(15, 30, 30), (14, 29, 28), (16, 31, 27), (17, 33, 26)]


def _curation_widget(w, layer):
    """A fresh curation widget (as after a restart) wired as Setup & Tuning wires it."""
    from mousebrain.plugin_3d.curation_widget import CurationWidget
    viewer = SimpleNamespace(layers=SimpleNamespace(selection=SimpleNamespace(active=layer)))
    cw = CurationWidget(viewer)
    cw.candidate_confirmed.connect(w._on_candidate_confirmed)
    cw.candidate_rejected.connect(w._on_candidate_rejected)
    skipped = getattr(cw, "candidate_skipped", None)
    if skipped is not None:
        skipped.connect(w._on_candidate_skipped)
    w.curation_widget = cw
    return cw


def _press(cw, index, key):
    cw.review_mode = True
    cw.current_index = index
    {"C": cw.confirm_current, "X": cw.reject_current, "S": cw.skip_current}[key]()


def _review(tmp_path, monkeypatch, brain=BRAIN_A):
    w = curation_widget(tmp_path, monkeypatch, brain)
    w.create_training_layers()
    layer = candidates(w, P, brain)
    cw = _curation_widget(w, layer)
    cw.use_selected_layer()
    return w, layer, cw


def _restart(w, layer):
    cw = _curation_widget(w, layer)
    w.create_training_layers()
    cw.use_selected_layer()
    return cw


def test_a_skipped_candidate_is_not_asked_again_after_a_restart(tmp_path, monkeypatch):
    w, layer, cw = _review(tmp_path, monkeypatch)
    _press(cw, 0, "C")
    _press(cw, 1, "S")
    cw = _restart(w, layer)
    assert cw.confirmed == {0}
    assert cw.skipped == {1}
    assert cw.current_index == 2               # neither answered nor skipped


def test_a_skip_is_never_training_data(tmp_path, monkeypatch):
    w, layer, cw = _review(tmp_path, monkeypatch)
    _press(cw, 1, "S")
    assert all_saved(w.models, "cells") == []
    assert all_saved(w.models, "non_cells") == []
    assert w._get_layer_by_type("train_cells") is None or len(w._get_layer_by_type("train_cells").data) == 0


def test_an_answer_after_a_skip_counts_and_ends_the_skip(tmp_path, monkeypatch):
    w, layer, cw = _review(tmp_path, monkeypatch)
    _press(cw, 1, "S")
    _press(cw, 1, "X")
    cw = _restart(w, layer)
    assert cw.rejected == {1}
    assert cw.skipped == set()


def test_a_skip_never_overrides_an_answer(tmp_path, monkeypatch):
    w, layer, cw = _review(tmp_path, monkeypatch)
    _press(cw, 2, "C")
    _press(cw, 2, "S")                         # went back to it and pressed S
    cw = _restart(w, layer)
    assert cw.confirmed == {2}
    assert cw.skipped == set()
    assert all_saved(w.models, "cells") == [P[2]]


def test_another_brains_skips_are_not_restored(tmp_path, monkeypatch):
    w, layer, cw = _review(tmp_path, monkeypatch)
    _press(cw, 1, "S")
    select_brain(w, tmp_path, BRAIN_B)
    other = candidates(w, P, BRAIN_B)          # same coordinates, other brain
    cw = _restart(w, other)
    assert cw.skipped == set()
