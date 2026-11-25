"""A curation review resumes where it stopped: answered candidates are not asked again.

Which candidates are answered must not be kept only in memory, keyed by their
position in the layer, and emptied whenever a layer is chosen (again) with
"Use Selected Layer". After a restart of napari, or simply choosing the layer
again, every candidate would come up as unanswered although the answers were
saved.
"""
from types import SimpleNamespace

from qtpy.QtWidgets import QApplication

from curation_helpers import BRAIN_A, BRAIN_B, candidates, curation_widget, select_brain

_APP = QApplication.instance() or QApplication([])

P = [(15, 30, 30), (14, 29, 28), (16, 31, 27)]


def _new_curation_widget(w, layer):
    """A fresh curation widget (as after a restart) with ``layer`` selected."""
    from mousebrain.plugin_3d.curation_widget import CurationWidget
    viewer = SimpleNamespace(layers=SimpleNamespace(selection=SimpleNamespace(active=layer)))
    cw = CurationWidget(viewer)
    cw.candidate_confirmed.connect(w._on_candidate_confirmed)
    cw.candidate_rejected.connect(w._on_candidate_rejected)
    w.curation_widget = cw
    return cw


def _answer_two(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    layer = candidates(w, P, BRAIN_A)
    w._on_candidate_confirmed(0)
    w._on_candidate_rejected(1)
    return w, layer


def test_answered_candidates_are_not_asked_again_after_a_restart(tmp_path, monkeypatch):
    w, layer = _answer_two(tmp_path, monkeypatch)
    cw = _new_curation_widget(w, layer)
    w.create_training_layers()                 # the brain's answers, set up again
    cw.use_selected_layer()
    assert cw.confirmed == {0}
    assert cw.rejected == {1}
    assert cw.current_index == 2               # the first candidate still to answer


def test_choosing_the_layer_again_keeps_the_progress(tmp_path, monkeypatch):
    w, layer = _answer_two(tmp_path, monkeypatch)
    cw = _new_curation_widget(w, layer)
    w.create_training_layers()
    cw.use_selected_layer()
    cw.use_selected_layer()                    # clicked again
    assert (cw.confirmed, cw.rejected) == ({0}, {1})


def test_another_brains_answers_are_not_restored(tmp_path, monkeypatch):
    w, layer = _answer_two(tmp_path, monkeypatch)
    select_brain(w, tmp_path, BRAIN_B)
    other = candidates(w, P, BRAIN_B)          # same coordinates, other brain
    cw = _new_curation_widget(w, other)
    w.create_training_layers()
    cw.use_selected_layer()
    assert (cw.confirmed, cw.rejected) == (set(), set())
