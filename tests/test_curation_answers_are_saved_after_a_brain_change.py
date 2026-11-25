"""After another brain is chosen, curation answers are still saved -- or the person is told.

Choosing another brain clears the training folder (so the last brain's pool
is no longer written to). The C / X handlers must then not return without
saving and without a word while the tab still says "Auto-save: <the last
brain's folder>". The curation widget must also drop the last brain's
candidates, so answers given on them are not shown as answered.
"""
from types import SimpleNamespace

import numpy as np
from qtpy.QtWidgets import QApplication

from curation_helpers import (BRAIN_A, BRAIN_B, _Label, all_saved, candidates,
                              curation_widget, layer_points, select_brain)

_APP = QApplication.instance() or QApplication([])

PA = (15, 30, 30)
PB = (14, 29, 28)


def test_the_first_answer_after_a_brain_change_is_saved(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    select_brain(w, tmp_path, BRAIN_B)
    candidates(w, [PB], BRAIN_B)
    w._on_candidate_confirmed(0)
    assert all_saved(w.models, "cells") == [PB]
    assert BRAIN_B in str(w.training_path_label.text())


def _change_brain(w, tmp_path, brain):
    """Drive on_brain_changed with the rest of the window stubbed out."""
    w.brain_combo = SimpleNamespace(currentData=lambda: str(tmp_path / "M" / brain))
    w.brain_status_label = _Label()
    for name in ("_update_counting_source_label", "load_metadata",
                 "parse_voxel_from_filename", "_refresh_context_runs",
                 "_refresh_prefilter_runs", "_update_registration_status",
                 "_update_paradigm_status", "_offer_paradigm_settings"):
        setattr(w, name, lambda *a, **k: None)
    w._set_swap_from_metadata = lambda: None
    w.on_brain_changed(brain)


def test_choosing_another_brain_clears_the_last_brains_curation(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, [PA], BRAIN_A)
    w._on_candidate_confirmed(0)
    assert "Auto-save" in w.training_path_label.text()
    _change_brain(w, tmp_path, BRAIN_B)
    assert "Auto-save" not in w.training_path_label.text()
    assert w.training_data_path.text() == ""
    assert layer_points(w, "train_cells") == []          # A's answer is not shown as B's
    assert all_saved(w.models, "cells") == [PA]          # ... and is still saved


def test_an_answer_on_the_last_brains_candidates_is_refused(tmp_path, monkeypatch):
    from mousebrain.plugin_3d.curation_widget import CurationWidget
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    cw = CurationWidget(SimpleNamespace(layers=[]))
    cw.candidate_confirmed.connect(w._on_candidate_confirmed)
    w.curation_widget = cw
    layer = candidates(w, [PA], BRAIN_A)                 # A's candidates ...
    cw.candidates, cw.candidates_layer = layer.data.copy(), layer
    select_brain(w, tmp_path, BRAIN_B)                   # ... then B is selected
    cw.review_mode = True
    cw.current_index = 0
    cw.confirm_current()
    assert all_saved(w.models, "cells") == []
    assert 0 not in cw.confirmed                         # not shown as answered
    assert any(kind == "warning" for kind, _, _ in w.msgs.seen)
