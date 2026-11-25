"""'Add Confirmed / Add Rejected' with no points selected asks before using the whole layer.

With no point selected in the chosen layer, the buttons would take EVERY
point of the layer. Since the last answer wins, choosing 'Train: Non-Cells'
and clicking 'Add Confirmed to Cells Bucket' would turn every saved rejection
of the brain into a cell. So the whole layer is used only after the person
confirms a question that says how many points that is and how many answers it
would change (default No). Selected points are saved without a question.
"""
import mousebrain.plugin_3d.tuning_widget as tw

from curation_helpers import BRAIN_A, all_saved, candidates, curation_widget
from tuning_widget_stand_in import _Messages

P = [(15, 30, 30), (14, 29, 28)]


class _AnswerNo(_Messages):
    def _rec(self, kind):
        def f(parent, title, text, *a, **k):
            self.seen.append((kind, title, text))
            return self.No if kind == "question" else None
        return f


def _two_rejections(tmp_path, monkeypatch):
    w = curation_widget(tmp_path, monkeypatch, BRAIN_A)
    w.create_training_layers()
    candidates(w, P, BRAIN_A)
    w._on_candidate_rejected(0)
    w._on_candidate_rejected(1)
    non_cells = w._get_layer_by_type("train_non_cells")
    non_cells.selected_data = set()
    w.viewer.layers.selection = [non_cells]
    return w, non_cells


def _questions(msgs):
    return [text for kind, _, text in msgs.seen if kind == "question"]


def test_no_selection_and_no_to_the_question_changes_nothing(tmp_path, monkeypatch):
    w, _ = _two_rejections(tmp_path, monkeypatch)
    msgs = _AnswerNo()
    monkeypatch.setattr(tw, "QMessageBox", msgs)
    w.add_confirmed_to_training()
    assert sorted(all_saved(w.models, "non_cells")) == sorted(P)
    assert all_saved(w.models, "cells") == []
    asked = _questions(msgs)
    assert len(asked) == 1
    assert "ALL 2 points" in asked[0] and "2 of them" in asked[0]


def test_no_selection_and_yes_relabels_the_whole_layer(tmp_path, monkeypatch):
    w, _ = _two_rejections(tmp_path, monkeypatch)
    w.add_confirmed_to_training()
    assert len(_questions(w.msgs)) == 1
    assert sorted(all_saved(w.models, "cells")) == sorted(P)
    assert all_saved(w.models, "non_cells") == []


def test_selected_points_are_saved_without_a_question(tmp_path, monkeypatch):
    w, layer = _two_rejections(tmp_path, monkeypatch)
    first = tuple(int(v) for v in layer.data[0])
    layer.selected_data = {0}
    w.add_confirmed_to_training()
    assert _questions(w.msgs) == []
    assert all_saved(w.models, "cells") == [first]
    assert len(all_saved(w.models, "non_cells")) == 1
