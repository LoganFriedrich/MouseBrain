"""Marking a Best detection run in Setup & Tuning changes only what it should.

The curation tab's Mark as Best / Promote Recent / Keep Best buttons used to
find "the Best layer" and "the recent layer" by name as well as metadata, and
in the whole viewer, and they ARCHIVED the previous Best in the tracker (which
also ends its Paradigm Best). Four ways that went wrong, each pinned here:

* the Best layer kept the session marker of a new run in its name, so it was
  also found as the "recent" layer: Keep Best hid the Best, and Promote
  archived the very run it then marked best;
* the Best layer of ANOTHER brain open in the viewer was archived (tracker
  included) when this brain's Best was marked;
* the run list's "Mark as Best" and the curation tab's left the tracker in
  different states;
* marking a layer that is not a tracked run archived the real Best and marked
  nothing.

The widget is driven through its own methods on a stand-in object (no Qt
window), with real napari points layers and a real tracker CSV in tmp_path.
"""
import types

import numpy as np
import pytest

pytest.importorskip("mousebrain.plugin_3d.tuning_widget")
pytest.importorskip("napari")

from tuning_widget_stand_in import (BRAIN_A, BRAIN_B, _row, _run_layer,  # noqa: E402
                                    _tracker, _widget, napari)

# --- the Best layer is never also the "recent" one ----------------------------

def test_keep_best_never_hides_the_best(tmp_path, monkeypatch):
    t = _tracker(tmp_path, [{"exp_id": "r_best", "brain": BRAIN_B, "marked_best": "True"}])
    w = _widget(tmp_path, monkeypatch, t)
    # A run made this session and then marked best keeps its session marker.
    best = _run_layer(w, BRAIN_B, "r_best", "det_best", "BEST: ● Run 10:00:00 (2 cells)")
    w.keep_current_best()
    assert best.metadata["cell_type"] == "det_best"
    assert not best.name.startswith("[archived]")


def test_promote_never_archives_the_best_it_found_by_name(tmp_path, monkeypatch):
    t = _tracker(tmp_path, [{"exp_id": "r_best", "brain": BRAIN_B, "marked_best": "True",
                             "paradigm_best": "True"}])
    w = _widget(tmp_path, monkeypatch, t)
    _run_layer(w, BRAIN_B, "r_best", "det_best", "BEST: ● Run 10:00:00 (2 cells)")
    w.promote_recent_to_best()
    row = _row(t, "r_best")
    assert row["archived_at"] == ""
    assert row["paradigm_best"] == "True" and row["marked_best"] == "True"


# --- another brain's Best is never touched -----------------------------------

def test_marking_this_brains_best_leaves_another_brains_best(tmp_path, monkeypatch):
    t = _tracker(tmp_path, [
        {"exp_id": "a_best", "brain": BRAIN_A, "marked_best": "True", "paradigm_best": "True"},
        {"exp_id": "b_new", "brain": BRAIN_B}])
    w = _widget(tmp_path, monkeypatch, t, brain=BRAIN_B)
    a_layer = _run_layer(w, BRAIN_A, "a_best", "det_best", "BEST: Run A (2 cells)")
    b_layer = _run_layer(w, BRAIN_B, "b_new", "det_recent", "RECENT: Run B (2 cells)")
    w.viewer.layers.selection = [b_layer]
    w.mark_current_as_best()
    a = _row(t, "a_best")
    assert (a["marked_best"], a["paradigm_best"], a["archived_at"]) == ("True", "True", "")
    assert a_layer.metadata["cell_type"] == "det_best"
    assert _row(t, "b_new")["marked_best"] == "True"
    assert b_layer.metadata["cell_type"] == "det_best"


def test_another_brains_layer_cannot_be_marked_here(tmp_path, monkeypatch):
    t = _tracker(tmp_path, [{"exp_id": "a_run", "brain": BRAIN_A},
                            {"exp_id": "b_best", "brain": BRAIN_B, "marked_best": "True"}])
    w = _widget(tmp_path, monkeypatch, t, brain=BRAIN_B)
    a_layer = _run_layer(w, BRAIN_A, "a_run", "det_recent", "RECENT: Run A (2 cells)")
    w.viewer.layers.selection = [a_layer]
    w.mark_current_as_best()
    assert _row(t, "a_run")["marked_best"] == ""
    assert _row(t, "b_best")["marked_best"] == "True"
    assert any(kind == "warning" for kind, _, _ in w.msgs.seen)


# --- both Mark as Best buttons leave the same state ----------------------------

def _two_runs(tmp_path):
    return _tracker(tmp_path, [
        {"exp_id": "old", "brain": BRAIN_B, "marked_best": "True", "paradigm_best": "True"},
        {"exp_id": "new", "brain": BRAIN_B}])


def _after_marking(tmp_path, monkeypatch, path):
    t = _two_runs(tmp_path)
    w = _widget(tmp_path, monkeypatch, t)
    old_layer = _run_layer(w, BRAIN_B, "old", "det_best", "BEST: Run old (2 cells)")
    new_layer = _run_layer(w, BRAIN_B, "new", "det_recent", "RECENT: Run new (2 cells)")
    if path == "run list":
        w._get_selected_run_id = lambda: "new"
        w._mark_selected_as_best()
    else:
        w.viewer.layers.selection = [new_layer]
        w.mark_current_as_best()
    rows = {k: {f: _row(t, k)[f] for f in ("marked_best", "paradigm_best", "archived_at")}
            for k in ("old", "new")}
    return rows, old_layer, new_layer


@pytest.mark.parametrize("path", ["run list", "curation tab"])
def test_both_mark_buttons_move_only_the_best_mark(tmp_path, monkeypatch, path):
    rows, old_layer, new_layer = _after_marking(tmp_path, monkeypatch, path)
    assert rows["old"] == {"marked_best": "", "paradigm_best": "True", "archived_at": ""}
    assert rows["new"]["marked_best"] == "True"
    assert new_layer.metadata["cell_type"] == "det_best"
    assert old_layer.metadata["cell_type"] != "det_best"


def test_paradigm_best_restyles_the_brains_layers_too(tmp_path, monkeypatch):
    # Paradigm Best also moves the brain's Best mark; the layer that showed the
    # old Best must stop being the Best layer the comparison buttons act on.
    t = _two_runs(tmp_path)
    w = _widget(tmp_path, monkeypatch, t)
    w.paradigm_status_label = types.SimpleNamespace(setText=lambda *a: None)
    old_layer = _run_layer(w, BRAIN_B, "old", "det_best", "BEST: Run old (2 cells)")
    new_layer = _run_layer(w, BRAIN_B, "new", "det_recent", "RECENT: Run new (2 cells)")
    w._get_selected_run_id = lambda: "new"
    w._mark_selected_as_paradigm_best()
    assert _row(t, "new")["paradigm_best"] == "True"
    assert _row(t, "old")["marked_best"] == ""
    assert old_layer.metadata["cell_type"] != "det_best"
    assert new_layer.metadata["cell_type"] == "det_best"


# --- a layer that is not a tracked run is refused --------------------------------

def test_a_layer_without_a_run_is_refused_and_the_best_kept(tmp_path, monkeypatch):
    t = _tracker(tmp_path, [{"exp_id": "old", "brain": BRAIN_B, "marked_best": "True",
                             "paradigm_best": "True"}])
    w = _widget(tmp_path, monkeypatch, t)
    old_layer = _run_layer(w, BRAIN_B, "old", "det_best", "BEST: Run old (2 cells)")
    hand = napari.layers.Points(np.zeros((1, 3)), name="hand-made points")
    w.viewer.layers.append(hand)
    w.viewer.layers.selection = [hand]
    w.mark_current_as_best()
    row = _row(t, "old")
    assert (row["marked_best"], row["paradigm_best"], row["archived_at"]) == ("True", "True", "")
    assert old_layer.metadata["cell_type"] == "det_best"
    assert any(kind == "warning" for kind, _, _ in w.msgs.seen)
