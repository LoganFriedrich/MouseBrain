"""Run Detection uses the insets only when they can be used: with StarDist,
and on the image (and stack slice) they were aligned to. Otherwise it runs
the ordinary detection and says the insets were ignored.

Insets beside an image are loaded automatically and 'use insets' is on by
default, so detection used to switch to the inset pipeline whenever such
files existed. That pipeline runs a StarDist model: with a threshold backend
it failed with a KeyError, and Cellpose would have been run as StarDist. The
inset manager also stayed bound to an earlier image when the next image
could not give it a base, and to the stack slice on screen when it was
built, so a detection could use another image's alignment.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from slice_widget_helpers import StubWorker, make_widget


def _with_insets(w, bound=True):
    manager = SimpleNamespace(insets={"inset_a": object()})
    w.inset_widget = SimpleNamespace(
        get_detection_settings=lambda: {"use_insets": True, "inset_manager": manager,
                                        "detect_in_base": True},
        bound_to_current_image=lambda: bound)
    used = []
    w._run_inset_detection = lambda params, settings: used.append(params)
    return used


@pytest.mark.parametrize("backend", ["Threshold", "Threshold+LoG", "Cellpose"])
def test_other_backends_run_the_ordinary_detection(tmp_path, monkeypatch, backend):
    w = make_widget(tmp_path, monkeypatch, "PRJ_01_01_S1", backend=backend)
    used = _with_insets(w)
    w._run_detection()
    assert used == []
    assert len(StubWorker.made) == 1        # the ordinary detection started
    assert "ignored" in w.status_label.text()
    assert "StarDist" in w.status_label.text()


def test_stardist_uses_the_insets_and_says_so(tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, "PRJ_01_01_S1", backend="StarDist")
    used = _with_insets(w)
    w.inset_status_label = SimpleNamespace(text="", setText=lambda t: setattr(
        w.inset_status_label, "text", t))
    w._run_detection()
    assert len(used) == 1 and used[0]["insets_used"] == 1
    assert w.inset_status_label.text == "Inset detection ON (1 insets)"


def test_insets_aligned_to_another_image_are_not_used(tmp_path, monkeypatch):
    w = make_widget(tmp_path, monkeypatch, "PRJ_01_01_S1", backend="StarDist")
    used = _with_insets(w, bound=False)
    w._run_detection()
    assert used == []
    assert len(StubWorker.made) == 1
    assert "another image" in w.status_label.text()


_APP = []      # the QApplication must outlive the widgets made with it


def _inset_widget(parent):
    from qtpy.QtWidgets import QApplication
    _APP.append(QApplication.instance() or QApplication([]))
    from mousebrain.plugin_2d.sliceatlas.napari_plugin.inset_widget import InsetWidget
    return InsetWidget(parent)


def _parent(red, current_file=None, step=0):
    return SimpleNamespace(
        viewer=SimpleNamespace(dims=SimpleNamespace(current_step=(step, 0, 0))),
        red_channel=red, green_channel=None, metadata={}, current_file=current_file,
        current_folder=None, status_label=SimpleNamespace(setText=lambda t: None),
        _get_current_slice=lambda arr: (None if arr is None else
                                        arr[0] if arr.ndim == 3 else arr))


def test_an_image_with_no_base_drops_the_previous_manager():
    parent = _parent(np.ones((16, 16)))
    widget = _inset_widget(parent)
    widget.on_base_loaded()
    assert widget.inset_manager is not None
    parent.red_channel = None              # the next image gives no base
    widget.on_base_loaded()
    assert widget.inset_manager is None


def test_a_manager_built_on_another_stack_slice_is_not_bound():
    parent = _parent(np.ones((3, 16, 16)), step=0)
    widget = _inset_widget(parent)
    widget.on_base_loaded()
    assert widget.bound_to_current_image()
    parent.viewer.dims.current_step = (2, 0, 0)
    assert not widget.bound_to_current_image()


def test_choosing_a_folder_without_loading_it_keeps_the_insets_bound():
    parent = _parent(np.ones((16, 16)))
    widget = _inset_widget(parent)
    widget.on_base_loaded()
    parent.current_folder = "a folder chosen but not loaded"
    assert widget.bound_to_current_image()
    parent.red_channel = np.ones((16, 16))   # another image's pixels
    assert not widget.bound_to_current_image()
