"""The saved atlas fit's 'elastix_warp_on_screen' is true only while a warp is drawn.

Region counts never use the Elastix warp (they use the Align tab's position,
rotation and scale), so the saved fit says whether one was on screen. That fact
must be right: after a position-slider, orientation or spin change the
unwarped atlas (without the warped outlines) is redrawn, after a
new image is loaded or another slice is shown the warp belongs to a different
image, and when Elastix returns no transform nothing was warped. A warp whose
run finishes after the atlas position or orientation was changed belongs to an
atlas slice no longer selected: it is neither drawn nor recorded. In none of
those cases may the record say a warp was on screen.

These tests drive the real AlignmentWidget methods on a stand-in (no window),
pressing Register Atlas (Elastix) through the real _run_elastix_registration
with a stand-in for the background worker. A field missing from the record is
read with .get() so that it fails as an assertion, not a KeyError.
"""
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mousebrain.plugin_2d.sliceatlas.napari_plugin import alignment_widget
from mousebrain.plugin_2d.sliceatlas.napari_plugin.alignment_widget import AlignmentWidget


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, _s):
        pass


class _Layers:
    def remove(self, _layer):
        pass


class _FakeElastixWorker:
    """Stands in for the Elastix thread: remembers the slot it must call."""
    last = None

    def __init__(self, **_kwargs):
        self.slot = None
        self.finished = SimpleNamespace(connect=self._connect)
        _FakeElastixWorker.last = self

    def _connect(self, slot):
        self.slot = slot

    def start(self):
        pass


@pytest.fixture(autouse=True)
def _no_elastix_thread(monkeypatch):
    monkeypatch.setattr(alignment_widget, "ElastixRegistrationWorker", _FakeElastixWorker)


class _Viewer:
    def __init__(self):
        self.layers = _Layers()
        self.dims = SimpleNamespace(current_step=(0, 0, 0))

    def add_image(self, *_a, **_k):
        return object()

    def add_labels(self, *_a, **_k):
        return object()


def _aligner(red):
    atlas = SimpleNamespace(
        resolution=(10.0, 10.0, 10.0),
    )
    manager = SimpleNamespace(
        brain_atlas_name="test_atlas_10um", brain_atlas=atlas,
        get_reference_slice=lambda *_a, **_k: np.zeros((6, 6), dtype=np.uint16),
        get_annotation_slice=lambda *_a, **_k: np.ones((6, 6), dtype=np.int32),
    )
    aw = SimpleNamespace(
        atlas_loaded=True,
        atlas_manager=manager,
        parent_widget=SimpleNamespace(red_channel=red),
        viewer=_Viewer(),
        current_orientation="coronal",
        current_atlas_position=640,
        transform_offset_x=0, transform_offset_y=0,
        transform_rotation=0, transform_scale=1.0,
        warped_annotation=None, deformation_field=None,
        _elastix_source=None, _warp_shown_for=None,
        atlas_ref_layer=None, atlas_labels_layer=None, boundaries_layer=None,
        elastix_btn=SimpleNamespace(setEnabled=lambda _b: None),
        elastix_status_label=_Label(),
        opacity_slider=SimpleNamespace(value=lambda: 50),
        show_reference_check=SimpleNamespace(isChecked=lambda: True),
        show_regions_check=SimpleNamespace(isChecked=lambda: True),
        show_boundaries_check=SimpleNamespace(isChecked=lambda: False),
    )
    # Bind every real method these tests use that this version of the widget
    # has; a method it lacks then shows up as wrong behaviour, not a setup error.
    for name in ("get_current_registration", "_current_image_key", "_key_is_current",
                 "_elastix_source_is_current", "_elastix_warp_on_screen",
                 "_get_current_image", "_run_elastix_registration",
                 "_on_elastix_finished", "_update_atlas_overlay", "_loaded_atlas_name",
                 "_on_opacity_changed", "_update_overlay_visibility"):
        if hasattr(AlignmentWidget, name):
            setattr(aw, name, types.MethodType(getattr(AlignmentWidget, name), aw))
    aw.drawn_warps = []
    aw.opacity_label = _Label()
    aw.show_reference_check = SimpleNamespace(isChecked=lambda: True)

    def _draw(result):
        aw.drawn_warps.append(result)
        return True, None

    aw._update_atlas_overlay_with_warp = _draw
    return aw


def _start_warp(aw):
    """Press Register Atlas (Elastix); returns the slot the thread will call."""
    aw._run_elastix_registration()
    return _FakeElastixWorker.last.slot


def _finish_warp(slot, applied=True):
    """The Elastix thread finishes with a warp."""
    result = {"warped_annotation": np.full((6, 6), 7, dtype=np.int32),
              "warped_reference": np.zeros((6, 6)),
              "deformation_field": None,
              "affine_params": {}, "success": True}
    if not applied:
        result["warp_applied"] = False
    slot(True, "Registration complete", result)


def _warp(aw, applied=True):
    """What pressing Register Atlas (Elastix) and waiting for it does."""
    _finish_warp(_start_warp(aw), applied)


def _on_screen(aw):
    return aw.get_current_registration().get("elastix_warp_on_screen")


def test_a_warp_just_drawn_is_recorded():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    _warp(aw)
    assert len(aw.drawn_warps) == 1
    assert _on_screen(aw) is True


def test_a_slider_change_after_a_warp_is_recorded_as_no_warp():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    _warp(aw)
    aw.transform_scale = 1.5
    aw._update_atlas_overlay()  # what the position slider, orientation and spin boxes call
    assert _on_screen(aw) is False


def test_a_new_image_after_a_warp_is_recorded_as_no_warp():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    _warp(aw)
    aw.parent_widget.red_channel = np.zeros((6, 6), dtype=np.uint16)
    assert _on_screen(aw) is False


def test_another_slice_of_a_stack_is_recorded_as_no_warp():
    aw = _aligner(np.zeros((3, 6, 6), dtype=np.uint16))
    _warp(aw)
    assert _on_screen(aw) is True
    aw.viewer.dims.current_step = (2, 0, 0)
    assert _on_screen(aw) is False


def test_elastix_returning_no_transform_is_recorded_as_no_warp():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    _warp(aw, applied=False)
    assert _on_screen(aw) is False


def test_a_warp_finishing_after_the_position_moved_is_not_drawn_or_recorded():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    slot = _start_warp(aw)
    # While Elastix runs, the person moves the position slider.
    aw.current_atlas_position = 700
    aw._update_atlas_overlay()
    _finish_warp(slot)
    assert aw.drawn_warps == []
    assert _on_screen(aw) is False


def test_a_warp_finishing_after_the_orientation_changed_is_not_drawn_or_recorded():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    slot = _start_warp(aw)
    aw.current_orientation = "sagittal"
    aw._update_atlas_overlay()
    _finish_warp(slot)
    assert aw.drawn_warps == []
    assert _on_screen(aw) is False


def test_a_finished_run_does_not_keep_the_image_it_ran_on():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    _warp(aw)
    assert getattr(aw, "_elastix_source", None) is None
    slot = _start_warp(aw)
    slot(False, "Elastix failed", None)
    assert getattr(aw, "_elastix_source", None) is None


def test_loading_another_image_after_a_warp_frees_the_previous_image():
    # Loading an image does not redraw the Align tab's overlay, so whatever
    # the tab holds to recognise the warped image must not keep that image
    # (possibly a whole stack) alive once another one replaces it.
    import gc
    import weakref

    first = np.zeros((3, 6, 6), dtype=np.uint16)
    first_ref = weakref.ref(first)
    aw = _aligner(first)
    _warp(aw)
    assert _on_screen(aw) is True
    del first
    aw.parent_widget.red_channel = np.zeros((3, 6, 6), dtype=np.uint16)
    gc.collect()
    assert first_ref() is None
    assert _on_screen(aw) is False


def test_going_back_to_the_warped_slice_shows_the_warp_again():
    aw = _aligner(np.zeros((3, 6, 6), dtype=np.uint16))
    _warp(aw)
    aw.viewer.dims.current_step = (2, 0, 0)
    assert _on_screen(aw) is False
    aw.viewer.dims.current_step = (0, 0, 0)
    assert _on_screen(aw) is True


def test_a_warp_finishing_after_another_atlas_was_loaded_is_not_drawn_or_recorded():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    slot = _start_warp(aw)
    # While Elastix runs, another atlas (another resolution) is loaded and the
    # slider ends on the same index number -- a different atlas slice.
    aw.atlas_manager = SimpleNamespace(
        brain_atlas_name="test_atlas_25um",
        brain_atlas=SimpleNamespace(resolution=(25.0, 25.0, 25.0)),
        get_reference_slice=aw.atlas_manager.get_reference_slice,
        get_annotation_slice=aw.atlas_manager.get_annotation_slice,
    )
    _finish_warp(slot)
    assert aw.drawn_warps == []
    assert _on_screen(aw) is False


def test_a_warp_that_cannot_be_drawn_is_not_recorded_as_on_screen():
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    # The real drawing code, on a viewer whose add_labels fails.
    aw._update_atlas_overlay_with_warp = types.MethodType(
        AlignmentWidget._update_atlas_overlay_with_warp, aw)
    aw._add_boundary_shapes = lambda *_a, **_k: None
    calls = {"n": 0}

    def _labels_fail(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("labels layer could not be made")
        return object()

    aw.viewer.add_labels = _labels_fail
    _warp(aw)

    assert _on_screen(aw) is False
    assert aw.warped_annotation is None
    assert "could not be drawn" in aw.elastix_status_label.text
    assert not aw.elastix_status_label.text.startswith("Registered")


def test_the_opacity_slider_and_show_checkboxes_leave_the_warp_on_screen():
    # These controls do not redraw the overlay, so the warp is still what is
    # on screen afterwards; the operator docs say so.
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    _warp(aw)
    aw._on_opacity_changed(20)
    aw._update_overlay_visibility()
    assert _on_screen(aw) is True


class _TrackedLayers(list):
    def remove(self, layer):
        if layer not in self:
            raise ValueError("not in viewer")
        list.remove(self, layer)


def test_a_redraw_after_a_warp_leaves_no_warped_outlines_on_screen():
    # 'Show region boundaries' draws outlines from the warp. A position-slider
    # or spin change redraws the unwarped atlas and records no warp on
    # screen, so the warped outlines must go too -- otherwise a warp is
    # visibly on screen while the saved fit says none is.
    aw = _aligner(np.zeros((6, 6), dtype=np.uint16))
    layers = _TrackedLayers()
    aw.viewer.layers = layers

    def _add(*_a, **_k):
        layer = object()
        layers.append(layer)
        return layer

    aw.viewer.add_image = _add
    aw.viewer.add_labels = _add
    aw.show_boundaries_check = SimpleNamespace(isChecked=lambda: True)

    def _outlines(_annotation, _deformation_field=None):
        aw.boundaries_layer = _add()

    aw._add_boundary_shapes = _outlines
    AlignmentWidget._update_atlas_overlay_with_warp(
        aw, {"warped_annotation": np.full((6, 6), 7, dtype=np.int32),
             "warped_reference": np.zeros((6, 6)), "deformation_field": None})
    outlines = aw.boundaries_layer
    assert outlines in layers

    aw.transform_offset_x = 3
    aw._update_atlas_overlay()  # what the position slider / spin boxes call

    assert outlines not in layers
    assert aw.boundaries_layer is None
    assert len(layers) == 2  # the unwarped reference and regions only
