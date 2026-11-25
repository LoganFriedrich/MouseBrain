"""The nucleus size filter (Min/Max Area on the Detect tab) is set from each
image's pixel size, until a person types a value; a typed value is then kept
for every image opened after it, until Auto size is clicked.

Setting the filter from the pixel size must not mark it as typed by hand
(setting the boxes fires the same signal a person's edit does): otherwise
after the first image it would never be set from a pixel size again and an
image with another pixel size would be filtered with the first image's
numbers. Opening an image with Browse File or Prev/Next must not clear the
mark, or it would overwrite what a person had really typed.
"""
from qtpy.QtWidgets import QApplication, QSpinBox

from slice_widget_helpers import Control, FakeWidget


def _widget(monkeypatch, pixel_size):
    from mousebrain.plugin_2d.sliceatlas.core import io as core_io
    app = QApplication.instance() or QApplication([])
    w = FakeWidget()
    w._size_manually_set = False
    w._pixel_size_um = pixel_size
    w._peeked_metadata = None
    w._area_um_label = Control()
    for name, value in (("min_area_spin", 10), ("max_area_spin", 5000),
                        ("diameter_spin", 30)):
        spin = QSpinBox()
        spin.setRange(1, 100000)
        spin.setValue(value)
        setattr(w, name, spin)
    w.min_area_spin.valueChanged.connect(w._on_area_spin_changed)
    w.max_area_spin.valueChanged.connect(w._on_area_spin_changed)
    w.next_pixel_size = pixel_size
    monkeypatch.setattr(core_io, "peek_metadata", lambda path: {
        "channels": [], "voxel_size_um": {"x": w.next_pixel_size}})
    return w, app


def test_each_pixel_size_gets_its_own_filter(monkeypatch, tmp_path):
    w, app = _widget(monkeypatch, 0.5)
    w._peek_and_configure(tmp_path / "PRJ_01_01_S1.tif")
    at_half = (w.min_area_spin.value(), w.max_area_spin.value())
    assert w._size_manually_set is False

    w.next_pixel_size = 0.8                 # next image: another objective
    w._peek_and_configure(tmp_path / "PRJ_01_01_S2.tif")
    at_one = (w.min_area_spin.value(), w.max_area_spin.value())
    assert at_one != at_half
    assert at_one[1] < at_half[1]           # bigger pixels, fewer of them


def test_a_typed_value_is_kept_for_the_next_image(monkeypatch, tmp_path):
    w, app = _widget(monkeypatch, 0.5)
    w._peek_and_configure(tmp_path / "PRJ_01_01_S1.tif")
    w.min_area_spin.setValue(123)           # typed by a person
    assert w._size_manually_set is True

    w.next_pixel_size = 0.8
    w._peek_and_configure(tmp_path / "PRJ_01_01_S2.tif")
    assert w.min_area_spin.value() == 123


def test_auto_size_returns_to_the_pixel_size(monkeypatch, tmp_path):
    w, app = _widget(monkeypatch, 0.5)
    w._peek_and_configure(tmp_path / "PRJ_01_01_S1.tif")
    auto = w.min_area_spin.value()
    w.min_area_spin.setValue(123)
    w._auto_size_filter()
    assert w.min_area_spin.value() == auto
    assert w._size_manually_set is False
