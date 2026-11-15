"""Rotate and flip in the slice annotator move the annotations with the image
and keep each channel's display settings.

Before this, rotate/flip turned only the image: every text label, box and
arrow stayed at its old screen position, now over a different part of the
tissue, and every channel's contrast, color and visibility went back to the
defaults.
"""
import numpy as np
import pytest

from annotator_helpers import make_widget

H, W = 6, 10            # deliberately not square, so a wrong turn shows
MARK = (1, 7)           # (y, x) of the one bright pixel the annotations sit on


def _load(widget, sections=None):
    """Load invented data through the same handler a finished load uses."""
    image = np.zeros((2, H, W), dtype=np.uint16)
    image[0] = 3
    image[0][MARK] = 1000
    image[1] = np.arange(H * W, dtype=np.uint16).reshape(H, W)
    if sections:
        data = np.zeros((sections, 2, H, W), dtype=np.uint16)
        data[sections - 1] = image
        meta = {"folder": "invented_folder", "n_channels": 2, "n_slices": sections,
                "channel_names": ["red", "green"], "shape": data.shape}
    else:
        data = image
        meta = {"file_path": "invented_section.nd2", "n_channels": 2,
                "channel_names": ["red", "green"], "shape": data.shape}
    widget._on_load_finished(True, "loaded", data, meta)


def _new_layer(widget, add):
    before = list(widget.viewer.layers)
    add()
    (layer,) = [l for l in widget.viewer.layers if l not in before]
    return layer


def _value_at(widget, coord):
    """Pixel of channel 0 under an annotation coordinate."""
    idx = tuple(int(round(c)) for c in coord)
    return widget.channel_layers[0].data[idx]


OPERATIONS = [
    ("rotate clockwise", lambda w: w._rotate_data(90)),
    ("rotate counter-clockwise", lambda w: w._rotate_data(-90)),
    ("rotate 180", lambda w: w._rotate_data(180)),
    ("flip horizontal", lambda w: w._flip_data('horizontal')),
    ("flip vertical", lambda w: w._flip_data('vertical')),
]


@pytest.mark.parametrize("name,operate", OPERATIONS, ids=[o[0] for o in OPERATIONS])
@pytest.mark.parametrize("sections", [None, 3], ids=["one section", "folder of sections"])
def test_annotations_stay_on_their_feature(name, operate, sections):
    widget, _ = make_widget()
    _load(widget, sections)
    lead = [] if sections is None else [sections - 1]

    label = _new_layer(widget, widget._add_text_label)
    label.data = np.array([lead + list(MARK)], dtype=float)
    label.features = {"text": ["cell body"]}
    box = _new_layer(widget, lambda: widget._add_shapes_layer('rectangle'))
    y, x = MARK
    box.add(np.array([lead + [y - 1, x - 1], lead + [y - 1, x + 1],
                      lead + [y + 1, x + 1], lead + [y + 1, x - 1]], dtype=float),
            shape_type='rectangle')
    assert _value_at(widget, label.data[0]) == 1000

    operate(widget)
    operate(widget)
    operate(widget)     # three times, so a turn that is merely undone is caught too

    assert _value_at(widget, label.data[0]) == 1000, name
    assert list(label.features["text"]) == ["cell body"], "the label keeps its text"
    corners = box.data[0]
    center = corners.mean(axis=0)
    assert _value_at(widget, center) == 1000, name
    assert box.shape_type == ["rectangle"]


def test_channel_display_settings_survive_rotate_and_flip():
    widget, _ = make_widget()
    _load(widget)
    red, green = widget.channel_layers
    red.contrast_limits = (5, 50)
    red.gamma = 1.7
    green.visible = False
    green.colormap = "magenta"

    widget._rotate_data(90)
    widget._flip_data('vertical')

    red2, green2 = widget.channel_layers
    assert red2.data.shape == (W, H), "the image really was turned"
    assert tuple(red2.contrast_limits) == (5, 50)
    assert red2.gamma == pytest.approx(1.7)
    assert green2.visible is False
    assert green2.colormap.name == "magenta"
    assert len([l for l in widget.viewer.layers if l in widget.channel_layers]) == 2
