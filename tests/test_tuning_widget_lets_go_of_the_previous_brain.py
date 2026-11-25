"""Selecting another brain in Setup & Tuning drops everything held for the last one.

If changing brain changed only current_brain, detection could run on the
previous brain's pixels and be logged under the new one, Save would write the
previous brain's metadata into the new brain's folders (and into every image
folder, forging "cropped"), and a pre-filter result could be saved into the
wrong brain.
"""
from pathlib import Path
from types import SimpleNamespace

from mousebrain.brain_context import (PER_BRAIN_DEFAULTS, layer_source_folder,
                                      metadata_save_target, pick_image_layers,
                                      reset_per_brain_state, tag_layer)

TW = (Path(__file__).resolve().parents[1] / "src" / "mousebrain" / "plugin_3d"
      / "tuning_widget.py")


def _method(name):
    text = TW.read_text(encoding="utf-8")
    body = text[text.index("    def %s(self" % name):]
    return body[:body.index("\n    def ", 10)]


def _layer(name, brain=None, folder=None):
    layer = SimpleNamespace(name=name, data=object(), metadata={})
    if brain is not None:
        tag_layer(layer, Path(brain), Path(folder) if folder else None)
    return layer


def test_reset_clears_every_per_brain_field():
    w = SimpleNamespace(metadata={"voxel": 1}, _prefilter_result="A's result",
                        training_data_dir="A's pool", last_run_id="det_A",
                        _metadata_source="A/crop")
    reset_per_brain_state(w)
    for name, default in PER_BRAIN_DEFAULTS.items():
        expected = default() if callable(default) else default
        assert getattr(w, name) == expected, name


def test_every_reset_field_is_one_the_widget_uses():
    text = TW.read_text(encoding="utf-8")
    for name in PER_BRAIN_DEFAULTS:
        assert "self.%s" % name in text, name


def test_changing_brain_resets_the_state():
    assert "reset_per_brain_state(self)" in _method("on_brain_changed")


def test_only_the_selected_brains_layers_are_used(tmp_path):
    a, b = tmp_path / "BRAIN_A", tmp_path / "BRAIN_B"
    layers = [_layer("BRAIN_A - Signal (cropped)", a, a / "crop"),
              _layer("BRAIN_A - Background (cropped)", a, a / "crop")]
    sig, bg, other = pick_image_layers(layers, b)
    assert sig is None and bg is None and other
    sig, bg, other = pick_image_layers(layers, a)
    assert sig is layers[0] and bg is layers[1] and not other
    assert layer_source_folder(sig) == str(a / "crop")


def test_an_untagged_layer_counts_as_another_brains(tmp_path):
    sig, bg, other = pick_image_layers([_layer("Signal")], tmp_path / "BRAIN_A")
    assert sig is None and other


def test_detection_refuses_another_brains_images_and_records_its_input():
    body = _method("run_test_detection")
    assert "pick_image_layers(" in body
    assert "input_path=input_path" in body


def test_classification_uses_only_this_brains_layers():
    assert "pick_image_layers(" in _method("run_classification")


def test_metadata_is_saved_only_where_it_was_loaded_from(tmp_path):
    assert metadata_save_target(None) is None
    assert metadata_save_target(tmp_path / "crop") == tmp_path / "crop"
    body = _method("save_metadata")
    assert "metadata_save_target(" in body
    assert 'for folder in ["2_Cropped_For_Registration_Manual"' not in body


def test_load_metadata_never_keeps_the_previous_brains():
    body = _method("load_metadata")
    assert "self.metadata = {}" in body
    assert "self._metadata_source = meta_path.parent" in body
