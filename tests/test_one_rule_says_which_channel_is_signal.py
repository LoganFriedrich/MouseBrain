"""Which channel holds the cells is decided by one rule
(brain_context.channel_roles) for Load, detection and classification.

Separate rules would disagree: Load taking ch1 if any of three metadata keys
said so (and missing the "ch1" text an older crop tool wrote), detection read
from disk following a checkbox that starts checked for every brain and is
never reset for the next brain, classification reading only
channels.signal_channel.
"""
import json

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import mousebrain.plugin_3d.tuning_widget as tw  # noqa: E402
from tuning_widget_stand_in import BRAIN_A, _tracker, _widget  # noqa: E402

CROP = "2_Cropped_For_Registration"


def test_the_rule_reads_every_key_and_refuses_to_pick_a_side():
    from mousebrain.brain_context import channel_roles
    assert channel_roles({}) == (0, 1, "")
    assert channel_roles({"channels": {"signal_channel": 1, "background_channel": 0}}) == (1, 0, "")
    assert channel_roles({"signal_channel": "ch1"}) == (1, 0, "")
    assert channel_roles({"channels_swapped": True}) == (1, 0, "")
    assert channel_roles({"channels": {"signal_channel": 1}, "channels_swapped": True}) == (1, 0, "")
    roles = channel_roles({"channels": {"signal_channel": 0}, "channels_swapped": True})
    assert roles.signal is None and "contradicts" in roles.conflict


class _Box:
    def __init__(self, checked):
        self.checked = checked

    def isChecked(self):
        return self.checked

    def setChecked(self, v):
        self.checked = bool(v)


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, _s):
        pass

    def setEnabled(self, _v):
        pass


def _brain_with(tmp_path, monkeypatch, metadata):
    w = _widget(tmp_path, monkeypatch, _tracker(tmp_path, []), brain=BRAIN_A)
    folder = w.current_brain / CROP
    for ch in ("ch0", "ch1"):
        (folder / ch).mkdir(parents=True)
        tifffile.imwrite(str(folder / ch / "Z0000.tif"), np.zeros((4, 4), np.uint16))
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return w


def _select(w):
    """on_brain_changed, with the panels it refreshes stubbed."""
    brain = w.current_brain
    w.brain_combo = type("C", (), {"currentData": lambda self: str(brain)})()
    w.brain_status_label = _Label()
    w.parse_voxel_from_filename = lambda: None
    for name in ("_refresh_prefilter_runs", "_update_registration_status",
                 "_update_paradigm_status", "_offer_paradigm_settings"):
        setattr(w, name, lambda: None)
    w.on_brain_changed(brain.name)


def test_selecting_a_brain_sets_the_swap_checkbox_from_its_metadata(tmp_path, monkeypatch):
    w = _brain_with(tmp_path, monkeypatch, {"channels": {"signal_channel": 0}})
    w.swap_channels = _Box(True)  # how the widget starts, or left by the last brain
    _select(w)
    assert w.swap_channels.isChecked() is False


def test_contradicting_metadata_is_shown_and_decides_nothing(tmp_path, monkeypatch):
    w = _brain_with(tmp_path, monkeypatch,
                    {"channels": {"signal_channel": 0}, "channels_swapped": True})
    w.swap_channels = _Box(False)
    _select(w)
    # as before the rule: the checkbox follows channels_swapped
    assert w.swap_channels.isChecked() is True
    assert "contradicts" in w.brain_status_label.text


def test_load_reads_the_crop_tools_ch1_as_the_signal(tmp_path, monkeypatch):
    made = []

    class _Loader:
        def __init__(self, signal_path, background_path):
            made.append((signal_path, background_path))
            self.progress = self.finished = type("S", (), {"connect": lambda s, f: None})()

        def start(self):
            pass

    monkeypatch.setattr(tw, "BrainLoaderWorker", _Loader)
    w = _brain_with(tmp_path, monkeypatch, {"signal_channel": "ch1"})
    w.view_mode = "cell_detection"
    w.brain_loader_worker = None
    w.brain_status_label = w.load_brain_btn = _Label()
    w.load_brain_into_napari()
    assert [p.name for p in made[0]] == ["ch1", "ch0"]
