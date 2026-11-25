"""The crop window's "which channel has the cells" answer reaches the crop's
metadata, under the key every later step reads -- and the window starts on the
brain's own answer, so leaving it alone changes nothing.

The window writes the answer into .crop_position.json, and the process that
does the crop must carry it into the crop's metadata too: copying the
extraction's metadata unchanged would drop the answer, and steps 3-5 would
keep reading channels.signal_channel from the extraction.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
BRAIN = "201_PROJ_02_03_2p5x_z5"


def _crop_tool():
    spec = importlib.util.spec_from_file_location(
        "crop_subprocess_signal_test", ROOT / "src/mousebrain/plugin_3d/crop_subprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extracted_brain(tmp_path, metadata):
    b = tmp_path / "M1" / BRAIN
    ext = b / "1_Extracted_Full"
    for ch in ("ch0", "ch1"):
        (ext / ch).mkdir(parents=True)
        for z in range(2):
            tifffile.imwrite(str(ext / ch / ("Z%04d.tif" % z)), np.zeros((10, 4), np.uint16))
    (ext / "metadata.json").write_text(json.dumps(metadata))
    return b


def _crop(b, **choice):
    (b / ".crop_position.json").write_text(json.dumps(
        dict({"y_crop": 6, "rows_kept": 6, "total_rows": 10}, **choice)))
    assert _crop_tool().do_crop(b)
    return json.loads((b / "2_Cropped_For_Registration" / "metadata.json").read_text())


def test_the_channel_chosen_in_the_crop_window_is_what_later_steps_read(tmp_path):
    b = _extracted_brain(tmp_path, {"channels": {"signal_channel": 0, "background_channel": 1}})
    meta = _crop(b, signal_channel="ch1", channels_swapped=True)
    assert meta["channels"]["signal_channel"] == 1
    assert meta["channels"]["background_channel"] == 0
    # nothing left in the crop's metadata that could say otherwise
    assert "channels_swapped" not in meta and "signal_channel" not in meta
    from mousebrain.brain_context import channel_roles
    assert channel_roles(meta) == (1, 0, "")


def test_leaving_the_choice_alone_changes_nothing_in_the_channels(tmp_path):
    """A brain with no background channel recorded keeps it that way."""
    chans = {"signal_channel": 0, "background_channel": None, "count": 2}
    b = _extracted_brain(tmp_path, {"channels": dict(chans)})
    meta = _crop(b, signal_channel="ch0", channels_swapped=False)
    assert meta["channels"] == chans


def test_choosing_the_other_channel_swaps_the_two_roles(tmp_path):
    b = _extracted_brain(tmp_path, {"channels": {"signal_channel": 1, "background_channel": 0}})
    meta = _crop(b, signal_channel="ch0", channels_swapped=False)
    assert (meta["channels"]["signal_channel"], meta["channels"]["background_channel"]) == (0, 1)


def test_a_crop_request_without_a_choice_keeps_the_extractions_answer(tmp_path):
    b = _extracted_brain(tmp_path, {"channels": {"signal_channel": 1, "background_channel": 0}})
    meta = _crop(b)
    assert meta["channels"]["signal_channel"] == 1


# --------------------------------------------------------- the crop window
class _Radio:
    """A radio button of an exclusive group: checking one unchecks the other."""
    def __init__(self):
        self.checked = False
        self.group = None

    def setChecked(self, v):
        if v and self.group is not None and self.group.exclusive:
            for r in self.group.radios:
                r.checked = False
        self.checked = bool(v)


class _Group:
    def __init__(self, a, b):
        self.radios = (a, b)
        self.exclusive = True
        a.group = b.group = self

    def setExclusive(self, v):
        self.exclusive = v

    def checkedId(self):
        for i, r in enumerate(self.radios):
            if r.checked:
                return i
        return -1


def _window(metadata, folder="brain_x"):
    ch0, ch1 = _Radio(), _Radio()
    ch0.checked = True  # how the window is built
    help_label = SimpleNamespace(text_=None)
    help_label.setText = lambda t: setattr(help_label, "text_", t)
    return SimpleNamespace(metadata=metadata, pipeline_folder=Path(folder), ch0_radio=ch0,
                           ch1_radio=ch1, channel_button_group=_Group(ch0, ch1),
                           channel_help=help_label)


def _preset(win):
    from mousebrain.plugin_3d import manual_crop_widget as mcw
    mcw.ManualCropWidget._preset_signal_choice(win)


def test_the_window_starts_on_the_brains_own_signal_channel():
    win = _window({"channels": {"signal_channel": 1}})
    _preset(win)
    assert win.channel_button_group.checkedId() == 1


def test_contradicting_metadata_leaves_the_choice_to_the_person():
    win = _window({"channels": {"signal_channel": 0}, "channels_swapped": True})
    _preset(win)
    assert win.channel_button_group.checkedId() == -1
    assert win._signal_choice_required is True
    assert "contradicts" in win.channel_help.text_


def test_a_persons_choice_is_not_reset_on_the_same_brain():
    win = _window({"channels": {"signal_channel": 0}})
    _preset(win)
    win.ch0_radio.checked, win.ch1_radio.checked = False, True  # the person picks ch1
    _preset(win)  # Add Crop Line on the same brain
    assert win.channel_button_group.checkedId() == 1


def test_a_signal_channel_the_window_does_not_show_is_kept_unless_changed():
    win = _window({"channels": {"signal_channel": 2}})
    _preset(win)
    assert win.channel_button_group.checkedId() == -1
    assert win._signal_choice_required is False
