"""Load shows the images the atlas was fitted to, and classification refuses
layers loaded from any other folder.

Load took the manual crop, else the crop, else the extraction -- by which
folder existed -- while detection-from-disk and classification read the
registered folder. Classification cut its cubes from whatever layer was
loaded, at positions in the registered images, and such a run can be
promoted to production.
"""
import numpy as np
import pytest

napari = pytest.importorskip("napari")
tifffile = pytest.importorskip("tifffile")

import mousebrain.plugin_3d.tuning_widget as tw  # noqa: E402
from mousebrain.config import record_registered_space  # noqa: E402
from tuning_widget_stand_in import BRAIN_A, _tracker, _widget  # noqa: E402
from test_a_classification_run_keeps_its_brain import _FakeClassification, _widget_for  # noqa: E402

MANUAL = "2_Cropped_For_Registration_Manual"
CROP = "2_Cropped_For_Registration"


def _planes(folder):
    for ch in ("ch0", "ch1"):
        (folder / ch).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(folder / ch / "Z0000.tif"), np.zeros((4, 4), np.uint16))


class _Btn:
    def setEnabled(self, _v): pass
    def setText(self, _t): pass
    def setStyleSheet(self, _s): pass


class _Sig:
    def connect(self, _fn): pass


class _FakeLoader:
    made = []

    def __init__(self, signal_path, background_path):
        self.signal_path = signal_path
        self.progress, self.finished = _Sig(), _Sig()
        _FakeLoader.made.append(self)

    def start(self):
        pass

    def isRunning(self):
        return False


def test_load_shows_the_registered_crop_not_a_later_manual_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(tw, "BrainLoaderWorker", _FakeLoader)
    _FakeLoader.made = []
    w = _widget(tmp_path, monkeypatch, _tracker(tmp_path, []), brain=BRAIN_A)
    _planes(w.current_brain / CROP)
    _planes(w.current_brain / MANUAL)  # made after the brain was registered
    record_registered_space(w.current_brain / "3_Registered_Atlas", "cropped",
                            w.current_brain / CROP)
    w.view_mode = "cell_detection"
    w.brain_loader_worker = None
    w.brain_status_label = w.load_brain_btn = _Btn()
    w.load_brain_into_napari()
    assert len(_FakeLoader.made) == 1, w.msgs.seen
    assert _FakeLoader.made[0].signal_path.parent == w.current_brain / CROP


def test_an_unregistered_brain_still_loads_its_manual_crop_first(tmp_path, monkeypatch):
    monkeypatch.setattr(tw, "BrainLoaderWorker", _FakeLoader)
    _FakeLoader.made = []
    w = _widget(tmp_path, monkeypatch, _tracker(tmp_path, []), brain=BRAIN_A)
    _planes(w.current_brain / CROP)
    _planes(w.current_brain / MANUAL)
    w.view_mode = "cell_detection"
    w.brain_loader_worker = None
    w.brain_status_label = w.load_brain_btn = _Btn()
    w.load_brain_into_napari()
    assert _FakeLoader.made[0].signal_path.parent == w.current_brain / MANUAL


def test_classification_refuses_layers_not_loaded_from_the_registered_folder(tmp_path, monkeypatch):
    w, _tracker_ = _widget_for(tmp_path, monkeypatch, layer_folder_name=MANUAL)
    record_registered_space(w.current_brain / "3_Registered_Atlas", "cropped",
                            w.current_brain / CROP)
    w.run_classification()
    assert _FakeClassification.made == []
    assert [m[1] for m in w.msgs.seen] == ["Loaded Images Are Not The Registered Ones"]


def test_classification_runs_on_layers_of_the_registered_folder(tmp_path, monkeypatch):
    w, _tracker_ = _widget_for(tmp_path, monkeypatch, layer_folder_name=CROP)
    record_registered_space(w.current_brain / "3_Registered_Atlas", "cropped",
                            w.current_brain / CROP)
    w.run_classification()
    assert len(_FakeClassification.made) == 1, w.msgs.seen
