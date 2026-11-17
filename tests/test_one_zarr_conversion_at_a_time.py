"""Two Zarr conversions never build the same cache over each other.

All builds of a channel shared one ``ch0.zarr.partial`` folder and each
started by deleting it; the widget kept the folder and channel being converted
in its own fields and let a second conversion (another Load, or Convert during
the automatic one) start and overwrite them. A build could then rename a folder
with planes missing under it and record it as complete.
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from mousebrain.zarr_cache import build, cache_state, zarr_path_for


def _tiffs(folder, n=6):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        tifffile.imwrite(str(folder / ("Z%04d.tif" % i)), np.full((4, 6), 10 + i, np.uint16))
    return folder


def test_a_second_build_started_during_the_first_leaves_a_complete_cache(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")
    started_second = []

    def read_plane_first(path):
        # Half-way through the first build, a second build of the same
        # channel runs from start to finish.
        if path.endswith("Z0003.tif") and not started_second:
            started_second.append(True)
            build(ch0)
        return tifffile.imread(path)

    z = build(ch0, read_plane=read_plane_first)
    assert started_second
    assert cache_state(ch0) == (True, "complete")
    data = zarr.open_array(str(z), mode="r")[:]
    # every plane is the TIFF's, none blank
    assert [int(data[i, 0, 0]) for i in range(6)] == [10, 11, 12, 13, 14, 15]
    # and no partial folder was left behind by either build
    assert not [p for p in tmp_path.iterdir() if ".partial" in p.name]


def test_an_abandoned_partial_is_removed_only_when_no_build_writes_to_it(tmp_path):
    from mousebrain.zarr_cache import ABANDONED_AFTER_S, remove_abandoned_partials
    ch0 = _tiffs(tmp_path / "ch0")
    zp = zarr_path_for(ch0)
    old = zp.with_name(zp.name + ".partial-oldbuild")
    old.mkdir()
    (old / "0.0.0").write_bytes(b"x")
    live = zp.with_name(zp.name + ".partial-livebuild")
    live.mkdir()
    (live / "0.0.0").write_bytes(b"x")
    long_ago = time.time() - ABANDONED_AFTER_S - 60
    for p in (old, old / "0.0.0"):
        os.utime(p, (long_ago, long_ago))

    removed = remove_abandoned_partials(zp)
    assert removed == [old]
    assert live.exists()


# ------------------------------------------------------------- the widget
class _RunningWorker:
    def isRunning(self):
        return True


def test_convert_refuses_while_a_conversion_is_running(tmp_path, monkeypatch):
    import mousebrain.plugin_3d.tuning_widget as tw
    from tuning_widget_stand_in import _Messages, _Widget

    brain = tmp_path / "M" / "301_PROJ_03_01_2p5x_z5"
    _tiffs(brain / "1_Extracted_Full" / "ch0")
    msgs = _Messages()
    monkeypatch.setattr(tw, "QMessageBox", msgs)
    made = []
    monkeypatch.setattr(tw, "ZarrConverterWorker", lambda *a, **k: made.append(a) or pytest.fail(
        "a second conversion was started"))

    class _Btn:
        def setEnabled(self, v): pass
        def setText(self, t): pass

    class _Label(_Btn):
        def setStyleSheet(self, s): pass

    w = _Widget()
    w.current_brain = brain
    w.zarr_worker = _RunningWorker()
    w.brain_loader_worker = None
    w.convert_zarr_btn = w.load_brain_btn = _Btn()
    w.brain_status_label = _Label()
    w.convert_to_zarr()
    assert not made
    assert [m[1] for m in msgs.seen if m[0] == "warning"] == ["Conversion Running"]


def test_one_worker_converts_every_channel_of_its_own_folder(tmp_path):
    from mousebrain.plugin_3d.tuning_widget import ZarrConverterWorker
    data = tmp_path / "2_Cropped_For_Registration"
    _tiffs(data / "ch0")
    _tiffs(data / "ch1")
    worker = ZarrConverterWorker(data)
    results = []
    worker.finished.connect(lambda ok, msg: results.append(ok))
    worker.run()  # in this thread
    assert results == [True]
    assert cache_state(data / "ch0")[0] and cache_state(data / "ch1")[0]
