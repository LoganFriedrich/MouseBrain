"""A ch*.zarr cache is used only when it is complete and current.

The widget treated "the cache folder exists" as "converted": an interrupted
or failed conversion (an empty folder, or a full-size array with planes never
written) was loaded as the brain, and a re-crop left the previous crop's
cache in use.
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from mousebrain.zarr_cache import build, cache_state, record_path, write_array, zarr_path_for

TW = (Path(__file__).resolve().parents[1] / "src" / "mousebrain" / "plugin_3d"
      / "tuning_widget.py")


def _tiffs(folder, n=5, value=1):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        tifffile.imwrite(str(folder / ("Z%04d.tif" % i)), np.full((4, 6), value + i, np.uint16))
    return folder


def test_a_finished_build_is_usable_and_holds_every_plane(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")
    z = build(ch0)
    assert cache_state(ch0) == (True, "complete")
    assert record_path(z).exists()
    assert zarr.open_array(str(z), mode="r")[4, 0, 0] == 5


def test_an_interrupted_build_leaves_nothing_usable(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return tifffile.imread(path)
    with pytest.raises(KeyboardInterrupt):
        build(ch0, read_plane=flaky)
    assert not zarr_path_for(ch0).exists()
    assert cache_state(ch0)[0] is False


def test_an_empty_or_unrecorded_cache_folder_is_not_used(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")
    zarr_path_for(ch0).mkdir()
    ok, why = cache_state(ch0)
    assert not ok and "completion record" in why


def test_a_cache_older_than_rewritten_tiffs_is_not_used(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")
    build(ch0)
    later = time.time() + 60
    os.utime(ch0 / "Z0002.tif", (later, later))
    assert cache_state(ch0)[0] is False


def test_a_cache_with_a_different_plane_count_is_not_used(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")
    build(ch0)
    (ch0 / "Z0004.tif").unlink()
    assert cache_state(ch0)[0] is False


def test_rebuilding_replaces_an_incomplete_cache(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")
    zarr_path_for(ch0).mkdir()
    build(ch0)
    assert cache_state(ch0)[0]


def test_a_failed_whole_volume_save_leaves_no_cache(tmp_path):
    ch0 = _tiffs(tmp_path / "ch0")

    def broken_save(path, volume):
        os.makedirs(path)  # what a failed save leaves: an empty folder
        raise MemoryError
    with pytest.raises(MemoryError):
        write_array(np.zeros((5, 4, 6), np.uint16), ch0, save=broken_save)
    assert cache_state(ch0)[0] is False
    write_array(np.zeros((5, 4, 6), np.uint16), ch0)
    assert cache_state(ch0)[0]


def test_the_widget_loads_and_skips_conversion_only_for_a_complete_cache():
    text = TW.read_text(encoding="utf-8")
    assert "sig_ok, sig_why = cache_state(" in text
    assert "if zarr_ch0.exists() and zarr_ch1.exists():" not in text
