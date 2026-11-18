"""The 2D batch never measures an image at a pixel size the file does not give.

Defect covered: an image whose metadata held no pixel size was processed at
1.0 um/px without a word. The nucleus diameter range is turned into pixel
areas with that number, so the batch counted objects of a size nobody chose
and recorded those areas in the tracker as if measured. The ND2 reader also
put 1.0 in itself when the file had no size, so the batch could not tell.

Now the reader says whether the size came from the file
(voxel_size_from_file), and the batch refuses such an image: the sample
fails with the reason, the detector is not run, no tracker row is written,
and -- because a reader that stopped finding the size says nothing about the
sample -- its earlier registry entry is kept.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

SAMPLE = "PRJ_01_02_S3_RA"

# Every value process_single needs. Test data only.
VALUES = dict(threshold=0, sigma_threshold=0, dilation=50, bg_percentile=10,
              soma_dilation=6, threshold_fraction=0.2, log_threshold=0.005,
              min_diameter_um=8.0, max_diameter_um=25.0)


@pytest.fixture
def env(tmp_path, monkeypatch):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    img = np.zeros((32, 32), dtype=np.float32)
    monkeypatch.setattr(io_mod, "extract_channels",
                        lambda data, red_idx=0, green_idx=1, **k: (img, img))
    detected = []

    def detect(*a, **k):
        detected.append(k.get("pixel_um"))
        return np.zeros((32, 32), dtype=np.int32), {"filtered_count": 0}

    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", detect)
    nd2 = tmp_path / f"{SAMPLE}.nd2"
    nd2.write_bytes(b"")
    return t, nd2, img, detected


@pytest.mark.parametrize("metadata", [
    {},                                                          # no size at all
    {"voxel_size_um": {"x": 1.0, "y": 1.0}, "voxel_size_from_file": False},  # reader's 1.0
    {"voxel_size_um": {"x": 0.0}},                               # not a size
    {"voxel_size_um": {"x": None}},
])
def test_an_image_without_a_pixel_size_is_not_measured(env, monkeypatch, tmp_path, metadata):
    t, nd2, img, detected = env
    monkeypatch.setattr(io_mod, "load_image",
                        lambda path, **k: (np.stack([img, img]), dict(metadata)))
    result = be.process_single(nd2, tmp_path / "out", **VALUES)
    assert result["status"] == "error"
    assert "pixel size" in result["error"]
    assert detected == []                       # the detector never ran
    assert t.search(run_type="detection") == []  # no row with made-up areas
    # Not the sample's fault in a way that should retire a good earlier entry.
    assert result["failure"] == "infrastructure"


def test_an_image_with_its_pixel_size_is_measured_at_it(env, monkeypatch, tmp_path):
    t, nd2, img, detected = env
    monkeypatch.setattr(io_mod, "load_image", lambda path, **k: (
        np.stack([img, img]),
        {"voxel_size_um": {"x": 0.65, "y": 0.65}, "voxel_size_from_file": True}))
    result = be.process_single(nd2, tmp_path / "out", **VALUES)
    assert result["status"] == "no_nuclei"
    assert detected == [0.65]


def _volume_meta(calibrated, calibration):
    """nd2 metadata whose first channel's volume says *calibrated*."""
    volume = SimpleNamespace(axesCalibrated=(calibrated, calibrated, False),
                             axesCalibration=calibration)
    return SimpleNamespace(channels=[SimpleNamespace(volume=volume)])


class _FakeND2File:
    """Stands in for nd2.ND2File: a 2-channel image whose voxel size and
    calibration metadata are given. *legacy_calibration* makes it a legacy
    file whose first frame records that raw "Calibration" value."""

    def __init__(self, voxel, metadata=None, legacy_calibration=None,
                 legacy=False):
        self._voxel = voxel
        self.sizes = {"C": 2, "Y": 8, "X": 8}
        self.attributes = SimpleNamespace()
        self.metadata = metadata
        self.is_legacy = legacy
        frame0 = {} if legacy_calibration is None else {
            "Calibration": legacy_calibration}
        self._rdr = SimpleNamespace(_frame0_meta=lambda: frame0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def asarray(self):
        return np.zeros((2, 8, 8), dtype=np.uint16)

    def voxel_size(self):
        return self._voxel


def _vs(x):
    """What nd2.ND2File.voxel_size() returns: a named tuple, never empty."""
    return SimpleNamespace(x=x, y=x, z=1.0)


# Each case is what the installed nd2 library really hands back. It never
# reports a missing calibration as empty: an uncalibrated file reads 1.0.
ND2_CASES = [
    # modern file, calibration flag off: the library still returns 1.0
    pytest.param(dict(voxel=_vs(1.0), metadata=_volume_meta(False, (1.0, 1.0, 1.0))),
                 False, id="modern-uncalibrated"),
    # no metadata or no channels: the library's (1, 1, 1) fallback
    pytest.param(dict(voxel=_vs(1), metadata=None), False, id="no-metadata"),
    pytest.param(dict(voxel=_vs(1), metadata=SimpleNamespace(channels=[])),
                 False, id="no-channels"),
    # legacy file with no (or a non-positive) raw Calibration: `or 1` / 1.0
    pytest.param(dict(voxel=_vs(1), metadata=_volume_meta(True, (1.0, 1.0, 1.0)),
                      legacy=True), False, id="legacy-no-calibration"),
    pytest.param(dict(voxel=_vs(1.0), metadata=_volume_meta(True, (1.0, 1.0, 1.0)),
                      legacy=True, legacy_calibration=-1), False,
                 id="legacy-negative-calibration"),
    pytest.param(dict(voxel=_vs(0.65), metadata=_volume_meta(False, (0.65, 0.65, 1.0)),
                      legacy=True, legacy_calibration=0.65), False,
                 id="legacy-calibrated-false"),
    # zeros
    pytest.param(dict(voxel=SimpleNamespace(x=0, y=0, z=0),
                      metadata=_volume_meta(True, (0, 0, 0))), False, id="zeros"),
    # really calibrated, modern and legacy
    pytest.param(dict(voxel=_vs(0.65), metadata=_volume_meta(True, (0.65, 0.65, 1.0))),
                 True, id="modern-calibrated"),
    pytest.param(dict(voxel=_vs(0.65), metadata=_volume_meta(True, (0.65, 0.65, 1.0)),
                      legacy=True, legacy_calibration=0.65), True,
                 id="legacy-calibrated"),
]


@pytest.mark.parametrize("nd2_file, from_file", ND2_CASES)
def test_the_nd2_reader_says_whether_the_size_came_from_the_file(
        tmp_path, monkeypatch, nd2_file, from_file):
    fake = SimpleNamespace(ND2File=lambda path: _FakeND2File(**nd2_file))
    monkeypatch.setattr(io_mod, "_nd2", fake)
    f = tmp_path / "image.nd2"
    f.write_bytes(b"")
    _data, metadata = io_mod.load_image(f)
    assert metadata.get("voxel_size_from_file") is from_file
    # Display code still gets a size to work with.
    assert metadata["voxel_size_um"]["x"] > 0
    if from_file:
        assert metadata["voxel_size_um"]["x"] == 0.65


@pytest.mark.parametrize("nd2_file, from_file", ND2_CASES)
def test_the_batch_measures_an_nd2_only_at_a_size_the_file_records(
        env, monkeypatch, tmp_path, nd2_file, from_file):
    """End to end through the real image reader: an uncalibrated ND2 (which
    the nd2 library reports as 1.0 um/px) is refused, not measured at 1.0."""
    t, nd2, img, detected = env
    fake = SimpleNamespace(ND2File=lambda path: _FakeND2File(**nd2_file))
    monkeypatch.setattr(io_mod, "_nd2", fake)
    # The batch's image reader is the real one (env only stubs channel split).
    result = be.process_single(nd2, tmp_path / "out", **VALUES)
    if from_file:
        assert result["status"] == "no_nuclei"
        assert detected == [0.65]
    else:
        assert result["status"] == "error"
        assert "pixel size" in result["error"]
        assert detected == []
        assert t.search(run_type="detection") == []
