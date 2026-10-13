"""A corrected delivery arrives as floats; cell detection requires integers.

Most brains arrive as 16-bit integers. One that has had a flat-field or
vignetting correction applied in the imaging software arrives as float, because
that correction multiplies each count by a gain and multiplication produces
fractions. Cell detection refuses a floating-point image outright -- which is the
good case, since the alternative would be detecting on something it cannot
interpret.

Observed 2026-10-13: a re-delivered brain was extracted and cached over ninety
minutes, and detection then declined it in under a second with
"signal_array must be integer datatype, but has datatype float32".

The underlying quantity is a photon count, so rounding recovers the integer the
value represents. What must never happen is a silent rescale or a silent clip:
a rescale changes what every downstream intensity threshold means, and a clip
takes the top off the brightest voxels, which are the cells.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "pipeline" / "2_extract_and_analyze.py")


@pytest.fixture(scope="module")
def extract():
    pytest.importorskip("h5py")
    spec = importlib.util.spec_from_file_location("extract_for_dtype", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_an_integer_delivery_is_left_completely_alone(extract):
    """The normal case must not be touched, and must not claim it was."""
    volume = np.array([[[1, 2], [3, 4]]], dtype=np.uint16)
    out, conversion = extract.to_integer_counts(volume, "ch0")
    assert out.dtype == np.uint16
    assert conversion is None
    assert np.array_equal(out, volume)


def test_floats_are_rounded_to_the_counts_they_represent(extract):
    volume = np.array([[[110.4, 110.6], [0.0, 1648.5]]], dtype=np.float32)
    out, conversion = extract.to_integer_counts(volume, "ch0")
    assert np.issubdtype(out.dtype, np.integer)
    assert out[0, 0, 0] == 110
    assert out[0, 0, 1] == 111
    assert conversion["method"].startswith("rounded")


def test_nothing_is_rescaled(extract):
    """A rescale would change what every downstream threshold means."""
    volume = np.array([[[10.0, 100.0, 1000.0]]], dtype=np.float32)
    out, _ = extract.to_integer_counts(volume, "ch0")
    assert list(out.ravel()) == [10, 100, 1000]


def test_the_conversion_is_recorded_so_it_is_not_invisible(extract):
    volume = np.array([[[1.5, 19570.2]]], dtype=np.float32)
    _, conversion = extract.to_integer_counts(volume, "ch0")
    assert conversion["source_dtype"] == "float32"
    assert conversion["written_dtype"] == "uint16"
    assert conversion["observed_max"] == pytest.approx(19570.2, rel=1e-5)
    assert "photon count" in conversion["reason"]


def test_a_volume_too_bright_for_the_type_stops_rather_than_clipping(extract):
    """Clipping would take the top off exactly the voxels that are cells."""
    volume = np.array([[[0.0, 5e9]]], dtype=np.float64)
    with pytest.raises(ValueError, match="clip"):
        extract.to_integer_counts(volume, "ch0")


def test_a_wider_type_is_used_before_giving_up(extract):
    """Refusing is for data that fits nothing, not for data that is merely big."""
    volume = np.array([[[0.0, 70000.0]]], dtype=np.float32)
    out, conversion = extract.to_integer_counts(volume, "ch0")
    assert conversion["written_dtype"] == "uint32"
    assert out[0, 0, 1] == 70000


def test_negative_values_stop_rather_than_wrapping(extract):
    """Unsigned casting turns a negative into a very large positive, silently."""
    volume = np.array([[[-3.0, 10.0]]], dtype=np.float32)
    with pytest.raises(ValueError, match="negative"):
        extract.to_integer_counts(volume, "ch0")
