"""Imaris header values are character arrays, and must be read as whole values.

An .ims file does not store the image width as a number. It stores it as an
array of single characters: 2924 is [b'2', b'9', b'2', b'4']. Reading element
[0] of that gets the first DIGIT, succeeds, and raises nothing -- so a brain
2924 voxels wide was recorded in its own metadata.json as 2 voxels wide, and a
brain 8179 voxels tall as 8.

That is wrong provenance for every brain ever extracted, and it is not inert:
the crop-optimising utility reads dimensions.y as the brain's real height.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "pipeline" / "2_extract_and_analyze.py")


@pytest.fixture(scope="module")
def extract():
    pytest.importorskip("h5py")
    spec = importlib.util.spec_from_file_location("extract_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ims(tmp_path):
    """A minimal .ims holding only the header attributes, stored as Imaris does."""
    h5py = pytest.importorskip("h5py")
    import numpy as np

    path = tmp_path / "101_PROJ_01_02_2.5x_z5.ims"
    with h5py.File(path, "w") as f:
        image = f.create_group("DataSetInfo/Image")
        for key, text in (("X", "2924"), ("Y", "8179"), ("Z", "1301")):
            image.attrs[key] = np.array([c.encode() for c in text], dtype="|S1")
        ch0 = f.create_group("DataSetInfo/Channel 0")
        ch0.attrs["Name"] = np.array([c.encode() for c in "488"], dtype="|S1")
    return path


def test_dimensions_are_the_whole_number_not_its_first_digit(extract, ims):
    info = extract.get_ims_info(ims)
    assert info["size_x"] == 2924
    assert info["size_y"] == 8179
    assert info["size_z"] == 1301


def test_a_channel_name_is_not_truncated_to_one_character(extract, ims):
    info = extract.get_ims_info(ims)
    assert info["channels"][0]["name"] == "488"


def test_a_missing_attribute_is_absent_rather_than_guessed(extract, tmp_path):
    """No header is a real case (a converted or trimmed file), and must not lie."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bare.ims"
    with h5py.File(path, "w") as f:
        f.create_group("DataSetInfo/Image")
    info = extract.get_ims_info(path)
    assert info["size_x"] is None


def test_reading_an_attribute_that_is_not_there_returns_nothing(extract):
    assert extract.read_ims_attribute({}, "X") is None


def test_a_plain_string_attribute_still_reads(extract):
    """Not every writer uses character arrays, so both forms have to work."""
    assert extract.read_ims_attribute({"X": b"2924"}, "X") == "2924"
    assert extract.read_ims_attribute({"X": "2924"}, "X") == "2924"
