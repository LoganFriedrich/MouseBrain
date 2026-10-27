"""The orientation check: BrainGlobe's "Check orientation", drawn as a picture.

Registration has to be told how the stack lies ("iar", ...). Told wrongly it
registers anyway, for over an hour, and every cell lands in the wrong region
while every number looks reasonable. BrainGlobe's own window has a button that
shows the stack turned into the atlas's orientation as declared, beside the
atlas; this check draws the same thing as a PNG so the pipeline writes it for
every brain before registering.

A stand-in atlas keeps these tests independent of any atlas download.
"""
import json

import numpy as np
import pytest
import tifffile

import mousebrain.orientation_check as oc


class FakeAtlas:
    """What the check uses of a BrainGlobeAtlas: space, reference, resolution."""

    def __init__(self, name, check_latest=False):
        from brainglobe_space import AnatomicalSpace
        self.resolution = (25.0, 25.0, 25.0)
        self.reference = np.random.default_rng(0).random((40, 24, 36)).astype(np.float32)
        self.space = AnatomicalSpace("asr", shape=self.reference.shape, resolution=self.resolution)


def stack(folder, shape=(12, 30, 20)):
    """A small asymmetric stack written as Z0000.tif ... (z, y, x)."""
    folder.mkdir(parents=True)
    vol = np.zeros(shape, np.uint16)
    vol[2:10, 5:25, 3:9] = 500                      # a block off to one side
    for i, plane in enumerate(vol):
        tifffile.imwrite(str(folder / ("Z%04d.tif" % i)), plane)
    return folder


@pytest.fixture
def atlas(monkeypatch):
    import brainglobe_atlasapi
    monkeypatch.setattr(brainglobe_atlasapi, "BrainGlobeAtlas", FakeAtlas)
    monkeypatch.setattr(oc, "check_atlas_name", lambda name: name)


def test_the_check_is_drawn_with_the_brains_real_size(tmp_path, atlas):
    folder = stack(tmp_path / "ch1")
    out = tmp_path / "check.png"
    s = oc.orientation_check(folder, "iar", (25.0, 25.0, 25.0), "fake_atlas_25um", out)
    assert out.is_file() and out.stat().st_size > 0
    assert s["declared_orientation"] == "iar" and s["atlas_orientation"] == "asr"
    # 'iar': slices run inferior->superior, rows anterior->posterior, columns right->left;
    # in the atlas's 'asr' that is (front-back, top-bottom, left-right) = (rows, slices, columns)
    assert s["brain_extent_mm"] == [0.75, 0.3, 0.5]


def test_the_stack_is_read_at_about_the_atlas_resolution(tmp_path):
    folder = stack(tmp_path / "ch0", shape=(30, 40, 40))
    vol, res = oc.read_stack_subsampled(folder, (4.0, 4.0, 4.0), target_um=25.0, threads=2)
    assert res == (24.0, 24.0, 24.0)                 # every 6th slice and pixel
    assert vol.shape == (5, 7, 7)


def _brain(tmp_path, orientation="iar"):
    p = tmp_path / "401_PROJ_01_02_1p625x_z4"
    stack(p / "1_Extracted_Full" / "ch1")
    stack(p / "1_Extracted_Full" / "ch0")
    (p / "1_Extracted_Full" / "metadata.json").write_text(json.dumps({
        "voxel_size_um": {"x": 25.0, "y": 25.0, "z": 25.0},
        "channels": {"signal_channel": 0, "background_channel": 1},
        "orientation": orientation}), encoding="utf-8")
    return p


def test_a_brain_is_checked_with_what_registration_uses(tmp_path, atlas):
    p = _brain(tmp_path)
    inputs = oc.brain_inputs(p)
    assert inputs["channel"] == 1 and inputs["orientation"] == "iar"   # background channel
    s = oc.check_brain(p, register=False)
    assert s["png"].endswith("3_Registered_Atlas\\QC_orientation_check.png") or \
        s["png"].endswith("3_Registered_Atlas/QC_orientation_check.png")


def test_a_trial_orientation_never_overwrites_the_declared_ones_picture(tmp_path, atlas):
    p = _brain(tmp_path)
    s = oc.check_brain(p, orientation="sar", register=False)
    assert s["png"].endswith("QC_orientation_check_sar.png")
    assert s["declared_orientation"] == "sar"
    assert not (p / "3_Registered_Atlas" / oc.CHECK_FILENAME).exists()


def test_the_lighter_sibling_atlas_is_used_when_downloaded(monkeypatch):
    import brainglobe_atlasapi.list_atlases as la
    monkeypatch.setattr(la, "get_downloaded_atlases", lambda: ["allen_mouse_25um"])
    assert oc.check_atlas_name("allen_mouse_10um") == "allen_mouse_25um"
    monkeypatch.setattr(la, "get_downloaded_atlases", lambda: [])
    assert oc.check_atlas_name("allen_mouse_10um") == "allen_mouse_10um"
