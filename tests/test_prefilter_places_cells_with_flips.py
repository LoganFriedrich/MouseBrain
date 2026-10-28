"""The prefilter looks each candidate up where registration put that spot.

brainreg turns the brain into the atlas's orientation, so an axis whose letter
differs from the atlas's ('i' vs 's', 'p' vs 'a') is reversed as well as
reordered. The prefilter used to reorder only, and looked candidates up
millimetres away (up-down on 'iar' brains, front-back too on 'ipr').
"""
from types import SimpleNamespace

import numpy as np
import tifffile

from mousebrain.prefilter import _raw_to_downsampled


def _brain(tmp_path, shape):
    """A pipeline folder whose full extraction is a tiny stack of the given shape,
    recorded as the registered images."""
    from mousebrain.config import record_registered_space
    brain = tmp_path / "1_CNT_01_01_1p625x_z4"
    ch0 = brain / "1_Extracted_Full" / "ch0"
    ch0.mkdir(parents=True)
    for z in range(shape[0]):
        tifffile.imwrite(str(ch0 / f"Z{z:04d}.tif"), np.zeros(shape[1:], np.uint16))
    reg = brain / "3_Registered_Atlas"
    record_registered_space(reg, "full", brain / "1_Extracted_Full")
    return reg


ATLAS = SimpleNamespace(space=SimpleNamespace(origin="asr"), resolution=(10.0, 10.0, 10.0))


def test_ipr_brain_is_reversed_on_two_axes(tmp_path):
    # raw (z, y, x) = (i, p, r); atlas (asr) = (a->p, s->i, r->l)
    reg = _brain(tmp_path, (4, 6, 8))
    corner = _raw_to_downsampled([(0, 0, 0)], reg, "ipr", [10, 10, 10], (6, 4, 8), ATLAS)[0]
    # raw index 0 on the 'p' axis is the most posterior -> far end of atlas axis 0
    assert corner[0] > 4.5
    # raw index 0 on the 'i' axis is the most inferior -> far end of atlas axis 1
    assert corner[1] > 2.5
    # 'r' matches the atlas: not reversed
    assert corner[2] < 1


def test_iar_brain_is_reversed_up_down_only(tmp_path):
    reg = _brain(tmp_path, (4, 6, 8))
    corner = _raw_to_downsampled([(0, 0, 0)], reg, "iar", [10, 10, 10], (6, 4, 8), ATLAS)[0]
    assert corner[0] < 1
    assert corner[1] > 2.5
    assert corner[2] < 1
