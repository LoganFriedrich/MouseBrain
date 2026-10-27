"""The registration QC picture: what a person looks at before approving.

Its rows are coronal slices, front to back -- brainreg writes the brain in the
atlas's orientation, and every BrainGlobe atlas runs front to back along its
first axis. The rows are spread over the stretch the atlas covers: a brain kept
with more brainstem than the atlas has used to get its last row drawn past the
atlas's end, where an empty atlas beside real tissue reads as a failed
registration.
"""
import numpy as np

from mousebrain.config import import_util

qc = import_util("util_registration_qc")


def test_rows_stay_where_the_atlas_is():
    brain = np.ones((127, 4, 4), np.uint16)
    atlas = np.zeros(brain.shape, np.uint32)
    atlas[:115] = 315                       # the atlas ends 12 slices before the tissue does
    levels = qc.choose_levels(brain, atlas, 5)
    assert len(levels) == 5
    assert levels.min() >= 0 and levels.max() <= 114
    assert levels.max() < 0.9 * 115 + 1     # not on the atlas's last slices either


def test_rows_start_where_the_atlas_starts():
    brain = np.ones((100, 4, 4), np.uint16)
    atlas = np.zeros(brain.shape, np.uint32)
    atlas[40:90] = 698
    levels = qc.choose_levels(brain, atlas, 3)
    assert list(levels) == [44, 64, 84]     # 10% to 90% of slices 40-89


def test_without_an_atlas_the_rows_span_the_image():
    brain = np.ones((100, 4, 4), np.uint16)
    assert list(qc.choose_levels(brain, None, 3)) == [9, 49, 89]
