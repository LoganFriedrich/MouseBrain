"""DeepSlice predictions are reused only for the images they were made from.

Before this, the folder's prediction cache was returned whenever it existed.
An image added to the folder afterwards got no atlas position, and an image
re-exported under the same name kept the position predicted for its old
pixels. These tests drive the real predict_folder with a stand-in for the
DeepSlice network (which is not needed to show what the cache does).
"""
import json
import os
from pathlib import Path

from mousebrain.plugin_2d.sliceatlas.core.deepslice_wrapper import DeepSliceWrapper


class _FakeModel:
    """Stands in for DeepSlice's DSModel: one prediction per image, numbered by call."""

    def __init__(self):
        self.calls = 0
        self._last_path = None

    def predict(self, path, ensemble=True, section_numbers=True):
        self.calls += 1
        self._last_path = Path(path)

    def save_predictions(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = sorted(f.name for f in self._last_path.iterdir()
                       if f.is_file() and f.suffix.lower() == '.png')
        slices = [{'filename': n, 'oz': 1000.0 * self.calls} for n in names]
        (out / 'predictions.json').write_text(json.dumps({'slices': slices}))


def _wrapper():
    w = DeepSliceWrapper.__new__(DeepSliceWrapper)   # skip the DeepSlice import
    w.model = _FakeModel()
    w.species = 'mouse'
    w._predictions = None
    w._predictions_by_filename = {}
    return w


def _png(path, payload=b'\x89PNG fake'):
    path.write_bytes(payload)
    return path


def test_an_unchanged_folder_is_answered_from_the_cache(tmp_path):
    _png(tmp_path / 'SUBJ_A_S1.png')
    w = _wrapper()
    first = w.predict_folder(tmp_path, use_cache=True)
    second = w.predict_folder(tmp_path, use_cache=True)
    assert w.model.calls == 1
    assert [s['filename'] for s in second['slices']] == ['SUBJ_A_S1.png']
    assert second['slices'][0]['oz'] == first['slices'][0]['oz']


def test_an_added_image_is_predicted_not_answered_from_the_old_cache(tmp_path):
    _png(tmp_path / 'SUBJ_A_S1.png')
    w = _wrapper()
    w.predict_folder(tmp_path, use_cache=True)

    _png(tmp_path / 'SUBJ_A_S2.png')
    again = w.predict_folder(tmp_path, use_cache=True)

    assert w.model.calls == 2
    assert sorted(s['filename'] for s in again['slices']) == ['SUBJ_A_S1.png', 'SUBJ_A_S2.png']
    assert w.get_position_for_slice('SUBJ_A_S2.png') is not None


def test_a_reexported_image_is_predicted_again(tmp_path):
    img = _png(tmp_path / 'SUBJ_A_S1.png')
    w = _wrapper()
    w.predict_folder(tmp_path, use_cache=True)

    _png(img, b'\x89PNG re-exported with different pixels')
    st = img.stat()
    os.utime(img, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    again = w.predict_folder(tmp_path, use_cache=True)

    assert w.model.calls == 2
    assert again['slices'][0]['oz'] == 2000.0     # the second prediction, not the cached one


def test_a_removed_image_is_not_kept_from_the_old_cache(tmp_path):
    _png(tmp_path / 'SUBJ_A_S1.png')
    gone = _png(tmp_path / 'SUBJ_A_S2.png')
    w = _wrapper()
    w.predict_folder(tmp_path, use_cache=True)

    gone.unlink()
    again = w.predict_folder(tmp_path, use_cache=True)

    assert [s['filename'] for s in again['slices']] == ['SUBJ_A_S1.png']


def test_a_cache_without_a_fingerprint_is_not_trusted(tmp_path):
    """A cache written before fingerprints were stored cannot say what it describes."""
    _png(tmp_path / 'SUBJ_A_S1.png')
    (tmp_path / DeepSliceWrapper.CACHE_FILENAME).write_text(json.dumps(
        {'species': 'mouse', 'n_slices': 1,
         'slices': [{'filename': 'SUBJ_A_OLD.png', 'oz': 5.0}]}))
    w = _wrapper()
    result = w.predict_folder(tmp_path, use_cache=True)
    assert w.model.calls == 1
    assert [s['filename'] for s in result['slices']] == ['SUBJ_A_S1.png']
