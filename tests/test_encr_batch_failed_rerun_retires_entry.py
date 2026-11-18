"""A re-run that fails because of the sample retires the sample's earlier entry;
a failure of the machinery around it does not.

Defect covered: a sample whose re-run ended in an error kept its earlier
registry entry -- with the earlier count, still marked current -- so the
registry reported a result whose newest attempt had failed on the sample.
Guarded the other way: a share or tracker failure, a missing or broken
library, or a defect in the code says nothing about the sample, so it must
never retire a good result -- and neither may a failure at a step that no
other sample of the run got past, nor ANYTHING raised while the file is being
read from disk: a file the reader does not recognize, or one whose end is
missing (cut off, or still being copied).
An entry whose files cannot be moved stays current and the summary says so.
"""

import sys

import numpy as np
import pytest

from example_method import write_example_approved_method
from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"
OTHER = "PRJ_01_02_S4_RA"   # a second sample of the same run


@pytest.fixture
def env(tmp_path, monkeypatch):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    # The batch has no built-in values: it needs an approved method to run.
    write_example_approved_method(tmp_path / "registry")
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    folder.mkdir(parents=True)
    (folder / f"{SAMPLE}.nd2").write_bytes(b"")
    img = np.zeros((32, 32), dtype=np.float32)
    monkeypatch.setattr(io_mod, "extract_channels",
                        lambda data, red_idx=0, green_idx=1, **k: (img, img))
    return t, tmp_path, img


def _earlier_entry(tmp_path, with_file=True):
    reg = AnalysisRegistry(analysis_name="PRJ_Detection",
                           registry_root=tmp_path / "registry")
    files = {}
    if with_file:
        csv_file = tmp_path / "measurements.csv"
        csv_file.write_text("label,is_positive\n1,True\n", encoding="utf-8")
        files = {"measurements": str(csv_file)}
    paths = reg.register_output(
        sample=SAMPLE, category="detection", files=files,
        results={"n_nuclei": 28, "n_positive": 15, "positive_fraction": 0.536},
        method_params={"sigma_threshold": 0}, source_files={})
    return reg, paths.get("measurements")


def _other_sample_succeeds(env, monkeypatch):
    """Add OTHER to the run and make its processing succeed.

    The batch retires a failed sample's entry only when another sample of
    the same run got past the step that failed; a completed sample got past
    every step.
    """
    t, tmp_path, _ = env
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    (folder / f"{OTHER}.nd2").write_bytes(b"")
    real = be.process_single

    def process(nd2_path, *a, **k):
        if nd2_path.stem == OTHER:
            return {"file": nd2_path.name, "path": str(nd2_path),
                    "status": "completed", "n_nuclei": 3, "n_positive": 1,
                    "positive_fraction": 0.3333, "background": 1.0}
        return real(nd2_path, *a, **k)
    monkeypatch.setattr(be, "process_single", process)


def _loads(monkeypatch, img):
    monkeypatch.setattr(io_mod, "load_image",
                        lambda path, **k: (np.stack([img, img]), {"voxel_size_um": {"x": 1.0}}))


def _detects_one_nucleus(monkeypatch):
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[10:14, 10:14] = 1
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        lambda *a, **k: (labels, {"filtered_count": 1}))


def _detects_no_nuclei_in(monkeypatch, img, sample):
    """Make `sample` load and end with no nuclei; other samples run as patched."""
    real = det_mod.detect_with_log_augmentation
    state = {"current": None}
    real_load = io_mod.load_image

    def load(path, **k):
        state["current"] = path.stem
        return real_load(path, **k)

    def detect(*a, **k):
        if state["current"] == sample:
            return np.zeros((32, 32), dtype=np.int32), {"filtered_count": 0}
        return real(*a, **k)
    monkeypatch.setattr(io_mod, "load_image", load)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", detect)


def _files_cannot_be_moved(monkeypatch):
    """As a file held open by another program: every archive move fails."""
    from mousebrain import analysis_registry as ar
    monkeypatch.setattr(ar, "_rename_only",
                        _raise(PermissionError("the file is in use")))


def _assert_kept(reg, old_file):
    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["results"]["n_nuclei"] == 28
    assert old_file.exists()
    assert not list(old_file.parent.glob("_archived/*/" + old_file.name))


def _run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out"),
                                      "--ignore-tracker"])
    be.main()


def _raise(exc):
    def f(*a, **k):
        raise exc
    return f


def test_an_image_with_the_wrong_channels_retires_the_earlier_entry(env, monkeypatch, capsys):
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    _loads(monkeypatch, img)
    # What the real channel extraction raises for an image with one channel.
    monkeypatch.setattr(io_mod, "extract_channels",
                        _raise(ValueError("red_idx 1 out of range for 1 channels")))
    _run(monkeypatch, tmp_path)

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is False
    assert not old_file.exists()
    assert len(list(old_file.parent.glob("_archived/*/" + old_file.name))) == 1
    out = capsys.readouterr().out
    assert f"{SAMPLE}: earlier registry entry retired" in out


def test_a_detector_failure_retires_the_earlier_entry(env, monkeypatch):
    t, tmp_path, img = env
    reg, _ = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    _loads(monkeypatch, img)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        _raise(ValueError("image must be 2D")))
    _run(monkeypatch, tmp_path)
    assert reg.get_entry(SAMPLE)["is_current"] is False


def test_a_share_that_cannot_be_reached_keeps_the_earlier_entry(env, monkeypatch, capsys):
    t, tmp_path, _ = env
    reg, old_file = _earlier_entry(tmp_path)
    monkeypatch.setattr(io_mod, "load_image",
                        _raise(OSError("the network path was not found")))
    _run(monkeypatch, tmp_path)

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["results"]["n_nuclei"] == 28
    assert old_file.exists()
    assert "kept current" in capsys.readouterr().out


def test_a_tracker_that_cannot_be_written_keeps_the_earlier_entry(env, monkeypatch):
    t, tmp_path, img = env
    reg, _ = _earlier_entry(tmp_path)
    monkeypatch.setattr(io_mod, "load_image",
                        lambda path, **k: (np.stack([img, img]), {"voxel_size_um": {"x": 1.0}}))
    monkeypatch.setattr(t, "log_detection",
                        _raise(PermissionError("file is open in another program")))
    _run(monkeypatch, tmp_path)
    assert reg.get_entry(SAMPLE)["is_current"] is True


def test_no_nuclei_summary_says_archived_only_when_files_were_moved(env, monkeypatch, capsys):
    t, tmp_path, img = env
    _earlier_entry(tmp_path, with_file=False)
    monkeypatch.setattr(io_mod, "load_image",
                        lambda path, **k: (np.stack([img, img]), {"voxel_size_um": {"x": 1.0}}))
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        lambda *a, **k: (np.zeros((32, 32), dtype=np.int32),
                                         {"filtered_count": 0}))
    _run(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "replaced by this run's 0 nuclei; no files to archive" in out
    assert "moved to _archived" not in out


def test_a_missing_reader_library_keeps_the_earlier_entry(env, monkeypatch, capsys):
    t, tmp_path, _ = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    monkeypatch.setattr(io_mod, "load_image",
                        _raise(ModuleNotFoundError("No module named 'nd2'")))
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)
    out = capsys.readouterr().out
    assert "infrastructure failure while read" in out
    assert "earlier registry entry retired" not in out


def test_an_import_error_while_measuring_keeps_the_earlier_entry(env, monkeypatch):
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    _loads(monkeypatch, img)
    _detects_one_nucleus(monkeypatch)
    from mousebrain.plugin_2d.sliceatlas.core import colocalization as coloc_mod
    monkeypatch.setattr(coloc_mod, "ColocalizationAnalyzer",
                        _raise(ImportError("No module named 'sklearn'")))
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)


@pytest.mark.parametrize("exc", [TypeError("unexpected keyword argument"),
                                 AttributeError("module has no attribute"),
                                 RuntimeError("library internals failed"),
                                 EOFError("read cut off")])
def test_a_code_or_library_error_while_detecting_keeps_the_earlier_entry(env, monkeypatch, exc):
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    _loads(monkeypatch, img)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", _raise(exc))
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)


def test_a_run_where_no_sample_got_past_the_step_retires_nothing(env, monkeypatch, capsys):
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    # A data error on the only sample, but nothing in the run succeeded.
    _loads(monkeypatch, img)
    monkeypatch.setattr(io_mod, "extract_channels",
                        _raise(ValueError("unusable image")))
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)
    out = capsys.readouterr().out
    assert (f"{SAMPLE}: earlier registry entry kept current -- no other sample "
            "of this run got past the 'channels' step") in out
    assert "earlier registry entry retired" not in out


def test_a_file_the_reader_does_not_recognize_keeps_the_earlier_entry(env, monkeypatch, capsys):
    """Through the REAL reader: it reports such a file with an OSError, which
    cannot be told from a share dropping out, so the entry stays current."""
    pytest.importorskip("nd2")
    t, tmp_path, _ = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    (folder / f"{SAMPLE}.nd2").write_bytes(bytes(range(256)) * 64)
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)
    out = capsys.readouterr().out
    assert "infrastructure failure while read" in out
    assert (f"{SAMPLE}: earlier registry entry kept current -- the failure was "
            "not shown to be the sample's") in out


def _cut_off_nd2():
    """The start of an ND2 file with everything after it missing.

    A correct first block (magic number, signature, version), then zeros
    where the image data and the table of contents at the END of the file
    would be -- what a copy that stopped part-way, or one still under way
    into a pre-sized file, leaves behind.
    """
    import struct
    name = b"ND2 FILE SIGNATURE CHUNK NAME01!"
    version = b"Ver3.0".ljust(64, b"\x00")
    head = struct.pack("<IIQ", 0x0ABECEDA, len(name), len(version)) + name + version
    return head + b"\x00" * 8192


def test_a_cut_off_file_keeps_the_earlier_entry(env, monkeypatch, capsys):
    """Through the REAL reader: it reports a file whose end is missing with a
    ValueError -- one of the data-error types. Raised while READING, it must
    not retire the entry, even though another sample of the run completed."""
    nd2 = pytest.importorskip("nd2")
    t, tmp_path, _ = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    cut_off = folder / f"{SAMPLE}.nd2"
    cut_off.write_bytes(_cut_off_nd2())
    # The premise: the reader takes the start for an ND2 and then fails with
    # a ValueError (not the OSError of an unrecognized file).
    with pytest.raises(ValueError):
        with nd2.ND2File(cut_off) as f:
            f.asarray()

    _run(monkeypatch, tmp_path)

    _assert_kept(reg, old_file)
    out = capsys.readouterr().out
    assert "infrastructure failure while read" in out
    assert (f"{SAMPLE}: earlier registry entry kept current -- the failure was "
            "not shown to be the sample's") in out
    assert "earlier registry entry retired" not in out


@pytest.mark.parametrize("exc", [ValueError("Invalid ChunkMap signature"),
                                 KeyError("missing chunk"),
                                 IndexError("offset past the end of the file")])
def test_no_error_raised_while_reading_the_file_retires_the_entry(env, monkeypatch, exc):
    """Whatever reader is installed, and whichever error it raises."""
    t, tmp_path, _ = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    monkeypatch.setattr(io_mod, "load_image", _raise(exc))
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)


def test_a_no_nuclei_sample_does_not_vouch_for_the_measuring_step(env, monkeypatch, capsys):
    """The run's only other sample ended with no nuclei, so nothing in the run
    got past measuring: a data error there may be the software's."""
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    (folder / f"{OTHER}.nd2").write_bytes(b"")
    _loads(monkeypatch, img)
    _detects_one_nucleus(monkeypatch)
    _detects_no_nuclei_in(monkeypatch, img, OTHER)
    from mousebrain.plugin_2d.sliceatlas.core import colocalization as coloc_mod
    monkeypatch.setattr(coloc_mod, "ColocalizationAnalyzer",
                        _raise(ValueError("invalid parameter")))
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)
    out = capsys.readouterr().out
    assert "No nuclei: 1" in out
    assert "sample failure while colocalize" in out
    assert (f"{SAMPLE}: earlier registry entry kept current -- no other sample "
            "of this run got past the 'colocalize' step") in out


def test_a_no_nuclei_sample_vouches_for_the_detecting_step(env, monkeypatch):
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    (folder / f"{OTHER}.nd2").write_bytes(b"")
    _loads(monkeypatch, img)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        _raise(ValueError("image must be 2D")))
    _detects_no_nuclei_in(monkeypatch, img, OTHER)
    _run(monkeypatch, tmp_path)
    assert reg.get_entry(SAMPLE)["is_current"] is False


def test_an_entry_whose_files_cannot_be_moved_is_reported_still_current(env, monkeypatch, capsys):
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    _loads(monkeypatch, img)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        _raise(ValueError("image must be 2D")))
    _files_cannot_be_moved(monkeypatch)
    _run(monkeypatch, tmp_path)
    _assert_kept(reg, old_file)
    out = capsys.readouterr().out
    assert (f"[!] {SAMPLE}: the sample's own re-run failed, but its earlier "
            "registry entry could not be retired and is STILL CURRENT") in out
    assert "earlier registry entry retired" not in out


def test_a_library_that_does_not_import_stops_the_batch_first(env, monkeypatch, capsys):
    t, tmp_path, _ = env
    reg, old_file = _earlier_entry(tmp_path)
    monkeypatch.setattr(be, "REQUIRED_MODULES",
                        be.REQUIRED_MODULES + ("a_library_that_is_not_installed",))
    attempted = []
    monkeypatch.setattr(be, "process_single",
                        lambda *a, **k: attempted.append(a) or {"status": "error"})
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, tmp_path)
    assert stop.value.code == 1
    assert attempted == []
    _assert_kept(reg, old_file)
    assert "a_library_that_is_not_installed" in capsys.readouterr().out


def _retired_records(reg, sample):
    """The invalidate_entry records the registry's audit log holds for sample."""
    import json
    log = (reg.logs_dir / "PRJ_Detection.log").read_text(encoding="utf-8")
    records = []
    for line in log.splitlines():
        fields = line.split("\t")
        if fields[2] == "invalidate_entry" and fields[3] == sample:
            records.append(json.loads(fields[5]))
    return records


def test_the_retired_counts_survive_in_the_log_after_a_re_registration(env, monkeypatch):
    # The re-run that follows a retirement OVERWRITES the entry's record in
    # registry.json; the earlier counts and parameters must still be readable
    # from the registry's audit log.
    t, tmp_path, img = env
    reg, _ = _earlier_entry(tmp_path)
    earlier_hash = reg.get_entry(SAMPLE)["method_hash"]
    _other_sample_succeeds(env, monkeypatch)
    _loads(monkeypatch, img)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        _raise(ValueError("image must be 2D")))
    _run(monkeypatch, tmp_path)
    assert reg.get_entry(SAMPLE)["is_current"] is False

    reg.register_output(
        sample=SAMPLE, category="detection", files={},
        results={"n_nuclei": 30, "n_positive": 12, "positive_fraction": 0.4},
        method_params={"sigma_threshold": 1}, source_files={})
    assert reg.get_entry(SAMPLE)["results"]["n_nuclei"] == 30

    records = _retired_records(reg, SAMPLE)
    assert len(records) == 1
    assert records[0]["results"] == {"n_nuclei": 28, "n_positive": 15,
                                     "positive_fraction": 0.536}
    assert records[0]["method_params"] == {"sigma_threshold": 0}
    assert records[0]["method_hash"] == earlier_hash
    assert records[0]["archived_to"].startswith("_archived/")
