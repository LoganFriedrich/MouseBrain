"""Retiring a sample's entry archives EVERY file in its own registry folder.

Defect covered: when the batch retired an entry (a re-run that found no
nuclei, or a re-run that failed on the sample), only the files the entry
listed among its outputs were moved to _archived/. A sample folder can hold
more files of that earlier run than the entry lists (extra overlays,
histograms copied by an earlier version of the batch). They stayed in the
active folder, beside the new 0-nuclei entry or beside no current entry at
all, and anything mirroring the folder showed them as current results.
All or nothing still holds: if one of those files will not move, nothing
stays moved and the entry stays current.
"""

import sys

import numpy as np
import pytest

from example_method import write_example_approved_method
from mousebrain import analysis_registry as ar
from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"
OTHER = "PRJ_01_02_S4_RA"
STRAYS = ("PRJ_01_02_S3_RA_overlay.png", "PRJ_01_02_S3_RA_fold_change_histogram.png")


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
    monkeypatch.setattr(io_mod, "load_image",
                        lambda path, **k: (np.stack([img, img]), {"voxel_size_um": {"x": 1.0}}))
    monkeypatch.setattr(io_mod, "extract_channels",
                        lambda data, red_idx=0, green_idx=1, **k: (img, img))
    return tmp_path


def _earlier_entry_with_strays(tmp_path):
    """An earlier entry listing one file, with two more files of the same
    run in its sample folder that the entry does not list."""
    reg = AnalysisRegistry(analysis_name="PRJ_Detection",
                           registry_root=tmp_path / "registry")
    csv_file = tmp_path / "measurements.csv"
    csv_file.write_text("label,is_positive\n1,True\n", encoding="utf-8")
    paths = reg.register_output(
        sample=SAMPLE, category="detection", files={"measurements": str(csv_file)},
        results={"n_nuclei": 28, "n_positive": 15, "positive_fraction": 0.536},
        method_params={"sigma_threshold": 0}, source_files={})
    folder = paths["measurements"].parent
    for name in STRAYS:
        (folder / name).write_bytes(b"earlier run")
    return reg, folder


def _run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out"),
                                      "--ignore-tracker"])
    be.main()


def _active_files(folder):
    return sorted(p.name for p in folder.rglob("*")
                  if p.is_file() and "_archived" not in p.relative_to(folder).parts)


def _archived_files(folder):
    stamps = list((folder / "_archived").iterdir())
    assert len(stamps) == 1, "everything goes into ONE _archived/<timestamp>/"
    return sorted(p.name for p in stamps[0].iterdir())


def test_no_nuclei_rerun_archives_files_the_entry_did_not_list(env, monkeypatch, capsys):
    tmp_path = env
    reg, folder = _earlier_entry_with_strays(tmp_path)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        lambda *a, **k: (np.zeros((32, 32), dtype=np.int32),
                                         {"filtered_count": 0}))
    _run(monkeypatch, tmp_path)

    assert reg.get_entry(SAMPLE)["results"]["n_nuclei"] == 0
    assert _active_files(folder) == []
    assert _archived_files(folder) == sorted(STRAYS + ("measurements.csv",))
    assert ("replaced by this run's 0 nuclei; 3 file(s) moved to _archived/"
            in capsys.readouterr().out)


def test_failed_rerun_archives_files_the_entry_did_not_list(env, monkeypatch, capsys):
    tmp_path = env
    reg, folder = _earlier_entry_with_strays(tmp_path)
    # Another sample of the run completes, so the failure is shown to be
    # this sample's and its entry is retired.
    (tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions" / f"{OTHER}.nd2").write_bytes(b"")
    real = be.process_single

    def process(nd2_path, *a, **k):
        if nd2_path.stem == OTHER:
            return {"file": nd2_path.name, "path": str(nd2_path),
                    "status": "completed", "n_nuclei": 3, "n_positive": 1,
                    "positive_fraction": 0.3333, "background": 1.0}
        return real(nd2_path, *a, **k)
    monkeypatch.setattr(be, "process_single", process)

    def detector_fails(*a, **k):
        raise ValueError("image must be 2D")
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", detector_fails)
    _run(monkeypatch, tmp_path)

    assert reg.get_entry(SAMPLE)["is_current"] is False
    assert _active_files(folder) == []
    assert _archived_files(folder) == sorted(STRAYS + ("measurements.csv",))
    out = capsys.readouterr().out.split("BATCH COMPLETE")[1]
    assert (f"{SAMPLE}: earlier registry entry retired (no longer current; "
            "3 file(s) moved to _archived/)") in out


def test_failed_rerun_of_an_entry_without_files_says_so(env, monkeypatch, capsys):
    """A 0-nuclei entry lists no files; retiring it moves nothing, and the
    summary must not claim a move."""
    tmp_path = env
    reg = AnalysisRegistry(analysis_name="PRJ_Detection",
                           registry_root=tmp_path / "registry")
    reg.register_output(sample=SAMPLE, category="detection", files={},
                        results={"n_nuclei": 0}, method_params={}, source_files={})
    (tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions" / f"{OTHER}.nd2").write_bytes(b"")
    real = be.process_single

    def process(nd2_path, *a, **k):
        if nd2_path.stem == OTHER:
            return {"file": nd2_path.name, "path": str(nd2_path),
                    "status": "completed", "n_nuclei": 3, "n_positive": 1,
                    "positive_fraction": 0.3333, "background": 1.0}
        return real(nd2_path, *a, **k)
    monkeypatch.setattr(be, "process_single", process)

    def detector_fails(*a, **k):
        raise ValueError("image must be 2D")
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation", detector_fails)
    _run(monkeypatch, tmp_path)

    assert reg.get_entry(SAMPLE)["is_current"] is False
    out = capsys.readouterr().out.split("BATCH COMPLETE")[1]
    assert (f"{SAMPLE}: earlier registry entry retired (no longer current; "
            "no files to archive)") in out
    assert "moved to _archived" not in out


def test_a_stray_file_that_will_not_move_keeps_the_entry_current(env, monkeypatch, capsys):
    tmp_path = env
    reg, folder = _earlier_entry_with_strays(tmp_path)
    real_rename = ar._rename_only

    def stray_in_use(src, dest):
        if src.name == STRAYS[1]:
            raise PermissionError("the file is in use")
        return real_rename(src, dest)
    monkeypatch.setattr(ar, "_rename_only", stray_in_use)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        lambda *a, **k: (np.zeros((32, 32), dtype=np.int32),
                                         {"filtered_count": 0}))
    _run(monkeypatch, tmp_path)

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["results"]["n_nuclei"] == 28
    # Everything already moved was put back.
    assert _active_files(folder) == sorted(STRAYS + ("measurements.csv",))
    assert "is KEPT CURRENT" in capsys.readouterr().out


def test_folders_shared_with_other_samples_are_never_swept(tmp_path):
    """An ROI analysis files several samples in one animal/region folder:
    retiring one of them must not take the others' files."""
    reg = AnalysisRegistry(analysis_name="ROI_Test", registry_root=tmp_path / "registry")
    files = {}
    for s in ("AN_01_02_S3_DCN", "AN_01_02_S4_DCN"):
        f = tmp_path / f"{s}.csv"
        f.write_text("x\n1\n", encoding="utf-8")
        files[s] = reg.register_output(
            sample=s, category="roi_analysis", files={"measurements": str(f)},
            results={}, method_params={}, source_files={})["measurements"]
    assert reg.invalidate(sample="AN_01_02_S3_DCN", include_own_folder=True)
    assert not files["AN_01_02_S3_DCN"].exists()
    assert files["AN_01_02_S4_DCN"].exists()
    assert reg.get_entry("AN_01_02_S4_DCN")["is_current"] is True
