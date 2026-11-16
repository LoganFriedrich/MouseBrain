"""A re-run that finds no nuclei replaces the sample's earlier registry entry.

Defect covered: only samples that completed with nuclei were registered. A
sample whose re-run found none kept its earlier entry -- with the earlier
count, still marked current -- so the registry reported a result the newest
run contradicts.
"""

import json
import sys

import numpy as np
import pytest

from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.plugin_2d.sliceatlas import tracker as tracker_pkg
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be
from mousebrain.plugin_2d.sliceatlas.core import detection as det_mod
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"


@pytest.fixture
def env(tmp_path, monkeypatch):
    t = SliceTracker(csv_path=tmp_path / "runs.csv")
    monkeypatch.setattr(tracker_pkg, "SliceTracker", lambda *a, **k: t)
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
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
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        lambda *a, **k: (np.zeros((32, 32), dtype=np.int32),
                                         {"filtered_count": 0}))
    return tmp_path


def _earlier_entry(tmp_path):
    reg = AnalysisRegistry(analysis_name="ENCR_Detection",
                           registry_root=tmp_path / "registry")
    csv_file = tmp_path / "measurements.csv"
    csv_file.write_text("label,is_positive\n1,True\n", encoding="utf-8")
    paths = reg.register_output(
        sample=SAMPLE, category="detection", files={"measurements": str(csv_file)},
        results={"n_nuclei": 28, "n_positive": 15, "positive_fraction": 0.536},
        method_params={"sigma_threshold": 0}, source_files={})
    return reg, paths["measurements"]


def _run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out"),
                                      "--ignore-tracker"])
    be.main()


def test_no_nuclei_rerun_replaces_the_earlier_count(env, monkeypatch, capsys):
    tmp_path = env
    reg, old_file = _earlier_entry(tmp_path)
    _run(monkeypatch, tmp_path)

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["results"]["n_nuclei"] == 0
    assert entry["results"]["positive_fraction"] is None
    assert entry["outputs"] == {}
    # Registered with the parameters this run used, like a completed sample.
    assert entry["method_params"]["size_filter_um"] == list(be.BATCH_NUCLEUS_DIAMETER_UM)
    # The earlier run's file is archived, not left beside the new entry.
    assert not old_file.exists()
    archived = list(old_file.parent.glob("_archived/*/" + old_file.name))
    assert len(archived) == 1
    assert ("earlier registry entry (28 nuclei) replaced by this run's 0 nuclei; "
            "1 file(s) moved to _archived/") in capsys.readouterr().out


def test_files_that_cannot_be_moved_keep_the_earlier_entry(env, monkeypatch, capsys):
    """A file held open by another program: the 0-nuclei result must not be
    registered over an entry whose files are still in the active folder."""
    tmp_path = env
    reg, old_file = _earlier_entry(tmp_path)
    from mousebrain import analysis_registry as ar

    def in_use(*a, **k):
        raise PermissionError("the file is in use")
    monkeypatch.setattr(ar, "_rename_only", in_use)
    _run(monkeypatch, tmp_path)

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["results"]["n_nuclei"] == 28
    assert entry["outputs"]
    assert old_file.exists()
    out = capsys.readouterr().out
    assert (f"[!] {SAMPLE}: this run found 0 nuclei, but the earlier registry "
            "entry (28 nuclei) is KEPT CURRENT") in out
    assert "moved to _archived/" not in out.split("BATCH COMPLETE")[1].replace(
        "could not be moved to _archived/", "")
    assert "replaced by this run's 0 nuclei" not in out


def test_retired_then_not_registered_is_reported_as_exactly_that(env, monkeypatch, capsys):
    """The earlier entry's files move to _archived/, then writing the
    0-nuclei entry fails (the registry file is in use, the share drops out).
    The sample is left with no current entry; the summary must say so and
    must not say the entry is as it was before the run."""
    tmp_path = env
    reg, old_file = _earlier_entry(tmp_path)
    from mousebrain import analysis_registry as ar

    def cannot_write(self, *a, **k):
        raise PermissionError("registry.json is in use")
    monkeypatch.setattr(ar.AnalysisRegistry, "register_output", cannot_write)
    _run(monkeypatch, tmp_path)

    # The state the summary has to describe.
    assert reg.get_entry(SAMPLE)["is_current"] is False
    assert not old_file.exists()
    assert len(list(old_file.parent.glob("_archived/*/" + old_file.name))) == 1

    summary = capsys.readouterr().out.split("BATCH COMPLETE")[1]
    assert (f"[!] {SAMPLE}: earlier registry entry (28 nuclei) was retired, "
            "1 file(s) moved to _archived/, but this run's 0-nuclei result "
            "could NOT be registered: registry.json is in use") in summary
    assert "has no current entry now; run the batch again" in summary
    assert "its entry is as it was before this run" not in summary
    assert "replaced by this run's 0 nuclei" not in summary


def test_no_nuclei_first_run_is_registered(env, monkeypatch):
    tmp_path = env
    _run(monkeypatch, tmp_path)
    manifest = json.loads(
        (tmp_path / "registry" / "exports" / "ENCR_Detection" / "registry.json")
        .read_text(encoding="utf-8"))
    assert manifest["entries"][SAMPLE]["results"]["n_nuclei"] == 0


def _invalidate_entry_records(tmp_path):
    """The retired records in the registry's audit log, oldest first."""
    log = tmp_path / "registry" / "logs" / "ENCR_Detection.log"
    records = []
    for line in log.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if cols[2] == "invalidate_entry" and cols[3] == SAMPLE:
            records.append(json.loads(cols[5]))
    return records


def test_earlier_entry_with_missing_files_is_retired_and_logged(env, monkeypatch, capsys):
    """A current entry whose files are gone has nothing to move, but it must
    still be retired through invalidate, so its counts and parameters reach
    the log before the 0-nuclei registration overwrites its record."""
    tmp_path = env
    reg, old_file = _earlier_entry(tmp_path)
    old_file.unlink()
    _run(monkeypatch, tmp_path)

    assert reg.get_entry(SAMPLE)["results"]["n_nuclei"] == 0
    records = _invalidate_entry_records(tmp_path)
    assert len(records) == 1
    assert records[0]["results"]["n_nuclei"] == 28
    assert records[0]["method_params"] == {"sigma_threshold": 0}
    assert ("earlier registry entry (28 nuclei) replaced by this run's 0 nuclei; "
            "no files to archive") in capsys.readouterr().out


def test_earlier_no_nuclei_entry_with_other_parameters_is_logged(env, monkeypatch):
    """An earlier 0-nuclei entry lists no files. Re-running with different
    parameters must still leave the earlier parameters in the log."""
    tmp_path = env
    reg = AnalysisRegistry(analysis_name="ENCR_Detection",
                           registry_root=tmp_path / "registry")
    reg.register_output(
        sample=SAMPLE, category="detection", files={},
        results={"n_nuclei": 0, "n_positive": 0, "positive_fraction": None,
                 "background": None},
        method_params={"sigma_threshold": 7, "log_threshold": 0.5},
        source_files={})
    _run(monkeypatch, tmp_path)

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["method_params"]["size_filter_um"] == list(be.BATCH_NUCLEUS_DIAMETER_UM)
    records = _invalidate_entry_records(tmp_path)
    assert len(records) == 1
    assert records[0]["results"]["n_nuclei"] == 0
    assert records[0]["method_params"] == {"sigma_threshold": 7, "log_threshold": 0.5}


def test_record_is_printed_when_the_log_cannot_be_written(tmp_path, capsys):
    """A failed invalidate_entry log write is reported, with the record, so
    the retired counts are not lost without anyone knowing."""
    reg = AnalysisRegistry(analysis_name="ENCR_Detection",
                           registry_root=tmp_path / "registry")
    reg.register_output(
        sample=SAMPLE, category="detection", files={},
        results={"n_nuclei": 41}, method_params={"sigma_threshold": 3},
        source_files={})
    log = tmp_path / "registry" / "logs" / "ENCR_Detection.log"
    log.unlink()
    log.mkdir()   # a folder where the log should be: every write fails
    assert reg.invalidate(sample=SAMPLE) == [SAMPLE]

    out = capsys.readouterr().out
    assert f"[!] {SAMPLE} was invalidated, but its record could not be written" in out
    printed = json.loads(out.split("Its record is: ")[1].splitlines()[0])
    assert printed["results"] == {"n_nuclei": 41}
    assert printed["method_params"] == {"sigma_threshold": 3}
