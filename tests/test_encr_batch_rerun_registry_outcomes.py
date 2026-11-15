"""What a batch re-run does to a sample's earlier registry entry.

Defect covered: only completed samples touched the registry, so a sample whose
re-run now finds 0 nuclei kept its earlier counts marked current, and those
kept being exported as its result.

Guarded against as well: a FAILED re-run (unreadable file, a tracker CSV locked
by another program) is not a result and must never retire a good earlier
entry -- an infrastructure failure is not a scientific outcome.
"""

import numpy as np

from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.plugin_2d.sliceatlas.core import io as io_mod
from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

from test_encr_batch_runs_and_records_its_parameters import (  # noqa: F401  (fixture)
    PROJECT, SAMPLE, _fake_loader, _nd2, _run_main, _synthetic_image, env)

ARGS = ["--project", PROJECT, "--ignore-tracker"]


def _first_run(tmp_path, monkeypatch):
    _fake_loader(monkeypatch, *_synthetic_image())
    _nd2(tmp_path)
    _run_main(monkeypatch, ARGS + ["--output", str(tmp_path / "out")])
    entry = AnalysisRegistry(analysis_name="ENCR_Detection").get_entry(SAMPLE)
    assert entry["is_current"] and entry["results"]["n_nuclei"] > 0
    return entry


def test_rerun_with_no_nuclei_replaces_the_old_counts(env, monkeypatch, capsys):
    t, tmp_path = env
    old = _first_run(tmp_path, monkeypatch)
    blank = np.full((240, 240), 10.0, dtype=np.float32)
    _fake_loader(monkeypatch, blank, blank)
    _run_main(monkeypatch, ARGS + ["--output", str(tmp_path / "out")])
    entry = AnalysisRegistry(analysis_name="ENCR_Detection").get_entry(SAMPLE)
    assert entry["is_current"]
    assert entry["results"]["n_nuclei"] == 0
    assert entry["outputs"] == {}
    # The replaced record stays in the manifest, with why it was retired.
    superseded = entry.get("superseded", [])
    assert superseded and superseded[0]["results"] == old["results"]
    assert "no nuclei" in superseded[0]["invalidated_reason"]


def _assert_kept(tmp_path, old, capsys):
    reg_root = tmp_path / "registry"
    entry = AnalysisRegistry(analysis_name="ENCR_Detection").get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["results"] == old["results"]
    assert all((reg_root / rel).exists() for rel in old["outputs"].values())
    out = capsys.readouterr().out
    assert "[!] not retired" in out
    assert f"earlier result kept as current -- re-run these (1): {SAMPLE}" in out


def test_unreadable_file_leaves_the_earlier_entry_current(env, monkeypatch, capsys):
    t, tmp_path = env
    old = _first_run(tmp_path, monkeypatch)

    def unreadable(path, **k):
        raise OSError("share not reachable")

    monkeypatch.setattr(io_mod, "load_image", unreadable)
    _run_main(monkeypatch, ARGS + ["--output", str(tmp_path / "out")])
    _assert_kept(tmp_path, old, capsys)


def test_locked_tracker_leaves_the_earlier_entry_current(env, monkeypatch, capsys):
    t, tmp_path = env
    old = _first_run(tmp_path, monkeypatch)

    def locked(self, row):
        raise PermissionError(13, "file is open in another program")

    monkeypatch.setattr(SliceTracker, "_write_row", locked)
    _run_main(monkeypatch, ARGS + ["--output", str(tmp_path / "out")])
    _assert_kept(tmp_path, old, capsys)
