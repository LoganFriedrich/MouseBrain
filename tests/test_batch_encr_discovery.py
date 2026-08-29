"""batch_encr ND2 discovery: both slice layouts, priority between them, and the
not-found exit.

WHY these exist: the documented invocation once searched only the 1_Subjects
layout, found 0 files at a site whose raw slices still sat in the legacy
per-project folder, and exited 0 as if the batch had run.
"""
import sys
from pathlib import Path

import pytest

from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be

PROJECT = "PRJ"


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _make_new_layout(root: Path):
    """<root>/PRJ_01_01/0_Raw_HD, /0_Raw and a second subject; returns expected order."""
    files = [
        _touch(root / "PRJ_01_01" / "0_Raw_HD" / "P01_01_S01_DCN.nd2"),
        _touch(root / "PRJ_01_01" / "0_Raw_HD" / "P01_01_S02_DCN.nd2"),
        _touch(root / "PRJ_01_01" / "0_Raw" / "P01_01_S03.nd2"),
        _touch(root / "PRJ_01_02" / "0_Raw_HD" / "P01_02_S01_DCN.nd2"),
    ]
    # decoys that must not be picked up
    _touch(root / "PRJ_01_01" / "0_Raw_HD" / "notes.txt")
    _touch(root / "OTHER_01_01" / "0_Raw_HD" / "O01_01_S01.nd2")
    return files


def _make_legacy_layout(root: Path):
    """<root>/PRJ_01_01_HD_Regions[/Corrected]; returns expected order."""
    hd = root / "PRJ_01_01_HD_Regions"
    corrected_a = _touch(hd / "Corrected" / "P01_01_S01_DCN.nd2")
    _touch(hd / "P01_01_S01_DCN.nd2")            # has a Corrected copy -> skipped
    base_b = _touch(hd / "P01_01_S02_DCN.nd2")   # no Corrected copy -> kept
    hd2 = root / "PRJ_01_02_HD_Regions"
    base_c = _touch(hd2 / "P01_02_S01_DCN.nd2")  # no Corrected folder at all
    # decoys
    _touch(root / "PRJ_01_01_ATLAS" / "P01_01_atlas.nd2")   # not an HD_Regions folder
    _touch(root / "PRJ_01_01" / "P01_01_loose.nd2")         # bare subject folder, no 0_Raw*
    return [corrected_a, base_b, base_c]


def test_new_layout_found(tmp_path):
    root = tmp_path / "1_Subjects" / PROJECT
    expected = _make_new_layout(root)
    assert be.find_nd2_files(root, PROJECT) == expected


def test_legacy_layout_corrected_wins(tmp_path):
    root = tmp_path / PROJECT
    expected = _make_legacy_layout(root)
    assert be.find_nd2_files(root, PROJECT) == expected


def test_missing_root_is_empty(tmp_path):
    assert be.find_nd2_files(tmp_path / "nowhere", PROJECT) == []


def test_discover_prefers_new_layout_root(tmp_path):
    new_root = tmp_path / "1_Subjects" / PROJECT
    legacy_root = tmp_path / PROJECT
    expected_new = _make_new_layout(new_root)
    _make_legacy_layout(legacy_root)
    files, used, layout = be.discover_nd2_files([new_root, legacy_root], PROJECT)
    assert (files, used, layout) == (expected_new, new_root, be.LAYOUT_NEW)


def test_discover_falls_back_to_legacy_root(tmp_path):
    new_root = tmp_path / "1_Subjects" / PROJECT   # exists but holds no ND2
    new_root.mkdir(parents=True)
    legacy_root = tmp_path / PROJECT
    expected_legacy = _make_legacy_layout(legacy_root)
    files, used, layout = be.discover_nd2_files([new_root, legacy_root], PROJECT)
    assert (files, used, layout) == (expected_legacy, legacy_root, be.LAYOUT_LEGACY)


def test_discover_nothing(tmp_path):
    assert be.discover_nd2_files([tmp_path / "a", tmp_path / "b"], PROJECT) == ([], None, None)


def test_input_roots_order_and_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    assert be.input_roots(PROJECT) == [tmp_path / "1_Subjects" / PROJECT, tmp_path / PROJECT]
    assert be.input_roots(PROJECT, tmp_path / "x") == [tmp_path / "x"]
    assert be.input_roots(None) == []


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["batch_encr"] + argv)
    return be.main()


def test_main_not_found_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["--project", "NOPE", "--dry-run", "--ignore-tracker",
                                "--output", str(tmp_path / "out")])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "[!] no ND2 files under " in out
    assert str(tmp_path / "1_Subjects" / "NOPE") in out
    assert str(tmp_path / "NOPE") in out
    assert "Input:     (no ND2 files found)" in out
    assert not (tmp_path / "out").exists()


def test_main_explicit_input_not_found_exits_1(tmp_path, monkeypatch, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["--project", PROJECT, "--input", str(empty), "--dry-run",
                                "--ignore-tracker", "--output", str(tmp_path / "out")])
    assert exc.value.code == 1
    assert f"[!] no ND2 files under {empty}" in capsys.readouterr().out


def test_main_dry_run_reports_legacy_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    legacy_root = tmp_path / PROJECT
    expected = _make_legacy_layout(legacy_root)
    assert _run_main(monkeypatch, ["--project", PROJECT, "--dry-run", "--ignore-tracker",
                                   "--output", str(tmp_path / "out")]) is None
    out = capsys.readouterr().out
    assert f"Input:     {legacy_root}  [{be.LAYOUT_LEGACY} layout]" in out
    assert f"Found {len(expected)} ND2 files" in out
    for f in expected:
        assert str(f.relative_to(legacy_root)) in out
    assert not (tmp_path / "out").exists()   # dry-run writes nothing
