"""Tests for mousebrain.analysis_registry.

Everything on disk goes through pytest's tmp_path; the environment variables
and the pipeline root are monkeypatched so no real installation is touched.

Run from the repository root:
    PYTHONPATH=src python -m pytest -q tests/test_analysis_registry.py
"""

import json
import warnings
from pathlib import Path

import pytest

import mousebrain.analysis_registry as ar
from mousebrain.analysis_registry import (
    DEFAULT_METHOD,
    AnalysisRegistry,
    default_registry_root,
    get_approved_method,
    roi_results_from_rows,
)


SAMPLE = "E01_01_S13_DCN"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Start every test with no registry configuration at all.

    Clears the environment variables the module reads and points the lazily
    imported mousebrain.config.PIPELINE_ROOT at nothing.
    """
    for var in ("MOUSEBRAIN_REGISTRY_ROOT", "MOUSEBRAIN_APPROVED_METHOD"):
        monkeypatch.delenv(var, raising=False)
    import mousebrain.config as cfg
    monkeypatch.setattr(cfg, "PIPELINE_ROOT", None)
    yield


def _make_source_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    csv_file = src / "measurements.csv"
    csv_file.write_text("a,b\n1,2\n", encoding="utf-8")
    fig_file = src / f"{SAMPLE}_coloc_result.png"
    fig_file.write_bytes(b"\x89PNG not really an image")
    return csv_file, fig_file


# ---------------------------------------------------------------------------
# (a) root resolution
# ---------------------------------------------------------------------------

def test_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "reg"))
    assert default_registry_root() == tmp_path / "reg"


def test_root_from_pipeline_root(monkeypatch, tmp_path):
    import mousebrain.config as cfg
    monkeypatch.setattr(cfg, "PIPELINE_ROOT", tmp_path / "pipe")
    assert default_registry_root() == tmp_path / "pipe" / "Registry"


def test_root_none_when_unconfigured():
    assert default_registry_root() is None


def test_constructor_refuses_without_root():
    with pytest.raises(RuntimeError) as exc:
        AnalysisRegistry(analysis_name="A")
    msg = str(exc.value)
    assert "MOUSEBRAIN_REGISTRY_ROOT" in msg
    assert "CONNECTOME_ROOT" in msg


def test_constructor_creates_exports_and_logs_but_not_figures(tmp_path):
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    assert reg.registry_root == tmp_path
    assert reg.db_root == tmp_path  # attribute alias kept for older callers
    assert reg.exports_dir == tmp_path / "exports" / "A"
    assert reg.exports_dir.is_dir()
    assert reg.logs_dir == tmp_path / "logs"
    assert reg.logs_dir.is_dir()
    assert not (tmp_path / "figures").exists()


def test_constructor_db_root_keyword_alias(tmp_path):
    reg = AnalysisRegistry(analysis_name="A", db_root=tmp_path)
    assert reg.registry_root == tmp_path


def test_constructor_uses_env_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "reg"))
    reg = AnalysisRegistry(analysis_name="A")
    assert reg.registry_root == tmp_path / "reg"
    assert (tmp_path / "reg" / "exports" / "A").is_dir()


# ---------------------------------------------------------------------------
# (b) register_output
# ---------------------------------------------------------------------------

def test_register_output_data_file_goes_to_exports_sample_dir(tmp_path):
    root = tmp_path / "Registry"
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=root)

    dest = reg.register_output(
        sample=SAMPLE,
        category="detection",
        files={"measurements": str(csv_file)},
        results={"n_nuclei": 28, "n_positive": 15},
        method_params=DEFAULT_METHOD,
        source_files={"nd2": str(tmp_path / "src" / f"{SAMPLE}.nd2")},
    )

    expected = root / "exports" / "A" / SAMPLE / "measurements.csv"
    assert dest["measurements"] == expected
    assert expected.is_file()
    assert expected.read_text(encoding="utf-8") == csv_file.read_text(encoding="utf-8")
    assert not (root / "figures").exists()  # no figure written -> no figures tree

    data = json.loads((root / "exports" / "A" / "registry.json").read_text(encoding="utf-8"))
    assert data["analysis_name"] == "A"
    entry = data["entries"][SAMPLE]
    assert entry["is_current"] is True
    assert entry["animal"] == "E01_01"
    assert entry["region"] == "DCN"
    assert entry["method_hash"] == AnalysisRegistry.get_method_hash(DEFAULT_METHOD)
    rel = entry["outputs"]["measurements"]
    assert not Path(rel).is_absolute()
    assert (root / rel) == expected
    assert (root / "logs" / "A.log").is_file()


def test_register_output_figure_goes_to_figures_animal_region_dir(tmp_path):
    root = tmp_path / "Registry"
    _, fig_file = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=root)

    dest = reg.register_output(
        sample=SAMPLE,
        category="roi_analysis",
        files={"figure": str(fig_file)},
        results={"n_nuclei": 13, "n_positive": 12, "positive_fraction": 0.923},
        method_params=DEFAULT_METHOD,
    )

    expected = root / "figures" / "A" / "E01_01" / "DCN" / fig_file.name
    assert dest["figure"] == expected
    assert expected.is_file()

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    rel = entry["outputs"]["figure"]
    assert not Path(rel).is_absolute()
    assert (root / rel).is_file()


def test_register_output_skips_missing_source_file(tmp_path):
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    dest = reg.register_output(
        sample=SAMPLE, category="detection",
        files={"measurements": str(tmp_path / "does_not_exist.csv")},
        results={}, method_params=DEFAULT_METHOD,
    )
    assert dest == {}
    assert reg.get_entry(SAMPLE)["outputs"] == {}


# ---------------------------------------------------------------------------
# (c) staleness
# ---------------------------------------------------------------------------

def test_check_staleness(tmp_path):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    reg.register_output(
        sample=SAMPLE, category="detection",
        files={"measurements": str(csv_file)}, results={},
        method_params=DEFAULT_METHOD,
    )

    assert reg.check_staleness(DEFAULT_METHOD) == []
    assert reg.check_staleness(dict(DEFAULT_METHOD)) == []  # same content, new dict

    changed = dict(DEFAULT_METHOD, soma_dilation=8)
    stale = reg.check_staleness(changed)
    assert len(stale) == 1
    assert stale[0]["sample"] == SAMPLE
    assert stale[0]["diff_keys"] == ["soma_dilation"]
    assert stale[0]["registered_hash"] == AnalysisRegistry.get_method_hash(DEFAULT_METHOD)
    assert stale[0]["current_hash"] == AnalysisRegistry.get_method_hash(changed)
    assert reg.get_stale_samples(changed) == [SAMPLE]


# ---------------------------------------------------------------------------
# (d) invalidate
# ---------------------------------------------------------------------------

def test_invalidate_archives_files_and_flags_entry(tmp_path):
    root = tmp_path / "Registry"
    csv_file, fig_file = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=root)
    dest = reg.register_output(
        sample=SAMPLE, category="roi_analysis",
        files={"roi_counts": str(csv_file), "figure": str(fig_file)},
        results={}, method_params=DEFAULT_METHOD,
    )

    invalidated = reg.invalidate(sample=SAMPLE)
    assert invalidated == [SAMPLE]

    for path in dest.values():
        assert not path.exists()
        archived = list((path.parent / "_archived").glob("*/" + path.name))
        assert len(archived) == 1, path
        assert archived[0].parent.parent.name == "_archived"

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is False
    assert "invalidated_at" in entry

    # Stale entries are not reported again, and a second invalidate is a no-op.
    assert reg.check_staleness(dict(DEFAULT_METHOD, soma_dilation=8)) == []
    assert reg.invalidate(sample=SAMPLE) == []


def test_invalidate_all(tmp_path):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    for sample in ("E01_01_S13_DCN", "E01_01_S14_DCN"):
        reg.register_output(
            sample=sample, category="detection",
            files={"measurements": str(csv_file)}, results={},
            method_params=DEFAULT_METHOD,
        )
    assert sorted(reg.invalidate()) == ["E01_01_S13_DCN", "E01_01_S14_DCN"]
    assert all(not e["is_current"] for e in reg.get_all_entries().values())


# ---------------------------------------------------------------------------
# (e) approved method
# ---------------------------------------------------------------------------

def test_get_approved_method_default_is_a_copy():
    method = get_approved_method()
    assert method == DEFAULT_METHOD
    assert method is not DEFAULT_METHOD


def test_get_approved_method_json_under_root(monkeypatch, tmp_path):
    override = dict(DEFAULT_METHOD, threshold_fraction=0.25)
    (tmp_path / "approved_method.json").write_text(json.dumps(override), encoding="utf-8")

    # explicit root
    assert get_approved_method(tmp_path) == override
    # default root via the environment
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path))
    assert get_approved_method() == override
    # a fresh registry under that root records the override as its approved method
    reg = AnalysisRegistry(analysis_name="A")
    skeleton = reg._read_registry()
    assert skeleton["approved_method"] == override
    assert skeleton["approved_method_hash"] == AnalysisRegistry.get_method_hash(override)


def test_get_approved_method_env_file_wins(monkeypatch, tmp_path):
    root_override = dict(DEFAULT_METHOD, threshold_fraction=0.25)
    (tmp_path / "approved_method.json").write_text(json.dumps(root_override), encoding="utf-8")
    env_override = dict(DEFAULT_METHOD, soma_dilation=9)
    env_file = tmp_path / "method.json"
    env_file.write_text(json.dumps(env_override), encoding="utf-8")
    monkeypatch.setenv("MOUSEBRAIN_APPROVED_METHOD", str(env_file))

    got = get_approved_method(tmp_path)
    assert got == env_override
    # JSON round-trip keeps the hash comparable to the in-memory dict
    assert AnalysisRegistry.get_method_hash(got) == AnalysisRegistry.get_method_hash(env_override)


def test_get_approved_method_bad_file_raises(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setenv("MOUSEBRAIN_APPROVED_METHOD", str(bad))
    with pytest.raises(RuntimeError, match="pproved-method"):
        get_approved_method()

    monkeypatch.setenv("MOUSEBRAIN_APPROVED_METHOD", str(tmp_path / "missing.json"))
    with pytest.raises(RuntimeError, match="pproved-method"):
        get_approved_method()


# ---------------------------------------------------------------------------
# ROI count rows (the shape count_cells_in_rois actually returns)
# ---------------------------------------------------------------------------

def test_roi_results_from_rows_single_channel():
    rows = [
        {"name": "Left", "total": 12, "positive": 11, "negative": 1, "fraction": 0.9167, "_scratch": 1},
        {"name": "Outside", "total": 0, "positive": 0, "negative": 0, "fraction": 0.0},
        {"name": "TOTAL", "total": 12, "positive": 11, "negative": 1, "fraction": 0.9167},
    ]
    d = roi_results_from_rows(rows)
    assert list(d) == ["Left", "Outside", "TOTAL"]
    assert d["TOTAL"] == {"total": 12, "positive": 11, "negative": 1, "fraction": 0.9167}
    assert "_scratch" not in d["Left"]


def test_roi_results_from_rows_dual_channel():
    rows = [{"name": "TOTAL", "total": 10, "dual": 4, "red_only": 3, "green_only": 2, "neither": 1}]
    d = roi_results_from_rows(rows)
    assert d["TOTAL"]["positive"] == 4
    assert d["TOTAL"]["negative"] == 6
    assert d["TOTAL"]["fraction"] == 0.4
    assert d["TOTAL"]["dual"] == 4  # original keys kept


def test_register_roi_counts_accepts_row_list(tmp_path):
    rows = [
        {"name": "Left", "total": 12, "positive": 11, "negative": 1, "fraction": 0.9167},
        {"name": "Outside", "total": 1, "positive": 0, "negative": 1, "fraction": 0.0},
        {"name": "TOTAL", "total": 13, "positive": 11, "negative": 2, "fraction": 0.8462},
    ]
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    counts_path = reg.register_roi_counts(
        sample=SAMPLE, region="DCN", roi_results=rows,
        method_params=DEFAULT_METHOD,
        source_files={"nd2": str(tmp_path / f"{SAMPLE}.nd2")},
    )

    assert counts_path == tmp_path / "exports" / "A" / "E01_01" / "DCN" / f"{SAMPLE}_roi_counts.csv"
    lines = counts_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "roi,total,positive,negative,fraction"
    assert lines[1] == "Left,12,11,1,0.9167"

    summary = (tmp_path / "exports" / "A" / "roi_summary_DCN.csv").read_text(encoding="utf-8").splitlines()
    assert summary[0] == "sample,roi,total,positive,negative,fraction"
    assert summary[1] == f"{SAMPLE},TOTAL,13,11,2,0.8462"

    entry = reg.get_all_entries()[f"{SAMPLE}__roi_counts"]
    assert entry["category"] == "roi_counts"
    assert entry["results"]["TOTAL"]["positive"] == 11
    assert not Path(entry["outputs"]["roi_counts"]).is_absolute()
    assert (tmp_path / entry["outputs"]["roi_counts"]) == counts_path


def test_register_roi_counts_keeps_dual_columns(tmp_path):
    rows = [{"name": "TOTAL", "total": 10, "dual": 4, "red_only": 3, "green_only": 2, "neither": 1}]
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    counts_path = reg.register_roi_counts(
        sample=SAMPLE, region="DCN", roi_results=rows, method_params=DEFAULT_METHOD,
    )
    lines = counts_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "roi,total,positive,negative,fraction,dual,green_only,neither,red_only"
    assert lines[1] == "TOTAL,10,4,6,0.4,4,2,1,3"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_main_root_and_hidden_alias(tmp_path, capsys):
    assert ar.main(["--name", "A", "--root", str(tmp_path), "--stale"]) == 0
    out = capsys.readouterr().out
    assert "Total entries: 0" in out
    assert str(tmp_path) in out
    assert "All entries are current." in out

    assert ar.main(["--name", "A", "--db-root", str(tmp_path)]) == 0
    assert ar._cli_main is ar.main


# ---------------------------------------------------------------------------
# An unreadable manifest is set aside, never emptied
# ---------------------------------------------------------------------------

def test_an_unreadable_manifest_is_kept_not_replaced_by_an_empty_one(tmp_path):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    damaged = '{"entries": {"E01_01_S01_DCN": {"sample": "E01_01_S01_DCN"'  # cut off
    reg.registry_path.write_text(damaged, encoding="utf-8")

    with pytest.raises(RuntimeError, match="unreadable"):
        reg.register_output(sample=SAMPLE, category="detection",
                            files={"measurements": str(csv_file)}, results={},
                            method_params=DEFAULT_METHOD)

    kept = list(reg.exports_dir.glob("registry.json.unreadable-*"))
    assert len(kept) == 1
    assert kept[0].read_text(encoding="utf-8") == damaged  # byte for byte
    assert not reg.registry_path.exists()

    # The next registration starts a new manifest; the kept file is untouched.
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    assert list(reg.get_all_entries()) == [SAMPLE]
    assert kept[0].read_text(encoding="utf-8") == damaged


def test_reading_an_unreadable_manifest_raises_and_moves_nothing(tmp_path, capsys):
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg.registry_path.write_text('["not", "a", "manifest"]', encoding="utf-8")

    with pytest.raises(RuntimeError):
        reg.get_all_entries()
    assert ar.main(["--name", "A", "--root", str(tmp_path)]) == 1
    assert "[FAIL]" in capsys.readouterr().out
    # Inspection never renames: the file is exactly where and what it was.
    assert reg.registry_path.read_text(encoding="utf-8") == '["not", "a", "manifest"]'
    assert not list(reg.exports_dir.glob("registry.json.unreadable-*"))


# ---------------------------------------------------------------------------
# Two registrations at once both land (the manifest is written under a lock)
# ---------------------------------------------------------------------------

def test_two_registrations_at_once_both_land(tmp_path, monkeypatch):
    import threading
    import time

    csv_file, _ = _make_source_files(tmp_path)
    reg_a = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg_b = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)

    # Widen the gap between reading the manifest and writing it back, the
    # window in which a second writer used to read the same old manifest.
    real_write = AnalysisRegistry._write_registry

    def slow_write(self, data):
        time.sleep(0.6)
        real_write(self, data)

    monkeypatch.setattr(AnalysisRegistry, "_write_registry", slow_write)

    errors = []

    def register(reg, sample):
        try:
            reg.register_output(sample=sample, category="detection",
                                files={"measurements": str(csv_file)}, results={},
                                method_params=DEFAULT_METHOD)
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=register, args=(reg_a, "E01_01_S01_DCN")),
               threading.Thread(target=register, args=(reg_b, "E01_01_S02_DCN"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Without the lock one entry was lost: either written over by the other
    # writer's stale manifest, or refused when both swapped the file at once.
    assert sorted(reg_a.get_all_entries()) == ["E01_01_S01_DCN", "E01_01_S02_DCN"]
    assert errors == []
    assert not (reg_a.exports_dir / "registry.json.lock").exists()  # let go after use


def test_a_lock_left_by_a_crashed_writer_is_taken_over(tmp_path):
    import os
    import time

    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    lock = reg.exports_dir / "registry.json.lock"
    lock.write_text("other-machine pid 1 at earlier token x\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))

    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    assert list(reg.get_all_entries()) == [SAMPLE]
    assert not lock.exists()


# ---------------------------------------------------------------------------
# invalidate(): only what really moved, recorded one entry at a time
# ---------------------------------------------------------------------------

def _register_two(tmp_path):
    _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    dests = {}
    for sample in ("E01_01_S01_DCN", "E01_01_S02_DCN"):
        one_csv = tmp_path / "src" / f"{sample}_measurements.csv"
        one_csv.write_text("a\n1\n", encoding="utf-8")
        one_fig = tmp_path / "src" / f"{sample}_overlay.png"
        one_fig.write_bytes(b"png")
        dests[sample] = reg.register_output(
            sample=sample, category="roi_analysis",
            files={"figure": str(one_fig), "measurements": str(one_csv)},
            results={}, method_params=DEFAULT_METHOD)
    return reg, dests


def test_an_entry_whose_file_cannot_be_moved_stays_current(tmp_path, monkeypatch):
    reg, dests = _register_two(tmp_path)
    stuck = dests["E01_01_S02_DCN"]["measurements"]
    real_move = ar.shutil.move

    def move(src, dst, *a, **k):
        if Path(src) == stuck:
            raise PermissionError(13, "held open by another program")
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr(ar.shutil, "move", move)
    assert reg.invalidate() == ["E01_01_S01_DCN"]

    on_disk = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"]
    assert on_disk["E01_01_S01_DCN"]["is_current"] is False
    assert on_disk["E01_01_S02_DCN"]["is_current"] is True
    # Every file of the entry left current is in the active path -- the
    # figure, moved before the failure, was put back.
    for path in dests["E01_01_S02_DCN"].values():
        assert path.is_file(), path


def test_each_invalidated_entry_is_recorded_before_the_next_is_moved(tmp_path, monkeypatch):
    reg, dests = _register_two(tmp_path)
    second = dests["E01_01_S02_DCN"]["measurements"]
    real_move = ar.shutil.move

    def move(src, dst, *a, **k):
        if Path(src) == second:
            raise KeyboardInterrupt  # the process stops part-way through the pass
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr(ar.shutil, "move", move)
    with pytest.raises(KeyboardInterrupt):
        reg.invalidate()

    # The first entry's files were archived AND the manifest says so.
    for path in dests["E01_01_S01_DCN"].values():
        assert not path.exists()
    on_disk = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"]
    assert on_disk["E01_01_S01_DCN"]["is_current"] is False
