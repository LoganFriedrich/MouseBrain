"""Tests for mousebrain.analysis_registry.

Everything on disk goes through pytest's tmp_path; the environment variables
and the pipeline root are monkeypatched so no real installation is touched.

Run from the repository root:
    PYTHONPATH=src python -m pytest -q tests/test_analysis_registry.py
"""

import json
import os
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

    kept = reg.set_aside_files()
    # Kept outside the mirrored exports folder (see test_the_mirrored_...).
    assert not list(reg.exports_dir.glob("registry.json.unreadable-*"))
    assert len(kept) == 1
    assert kept[0].read_text(encoding="utf-8") == damaged  # byte for byte

    # A new manifest is in place at once -- the analysis folder is never
    # left without one, so an integrator that finds analyses by their
    # registry.json still lists it -- and its header names the kept file.
    started = json.loads(reg.registry_path.read_text(encoding="utf-8"))
    assert started["entries"] == {}
    assert [r["file"] for r in started["set_aside_manifests"]] == [kept[0].name]

    # The next registration adds to it; the kept file is untouched.
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    assert list(reg.get_all_entries()) == [SAMPLE]
    assert kept[0].read_text(encoding="utf-8") == damaged


def test_if_no_new_manifest_can_be_written_the_damaged_one_is_put_back(tmp_path, monkeypatch):
    # Setting a damaged manifest aside must never leave the analysis folder
    # without a registry.json: if the new one cannot be written (the share
    # dropped out), the damaged file goes back exactly as it was.
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    damaged = '{"entries": {"E01_01_S01_DCN": {"sample"'
    reg.registry_path.write_text(damaged, encoding="utf-8")

    def refuse(*a, **k):
        raise OSError(5, "the network drive is not available")

    monkeypatch.setattr(AnalysisRegistry, "_write_registry", refuse)
    with pytest.raises(RuntimeError, match="no new manifest"):
        reg.register_output(sample=SAMPLE, category="detection",
                            files={"measurements": str(csv_file)}, results={},
                            method_params=DEFAULT_METHOD)
    assert reg.registry_path.read_text(encoding="utf-8") == damaged
    assert reg.set_aside_files() == []
    assert not (reg.exports_dir / SAMPLE).exists()


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
    assert reg.set_aside_files() == []


@pytest.mark.parametrize("flag", ["--refresh-approved", "--no-approved-method",
                                  "--follows-approved-method"])
def test_refreshing_the_header_of_an_unreadable_manifest_moves_nothing(tmp_path, capsys, flag):
    # The guide tells a person to run --refresh-approved to check a repaired
    # manifest. It registers nothing, so it must never set the file aside
    # (and print that a registration was not recorded).
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg.registry_path.write_text('{"entries": {"x"', encoding="utf-8")

    assert ar.main(["--name", "A", "--root", str(tmp_path), flag]) == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out
    assert "registration was not recorded" not in out
    assert reg.registry_path.read_text(encoding="utf-8") == '{"entries": {"x"'
    assert reg.set_aside_files() == []


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

    def slow_write(self, data, *args, **kwargs):
        time.sleep(0.6)
        real_write(self, data, *args, **kwargs)

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
    assert not reg_a.lock_path.exists()  # let go after use


def test_a_registration_copying_one_slow_file_keeps_its_lock(tmp_path, monkeypatch):
    # One file copy that lasts longer than the lock's stale age (a large
    # figure over a slow share) must not let a second writer take the lock
    # over: the second writer's entry was then written over by the first
    # writer's manifest and lost. Stale age scaled down to 1 s, one copy
    # slowed to 2.5 s.
    import threading
    import time

    monkeypatch.setattr(AnalysisRegistry, "LOCK_STALE_SECONDS", 1, raising=False)
    src = tmp_path / "src"
    src.mkdir()
    slow_src = src / "slow_measurements.csv"
    slow_src.write_text("a\n1\n", encoding="utf-8")
    fast_src = src / "fast_measurements.csv"
    fast_src.write_text("a\n2\n", encoding="utf-8")
    real_copy2 = ar.shutil.copy2
    copying = threading.Event()

    def copy2(s, d, *a, **k):
        if Path(s) == slow_src:
            copying.set()
            time.sleep(2.5)
        return real_copy2(s, d, *a, **k)

    monkeypatch.setattr(ar.shutil, "copy2", copy2)
    reg_a = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    reg_b = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    errors = []

    def register(reg, sample, f):
        try:
            reg.register_output(sample=sample, category="detection",
                                files={"measurements": str(f)}, results={},
                                method_params=DEFAULT_METHOD)
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    first = threading.Thread(target=register, args=(reg_a, "E01_01_S01_DCN", slow_src))
    first.start()
    assert copying.wait(10)
    time.sleep(0.2)
    second = threading.Thread(target=register, args=(reg_b, "E01_01_S02_DCN", fast_src))
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert sorted(reg_a.get_all_entries()) == ["E01_01_S01_DCN", "E01_01_S02_DCN"]


def test_a_lock_left_by_a_crashed_writer_is_taken_over(tmp_path):
    import os
    import time

    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    lock = reg.lock_path
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("other-machine pid 1 at earlier token x\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))

    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    assert list(reg.get_all_entries()) == [SAMPLE]
    assert not lock.exists()


_WRITER = """
import sys
from mousebrain.analysis_registry import AnalysisRegistry, DEFAULT_METHOD
root, src, worker, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
reg = AnalysisRegistry(analysis_name="A", registry_root=root)
for i in range(n):
    reg.register_output(sample="E%02d_01_S%02d_DCN" % (worker, i + 1),
                        category="detection", files={"measurements": src},
                        results={"i": i}, method_params=DEFAULT_METHOD)
"""

_READER = """
import sys, time
from pathlib import Path
from mousebrain.analysis_registry import AnalysisRegistry
reg = AnalysisRegistry(analysis_name="A", registry_root=sys.argv[1])
# Wherever this version keeps the lock (older versions: beside the manifest).
lock = getattr(reg, "lock_path", reg.registry_path.with_name("registry.json.lock"))
stop = Path(sys.argv[2])
while not stop.exists():
    # What an inspector and every waiting writer do: read the manifest and
    # look at the lock, holding each open for a moment.
    try:
        reg.get_all_entries()
    except Exception:
        pass
    try:
        with open(lock, "rb"):
            time.sleep(0.01)
    except OSError:
        pass
"""


def test_several_processes_registering_at_once_all_land_promptly(tmp_path):
    # Separate PROCESSES (not threads), more than two writers, and a reader
    # opening the manifest and the lock all the while. Without a lock entries
    # were lost; with a lock whose release failed while a reader had it open,
    # writers stalled in two-minute steps until it aged out.
    import os
    import subprocess
    import sys
    import time

    csv_file, _ = _make_source_files(tmp_path)
    root = tmp_path / "Registry"
    AnalysisRegistry(analysis_name="A", registry_root=root)
    env = dict(os.environ)
    src_dir = str(Path(ar.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("MOUSEBRAIN_REGISTRY_ROOT", None)
    env.pop("MOUSEBRAIN_APPROVED_METHOD", None)
    stop = tmp_path / "stop"
    writers, per_writer = 5, 6

    reader = subprocess.Popen([sys.executable, "-c", _READER, str(root), str(stop)],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = time.monotonic()
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(root), str(csv_file),
                               str(w + 1), str(per_writer)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
             for w in range(writers)]
    outputs = []
    try:
        for proc in procs:
            # Far below the two-minute stale age: a release left behind would
            # show up as a writer still waiting here.
            out, _ = proc.communicate(timeout=max(1.0, 90 - (time.monotonic() - started)))
            outputs.append((proc.returncode, out.decode("utf-8", "replace")))
    finally:
        stop.write_text("", encoding="utf-8")
        for proc in procs + [reader]:
            if proc.poll() is None:
                proc.kill()
        reader.wait(timeout=30)

    assert [code for code, _ in outputs] == [0] * writers, outputs
    reg = AnalysisRegistry(analysis_name="A", registry_root=root)
    assert len(reg.get_all_entries()) == writers * per_writer
    assert not any("Took over a lock" in out for _, out in outputs), outputs
    assert not reg.lock_path.exists()


def test_the_mirrored_exports_folder_holds_only_outputs_and_the_manifest(tmp_path, monkeypatch):
    # An integrator mirrors every file under exports/<analysis>/. The lock,
    # the server-clock probes and the temporary manifest used to be created
    # there: one caught by a nightly pull was copied as a result, and one that
    # vanished between the pull listing the folder and reading it stopped
    # the pull. Look at the folder at the instant the manifest is swapped in,
    # while the lock is held and the temporary manifest exists.
    csv_file, fig_file = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    seen = []
    real_replace = ar.os.replace

    def replace(src, dst, *a, **k):
        seen.append(sorted(p.name for p in (reg.registry_root / "exports").rglob("*")
                           if p.is_file()))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(ar.os, "replace", replace)
    reg.register_output(sample=SAMPLE, category="roi_analysis",
                        files={"measurements": str(csv_file), "figure": str(fig_file)},
                        results={}, method_params=DEFAULT_METHOD)
    assert seen
    for names in seen:
        assert set(names) <= {"measurements.csv", "registry.json"}, names


def test_a_set_aside_manifest_is_kept_outside_the_mirrored_folder(tmp_path):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    reg.registry_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError):
        reg.register_output(sample=SAMPLE, category="detection",
                            files={"measurements": str(csv_file)}, results={},
                            method_params=DEFAULT_METHOD)
    exported = [p.name for p in (reg.registry_root / "exports").rglob("*") if p.is_file()]
    assert exported == ["registry.json"]  # the new, empty manifest only
    kept = reg.set_aside_files()
    assert [p.read_text(encoding="utf-8") for p in kept] == ["{not json"]
    assert kept[0].parent == reg.registry_root / "logs"


def test_the_registry_adds_no_folders_of_its_own(tmp_path):
    # The lock, its probes, temporary files and set-aside manifests live in
    # the existing logs/ folder: a registry people browse gains no new
    # folders, and none of those names can be taken for an audit log.
    csv_file, fig_file = _make_source_files(tmp_path)
    root = tmp_path / "Registry"
    reg = AnalysisRegistry(analysis_name="A", registry_root=root)
    reg.register_output(sample=SAMPLE, category="roi_analysis",
                        files={"measurements": str(csv_file), "figure": str(fig_file)},
                        results={}, method_params=DEFAULT_METHOD)
    reg.register_roi_counts(SAMPLE, "DCN", {"TOTAL": {"total": 1, "positive": 1,
                                                      "negative": 0, "fraction": 1.0}},
                            method_params=DEFAULT_METHOD)
    reg.registry_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError):
        reg.register_output(sample=SAMPLE, category="detection",
                            files={"measurements": str(csv_file)}, results={},
                            method_params=DEFAULT_METHOD)
    assert sorted(p.name for p in root.iterdir()) == ["exports", "figures", "logs"]
    assert [p.name for p in (root / "logs").iterdir() if p.is_dir()] == []
    logs = sorted(p.name for p in (root / "logs").iterdir())
    assert [n for n in logs if n.endswith(".log")] == ["A.log"]


# ---------------------------------------------------------------------------
# A set-aside manifest stays visible until a person deals with it
# ---------------------------------------------------------------------------

def _set_one_aside(tmp_path, damaged):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    reg.registry_path.write_text(damaged, encoding="utf-8")
    with pytest.raises(RuntimeError):
        reg.register_output(sample=SAMPLE, category="detection",
                            files={"measurements": str(csv_file)}, results={},
                            method_params=DEFAULT_METHOD)
    return reg, csv_file


def test_a_set_aside_manifest_is_reported_before_the_next_registration(tmp_path, capsys):
    # Until the earlier entries are merged back, the registry holds fewer
    # entries than were registered: every reader says so, not only writers.
    reg, _ = _set_one_aside(tmp_path, '{"entries": {"a": {"registered_at": "x"}, "b"')
    capsys.readouterr()
    assert ar.main(["--name", "A", "--root", str(reg.registry_root)]) == 0
    out = capsys.readouterr().out
    assert "[!] 1 damaged manifest" in out
    assert reg.set_aside_files()[0].name in out

    AnalysisRegistry(analysis_name="A", registry_root=reg.registry_root).get_all_entries()
    out = capsys.readouterr().out
    assert "[!]" in out and reg.set_aside_files()[0].name in out


def test_the_next_manifest_records_each_set_aside_one_until_it_is_dealt_with(tmp_path, capsys):
    damaged = ('{"entries": {"a": {"registered_at": "x"}, "b": {"registered_at": "y"}, '
               '"c": {"regis')
    reg, csv_file = _set_one_aside(tmp_path, damaged)
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    header = json.loads(reg.registry_path.read_text(encoding="utf-8"))
    [record] = header["set_aside_manifests"]
    assert record["file"] == reg.set_aside_files()[0].name
    assert record["entries_estimate"] == 2
    assert record["set_aside_at"]
    assert record["size_bytes"] == len(damaged.encode("utf-8"))

    # After a person has repaired, merged and removed it, the next write
    # drops the record.
    reg.set_aside_files()[0].unlink()
    reg.register_output(sample="E01_01_S02_DCN", category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    header = json.loads(reg.registry_path.read_text(encoding="utf-8"))
    assert "set_aside_manifests" not in header


# ---------------------------------------------------------------------------
# A registration that is refused changes no file: the registry's files and
# the manifest entry describing them always come from the same run
# ---------------------------------------------------------------------------

def _work_leftovers(reg):
    """Working files left in logs/ (anything but the audit logs and a
    set-aside manifest)."""
    return sorted(p.name for p in reg.work_dir.iterdir()
                  if not p.name.endswith(".log")
                  and not p.name.startswith(reg.set_aside_prefix))


def _run(tmp_path, n):
    """A source CSV holding run *n*'s content."""
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    f = src / "measurements.csv"
    f.write_text("run\n%d\n" % n, encoding="utf-8")
    return f


def _register_run(reg, tmp_path, n, sample=SAMPLE):
    return reg.register_output(sample=sample, category="detection",
                               files={"measurements": str(_run(tmp_path, n))},
                               results={"run": n}, method_params={"k": n})


def _registered_csv(reg, sample=SAMPLE):
    return (reg.registry_root / reg.get_entry(sample)["outputs"]["measurements"]).read_text(
        encoding="utf-8")


def test_a_damaged_manifest_refuses_before_replacing_the_earlier_files(tmp_path):
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    dest = _register_run(reg, tmp_path, 1)["measurements"]
    good = reg.registry_path.read_text(encoding="utf-8")
    reg.registry_path.write_text(good[:-20], encoding="utf-8")  # cut off

    with pytest.raises(RuntimeError):
        _register_run(reg, tmp_path, 2)
    # The file is still run 1's, so the set-aside manifest (which says run 1)
    # describes it correctly once repaired and merged back.
    assert dest.read_text(encoding="utf-8") == "run\n1\n"
    assert _work_leftovers(reg) == []  # nothing staged or kept behind


def test_a_broken_approved_method_file_refuses_before_replacing_the_earlier_files(tmp_path):
    # Also for an analysis that does not follow the shared approved method:
    # a file that cannot be parsed might name it, so its header cannot be
    # resolved and nothing may change.
    root = tmp_path / "Registry"
    reg = AnalysisRegistry(analysis_name="A", registry_root=root,
                           follows_approved_method=False)
    _register_run(reg, tmp_path, 1)
    (root / "approved_method.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Approved-method file"):
        _register_run(reg, tmp_path, 2)
    (root / "approved_method.json").unlink()
    assert reg.get_entry(SAMPLE)["results"] == {"run": 1}
    assert _registered_csv(reg) == "run\n1\n"


def test_a_failed_manifest_write_puts_the_earlier_files_back(tmp_path, monkeypatch):
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    _register_run(reg, tmp_path, 1)
    real_write = AnalysisRegistry._write_registry

    def failing_write(self, data, *args, **kwargs):
        raise OSError("the share dropped out")

    monkeypatch.setattr(AnalysisRegistry, "_write_registry", failing_write)
    with pytest.raises(OSError):
        _register_run(reg, tmp_path, 2)
    with pytest.raises(OSError):  # a new sample: its file is removed again
        _register_run(reg, tmp_path, 2, sample="E01_01_S14_DCN")
    monkeypatch.setattr(AnalysisRegistry, "_write_registry", real_write)

    assert reg.get_entry(SAMPLE)["results"] == {"run": 1}
    assert _registered_csv(reg) == "run\n1\n"
    assert not (reg.exports_dir / "E01_01_S14_DCN" / "measurements.csv").exists()
    assert _work_leftovers(reg) == []  # nothing kept behind in logs/


def test_a_lock_that_never_comes_free_changes_no_file(tmp_path, monkeypatch):
    from mousebrain.file_lock import exclusive_file_lock
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path / "Registry")
    dest = _register_run(reg, tmp_path, 1)["measurements"]
    reg.lock_path.write_text("busy-host pid 2 token b\n", encoding="utf-8")  # a live holder
    monkeypatch.setattr(
        AnalysisRegistry, "_locked",
        lambda self: exclusive_file_lock(self.registry_path, "Registry", stale_seconds=120,
                                         wait_seconds=0.5, lock_path=self.lock_path))
    with pytest.raises(TimeoutError):
        _register_run(reg, tmp_path, 2)
    assert dest.read_text(encoding="utf-8") == "run\n1\n"


def test_refused_roi_counts_leave_the_earlier_counts_and_summary(tmp_path):
    reg = AnalysisRegistry(analysis_name="R", registry_root=tmp_path / "Registry")
    first = {"TOTAL": {"total": 10, "positive": 5, "negative": 5, "fraction": 0.5}}
    counts_path = reg.register_roi_counts(SAMPLE, "DCN", first, method_params={"k": 1})
    summary = reg.exports_dir / "roi_summary_DCN.csv"
    counts_before = counts_path.read_text(encoding="utf-8")
    summary_before = summary.read_text(encoding="utf-8")
    reg.registry_path.write_text("{cut off", encoding="utf-8")

    second = {"TOTAL": {"total": 99, "positive": 1, "negative": 98, "fraction": 0.01}}
    with pytest.raises(RuntimeError):
        reg.register_roi_counts(SAMPLE, "DCN", second, method_params={"k": 2})
    assert counts_path.read_text(encoding="utf-8") == counts_before
    assert summary.read_text(encoding="utf-8") == summary_before


def test_roi_counts_are_replaced_when_the_registration_succeeds(tmp_path):
    reg = AnalysisRegistry(analysis_name="R", registry_root=tmp_path / "Registry")
    reg.register_roi_counts(SAMPLE, "DCN",
                            {"TOTAL": {"total": 10, "positive": 5, "negative": 5,
                                       "fraction": 0.5}}, method_params={"k": 1})
    counts_path = reg.register_roi_counts(
        SAMPLE, "DCN", {"TOTAL": {"total": 4, "positive": 1, "negative": 3,
                                  "fraction": 0.25}}, method_params={"k": 2})
    assert "4,1,3,0.25" in counts_path.read_text(encoding="utf-8")
    summary = (reg.exports_dir / "roi_summary_DCN.csv").read_text(encoding="utf-8")
    assert "4,1,3,0.25" in summary and "10,5,5" not in summary
    assert reg.get_entry(SAMPLE + "__roi_counts")["method_params"] == {"k": 2}
    assert _work_leftovers(reg) == []


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
    monkeypatch.setattr(AnalysisRegistry, "REPLACE_RETRY_SECONDS", 0.3, raising=False)
    stuck = dests["E01_01_S02_DCN"]["measurements"]
    real_rename = ar.os.rename

    def move(src, dst, *a, **k):
        if Path(src) == stuck:
            raise PermissionError(13, "held open by another program")
        return real_rename(src, dst, *a, **k)

    monkeypatch.setattr(ar.os, "rename", move)
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
    real_rename = ar.os.rename

    def move(src, dst, *a, **k):
        if Path(src) == second:
            raise KeyboardInterrupt  # the process stops part-way through the pass
        return real_rename(src, dst, *a, **k)

    monkeypatch.setattr(ar.os, "rename", move)
    with pytest.raises(KeyboardInterrupt):
        reg.invalidate()

    # The first entry's files were archived AND the manifest says so.
    for path in dests["E01_01_S01_DCN"].values():
        assert not path.exists()
    on_disk = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"]
    assert on_disk["E01_01_S01_DCN"]["is_current"] is False
    # The interrupted entry: its figure had already moved when the stop came.
    # It is put back, so every file is where the manifest (current) says.
    assert on_disk["E01_01_S02_DCN"]["is_current"] is True
    for path in dests["E01_01_S02_DCN"].values():
        assert path.is_file(), path


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to move a file held open")
def test_a_file_held_open_is_left_in_place_with_no_copy_in_the_archive(tmp_path, monkeypatch):
    # A file another program has open cannot be renamed on Windows. Moving it
    # with a copy-then-delete fallback copied it into _archived/ and then
    # failed to delete the original: the entry rightly stayed current, but a
    # stray copy sat in _archived/ for ever, mirrored as an archived result.
    reg, dests = _register_two(tmp_path)
    monkeypatch.setattr(AnalysisRegistry, "REPLACE_RETRY_SECONDS", 0.3, raising=False)
    held = dests["E01_01_S02_DCN"]["measurements"]
    with open(held, "rb"):  # a real handle, as a spreadsheet program holds one
        try:
            reg.invalidate()
        except OSError:
            pass
    stray = sorted(p for p in held.parent.rglob("*") if p.is_file() and p.name == held.name
                   and p != held)
    assert stray == [], "a copy of the held file was left in the archive: %s" % stray
    assert held.is_file()
    on_disk = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"]
    assert on_disk["E01_01_S02_DCN"]["is_current"] is True
    for path in dests["E01_01_S02_DCN"].values():
        assert path.is_file(), path


def test_invalidate_leaves_a_damaged_manifest_where_it_is(tmp_path):
    # invalidate records nothing of its own, so it must not start the
    # "no manifest" window a set-aside opens: it refuses and moves nothing.
    reg, dests = _register_two(tmp_path)
    reg.registry_path.write_text("{cut off", encoding="utf-8")
    with pytest.raises(RuntimeError):
        reg.invalidate()
    assert reg.registry_path.read_text(encoding="utf-8") == "{cut off"
    assert reg.set_aside_files() == []
    for files in dests.values():
        for path in files.values():
            assert path.is_file(), path


# ---------------------------------------------------------------------------
# An approved method per analysis; analyses with none are never "stale"
# ---------------------------------------------------------------------------


def _stale_report(root, name, capsys):
    capsys.readouterr()
    assert ar.main(["--name", name, "--root", str(root), "--stale"]) == 0
    return capsys.readouterr().out


def test_an_analysis_named_in_the_approved_file_is_judged_by_its_own_method(
        tmp_path, capsys):
    own = {"atlas": "atlas_x", "ball_xy": 6, "threshold": 10}
    (tmp_path / "approved_method.json").write_text(
        json.dumps({"analyses": {"B": own, "C": None}}), encoding="utf-8")
    csv_file, _ = _make_source_files(tmp_path)

    reg = AnalysisRegistry(analysis_name="B", registry_root=tmp_path)
    reg.register_output(sample=SAMPLE, category="region_counts",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=own)
    assert reg._read_registry()["approved_method_hash"] == AnalysisRegistry.get_method_hash(own)
    assert "All entries are current." in _stale_report(tmp_path, "B", capsys)

    reg.register_output(sample="E01_01_S14_DCN", category="region_counts",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=dict(own, threshold=12))
    assert "1 stale entries" in _stale_report(tmp_path, "B", capsys)

    # null in the file: this analysis has no approved method at all.
    reg_c = AnalysisRegistry(analysis_name="C", registry_root=tmp_path)
    reg_c.register_output(sample=SAMPLE, category="detection",
                          files={"measurements": str(csv_file)}, results={},
                          method_params=DEFAULT_METHOD)
    assert reg_c._read_registry()["approved_method_hash"] is None
    assert "No approved method is declared" in _stale_report(tmp_path, "C", capsys)

    # An analysis the file does not name follows the shared method, which
    # is the built-in default when the file gives no "default".
    assert get_approved_method(tmp_path) == DEFAULT_METHOD
    assert ar.approved_method_for("D", tmp_path) == DEFAULT_METHOD
    assert ar.approved_method_for("D", tmp_path, follows_approved_method=False) is None


def test_a_malformed_per_analysis_table_is_refused(tmp_path):
    (tmp_path / "approved_method.json").write_text(
        json.dumps({"analyses": {"B": []}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pproved-method"):
        ar.approved_method_for("B", tmp_path)


def test_a_malformed_entry_for_another_analysis_stops_only_the_analyses_it_concerns(tmp_path):
    # Every registration resolves its analysis's approved method. A fault in
    # a part of the file an analysis is not judged by used to stop the
    # registrations of every analysis, the 3D ones included.
    root = tmp_path / "Registry"
    (root).mkdir()
    (root / "approved_method.json").write_text(json.dumps({
        "default": {"detection": "x"},
        "analyses": {"Broken": [], "Own": {"p": 1}}}), encoding="utf-8")
    own_free = AnalysisRegistry(analysis_name="ThreeD", registry_root=root,
                                follows_approved_method=False)
    csv_file, _ = _make_source_files(tmp_path)
    own_free.register_output(sample=SAMPLE, category="detection",
                             files={"measurements": str(csv_file)}, results={},
                             method_params={"atlas": "a"})
    assert own_free.get_entry(SAMPLE) is not None
    assert json.loads(own_free.registry_path.read_text(encoding="utf-8"))[
        "approved_method_hash"] is None
    assert ar.approved_method_for("Own", root) == {"p": 1}

    # The analysis the broken entry belongs to, and every analysis judged by
    # the shared method, still stop with the error.
    with pytest.raises(RuntimeError, match="Broken"):
        ar.approved_method_for("Broken", root, follows_approved_method=False)
    with pytest.raises(RuntimeError, match="pproved-method"):
        ar.approved_method_for("Slices", root, follows_approved_method=True)

    # A file that cannot be parsed might name any analysis: it stops them all.
    (root / "approved_method.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pproved-method"):
        ar.approved_method_for("ThreeD", root, follows_approved_method=False)


# ---------------------------------------------------------------------------
# The manifest's approved-method header follows the approved method
# ---------------------------------------------------------------------------

def _header(reg):
    return json.loads(reg.registry_path.read_text(encoding="utf-8"))


def test_the_header_follows_a_changed_approved_method_on_the_next_write(tmp_path):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    assert _header(reg)["approved_method_hash"] == AnalysisRegistry.get_method_hash(DEFAULT_METHOD)

    # The lab approves a new method; the next registration records it in the
    # header, so an integrator counting "stale vs approved" from the header
    # now sees the first entry as stale.
    changed = dict(DEFAULT_METHOD, soma_dilation=8)
    (tmp_path / "approved_method.json").write_text(json.dumps(changed), encoding="utf-8")
    reg.register_output(sample="E01_01_S14_DCN", category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=changed)

    header = _header(reg)
    assert header["approved_method"] == changed
    assert header["approved_method_hash"] == AnalysisRegistry.get_method_hash(changed)
    assert "approved_method_recorded_at" in header
    stale_by_header = [k for k, e in header["entries"].items()
                       if e["is_current"] and e["method_hash"] != header["approved_method_hash"]]
    assert stale_by_header == [SAMPLE]


def test_refresh_approved_rewrites_the_header_without_registering(tmp_path, capsys):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    changed = dict(DEFAULT_METHOD, soma_dilation=8)
    (tmp_path / "approved_method.json").write_text(json.dumps(changed), encoding="utf-8")
    entries_before = _header(reg)["entries"]

    assert ar.main(["--name", "A", "--root", str(tmp_path), "--refresh-approved"]) == 0
    assert "[OK]" in capsys.readouterr().out
    header = _header(reg)
    assert header["approved_method_hash"] == AnalysisRegistry.get_method_hash(changed)
    assert header["entries"] == entries_before

    # An analysis with no manifest is left alone, not created.
    assert ar.main(["--name", "Z", "--root", str(tmp_path), "--refresh-approved"]) == 0
    assert not (tmp_path / "exports" / "Z" / "registry.json").exists()


def test_a_producer_declaration_reaches_an_existing_manifest(tmp_path):
    # A manifest written before analyses declared whether they follow the
    # shared method: the producer's next write records its declaration.
    csv_file, _ = _make_source_files(tmp_path)
    old = AnalysisRegistry(analysis_name="B", registry_root=tmp_path)
    old.register_output(sample=SAMPLE, category="qc",
                        files={"measurements": str(csv_file)}, results={},
                        method_params={"check": "x", "version": "1"})
    assert _header(old)["approved_method_hash"] is not None

    producer = AnalysisRegistry(analysis_name="B", registry_root=tmp_path,
                                follows_approved_method=False)
    producer.register_output(sample="E01_01_S14_DCN", category="qc",
                             files={"measurements": str(csv_file)}, results={},
                             method_params={"check": "x", "version": "1"})
    assert _header(old)["approved_method_hash"] is None
    # ...and a reader that declares nothing keeps it.
    assert AnalysisRegistry(analysis_name="B", registry_root=tmp_path).approved_method() is None


def test_a_file_holding_only_a_default_is_the_per_analysis_form(tmp_path):
    # {"default": {...}} with no "analyses" table used to be read as the flat
    # form: a method with one parameter named "default", so every entry of
    # every analysis following the shared method showed as stale.
    shared = dict(DEFAULT_METHOD, soma_dilation=8)
    (tmp_path / "approved_method.json").write_text(
        json.dumps({"default": shared}), encoding="utf-8")
    assert get_approved_method(tmp_path) == shared
    assert ar.approved_method_for("A", tmp_path) == shared
    assert ar.approved_method_for("A", tmp_path, follows_approved_method=False) is None

    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=shared)
    assert reg.check_staleness(reg.approved_method()) == []
    assert _header(reg)["approved_method_hash"] == AnalysisRegistry.get_method_hash(shared)


def test_a_default_that_is_not_a_set_of_parameters_is_refused(tmp_path):
    (tmp_path / "approved_method.json").write_text(
        json.dumps({"default": "threshold+log"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pproved-method"):
        get_approved_method(tmp_path)


# ---------------------------------------------------------------------------
# invalidate(): a manifest write that fails puts the entry's files back
# ---------------------------------------------------------------------------

def test_a_failed_manifest_write_puts_the_files_back_and_keeps_the_entry_current(
        tmp_path, monkeypatch):
    reg, dests = _register_two(tmp_path)
    monkeypatch.setattr(AnalysisRegistry, "REPLACE_RETRY_SECONDS", 0.3, raising=False)
    real_replace = ar.os.replace

    def replace(src, dst, *a, **k):
        # registry.json held open by a reader for longer than the retry time
        if Path(dst).name == "registry.json":
            raise PermissionError(13, "held open by another program")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(ar.os, "replace", replace)
    with pytest.raises(PermissionError):
        reg.invalidate()

    monkeypatch.setattr(ar.os, "replace", real_replace)
    on_disk = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"]
    assert all(e["is_current"] for e in on_disk.values())
    # The files of every entry the manifest still calls current are in the
    # active path, not in _archived/.
    for paths in dests.values():
        for path in paths.values():
            assert path.is_file(), path
    log = (reg.logs_dir / "A.log").read_text(encoding="utf-8")
    assert "invalidate_failed" in log


def test_a_broken_approved_method_file_stops_invalidate_before_anything_moves(tmp_path):
    reg, dests = _register_two(tmp_path)
    (reg.registry_root / "approved_method.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pproved-method"):
        reg.invalidate()
    on_disk = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"]
    assert all(e["is_current"] for e in on_disk.values())
    for paths in dests.values():
        for path in paths.values():
            assert path.is_file(), path


def test_a_manifest_held_open_briefly_is_written_on_a_retry(tmp_path, monkeypatch):
    reg, dests = _register_two(tmp_path)
    real_replace = ar.os.replace
    refusals = {"left": 2}

    def replace(src, dst, *a, **k):
        if Path(dst).name == "registry.json" and refusals["left"]:
            refusals["left"] -= 1
            raise PermissionError(13, "held open by another program")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(ar.os, "replace", replace)
    assert sorted(reg.invalidate()) == sorted(dests)
    on_disk = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"]
    assert not any(e["is_current"] for e in on_disk.values())


# ---------------------------------------------------------------------------
# The command line can record an analysis's declaration in an old manifest
# ---------------------------------------------------------------------------

def test_the_cli_records_that_an_old_manifest_has_no_approved_method(tmp_path, capsys):
    # A manifest written before analyses declared anything follows the shared
    # method, and --refresh-approved alone keeps that; --no-approved-method
    # records the declaration without waiting for the producer's next run.
    csv_file, _ = _make_source_files(tmp_path)
    old = AnalysisRegistry(analysis_name="B", registry_root=tmp_path)
    old.register_output(sample=SAMPLE, category="qc",
                        files={"measurements": str(csv_file)}, results={},
                        method_params={"check": "x", "version": "1"})
    assert "1 stale entries" in _stale_report(tmp_path, "B", capsys)

    assert ar.main(["--name", "B", "--root", str(tmp_path), "--refresh-approved"]) == 0
    assert _header(old)["approved_method_hash"] is not None

    assert ar.main(["--name", "B", "--root", str(tmp_path), "--no-approved-method"]) == 0
    header = _header(old)
    assert header["approved_method_hash"] is None
    assert header["follows_approved_method"] is False
    assert "No approved method is declared" in _stale_report(tmp_path, "B", capsys)

    assert ar.main(["--name", "B", "--root", str(tmp_path),
                    "--follows-approved-method"]) == 0
    assert _header(old)["approved_method_hash"] is not None


def test_the_stale_report_names_a_broken_approved_method_file(tmp_path, capsys):
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    (tmp_path / "approved_method.json").write_text("{not json", encoding="utf-8")
    capsys.readouterr()
    assert ar.main(["--name", "A", "--root", str(tmp_path), "--stale"]) == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_the_header_records_where_the_approved_method_came_from(tmp_path, monkeypatch):
    # The environment variable wins over approved_method.json and can differ
    # per machine; since the header is rewritten on every write, it records
    # the source and the machine so a flip between two values is visible.
    csv_file, _ = _make_source_files(tmp_path)
    reg = AnalysisRegistry(analysis_name="A", registry_root=tmp_path)
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)}, results={},
                        method_params=DEFAULT_METHOD)
    assert _header(reg)["approved_method_source"] == "built-in parameters"
    assert _header(reg)["approved_method_recorded_on"] == ar._get_hostname()

    (tmp_path / "approved_method.json").write_text(json.dumps(DEFAULT_METHOD),
                                                   encoding="utf-8")
    reg.refresh_approved_method()
    assert _header(reg)["approved_method_source"] == str(tmp_path / "approved_method.json")

    elsewhere = tmp_path / "machine_override.json"
    elsewhere.write_text(json.dumps(DEFAULT_METHOD), encoding="utf-8")
    monkeypatch.setenv("MOUSEBRAIN_APPROVED_METHOD", str(elsewhere))
    reg.refresh_approved_method()
    assert _header(reg)["approved_method_source"] == "MOUSEBRAIN_APPROVED_METHOD=%s" % elsewhere
