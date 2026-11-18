"""A completed sample whose result could not be registered is listed in the
end-of-batch summary, and so is a registry that could not be opened at all.

Defect covered: when registering a completed sample raised (the registry
file held open by another program, the share dropping out), the only trace
was one "Registry warning" line among that sample's own output. The summary
still counted the sample under "Completed", so a person reading it took the
result for registered while the registry still held the earlier run's count
-- or nothing. The same held when the registry could not be opened: every
sample was processed and none was registered, with one early warning line.
"""

import sys

import pytest

from example_method import write_example_approved_method
from mousebrain import analysis_registry as ar
from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be

PROJECT = "PRJ"
SAMPLE = "PRJ_01_02_S3_RA"
OTHER = "PRJ_01_02_S4_RA"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    # The batch has no built-in values: it needs an approved method to run.
    write_example_approved_method(tmp_path / "registry")
    monkeypatch.setattr(be, "DATA_DIR", tmp_path / "1_Subjects")
    monkeypatch.setattr(be, "SLICES_2D_DIR", tmp_path)
    folder = tmp_path / PROJECT / f"{PROJECT}_01_02_HD_Regions"
    folder.mkdir(parents=True)
    for name in (SAMPLE, OTHER):
        (folder / f"{name}.nd2").write_bytes(b"")
    # Every sample completes; this test is about what happens afterwards.
    monkeypatch.setattr(
        be, "process_single",
        lambda nd2_path, *a, **k: {
            "file": nd2_path.name, "path": str(nd2_path), "status": "completed",
            "n_nuclei": 3, "n_positive": 1, "positive_fraction": 0.3333,
            "background": 1.0})
    return tmp_path


def _summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT,
                                      "--output", str(tmp_path / "out"),
                                      "--ignore-tracker"])
    be.main()
    return capsys.readouterr().out.split("BATCH COMPLETE")[1]


def test_a_completed_sample_that_could_not_be_registered_is_listed(env, monkeypatch, capsys):
    tmp_path = env
    reg = AnalysisRegistry(analysis_name="PRJ_Detection",
                           registry_root=tmp_path / "registry")
    reg.register_output(
        sample=SAMPLE, category="detection", files={},
        results={"n_nuclei": 28, "n_positive": 15, "positive_fraction": 0.536},
        method_params={"sigma_threshold": 0}, source_files={})
    real = ar.AnalysisRegistry.register_output

    def in_use_for_one(self, sample, *a, **k):
        if sample == SAMPLE:
            raise PermissionError("registry.json is in use")
        return real(self, sample, *a, **k)
    monkeypatch.setattr(ar.AnalysisRegistry, "register_output", in_use_for_one)

    summary = _summary(monkeypatch, tmp_path, capsys)

    # The earlier entry is untouched, and the summary says which one stands.
    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] is True
    assert entry["results"]["n_nuclei"] == 28
    assert (f"[!] {SAMPLE}: completed, but this run's result was NOT "
            "registered (registry.json is in use)") in summary
    assert "is still the current one" in summary
    # Only the sample it happened to; the other one registered.
    assert f"[!] {OTHER}" not in summary
    assert reg.get_entry(OTHER)["results"]["n_nuclei"] == 3


def test_a_registry_that_cannot_be_opened_is_said_in_the_summary(env, monkeypatch, capsys):
    tmp_path = env

    def cannot_open(self, *a, **k):
        raise OSError("the registry folder cannot be reached")
    monkeypatch.setattr(ar.AnalysisRegistry, "__init__", cannot_open)

    summary = _summary(monkeypatch, tmp_path, capsys)

    assert "[!] Registry: could not be opened" in summary
    assert "the registry folder cannot be reached" in summary
    assert "NO result of this run was registered" in summary
    # The samples were still processed and written to the summary file.
    assert (tmp_path / "out" / "summary.csv").exists()


def test_a_run_that_registers_everything_has_neither_line(env, monkeypatch, capsys):
    summary = _summary(monkeypatch, env, capsys)
    assert "NOT registered" not in summary
    assert "could not be opened" not in summary
