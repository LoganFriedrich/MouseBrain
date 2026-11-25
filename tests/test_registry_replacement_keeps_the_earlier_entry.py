"""Registering a sample again keeps the entry it replaces.

A completed re-run of a sample (other parameters, say) replaces its registry
entry but must not DELETE the files it replaces: the earlier counts,
parameters and files would then be kept nowhere (the audit log would get only
a "register" line with the new hash), whereas retiring an entry through
invalidate keeps both. So a replacement moves the earlier files to
_archived/<time>/ beside them, writes a "replace_entry" log line holding the
earlier entry's record, and the new entry says what it replaced. A
registration identical to the current entry replaces nothing.
"""
import json

from mousebrain.analysis_registry import AnalysisRegistry

SAMPLE = "PRJ_01_02_S3_RA"


def _csv(tmp_path, name, text):
    path = tmp_path / "src" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _log_lines(reg, action):
    log = reg.logs_dir / f"{reg.analysis_name}.log"
    return [line.rstrip("\n").split("\t") for line in log.read_text(encoding="utf-8").splitlines()
            if line.split("\t")[2] == action]


def _archived(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and "_archived" in p.parts)


def test_a_rerun_with_other_parameters_keeps_the_earlier_files_and_record(tmp_path):
    reg = AnalysisRegistry(analysis_name="PRJ_Detection", registry_root=tmp_path / "Registry")
    first = _csv(tmp_path, "measurements.csv", "label,is_positive\n1,True\n2,True\n")
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(first)},
                        results={"n_nuclei": 2}, method_params={"threshold": 2.0})
    second = _csv(tmp_path, "measurements.csv", "label,is_positive\n1,False\n")
    dests = reg.register_output(sample=SAMPLE, category="detection",
                                files={"measurements": str(second)},
                                results={"n_nuclei": 1}, method_params={"threshold": 1.5})

    # The new file is in place; the earlier one is archived beside it, not deleted.
    assert dests["measurements"].read_text(encoding="utf-8") == "label,is_positive\n1,False\n"
    archived = _archived(dests["measurements"].parent)
    assert [p.read_text(encoding="utf-8") for p in archived] == [
        "label,is_positive\n1,True\n2,True\n"]

    # The earlier counts and parameters are in the audit log.
    lines = _log_lines(reg, "replace_entry")
    assert len(lines) == 1
    record = json.loads(lines[0][5])
    assert record["results"] == {"n_nuclei": 2}
    assert record["method_params"] == {"threshold": 2.0}
    assert record["archived_to"].startswith("_archived/")

    # The new entry says what it replaced.
    entry = reg.get_entry(SAMPLE)
    assert entry["results"] == {"n_nuclei": 1}
    assert entry["replaced_entry"]["results"] == {"n_nuclei": 2}
    assert entry["replaced_entry"]["archived_to"] == record["archived_to"]


def test_an_earlier_file_the_new_entry_does_not_list_leaves_the_active_folder(tmp_path):
    reg = AnalysisRegistry(analysis_name="PRJ_Detection", registry_root=tmp_path / "Registry")
    csv_file = _csv(tmp_path, "measurements.csv", "label\n1\n")
    overlay = tmp_path / "src" / "overlay.png"
    overlay.write_bytes(b"png")
    dests = reg.register_output(sample=SAMPLE, category="detection",
                                files={"measurements": str(csv_file), "overlay": str(overlay)},
                                results={"n_nuclei": 1}, method_params={"threshold": 2.0})
    csv_file.write_text("label\n1\n2\n", encoding="utf-8")
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(csv_file)},
                        results={"n_nuclei": 2}, method_params={"threshold": 1.5})
    folder = dests["measurements"].parent
    active = sorted(p.name for p in folder.iterdir() if p.is_file())
    assert active == ["measurements.csv"]
    assert sorted(p.name for p in _archived(folder)) == ["measurements.csv", "overlay.png"]


def test_an_identical_registration_archives_nothing(tmp_path):
    reg = AnalysisRegistry(analysis_name="PRJ_Window", registry_root=tmp_path / "Registry")
    csv_file = _csv(tmp_path, "measurements.csv", "label\n1\n")
    for _ in range(3):
        dests = reg.register_output(sample=SAMPLE, category="detection",
                                    files={"measurements": str(csv_file)},
                                    results={"n_nuclei": 1}, method_params={"t": 1})
    assert _archived(dests["measurements"].parent) == []
    assert _log_lines(reg, "replace_entry") == []
    assert "replaced_entry" not in reg.get_entry(SAMPLE)


def test_replaced_roi_counts_are_archived_and_recorded(tmp_path):
    reg = AnalysisRegistry(analysis_name="PRJ_ROI_Analysis", registry_root=tmp_path / "Registry")
    reg.register_roi_counts(SAMPLE, "RA", {"TOTAL": {"total": 10, "positive": 5,
                                                     "negative": 5, "fraction": 0.5}},
                            method_params={"k": 1})
    counts = reg.register_roi_counts(SAMPLE, "RA", {"TOTAL": {"total": 4, "positive": 1,
                                                              "negative": 3, "fraction": 0.25}},
                                     method_params={"k": 2})
    archived = _archived(counts.parent)
    assert len(archived) == 1 and "10,5,5,0.5" in archived[0].read_text(encoding="utf-8")
    record = json.loads(_log_lines(reg, "replace_entry")[0][5])
    assert record["results"]["TOTAL"]["total"] == 10
    # The region summary is derived: it shows the new counts only.
    summary = (reg.exports_dir / "roi_summary_RA.csv").read_text(encoding="utf-8")
    assert "4,1,3,0.25" in summary and "10,5,5" not in summary
