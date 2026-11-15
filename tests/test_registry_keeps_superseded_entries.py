"""Re-registering an invalidated sample keeps the invalidated record, and the
reason it was invalidated is written into the manifest.

Defect covered: register_output replaced the sample's entry outright, so the
invalidated record (its results, method and when it was retired) vanished from
registry.json; only the moved files and an audit-log line were left.
"""

from mousebrain.analysis_registry import AnalysisRegistry

SAMPLE = "Q01_01_S1_RA"
METHOD = {"detection": "threshold+log", "sigma_threshold": 0}


def _register(reg, tmp_path, n, tag):
    f = tmp_path / f"{tag}.csv"
    f.write_text("label\n1\n", encoding="utf-8")
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(f)},
                        results={"n_nuclei": n}, method_params=METHOD)


def test_reregistering_an_invalidated_sample_keeps_its_old_record(tmp_path):
    reg = AnalysisRegistry(analysis_name="Q_Detection", registry_root=tmp_path / "reg")
    _register(reg, tmp_path, 12, "first")
    reg.invalidate(sample=SAMPLE, reason="re-run found no nuclei")
    _register(reg, tmp_path, 0, "second")

    entry = reg.get_entry(SAMPLE)
    assert entry["is_current"] and entry["results"]["n_nuclei"] == 0
    old = entry.get("superseded", [])
    assert len(old) == 1
    assert old[0]["results"]["n_nuclei"] == 12
    assert old[0]["is_current"] is False
    assert old[0]["invalidated_reason"] == "re-run found no nuclei"


def test_overwriting_a_current_entry_keeps_no_history(tmp_path):
    reg = AnalysisRegistry(analysis_name="Q_Detection", registry_root=tmp_path / "reg")
    _register(reg, tmp_path, 12, "first")
    _register(reg, tmp_path, 13, "second")
    assert "superseded" not in reg.get_entry(SAMPLE)


def _register_content(reg, tmp_path, n, content):
    f = tmp_path / f"src_{n}.csv"
    f.write_text(content, encoding="utf-8")
    reg.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(f)},
                        results={"n": n}, method_params=METHOD)


def test_kept_record_points_at_its_own_archived_files(tmp_path):
    # invalidate() moves the files to _archived/; registering the sample again
    # writes NEW files at the old active path. The kept record must name the
    # archived files, not the active path that now holds the new result.
    reg = AnalysisRegistry(analysis_name="Q_Detection", registry_root=tmp_path / "reg")
    _register_content(reg, tmp_path, 1, "old")
    reg.invalidate(sample=SAMPLE)
    invalidated = reg.get_entry(SAMPLE)
    assert set(invalidated["archived_outputs"]) == {"measurements"}
    _register_content(reg, tmp_path, 2, "new")

    entry = reg.get_entry(SAMPLE)
    old = entry["superseded"][0]
    assert old["results"] == {"n": 1}
    assert old["outputs"], "the kept record names no files"
    for rel in old["outputs"].values():
        path = reg.registry_root / rel
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "old"
    for rel in entry["outputs"].values():
        assert (reg.registry_root / rel).read_text(encoding="utf-8") == "new"


def test_roi_counts_keep_an_invalidated_record_too(tmp_path):
    reg = AnalysisRegistry(analysis_name="Q_ROI", registry_root=tmp_path / "reg")
    rows = {"TOTAL": {"total": 4, "positive": 2, "negative": 2, "fraction": 0.5}}
    reg.register_roi_counts(sample=SAMPLE, region="RA", roi_results=rows,
                            method_params=METHOD)
    reg.invalidate(sample=SAMPLE, reason="ROIs redrawn")
    rows2 = {"TOTAL": {"total": 6, "positive": 3, "negative": 3, "fraction": 0.5}}
    reg.register_roi_counts(sample=SAMPLE, region="RA", roi_results=rows2,
                            method_params=METHOD)
    entry = reg.get_all_entries()[f"{SAMPLE}__roi_counts"]
    old = entry["superseded"][0]
    assert old["results"]["TOTAL"]["total"] == 4
    assert old["invalidated_reason"] == "ROIs redrawn"
    for rel in old["outputs"].values():
        text = (reg.registry_root / rel).read_text(encoding="utf-8")
        assert "4" in text.splitlines()[1]
