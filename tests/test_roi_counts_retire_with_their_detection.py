"""ROI counts are retired when the detection they were counted against changes.

Defect covered: the ROI counter ties a sample's counts to the measurements
it counted (their SHA-256) only at counting time. When the detection batch
then registered NEW measurements for the sample, found no nuclei in it, or
retired its detection entry after a failure, the sample's ROI counts -- and
its row in the region summary -- stayed current until someone counted
again. Now the batch retires them at that moment, unless the new
measurements are byte-for-byte the ones counted.
"""
import csv
import hashlib

from mousebrain.analysis_registry import AnalysisRegistry
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be

# The batch fixture and helpers of the failed re-run tests.
from test_encr_batch_failed_rerun_retires_entry import (  # noqa: F401
    SAMPLE, _detects_no_nuclei_in, _earlier_entry, _loads, _other_sample_succeeds,
    _raise, _run, det_mod, env, io_mod)

EARLIER = "label,is_positive\n1,True\n"


def _counted_against_the_earlier_measurements(tmp_path):
    """The sample's detection entry (measurements EARLIER) and ROI counts
    registered against exactly those measurements, the way the ROI counter
    registers them."""
    det, det_file = _earlier_entry(tmp_path)
    assert det_file.read_text(encoding="utf-8") == EARLIER
    sha = hashlib.sha256(det_file.read_bytes()).hexdigest()
    roi = AnalysisRegistry(analysis_name="PRJ_ROI_Analysis",
                           registry_root=tmp_path / "registry")
    roi.register_roi_counts(
        SAMPLE, "RA", {"TOTAL": {"total": 7, "positive": 3, "negative": 4,
                                 "fraction": 0.4286}},
        method_params={"sigma_threshold": 0},
        source_files={"measurements": str(det_file), "measurements_sha256": sha})
    return det, roi


def _summary_samples(roi):
    path = roi.exports_dir / "roi_summary_RA.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return [row["sample"] for row in csv.DictReader(f)]


def _rerun_completes_with(monkeypatch, measurements_text):
    """The sample's re-run completes and writes these measurements."""
    def process(nd2_path, output_dir, **k):
        folder = output_dir / nd2_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "measurements.csv").write_text(measurements_text, encoding="utf-8")
        return {"file": nd2_path.name, "path": str(nd2_path), "status": "completed",
                "n_nuclei": 2, "n_positive": 1, "positive_fraction": 0.5,
                "background": 1.0}
    monkeypatch.setattr(be, "process_single", process)


def test_new_measurements_retire_the_counts_made_from_the_old_ones(env, monkeypatch, capsys):
    t, tmp_path, img = env
    _, roi = _counted_against_the_earlier_measurements(tmp_path)
    assert _summary_samples(roi) == [SAMPLE]
    _rerun_completes_with(monkeypatch, "label,is_positive\n1,True\n2,False\n")
    _run(monkeypatch, tmp_path)

    entry = roi.get_entry(SAMPLE + "__roi_counts")
    assert entry["is_current"] is False
    assert _summary_samples(roi) == []
    assert f"{SAMPLE}: ROI counts in PRJ_ROI_Analysis retired" in capsys.readouterr().out


def test_the_same_measurements_leave_the_counts_current(env, monkeypatch):
    t, tmp_path, img = env
    _, roi = _counted_against_the_earlier_measurements(tmp_path)
    _rerun_completes_with(monkeypatch, EARLIER)
    _run(monkeypatch, tmp_path)
    assert roi.get_entry(SAMPLE + "__roi_counts")["is_current"] is True
    assert _summary_samples(roi) == [SAMPLE]


def test_a_retired_detection_retires_its_counts(env, monkeypatch):
    t, tmp_path, img = env
    det, roi = _counted_against_the_earlier_measurements(tmp_path)
    _other_sample_succeeds(env, monkeypatch)
    _loads(monkeypatch, img)
    monkeypatch.setattr(det_mod, "detect_with_log_augmentation",
                        _raise(ValueError("image must be 2D")))
    _run(monkeypatch, tmp_path)
    assert det.get_entry(SAMPLE)["is_current"] is False
    assert roi.get_entry(SAMPLE + "__roi_counts")["is_current"] is False
    assert _summary_samples(roi) == []


def test_a_rerun_with_no_nuclei_retires_the_counts(env, monkeypatch):
    t, tmp_path, img = env
    _, roi = _counted_against_the_earlier_measurements(tmp_path)
    _loads(monkeypatch, img)
    _detects_no_nuclei_in(monkeypatch, img, SAMPLE)
    _run(monkeypatch, tmp_path)
    assert roi.get_entry(SAMPLE + "__roi_counts")["is_current"] is False
    assert _summary_samples(roi) == []


def test_invalidating_roi_counts_takes_them_out_of_the_region_summary(tmp_path):
    roi = AnalysisRegistry(analysis_name="PRJ_ROI_Analysis", registry_root=tmp_path / "r")
    for sample, total in ((SAMPLE, 7), ("PRJ_01_02_S4_RA", 5)):
        roi.register_roi_counts(sample, "RA", {"TOTAL": {"total": total, "positive": 1,
                                                         "negative": total - 1,
                                                         "fraction": 0.2}},
                                method_params={"k": 1})
    roi.invalidate(sample=SAMPLE)
    assert _summary_samples(roi) == ["PRJ_01_02_S4_RA"]
