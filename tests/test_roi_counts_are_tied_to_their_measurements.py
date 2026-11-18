"""ROI counts record the detection run whose measurements they were counted against.

They used to be registered with the APPROVED method and without the
measurements file, so a re-detection (new measurements, new method) left the
old counts looking current.
"""

from example_method import write_example_approved_method
from mousebrain.analysis_registry import AnalysisRegistry, get_approved_method
from mousebrain.plugin_2d.sliceatlas.cli import run_roi

from roi_cli_helpers import (
    PROJECT, SAMPLE, batch_args, make_dataset, pin_registry, rois_of,
    write_measurements, write_rois,
)

# A detection method that is NOT the approved one.
DETECTION_METHOD = {"detection": "invented-method", "threshold": 7}


def _registered_detection(tmp_path, reg):
    src = write_measurements(tmp_path / "batch_out" / SAMPLE / "measurements.csv")
    det = AnalysisRegistry(f"{PROJECT}_Detection", registry_root=reg,
                           follows_approved_method=True)
    det.register_output(sample=SAMPLE, category="detection",
                        files={"measurements": str(src)}, results={"n_nuclei": 3},
                        method_params=DETECTION_METHOD, source_files={"nd2": "x.nd2"})
    return det


def _roi_entry(reg):
    return AnalysisRegistry(f"{PROJECT}_ROI_Analysis", registry_root=reg).get_entry(
        f"{SAMPLE}__roi_counts")


def test_counts_carry_the_detection_runs_method_and_name_the_measurements(tmp_path, monkeypatch):
    reg = pin_registry(monkeypatch, tmp_path)
    det = _registered_detection(tmp_path, reg)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))

    assert run_roi.cmd_batch(batch_args(base, count_only=True)) == 0

    entry = _roi_entry(reg)
    assert entry["method_hash"] == det.get_entry(SAMPLE)["method_hash"]
    assert entry["method_params"] == DETECTION_METHOD
    assert entry["source_files"]["measurements"].endswith("measurements.csv")
    assert len(entry["source_files"]["measurements_sha256"]) == 64


def test_counts_on_a_detection_that_is_not_approved_show_as_stale(tmp_path, monkeypatch):
    reg = pin_registry(monkeypatch, tmp_path)
    write_example_approved_method(reg)   # something to be stale against
    _registered_detection(tmp_path, reg)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))

    run_roi.cmd_batch(batch_args(base, count_only=True))

    roi_reg = AnalysisRegistry(f"{PROJECT}_ROI_Analysis", registry_root=reg)
    stale = roi_reg.get_stale_samples(get_approved_method(reg))
    assert stale == [SAMPLE]


def test_measurements_that_differ_from_the_registered_ones_are_not_tied(tmp_path, monkeypatch, capsys):
    reg = pin_registry(monkeypatch, tmp_path)
    write_example_approved_method(reg)   # something to be stale against
    det = _registered_detection(tmp_path, reg)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))
    # The file counted is not the one the detection run registered.
    other = tmp_path / "elsewhere"
    write_measurements(other / SAMPLE / "measurements.csv", cells=((5, 5, True),))

    run_roi.cmd_batch(batch_args(base, count_only=True, detection_dir=str(other)))

    entry = _roi_entry(reg)
    approved_hash = AnalysisRegistry.get_method_hash(get_approved_method(reg))
    assert entry["method_hash"] != det.get_entry(SAMPLE)["method_hash"]
    assert entry["method_hash"] != approved_hash
    assert "not tied to a registered detection run" in capsys.readouterr().out
