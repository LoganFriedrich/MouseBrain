"""brainslice-roi finds the detection results where the detection batch puts them.

The batch used to register every project's results under one fixed analysis
name, while the ROI counter looked under '<folder name>_Detection', and the
counter's fallback folder was not the batch's output folder. Results were
found only while a project's folder happened to carry that one name.
"""

import json
import sys

from example_method import write_example_approved_method
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr
from mousebrain.plugin_2d.sliceatlas.cli import run_roi
from mousebrain.plugin_2d.sliceatlas.core import config as slice_config

from roi_cli_helpers import (
    PROJECT, SAMPLE, batch_args, make_dataset, pin_registry, rois_of,
    write_measurements, write_rois,
)


def test_the_detection_batch_registers_under_its_own_projects_name(tmp_path, monkeypatch):
    reg = pin_registry(monkeypatch, tmp_path)
    write_example_approved_method(reg)   # the batch has no built-in values
    in_root = tmp_path / "raw" / PROJECT
    in_root.mkdir(parents=True)
    nd2 = in_root / f"{SAMPLE}.nd2"
    nd2.write_bytes(b"")
    out = tmp_path / "out"

    monkeypatch.setattr(batch_encr, "discover_nd2_files",
                        lambda roots, project=None: ([nd2], in_root, "legacy"))

    def fake_process(nd2_path, output_dir, **kw):
        write_measurements(output_dir / nd2_path.stem / "measurements.csv")
        return {"file": nd2_path.name, "status": "completed", "n_nuclei": 3,
                "n_positive": 2, "positive_fraction": 0.67, "background": 1.0}

    monkeypatch.setattr(batch_encr, "process_single", fake_process)
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", PROJECT, "--input",
                                      str(in_root), "--output", str(out),
                                      "--ignore-tracker"])
    batch_encr.main()

    manifest = reg / "exports" / f"{PROJECT}_Detection" / "registry.json"
    assert manifest.is_file()
    assert SAMPLE in json.loads(manifest.read_text())["entries"]
    # Nothing is filed under another project's analysis.
    assert [p.name for p in (reg / "exports").iterdir()] == [f"{PROJECT}_Detection"]


def test_results_in_the_batchs_default_output_folder_are_counted(tmp_path, monkeypatch, capsys):
    pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))
    batch_out = tmp_path / "2_Data_Summary" / "batch_results"
    write_measurements(batch_out / SAMPLE / "measurements.csv")
    monkeypatch.setattr(slice_config, "DEFAULT_BATCH_OUTPUT", batch_out, raising=False)
    monkeypatch.setattr(batch_encr, "DEFAULT_OUTPUT", batch_out)

    rc = run_roi.cmd_batch(batch_args(base, count_only=True))

    assert rc == 0
    assert "1 samples counted" in capsys.readouterr().out


def test_a_project_named_differently_from_its_folder_is_found_with_project(tmp_path, monkeypatch):
    reg = pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path, project="images_of_prjb")
    write_rois(rois_of(base))
    write_measurements(reg / "exports" / "PRJB_Detection" / SAMPLE / "measurements.csv")

    rc = run_roi.cmd_batch(batch_args(base, count_only=True, project="PRJB"))

    assert rc == 0
    assert (reg / "exports" / "PRJB_ROI_Analysis" / "roi_summary.csv").is_file()


def test_nothing_found_names_every_folder_searched(tmp_path, monkeypatch, capsys):
    reg = pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    batch_out = tmp_path / "2_Data_Summary" / "batch_results"
    monkeypatch.setattr(slice_config, "DEFAULT_BATCH_OUTPUT", batch_out, raising=False)

    rc = run_roi.cmd_batch(batch_args(base, count_only=True))

    out = capsys.readouterr().out
    assert rc == 1
    assert str(reg / "exports" / f"{PROJECT}_Detection") in out
    assert str(batch_out) in out
    assert str(base / "batch_results") in out
