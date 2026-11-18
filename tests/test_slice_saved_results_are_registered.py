"""What the 2D window and run_coloc save is registered in MouseBrain's own registry.

Defect covered: Save State / Count in ROI(s) (and the automatic save when
another image is opened) in the 2D window, and run_coloc --save, wrote their
results only beside the images or into the --save folder; nothing reached
the analysis registry, so whatever pulls the registry never saw them and
nothing recorded the settings that produced them. Now each registers what
it saved under its own analysis (<project>_Window_Results,
<project>_Coloc_CLI), never under the detection batch's analysis.
"""
import json
from argparse import Namespace

import numpy as np
import pandas as pd

from test_slice_saved_analysis_is_one_state import _open

PROJECT = "PRJ"


def _configure(tmp_path, monkeypatch, project=PROJECT):
    root = tmp_path / "Registry"
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(root))
    if project:
        monkeypatch.setenv("MOUSEBRAIN_2D_PROJECT", project)
    else:
        monkeypatch.delenv("MOUSEBRAIN_2D_PROJECT", raising=False)
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    return root


def _manifest(root, analysis):
    path = root / "exports" / analysis / "registry.json"
    assert path.is_file(), f"nothing was registered under {analysis}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_save_state_registers_the_saved_result_tables(tmp_path, monkeypatch):
    root = _configure(tmp_path, monkeypatch)
    w, image = _open(tmp_path, monkeypatch)
    # Measurements are saved beside the nuclei they were measured on.
    w.nuclei_labels = np.ones((8, 8), dtype=np.int32)
    w.cell_measurements = pd.DataFrame({"label": [1, 2, 3],
                                        "is_positive": [True, False, True]})
    w._roi_detail_data = pd.DataFrame({"roi": ["ROI_A"], "total": [3]})
    w._manual_save_state()

    entry = _manifest(root, "PRJ_Window_Results")["entries"].get(image.stem)
    assert entry is not None and entry["is_current"] is True
    assert set(entry["outputs"]) == {"measurements", "roi_detail"}
    registered = root / entry["outputs"]["measurements"]
    assert pd.read_csv(registered)["label"].tolist() == [1, 2, 3]
    assert entry["results"]["measurements"] == {"rows": 3, "positive": 2}
    assert entry["source_files"]["image"] == str(image)
    # The settings the window saved with are the method; the save time is not.
    assert "loaded_with" in entry["method_params"]
    assert "saved_at" not in entry["method_params"]
    # Never under the detection batch's analysis.
    assert not (root / "exports" / "PRJ_Detection").exists()


def test_saving_the_same_results_again_archives_nothing(tmp_path, monkeypatch):
    root = _configure(tmp_path, monkeypatch)
    w, image = _open(tmp_path, monkeypatch)
    w.cell_measurements = pd.DataFrame({"label": [1], "is_positive": [True]})
    w._manual_save_state()
    w._manual_save_state()
    folder = root / "exports" / "PRJ_Window_Results" / image.stem
    assert not (folder / "_archived").exists()


def test_results_gone_from_the_saved_state_retire_the_entry(tmp_path, monkeypatch):
    root = _configure(tmp_path, monkeypatch)
    w, image = _open(tmp_path, monkeypatch)
    w._roi_detail_data = pd.DataFrame({"roi": ["ROI_A"], "total": [3]})
    w._save_analysis_state()
    w._roi_detail_data = None            # counts dropped
    w._save_analysis_state()
    entry = _manifest(root, "PRJ_Window_Results")["entries"][image.stem]
    assert entry["is_current"] is False


def test_with_no_project_configured_nothing_is_registered_and_save_state_says_so(
        tmp_path, monkeypatch):
    root = _configure(tmp_path, monkeypatch, project=None)
    w, image = _open(tmp_path, monkeypatch)
    w.nuclei_labels = np.ones((8, 8), dtype=np.int32)
    w.cell_measurements = pd.DataFrame({"label": [1], "is_positive": [True]})
    w._manual_save_state()
    assert not (root / "exports").exists() or not any((root / "exports").iterdir())
    assert "not registered" in w.status_label.text()
    assert "MOUSEBRAIN_2D_PROJECT" in w.status_label.text()


def test_run_coloc_save_registers_what_it_wrote(tmp_path, monkeypatch):
    from mousebrain.plugin_2d.sliceatlas.cli import run_coloc

    root = _configure(tmp_path, monkeypatch)
    image = tmp_path / "imgs" / "PRJ_01_01_S2.nd2"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")
    args = Namespace(file=str(image), save=str(tmp_path / "saved"), method="zscore",
                     sigma_threshold=np.float64(2.0), dual=False)
    measurements = pd.DataFrame({"label": [1, 2], "is_positive": [True, False]})
    summary = {"positive_cells": np.int64(1), "negative_cells": 1,
               "positive_fraction": np.float64(0.5)}
    run_coloc._save_results(args.save, measurements, summary,
                            np.zeros((4, 4), dtype=np.int32), args)

    entry = _manifest(root, "PRJ_Coloc_CLI")["entries"].get(image.stem)
    assert entry is not None
    assert entry["results"]["positive_cells"] == 1
    assert entry["method_params"]["sigma_threshold"] == 2.0
    assert "file" not in entry["method_params"]
    assert set(entry["outputs"]) == {"measurements", "summary"}
    assert (root / entry["outputs"]["measurements"]).is_file()
