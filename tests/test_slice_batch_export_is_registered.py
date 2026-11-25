"""Export Batch Results registers what it wrote.

Export Batch Results writes the summary, detail and settings files to the
path a person chose and also registers them, with the batch settings, under
<project>_Window_Batch, keyed by the batch folder's name; without a registry
entry, whatever pulls the registry would never see a batch run from the 2D
window.
"""
import json

from slice_batch_helpers import (draw_roi_around_everything, export_batch,
                                 make_batch_widget, two_channel_image)


def test_export_batch_results_registers_the_exported_files(tmp_path, monkeypatch):
    root = tmp_path / "Registry"
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(root))
    monkeypatch.setenv("MOUSEBRAIN_2D_PROJECT", "PRJ")
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    images = {"A_01.tif": two_channel_image(1), "A_02.tif": None}
    w, folder = make_batch_widget(tmp_path, monkeypatch, images, folder_name="batch_one")
    w._batch_run_analysis()
    draw_roi_around_everything(w)
    w._batch_count_and_next()
    (tmp_path / "out").mkdir()
    export_batch(w, monkeypatch, tmp_path / "out" / "batch_one_summary.csv")

    manifest = root / "exports" / "PRJ_Window_Batch" / "registry.json"
    assert manifest.is_file(), "the batch export was not registered"
    entry = json.loads(manifest.read_text(encoding="utf-8"))["entries"].get("batch_one")
    assert entry is not None and entry["is_current"] is True
    assert {"summary", "settings"} <= set(entry["outputs"])
    assert entry["results"]["images"] == 2
    assert entry["results"]["counted"] == 1 and entry["results"]["failed"] == 1
    assert entry["source_files"]["batch_folder"] == str(folder)
    assert entry["method_params"]["threshold"] == w._batch_settings["threshold"]
    # The files are still where the person saved them.
    assert (tmp_path / "out" / "batch_one_summary.csv").is_file()
