"""Append to Folder CSV keeps the settings each sample was analyzed with.

The folder-wide CSV carries one '# Settings <sample>:' line per sample, not
ONE '# Settings:' line rewritten with the current settings on every append:
that would label every earlier sample's rows with settings they were not made
with.
"""
import json

import pandas as pd

from slice_widget_helpers import make_widget


def _settings_lines(path):
    return [l for l in path.read_text().splitlines() if l.startswith("#")]


def _append(w, folder, sample, threshold):
    w.current_file = folder / f"{sample}.tif"
    w._pa_results = pd.DataFrame({"label": [1, 2], "area": [10, 12]})
    w._pa_get_settings = lambda: {"threshold": threshold}
    w._pa_append_to_folder_csv()


def test_each_sample_keeps_its_own_settings_line(tmp_path, monkeypatch):
    folder = tmp_path / "SLIDES"
    folder.mkdir()
    w = make_widget(tmp_path, monkeypatch, "unused")
    _append(w, folder, "PRJ_01_01_S1", 40)
    _append(w, folder, "PRJ_01_01_S2", 55)

    master = folder / "SLIDES_particles.csv"
    lines = _settings_lines(master)
    by_sample = {l.split(":", 1)[0]: json.loads(l.split(":", 1)[1]) for l in lines}
    assert by_sample["# Settings PRJ_01_01_S1"] == {"threshold": 40}
    assert by_sample["# Settings PRJ_01_01_S2"] == {"threshold": 55}

    table = pd.read_csv(master, comment="#")
    assert sorted(table["sample"].unique()) == ["PRJ_01_01_S1", "PRJ_01_01_S2"]


def test_the_roi_results_csv_does_the_same(tmp_path, monkeypatch):
    folder = tmp_path / "SLIDES"
    folder.mkdir()
    w = make_widget(tmp_path, monkeypatch, "unused")
    w._roi_names = ["ROI_A"]
    for sample, threshold in (("PRJ_01_01_S1", 40), ("PRJ_01_01_S2", 55)):
        w.current_file = folder / f"{sample}.tif"
        w._roi_detail_data = pd.DataFrame({"roi": ["ROI_A"], "total": [3]})
        w._pa_get_settings = lambda t=threshold: {"threshold": t}
        w._append_to_folder_csv()

    lines = _settings_lines(folder / "SLIDES_roi_results.csv")
    assert any(l.startswith("# Settings PRJ_01_01_S1:") and '"threshold": 40' in l
               for l in lines)
    assert any(l.startswith("# Settings PRJ_01_01_S2:") and '"threshold": 55' in l
               for l in lines)


def test_an_older_files_single_settings_line_is_kept_and_named_for_what_it_is(tmp_path, monkeypatch):
    folder = tmp_path / "SLIDES"
    folder.mkdir()
    master = folder / "SLIDES_particles.csv"
    master.write_text('# Settings: {"threshold": 30}\nsample,label,area\nPRJ_01_01_S0,1,9\n')
    w = make_widget(tmp_path, monkeypatch, "unused")
    _append(w, folder, "PRJ_01_01_S1", 40)

    lines = _settings_lines(master)
    assert not any(l.startswith("# Settings: ") for l in lines)
    assert any("before each sample's settings were kept" in l and '"threshold": 30' in l
               for l in lines)
    assert any(l.startswith("# Settings PRJ_01_01_S1:") for l in lines)
    assert "PRJ_01_01_S0" in set(pd.read_csv(master, comment="#")["sample"])
