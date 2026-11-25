"""Exporting region counts replaces all three count files or none of them.

<sample>_cells.csv, _regions.csv and _summary.txt must not be overwritten one
after another: a failure on the second (for example the regions CSV open in a
spreadsheet program) would leave a new cells.csv beside the old regions and
summary: a folder mixing two runs.
"""
import os

import pandas as pd
import pytest

from mousebrain.plugin_2d.sliceatlas.core.quantification import RegionQuantifier

SAMPLE = "SUBJ_X_S03"


def test_the_export_is_all_or_nothing_when_a_count_file_is_locked(tmp_path, monkeypatch):

    out = tmp_path / "quant"
    out.mkdir()
    for suffix, text in (("_cells.csv", "old cells\n"), ("_regions.csv", "old regions\n"),
                         ("_summary.txt", "old summary\n")):
        (out / f"{SAMPLE}{suffix}").write_text(text)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    real_replace = os.replace

    def _regions_locked(src, dst):
        # A file open in a spreadsheet program cannot be renamed or replaced.
        if str(src).endswith(f"{SAMPLE}_regions.csv") or str(dst).endswith(f"{SAMPLE}_regions.csv"):
            raise PermissionError("regions CSV is open in another program")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _regions_locked)
    cells = pd.DataFrame({"label": [1, 2], "is_positive": [True, False], "region_id": [5, 5],
                          "region_name": ["R", "R"]})
    regions = pd.DataFrame({"region_id": [5], "region_name": ["R"], "total_cells": [2],
                            "positive_cells": [1], "negative_cells": [1], "positive_fraction": [0.5]})
    # Writing straight onto the locked file fails too.
    real_to_csv = regions.to_csv

    def _regions_to_csv(path, **kw):
        if str(path).endswith(f"{SAMPLE}_regions.csv"):
            raise PermissionError("regions CSV is open in another program")
        return real_to_csv(path, **kw)

    object.__setattr__(regions, "to_csv", _regions_to_csv)
    with pytest.raises(PermissionError):
        RegionQuantifier(None).export_results(cells, regions, out, SAMPLE)

    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert after == before


def test_the_export_replaces_all_three_files_when_nothing_is_locked(tmp_path):

    out = tmp_path / "quant"
    out.mkdir()
    (out / f"{SAMPLE}_cells.csv").write_text("old cells\n")
    cells = pd.DataFrame({"label": [1, 2], "is_positive": [True, False], "region_id": [5, 5],
                          "region_name": ["R", "R"]})
    regions = pd.DataFrame({"region_id": [5], "region_name": ["R"], "total_cells": [2],
                            "positive_cells": [1], "negative_cells": [1], "positive_fraction": [0.5]})
    paths = RegionQuantifier(None).export_results(cells, regions, out, SAMPLE)

    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"{SAMPLE}{s}" for s in ("_cells.csv", "_regions.csv", "_summary.txt"))
    assert "old cells" not in paths["cells"].read_text()
    assert pd.read_csv(paths["regions"])["total_cells"].tolist() == [2]
