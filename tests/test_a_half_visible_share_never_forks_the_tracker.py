"""A data drive that is only half visible never gets a second, empty tracker.

mousebrain picks its folder layout once, at import, by which folders exist. If
the configured root answers but its Tissue/MouseBrain_Pipeline folder does not
(a network share still reconnecting), falling back to the legacy layout would
create <root>/3_Nuclei_Detection/2_Data_Summary/calibration_runs.csv -- a
fresh, empty tracker that later runs would append to while the real record of
every calibration run sat unseen. So the legacy layout is chosen only when its
folder exists, and the default tracker is created only inside an existing
2_Data_Summary folder; otherwise it refuses and names the folder.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _fresh_import_and_open_tracker(root: Path, home: Path):
    """Import mousebrain anew with CONNECTOME_ROOT=root and open the default tracker."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MOUSEBRAIN_", "MOUSEDB_", "MOUSEREACH_"))
           and k not in ("CONNECTOME_ROOT", "SCI_CONNECTOME_ROOT", "PYTHONPATH")}
    env.update(CONNECTOME_ROOT=str(root), PYTHONPATH=str(SRC),
               USERPROFILE=str(home), HOME=str(home))
    code = ("from mousebrain.tracker import ExperimentTracker\n"
            "try:\n"
            "    t = ExperimentTracker()\n"
            "    print('OPENED', t.csv_path)\n"
            "except RuntimeError as e:\n"
            "    print('REFUSED', e)\n")
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(home),
                          capture_output=True, text=True, timeout=120)


def test_a_root_without_its_pipeline_folder_gets_no_new_tracker(tmp_path):
    root = tmp_path / "share"
    (root / "Tissue").mkdir(parents=True)        # the root answers, the pipeline does not
    home = tmp_path / "home"
    home.mkdir()
    out = _fresh_import_and_open_tracker(root, home)
    assert "REFUSED" in out.stdout, out.stdout + out.stderr
    made = [p for p in root.rglob("*") if p.name != "Tissue"]
    assert made == [], "nothing may be created under the root: %s" % made


def test_the_default_tracker_never_creates_its_own_folder(monkeypatch, tmp_path):
    from mousebrain import config as cfg, tracker as tk
    monkeypatch.setattr(cfg, "ROOT_PATH", tmp_path)
    target = tmp_path / "MouseBrain_Pipeline" / "3D_Cleared" / "2_Data_Summary" / "calibration_runs.csv"
    monkeypatch.setattr(tk, "DEFAULT_TRACKER_PATH", target)
    with pytest.raises(RuntimeError) as e:
        tk.ExperimentTracker()
    assert str(target.parent) in str(e.value), "the refusal names the folder"
    assert not target.parent.exists()


def test_an_existing_tracker_folder_still_starts_a_tracker(monkeypatch, tmp_path):
    from mousebrain import config as cfg, tracker as tk
    monkeypatch.setattr(cfg, "ROOT_PATH", tmp_path)
    target = tmp_path / "2_Data_Summary" / "calibration_runs.csv"
    target.parent.mkdir()
    monkeypatch.setattr(tk, "DEFAULT_TRACKER_PATH", target)
    tk.ExperimentTracker()
    assert target.exists()


def test_the_legacy_layout_is_still_used_where_it_exists(tmp_path):
    root = tmp_path / "old_install"
    summary = root / "3_Nuclei_Detection" / "2_Data_Summary"
    summary.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    out = _fresh_import_and_open_tracker(root, home)
    assert "OPENED" in out.stdout, out.stdout + out.stderr
    assert (summary / "calibration_runs.csv").exists()
