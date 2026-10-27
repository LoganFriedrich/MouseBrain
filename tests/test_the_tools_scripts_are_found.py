"""mousebrain finds its own scripts.

The reorganization moved the pipeline steps to scripts/pipeline and the
utilities to scripts/utils, but everything that started a script by name still
looked in the pipeline's 3D_Cleared/util_Scripts -- a data folder, now empty.
The napari pipeline dashboard, its registration QC button and batch_process all
stopped with "not found", and registration finished every brain without its QC
picture (it printed a warning nobody saw). Scripts are now found beside the
installed package, by name.
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import tifffile

from mousebrain import config

REPO = Path(config.__file__).resolve().parents[2]
CALLERS = [
    REPO / "src" / "mousebrain" / "plugin_3d" / "pipeline_widget.py",
    REPO / "src" / "mousebrain" / "plugin_3d" / "tuning_widget.py",
    REPO / "scripts" / "pipeline" / "batch_process.py",
]
ASKED_FOR = re.compile(r"""(?:run_script|run_step|find_script)\(\s*['"]([^'"]+\.py)['"]""")


def test_every_script_the_tool_starts_by_name_exists():
    names = set()
    for caller in CALLERS:
        names |= set(ASKED_FOR.findall(caller.read_text(encoding="utf-8")))
    # the dashboard's six steps, approval, and the tuning widget's own buttons
    assert {"3_register_to_atlas.py", "util_approve_registration.py",
            "util_registration_qc.py", "6_count_regions.py"} <= names
    missing = sorted(n for n in names if not config.find_script(n).is_file())
    assert missing == []


def test_a_script_that_does_not_exist_is_reported_missing():
    assert not config.find_script("no_such_script.py").exists()


def test_the_scripts_are_those_of_the_installed_code_not_the_data_folder():
    assert config.PIPELINE_SCRIPTS_DIR == REPO / "scripts" / "pipeline"
    assert config.UTIL_SCRIPTS_DIR == REPO / "scripts" / "utils"
    assert config.CLEARED_3D_DIR not in config.find_script("3_register_to_atlas.py").parents


def _step3():
    path = REPO / "scripts" / "pipeline" / "3_register_to_atlas.py"
    spec = importlib.util.spec_from_file_location("register_step", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registration_draws_its_qc_picture(tmp_path):
    """What step 3 does after brainreg finishes, on a small stand-in registration."""
    reg = tmp_path / "401_PROJ_01_02_1p625x_z4" / "3_Registered_Atlas"
    reg.mkdir(parents=True)
    brain = np.random.default_rng(0).integers(0, 1000, (40, 30, 36)).astype(np.uint16)
    atlas = np.zeros(brain.shape, np.uint32)
    atlas[5:35, 5:25, 4:18] = 315
    atlas[5:35, 5:25, 18:32] = 698
    tifffile.imwrite(str(reg / "downsampled.tiff"), brain)
    tifffile.imwrite(str(reg / "registered_atlas.tiff"), atlas)
    (reg / "brainreg.json").write_text(json.dumps({"atlas": "allen_mouse_25um"}), encoding="utf-8")

    png = _step3().draw_registration_qc(reg.parent)
    assert png is not None and Path(png).is_file()
    assert Path(png).name == "QC_registration_detailed.png"
