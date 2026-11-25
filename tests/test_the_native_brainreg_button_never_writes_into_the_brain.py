"""The dashboard's native brainreg button never registers into the brain.

It opens brainreg's own widget, which must not have its output folder preset
to the brain's 3_Registered_Atlas with a fixed orientation: a registration run
from there would bypass step 3 (no archive of the previous registration and
its approval, no record of the images used, no tracker row). The widget is for
trying settings; only step 3 makes the brain's registration. Identifiers are
invented.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

pw = pytest.importorskip("mousebrain.plugin_3d.pipeline_widget",
                         reason="the napari dashboard needs napari and Qt")


class _Field:
    def __init__(self, value=None):
        self.value = value


class _FakeBrainregWidget:
    def __init__(self):
        self.z_pixel_um = _Field()
        self.y_pixel_um = _Field()
        self.x_pixel_um = _Field()
        self.data_orientation = _Field("asr")  # brainreg's own default
        self.registration_output_folder = _Field(Path("chosen-by-the-person"))
        self.n_free_cpus = _Field()


def test_the_widget_is_not_pointed_at_the_brains_registration_folder(tmp_path, monkeypatch):
    brain = tmp_path / "M9" / "906_PROJ_09_06_2p5x_z5"
    brain.mkdir(parents=True)
    made = []

    def factory():
        made.append(_FakeBrainregWidget())
        return made[-1]
    monkeypatch.setattr(pw, "BRAINREG_AVAILABLE", True)
    monkeypatch.setattr(pw, "brainreg_register", factory, raising=False)
    docked = []
    fake_self = SimpleNamespace(
        current_brain=brain,
        get_brain_metadata=lambda: {"voxel_z": 5.0, "voxel_xy": 2.6},
        viewer=SimpleNamespace(window=SimpleNamespace(
            add_dock_widget=lambda w, **k: docked.append(w))),
        message_label=SimpleNamespace(setText=lambda t: None),
    )
    pw.PipelineDashboard.launch_brainreg(fake_self)
    assert docked, "the widget was not opened"
    w = made[0]
    assert w.z_pixel_um.value == 5.0  # the brain's voxel sizes are still filled in
    out = Path(str(w.registration_output_folder.value))
    assert brain not in out.parents and out != brain / "3_Registered_Atlas"
    assert w.data_orientation.value == "asr"  # not overwritten with a fixed guess
