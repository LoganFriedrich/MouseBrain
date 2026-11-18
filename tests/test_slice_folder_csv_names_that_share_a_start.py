"""A folder-wide CSV keeps the settings line of every sample, including one
whose name begins with another sample's name followed by ': '.

Each sample's line is '# Settings <sample>: {...}'. Saving sample 'A' used to
drop every line starting '# Settings A: ', which is also how the line of a
sample named 'A: x' starts, so that sample's settings were lost.
"""
import json

import pandas as pd

from mousebrain.plugin_2d.sliceatlas.napari_plugin.main_widget import write_folder_csv


def _settings_lines(path):
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("#")]


def test_saving_a_sample_keeps_the_line_of_a_sample_whose_name_extends_it(tmp_path):
    master = tmp_path / "SLIDES_particles.csv"
    rows = pd.DataFrame({"sample": ["X"], "label": [1]})
    write_folder_csv(master, rows, "A: x", {"threshold": 10})
    write_folder_csv(master, rows, "A", {"threshold": 20})

    lines = _settings_lines(master)
    assert '# Settings A: x: {"threshold": 10}' in lines
    assert '# Settings A: {"threshold": 20}' in lines
    assert len(lines) == 2


def test_saving_a_sample_again_still_replaces_its_own_line(tmp_path):
    master = tmp_path / "SLIDES_particles.csv"
    rows = pd.DataFrame({"sample": ["X"], "label": [1]})
    write_folder_csv(master, rows, "A", {"threshold": 20})
    write_folder_csv(master, rows, "A: x", {"threshold": 10})
    write_folder_csv(master, rows, "A", {"threshold": 30})

    lines = _settings_lines(master)
    own = [l for l in lines if l.startswith("# Settings A: {")]
    assert own == ['# Settings A: {"threshold": 30}']
    assert '# Settings A: x: {"threshold": 10}' in lines
    assert json.loads(own[0][len("# Settings A: "):]) == {"threshold": 30}
