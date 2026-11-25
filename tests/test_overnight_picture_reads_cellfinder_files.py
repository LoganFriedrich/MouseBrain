"""The overnight sample pictures read the cells in a cellfinder .xml.

cellfinder writes MarkerX, MarkerY and MarkerZ each on its own line, so the
reader must not look for a line holding all three together: every picture
would say 0 cells.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"

# The layout cellfinder (brainglobe_utils save_cells) writes.
CELLFINDER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<CellCounter_Marker_File>
  <Image_Properties>
    <Image_Filename>placeholder.tif</Image_Filename>
  </Image_Properties>
  <Marker_Data>
    <Current_Type>1</Current_Type>
    <Marker_Type>
      <Type>1</Type>
      <Marker>
        <MarkerX>3</MarkerX>
        <MarkerY>4</MarkerY>
        <MarkerZ>5</MarkerZ>
      </Marker>
      <Marker>
        <MarkerX>6</MarkerX>
        <MarkerY>7</MarkerY>
        <MarkerZ>8</MarkerZ>
      </Marker>
    </Marker_Type>
  </Marker_Data>
</CellCounter_Marker_File>
"""


@pytest.fixture
def overnight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the module opens its log file in the working folder
    spec = importlib.util.spec_from_file_location("overnight_parse_under_test",
                                                  PIPELINE / "overnight_batch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    for h in list(logging.root.handlers):
        if isinstance(h, logging.FileHandler) and str(tmp_path) in h.baseFilename:
            logging.root.removeHandler(h)
            h.close()


def test_every_cell_in_a_cellfinder_file_is_read(overnight, tmp_path):
    xml = tmp_path / "detected_cells.xml"
    xml.write_text(CELLFINDER_XML, encoding="utf-8")
    assert overnight.parse_cells_from_xml(xml) == [
        {"x": 3.0, "y": 4.0, "z": 5}, {"x": 6.0, "y": 7.0, "z": 8}]


def test_an_unreadable_file_gives_no_cells(overnight, tmp_path):
    xml = tmp_path / "detected_cells.xml"
    xml.write_text("<CellCounter_Marker_File><Marker>", encoding="utf-8")
    assert overnight.parse_cells_from_xml(xml) == []
    assert overnight.parse_cells_from_xml(tmp_path / "missing.xml") == []
