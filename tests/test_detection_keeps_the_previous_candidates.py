"""Step 4 archives the previous candidates and ties the new file to its run.

It wrote the fixed name detected_cells.xml over the previous one on every run,
and each tracker row named only the folder -- so after a second run nothing
said which run had written the file that went on to classification.
"""
import json
from pathlib import Path

from mousebrain.candidates_io import (CANDIDATES_NAME, archive_previous, provenance_path,
                                      restore, write_provenance)
from mousebrain.pipeline_status import find_candidates

STEP4 = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "4_detect_cells.py"


def _folder(tmp_path):
    det = tmp_path / "BRAIN_A" / "4_Cell_Candidates"
    det.mkdir(parents=True)
    return det


def test_a_second_run_archives_the_first_runs_file_and_record(tmp_path):
    det = _folder(tmp_path)
    (det / CANDIDATES_NAME).write_text("first")
    write_provenance(det / CANDIDATES_NAME, "det_first", {"threshold": 10})

    archived, prev_id = archive_previous(det)
    assert prev_id == "det_first"
    assert archived.read_text() == "first" and archived.parent.parent == det / "_archive"
    assert provenance_path(archived).exists()
    assert not (det / CANDIDATES_NAME).exists()

    (det / CANDIDATES_NAME).write_text("second")
    rec = write_provenance(det / CANDIDATES_NAME, "det_second", {}, archived)
    saved = json.loads(rec.read_text())
    assert saved["exp_id"] == "det_second"
    assert saved["previous_file_archived_to"] == str(archived)


def test_archived_candidates_are_never_picked_for_classification(tmp_path):
    det = _folder(tmp_path)
    (det / CANDIDATES_NAME).write_text("old")
    archive_previous(det)
    assert find_candidates(tmp_path / "BRAIN_A") is None


def test_a_failed_run_puts_the_previous_file_back(tmp_path):
    det = _folder(tmp_path)
    (det / CANDIDATES_NAME).write_text("first")
    write_provenance(det / CANDIDATES_NAME, "det_first")
    archived, _ = archive_previous(det)
    assert restore(archived, det)
    assert (det / CANDIDATES_NAME).read_text() == "first"
    assert json.loads(provenance_path(det / CANDIDATES_NAME).read_text())["exp_id"] == "det_first"
    assert not archived.parent.exists()


def test_nothing_to_archive_on_a_first_run(tmp_path):
    assert archive_previous(_folder(tmp_path)) == (None, None)


def test_step4_archives_and_records_the_writer():
    text = STEP4.read_text(encoding="utf-8")
    assert "archive_previous(det_folder)" in text
    assert "tracker.update_status(exp_id, output_path=str(cells_xml))" in text
    assert "tracker.update_status(prev_id, output_path=str(archived))" in text
