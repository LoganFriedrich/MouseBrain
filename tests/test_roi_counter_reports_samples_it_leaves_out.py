"""brainslice-roi says which samples it leaves out, and why.

A sample with no measurements.csv, or whose ND2 is not in the dataset folder,
used to be dropped without a word; the totals then counted only what was
left, so a missing image looked like a smaller dataset.
"""

from mousebrain.plugin_2d.sliceatlas.cli import run_roi

from roi_cli_helpers import (
    PROJECT, SAMPLE, batch_args, make_dataset, pin_registry, rois_of,
    write_measurements, write_rois,
)

NO_IMAGE = "Q01_02_S06_ZONE"
NO_MEASUREMENTS = "Q01_02_S07_ZONE"


def _setup(tmp_path, monkeypatch):
    reg = pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path, samples=(SAMPLE, NO_MEASUREMENTS))
    write_rois(rois_of(base))
    det = reg / "exports" / f"{PROJECT}_Detection"
    write_measurements(det / SAMPLE / "measurements.csv")
    write_measurements(det / NO_IMAGE / "measurements.csv")
    (det / NO_MEASUREMENTS).mkdir(parents=True)
    return base


def test_left_out_samples_are_named_with_their_reason(tmp_path, monkeypatch, capsys):
    base = _setup(tmp_path, monkeypatch)

    rc = run_roi.cmd_batch(batch_args(base, count_only=True))

    out = capsys.readouterr().out
    assert rc == 0
    assert "2 sample(s) left out" in out
    no_image_line = [l for l in out.splitlines() if NO_IMAGE in l]
    assert no_image_line and ".nd2" in no_image_line[0]
    no_meas_line = [l for l in out.splitlines() if NO_MEASUREMENTS in l]
    assert no_meas_line and "measurements.csv" in no_meas_line[0]


def test_the_final_summary_counts_the_left_out_samples(tmp_path, monkeypatch, capsys):
    base = _setup(tmp_path, monkeypatch)

    run_roi.cmd_batch(batch_args(base, count_only=True))

    summary = [l for l in capsys.readouterr().out.splitlines() if l.startswith("SUMMARY")]
    assert summary == ["SUMMARY: 1 samples counted, 2 left out"]


def test_a_sample_outside_the_chosen_region_is_not_reported(tmp_path, monkeypatch, capsys):
    reg = pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))
    det = reg / "exports" / f"{PROJECT}_Detection"
    write_measurements(det / SAMPLE / "measurements.csv")
    write_measurements(det / "Q01_02_S08_OTHER" / "measurements.csv")

    run_roi.cmd_batch(batch_args(base, count_only=True, region="ZONE"))

    assert "left out" not in capsys.readouterr().out


def test_the_registrys_archive_folder_is_not_a_left_out_sample(tmp_path, monkeypatch, capsys):
    """'_archived' (where invalidated results are moved) is registry
    bookkeeping, not a sample."""
    reg = pin_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))
    det = reg / "exports" / f"{PROJECT}_Detection"
    write_measurements(det / SAMPLE / "measurements.csv")
    (det / "_archived" / "20000101_000000" / SAMPLE).mkdir(parents=True)

    run_roi.cmd_batch(batch_args(base, count_only=True))

    out = capsys.readouterr().out
    assert "left out" not in out
    assert "SUMMARY: 1 samples counted" in out


def _no_registry(monkeypatch, tmp_path):
    from mousebrain.plugin_2d.sliceatlas.core import config as slice_config
    monkeypatch.delenv("MOUSEBRAIN_REGISTRY_ROOT", raising=False)
    monkeypatch.setattr(run_roi, "_registry_root", lambda: None)
    monkeypatch.setattr(slice_config, "DEFAULT_BATCH_OUTPUT", None, raising=False)


def test_counting_twice_with_no_registry_does_not_read_its_own_counts_as_samples(
        tmp_path, monkeypatch, capsys):
    """With no registry the counts are written beside the data; the next run
    must not take those folders for detection samples."""
    _no_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))
    write_measurements(base / "batch_results" / SAMPLE / "measurements.csv")

    assert run_roi.cmd_batch(batch_args(base, count_only=True)) == 0
    capsys.readouterr()
    assert run_roi.cmd_batch(batch_args(base, count_only=True)) == 0

    out = capsys.readouterr().out
    assert "left out" not in out
    assert any(l.startswith("SUMMARY: 1 samples counted") and "left out" not in l
               for l in out.splitlines())


def test_counts_left_by_an_older_version_in_batch_results_are_not_samples(
        tmp_path, monkeypatch, capsys):
    """Older versions wrote the counts into batch_results/<animal>/<region>/."""
    _no_registry(monkeypatch, tmp_path)
    base = make_dataset(tmp_path)
    write_rois(rois_of(base))
    write_measurements(base / "batch_results" / SAMPLE / "measurements.csv")
    old = base / "batch_results" / "Q01_02" / "ZONE"
    old.mkdir(parents=True)
    (old / f"{SAMPLE}_roi_counts.csv").write_text("name,total\nTOTAL,1\n")

    run_roi.cmd_batch(batch_args(base, count_only=True))

    assert "left out" not in capsys.readouterr().out
