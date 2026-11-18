"""A failed re-run that keeps the earlier entry current says so in the registry.

Defect covered: a re-run of one sample (a run where no other sample gets
past the step that failed) whose image raises a data error keeps the
sample's earlier entry current -- the failure may be the software's -- and
that was said only in a console summary line. Whoever pulled the registry
saw the earlier count as an unqualified current result although the newest
attempt on the sample failed. Now the entry carries newest_attempt_failed,
the audit log has an attempt_failed line, and the registry CLI lists it.
"""
import json

from mousebrain import analysis_registry as ar

# The batch fixture and helpers of the failed re-run tests.
from test_encr_batch_failed_rerun_retires_entry import (  # noqa: F401
    SAMPLE, _earlier_entry, _loads, _raise, _run, env, io_mod)


def test_a_kept_entry_carries_its_failed_attempt(env, monkeypatch, capsys):
    t, tmp_path, img = env
    reg, old_file = _earlier_entry(tmp_path)
    _loads(monkeypatch, img)
    monkeypatch.setattr(io_mod, "extract_channels", _raise(ValueError("unusable image")))
    _run(monkeypatch, tmp_path)

    entry = json.loads(reg.registry_path.read_text(encoding="utf-8"))["entries"][SAMPLE]
    # Still current, with its earlier count (the failure may be the software's) ...
    assert entry["is_current"] is True
    assert entry["results"]["n_nuclei"] == 28
    # ... but the registry itself says the newest attempt failed, and where.
    mark = entry.get("newest_attempt_failed")
    assert mark is not None, "the failed attempt is not recorded on the entry"
    assert mark["failed_at"] == "channels"
    assert "unusable image" in mark["error"]

    log = (reg.logs_dir / f"{reg.analysis_name}.log").read_text(encoding="utf-8")
    assert any(line.split("\t")[2] == "attempt_failed" and SAMPLE in line
               for line in log.splitlines())

    capsys.readouterr()
    assert ar.main(["--name", reg.analysis_name, "--root", str(reg.registry_root)]) == 0
    out = capsys.readouterr().out
    assert "Current, but the newest run failed: 1" in out
    assert f"{SAMPLE}: failed while channels" in out
