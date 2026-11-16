"""The batch's "write them into the approved-method file" hint says when the
method it would edit is shared.

Guard of the banner hint the batch prints when its numbers differ from the
approved method: when the approved-method file does not give the batch's
analysis its own entry, the batch is judged by the SHARED method, and editing
that also changes what every other analysis following it is judged against.
The person must be told so before editing it.
"""

import json

from mousebrain import analysis_registry as ar
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be

from test_encr_batch_marked_run_dilations_must_match import (  # noqa: F401  (fixture)
    OTHER, SAMPLE, _main, _marked, env)


def _approved_file(tmp_path, content):
    root = tmp_path / "registry"
    root.mkdir(exist_ok=True)
    (root / ar.APPROVED_METHOD_FILENAME).write_text(json.dumps(content), encoding="utf-8")


def test_hint_says_the_shared_method_is_every_analysis(env, monkeypatch, capsys):
    t, tmp_path, calls = env      # no approved-method file: the shared method
    _marked(t, SAMPLE, sigma=1.0)
    _main(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "write them into the approved-method file" in out
    assert "EVERY analysis it holds" in out
    assert be.approved_method_is_shared()


def test_a_flat_file_is_shared_too(env, monkeypatch, capsys):
    t, tmp_path, calls = env
    _approved_file(tmp_path, dict(ar.DEFAULT_METHOD))
    _marked(t, SAMPLE, sigma=1.0)
    _main(monkeypatch, tmp_path)
    assert "EVERY analysis it holds" in capsys.readouterr().out


def test_hint_is_not_given_when_the_file_names_the_batch_analysis(env, monkeypatch, capsys):
    t, tmp_path, calls = env
    _approved_file(tmp_path, {"analyses": {be.REGISTRY_ANALYSIS: dict(ar.DEFAULT_METHOD)}})
    _marked(t, SAMPLE, sigma=1.0)
    _main(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "write them into the approved-method file" in out
    assert "EVERY analysis it holds" not in out
    assert not be.approved_method_is_shared()


def test_typed_numbers_alone_also_get_the_shared_file_warning(env, monkeypatch, capsys):
    # No marked run: the differing number is typed. A person copying it into
    # the approved-method file changes every analysis's approved method just
    # the same, so the warning must not depend on where the number came from.
    t, tmp_path, calls = env      # no approved-method file: the shared method
    _main(monkeypatch, tmp_path, "--threshold", "1.5")
    out = capsys.readouterr().out
    assert "NOT the approved method" in out
    assert "write them into the approved-method file" in out
    assert "EVERY analysis it holds" in out


def test_typed_numbers_settling_disagreeing_marks_are_not_called_no_mark(env, monkeypatch, capsys):
    # Disagreeing marked runs, both values typed: the banner must not then
    # tell the person that no run is marked best and to go mark one.
    t, tmp_path, calls = env
    _marked(t, SAMPLE, sigma=1.0)
    _marked(t, OTHER, sigma=0.0)
    _main(monkeypatch, tmp_path, "--threshold", "0.5", "--bg-percentile", "10")
    out = capsys.readouterr().out
    assert "marked-best runs give different parameters" in out
    assert "no run of this project is marked best" not in out
    assert "marked-best runs disagree; none used" in out
