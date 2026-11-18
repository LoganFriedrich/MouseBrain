"""`mousebrain --check` prints ASCII only.

WHY: a Windows console whose code page cannot show a character stops the
whole command with UnicodeEncodeError, so a dependency check meant to tell a
person what is missing would itself crash. The section headings carried an em
dash. The dependency imports are answered "missing" here, so the test is fast
and does not depend on what this environment has installed.
"""
import builtins

from mousebrain import cli

_real_import = builtins.__import__


def _only_mousebrain_imports(name, *args, **kwargs):
    if name == "mousebrain" or name.startswith("mousebrain."):
        if name.startswith("mousebrain.plugin_"):
            raise ImportError(name)
        return _real_import(name, *args, **kwargs)
    if name in ("sys", "os", "importlib"):
        return _real_import(name, *args, **kwargs)
    raise ImportError(name)


def test_the_dependency_check_prints_ascii_only(capsys):
    # Patched only around the call: pytest itself imports while it reports.
    builtins.__import__ = _only_mousebrain_imports
    try:
        cli.check_dependencies()
    finally:
        builtins.__import__ = _real_import
    out = capsys.readouterr().out
    assert "pip install mousebrain[3d]" in out          # the headings were printed
    bad = sorted({c for c in out if ord(c) > 127})
    assert not bad, "non-ASCII printed: %r" % bad
