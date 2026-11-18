"""Suite-wide guard: no test may write into this machine's REAL lab folders.

What it refuses, where the real folders come from, and what a refusal looks
like: real_root_guard.py beside this file, and docs/pipeline/TEST_GUARD.md
for people who do not read code.

The real folders are collected HERE, when pytest loads this file -- before
any test patches an environment variable or a module constant -- and every
spelling of them is built now, so the audit hook only compares strings.
The hook refuses from that moment until the process ends: during a test, at
import and collection, and in session fixtures alike. Each attempt is also
recorded; a test during which one happened fails even if its code caught the
PermissionError.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_root_guard  # noqa: E402

# The repository this file belongs to: its own files are exempt, even when it
# sits inside a real lab root. Taken from this file's location, never from
# configuration, so a copy elsewhere exempts itself and nothing else.
REPO_ROOT = Path(__file__).resolve().parents[1]

REAL_ROOTS = real_root_guard.collect_real_roots()
GUARD = real_root_guard.Guard(
    REAL_ROOTS,
    exempt=[str(REPO_ROOT), os.path.abspath(__file__), tempfile.gettempdir()],
)
if GUARD:
    sys.addaudithook(GUARD)


@pytest.fixture(autouse=True)
def _no_writes_to_real_lab_folders():
    """Fail the test when it tried to write into a real lab folder, even if
    its code swallowed the refusal."""
    start = len(GUARD.hits)
    yield
    new = GUARD.hits[start:]
    if new:
        del GUARD.hits[start:]
        shown = "\n".join("  %s: %s" % hit for hit in new[:10])
        more = "\n  ... and %d more" % (len(new) - 10) if len(new) > 10 else ""
        pytest.fail(
            "TEST GUARD: this test tried to write into a REAL lab folder (the "
            "write was refused, nothing was written). Point the test at "
            "tmp_path instead:\n" + shown + more,
            pytrace=False)


def pytest_sessionfinish(session, exitstatus):
    """Attempts made outside any test (at import, collection or in a session
    fixture) were refused as well; make sure the run does not pass."""
    if GUARD.hits:
        shown = "\n".join("  %s: %s" % hit for hit in GUARD.hits[:10])
        sys.stderr.write(
            "\nTEST GUARD: writes into REAL lab folders were refused outside "
            "any test:\n" + shown + "\n")
        if session.exitstatus == 0:
            session.exitstatus = 1


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Second layer behind `-p no:cacheprovider` in pyproject.toml: when a run
    turns pytest's cache back on (for example `-o addopts=`), put the cache in
    the temp folder, never in the repository. WHY: the cache is written by
    renaming a pytest-cache-files-* folder into place; on a network share the
    rename can fail and leave a folder only an administrator can remove."""
    cache = getattr(config, "cache", None)
    if cache is None:
        return
    try:
        cache._cachedir = Path(tempfile.gettempdir()) / (
            "pytest-cache-" + REPO_ROOT.name)
    except Exception as exc:  # pytest changed its cache object: say so
        sys.stderr.write("TEST GUARD: could not move pytest's cache out of the "
                         "repository (%s); keep -p no:cacheprovider.\n" % exc)
