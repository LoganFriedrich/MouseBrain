"""The test guard refuses every write into a real lab folder, however the
folder is spelled and wherever its location comes from, while the
repository's own files stay writable -- and pytest's cache never lands in the
repository.

Each case runs a CHILD pytest in a throwaway copy of the repository layout:

    <tmp>/lab/                 <- a stand-in for a real lab root
    <tmp>/lab/repo/tests/      <- the repository: this suite's conftest is copied here
    <tmp>/lab/sibling/         <- a folder beside the repository, inside the root

The child's tests try to write into lab/sibling by every route; the parent
then looks on disk. Without the guard the files appear and the parent fails.
(A child process, because an audit hook cannot be removed once installed.)
"""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
GUARD_FILES = ("conftest.py", "real_root_guard.py")
SRC_DIR = TESTS_DIR.parent / "src"


def _lab(tmp_path, child_tests, ini="[pytest]\naddopts = -p no:cacheprovider\n"):
    lab = tmp_path / "lab"
    repo = lab / "repo"
    (repo / "tests").mkdir(parents=True)
    (lab / "sibling").mkdir()
    for name in GUARD_FILES:
        if (TESTS_DIR / name).is_file():
            shutil.copy2(TESTS_DIR / name, repo / "tests" / name)
    (repo / "pytest.ini").write_text(ini)
    (repo / "tests" / "test_child.py").write_text(textwrap.dedent(child_tests))
    return lab, repo


def _child_env(tmp_path, roots=None, extra=None):
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("MOUSE", "CONNECTOME_ROOT", "SCI_CONNECTOME_ROOT",
                                        "TEST_GUARD_ROOTS_FILE"))}
    if roots is not None:
        listing = tmp_path / "roots.txt"
        listing.write_text("# stand-in real roots\n" + "\n".join(str(r) for r in roots) + "\n")
        env["TEST_GUARD_ROOTS_FILE"] = str(listing)
    env["PYTHONPATH"] = str(SRC_DIR)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.update(extra or {})
    return env


def _run_child(repo, env, *args):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rA", "tests/test_child.py", *args],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=300)
    return proc.stdout + proc.stderr


EVERY_WRITE_ROUTE = """
    import os, shutil, sqlite3, tempfile
    from pathlib import Path

    SIB = Path(os.environ["CHILD_SIBLING"])

    def attempt(fn):
        try:
            fn()
        except PermissionError as exc:
            print("REFUSED", exc)

    def test_open_write():
        attempt(lambda: open(SIB / "open_w.txt", "w").close())

    def test_open_append():
        attempt(lambda: open(SIB / "open_a.txt", "a").close())

    def test_os_open_create():
        attempt(lambda: os.close(os.open(str(SIB / "os_open.txt"), os.O_CREAT | os.O_WRONLY)))

    def test_path_touch():
        attempt(lambda: (SIB / "touched.txt").touch())

    def test_write_text():
        attempt(lambda: (SIB / "write_text.txt").write_text("x"))

    def test_mkdir_and_makedirs():
        attempt(lambda: os.mkdir(SIB / "made_dir"))
        attempt(lambda: os.makedirs(SIB / "made" / "deep"))

    def test_mkstemp():
        attempt(lambda: tempfile.mkstemp(dir=str(SIB), prefix="mkstemp_"))

    def test_mkdtemp():
        attempt(lambda: tempfile.mkdtemp(dir=str(SIB), prefix="mkdtemp_"))

    def test_copy_into_sibling(tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("x")
        attempt(lambda: shutil.copyfile(src, SIB / "copied.txt"))
        attempt(lambda: shutil.copy2(src, SIB / "copied2.txt"))

    def test_rename_into_sibling(tmp_path):
        src = tmp_path / "to_move.txt"
        src.write_text("x")
        attempt(lambda: os.replace(src, SIB / "moved.txt"))
        attempt(lambda: shutil.move(str(src), str(SIB / "moved2.txt")))

    def test_remove_existing_file():
        attempt(lambda: os.remove(SIB / "keep_me.txt"))

    def test_rename_existing_file_away(tmp_path):
        attempt(lambda: os.replace(SIB / "keep_me.txt", tmp_path / "stolen.txt"))

    def test_sqlite_connect():
        attempt(lambda: sqlite3.connect(str(SIB / "db.sqlite")).close())

    def test_relative_path_after_chdir():
        os.chdir(SIB)
        attempt(lambda: open("relative.txt", "w").close())

    def test_dotdot_out_of_the_repository():
        repo_tests = Path(__file__).resolve().parent
        attempt(lambda: open(str(repo_tests) + "/../../sibling/dotdot.txt", "w").close())

    def test_swallowed_refusal():
        try:
            open(SIB / "swallowed.txt", "w").close()
        except Exception:
            pass

    def test_the_repository_own_files_stay_writable():
        here = Path(__file__).resolve().parent
        (here / "own_file.txt").write_text("ok")
        (here / "own_dir").mkdir()
        assert (here / "own_file.txt").read_text() == "ok"
"""

EXPECTED_ABSENT = (
    "open_w.txt", "open_a.txt", "os_open.txt", "touched.txt", "write_text.txt",
    "made_dir", "made", "copied.txt", "copied2.txt", "moved.txt", "moved2.txt",
    "db.sqlite", "relative.txt", "dotdot.txt", "swallowed.txt",
)


def _assert_nothing_written(lab, output):
    sibling = lab / "sibling"
    leftover = sorted(p.name for p in sibling.iterdir() if p.name != "keep_me.txt")
    assert not leftover, "written into the real-root stand-in: %s\n%s" % (leftover, output)
    assert (sibling / "keep_me.txt").read_text() == "real", output


def test_every_write_route_into_a_real_root_is_refused(tmp_path):
    lab, repo = _lab(tmp_path, EVERY_WRITE_ROUTE)
    (lab / "sibling" / "keep_me.txt").write_text("real")
    out = _run_child(repo, _child_env(tmp_path, roots=[lab],
                                      extra={"CHILD_SIBLING": str(lab / "sibling")}))
    _assert_nothing_written(lab, out)
    for name in EXPECTED_ABSENT:
        assert not (lab / "sibling" / name).exists(), name
    # The refusal is loud: the test that caught and swallowed it still fails,
    # and the message says what was refused.
    # (pytest reports a failure raised after the test body as an ERROR line.)
    assert "test_child.py::test_swallowed_refusal - Failed: TEST GUARD" in out, out
    # The repository's own files are exempt although it sits inside the root.
    assert "PASSED tests/test_child.py::test_the_repository_own_files_stay_writable" in out, out
    assert (repo / "tests" / "own_file.txt").read_text() == "ok"


SPELLING_ROUTES = """
    import os
    from pathlib import Path

    def attempt(target):
        try:
            open(target, "w").close()
        except PermissionError as exc:
            print("REFUSED", exc)

    def test_upper_case():
        attempt(os.environ["CHILD_SIBLING"].upper() + os.sep + "upper.txt")

    def test_forward_slashes():
        attempt(os.environ["CHILD_SIBLING"].replace(os.sep, "/") + "/slashes.txt")

    def test_long_path_prefix():
        if os.name != "nt":
            return
        attempt("\\\\\\\\?\\\\" + os.environ["CHILD_SIBLING"] + "\\\\prefixed.txt")

    def test_short_name():
        short = os.environ.get("CHILD_SHORT_SIBLING")
        if short:
            attempt(short + os.sep + "short.txt")
"""


def _short_name(path):
    if os.name != "nt":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 32768)
    if 0 < n < 32768 and os.path.normcase(buf.value) != os.path.normcase(str(path)):
        return buf.value
    return None


def test_a_real_root_is_refused_however_it_is_spelled(tmp_path):
    lab, repo = _lab(tmp_path, SPELLING_ROUTES)
    long_sibling = lab / "A Sibling Folder With A Long Name"
    long_sibling.mkdir()
    short = _short_name(long_sibling)
    # The root is listed in one spelling (long, back slashes); the writes use others.
    out = _run_child(repo, _child_env(tmp_path, roots=[lab], extra={
        "CHILD_SIBLING": str(long_sibling),
        "CHILD_SHORT_SIBLING": short or "",
    }))
    written = sorted(p.name for p in long_sibling.iterdir())
    assert not written, "written through another spelling: %s\n%s" % (written, out)
    assert "REFUSED" in out, out


ONE_WRITE = """
    import os

    def test_write():
        try:
            open(os.path.join(os.environ["CHILD_SIBLING"], "env_root.txt"), "w").close()
        except PermissionError as exc:
            print("REFUSED", exc)
"""


@pytest.mark.parametrize("variable", [
    "MouseReach_PROCESSING_ROOT",    # mixed case, as a real machine may carry it
    "mousebrain_registry_root",      # lower case
    "CONNECTOME_ROOT",
    "SCI_CONNECTOME_ROOT",           # the legacy name config.py still reads
])
def test_a_root_named_by_an_environment_variable_of_any_case_is_guarded(tmp_path, variable):
    lab, repo = _lab(tmp_path, ONE_WRITE)
    sibling = lab / "sibling"
    out = _run_child(repo, _child_env(tmp_path, extra={
        variable: str(lab), "CHILD_SIBLING": str(sibling)}))
    assert not (sibling / "env_root.txt").exists(), out
    assert "REFUSED" in out, out


def test_a_root_the_tool_finds_from_its_own_location_is_guarded(tmp_path):
    """With no variable at all, mousebrain.config finds the lab root from
    where the package sits (the pipeline layout above it). The guard asks the
    tool, so that root is guarded too."""
    lab, repo = _lab(tmp_path, ONE_WRITE)
    (lab / "Tissue" / "MouseBrain_Pipeline" / "3D_Cleared" / "util_Scripts").mkdir(parents=True)
    pkg_src = repo / "src"
    shutil.copytree(SRC_DIR / "mousebrain", pkg_src / "mousebrain",
                    ignore=shutil.ignore_patterns("__pycache__"))
    env = _child_env(tmp_path, extra={"CHILD_SIBLING": str(lab / "sibling")})
    env["PYTHONPATH"] = str(pkg_src)
    out = _run_child(repo, env)
    assert not (lab / "sibling" / "env_root.txt").exists(), out
    assert "REFUSED" in out, out


def test_activate_scripts_and_stored_variables_are_read():
    """The conda activate scripts and the stored environment are read for the
    same names, in every shell's syntax."""
    sys.path.insert(0, str(TESTS_DIR))
    import real_root_guard as g
    root = os.path.abspath(os.sep + "some_lab_root")
    bat = 'set CONNECTOME_ROOT=%s\nset "MouseBrain_REGISTRY_ROOT=%s"\nset OTHER=%s\n' % (root, root, root)
    ps1 = '$env:CONNECTOME_ROOT = "%s"\n' % root
    sh = "export MOUSEREACH_NAS_ROOT='%s'\n" % root
    assert g.roots_from_activate_script(bat) == [root, root]
    assert g.roots_from_activate_script(ps1) == [root]
    assert g.roots_from_activate_script(sh) == [root]
    pairs = [("MouseReach_NAS_DRIVE", root), ("Path", root), ("MOUSEBRAIN_INJURY_MODEL", "contusion")]
    assert g.roots_from_environment(pairs) == [root]


@pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows")
def test_a_mapped_drive_and_its_share_name_are_one_root():
    sys.path.insert(0, str(TESTS_DIR))
    import real_root_guard as g
    drives = {"Q:": "\\\\fileserver\\share"}
    guard = g.Guard(["Q:\\Lab"], drives=drives)
    assert guard.refused_path("\\\\FileServer\\Share\\Lab\\x.txt")
    assert guard.refused_path("\\\\?\\UNC\\fileserver\\share\\lab\\x.txt")
    assert guard.refused_path("q:/lab/x.txt")
    assert not guard.refused_path("\\\\fileserver\\share\\LabOther\\x.txt")
    guard_unc = g.Guard(["\\\\fileserver\\share\\Lab"], drives=drives)
    assert guard_unc.refused_path("Q:\\Lab\\sub\\x.txt")


def test_os_open_is_judged_by_its_flags():
    """The audit event of os.open carries mode None and int flags."""
    sys.path.insert(0, str(TESTS_DIR))
    import real_root_guard as g
    root = os.path.abspath(os.sep + "some_lab_root")
    guard = g.Guard([root], drives={})
    target = os.path.join(root, "f.txt")
    guard(("open"), (target, None, os.O_RDONLY))      # a read passes
    with pytest.raises(PermissionError):
        guard("open", (target, None, os.O_WRONLY | os.O_CREAT))
    with pytest.raises(PermissionError):
        guard("open", (target, "r+", os.O_RDWR))
