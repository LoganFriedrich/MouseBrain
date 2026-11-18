"""The test guard: no test may write into this machine's REAL lab folders.

Plain-language description for operators: docs/pipeline/TEST_GUARD.md.

WHY THIS EXISTS
---------------
Tests build their own folders under pytest's tmp_path and point the tool at
them. A test that forgets one setting acts on live lab data instead, and
nothing says so: the run passes, and a junk file sits in a real folder (this
has happened in a sister tool's suite -- junk review files written by tests
into a live review store). Telling people to unset variables before a run is
not enough either: a run once unset only the UPPER-case names while the
machine carried a mixed-case one, so the "isolated" run still pointed at the
live pipeline. The guard therefore does not trust the run's environment: it
finds the real folders itself and REFUSES writes there.

WHAT IT REFUSES
---------------
Every write or create, by any route Python audits (sys.addaudithook): open()
in a write, append, create or update mode; os.open with write/create/truncate
flags (the audit event for os.open carries mode None and the int flags, so
the flags are what is checked -- this also covers Path.touch and
tempfile.mkstemp); os.mkdir / os.makedirs / tempfile.mkdtemp; rename and
replace (source AND target: moving a real file away is a write too); remove,
rmdir, rmtree; shutil copies (target), move (both ends), archives; chmod,
utime, truncate, link, symlink; and sqlite3.connect (a connection to a real
database can write journals even when only reading). The refusal is a
PermissionError raised inside the call, so nothing is written. Each attempt
is also recorded, so a test whose code swallows the error still fails.

WHAT COUNTS AS A REAL FOLDER (collected once, when the test run starts)
----------------------------------------------------------------------
  1. Environment variables of ANY letter case whose name starts with MOUSE
     (MOUSEBRAIN_*, MOUSEREACH_*, MouseReach_*, MOUSEDB_* ...) or is
     CONNECTOME_ROOT / SCI_CONNECTOME_ROOT, when the value is an absolute path.
  2. The same names in the machine's PERSISTENT environment (on Windows the
     user and system environment in the registry), so a run that cleared
     them from its own shell is still guarded.
  3. The conda environment's activate scripts (<env>/etc/conda/activate.d),
     where this tool's CONNECTOME_ROOT is normally set.
  4. What mousebrain.config itself resolves (it can find the lab root from
     where the package sits, with no variable at all) and the folders it
     derives: the pipeline root, 3D_Cleared and its 2_Data_Summary (where
     calibration_runs.csv lives), 2D_Slices, and the analysis registry root.
  5. An optional list the person running the tests supplies: a text file,
     one absolute path per line, named by the variable TEST_GUARD_ROOTS_FILE.
     (A file and not a variable holding the paths, so the paths never appear
     in the run's environment, where an isolation check looks for them.)

Every root is matched however it is spelled: letter case, / or \\, 8.3 short
names (per folder level), the \\\\?\\ long-path prefix, a mapped drive letter
or its \\\\server\\share form (and the other way round), a subst drive, and
relative paths or ".." (resolved against the current folder). All spellings
are built here, ONCE, at import. The audit hook itself only does string work:
it runs on every file open of the whole run, and doing disk or network
lookups there would slow every test and could recurse into itself.

WHAT IS EXEMPT
--------------
The repository's own files (source, tests, docs, __pycache__), even when the
repository sits inside a real lab root -- its siblings stay guarded. The
system temp folder is exempt only when it lies inside a real root (a root set
to a whole drive), so tmp_path keeps working; a root that lies inside the
temp folder stays guarded.

LIMITS
------
A child process started by a test is not watched (audit hooks do not cross
processes). Paths given relative to a directory file descriptor (dir_fd) are
not resolved.
"""
import itertools
import os
import re
import sys
import tempfile

ROOTS_FILE_VARIABLE = "TEST_GUARD_ROOTS_FILE"

# Names whose values are lab locations for this tool or its sister tools.
# Matched case-INSENSITIVELY: a machine may carry MouseReach_PROCESSING_ROOT.
_ROOT_VARIABLE = re.compile(r"^(?:mouse[a-z0-9]*_\w+|(?:sci_)?connectome_root)$", re.I)

# One assignment in a conda activate script: cmd `set NAME=value` /
# `set "NAME=value"`, PowerShell `$env:NAME = "value"`, sh `export NAME=value`.
_ACTIVATE_ASSIGNMENT = re.compile(
    r"""^\s*(?:@?set\s+"?|\$env:|export\s+)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["']?([^"'\r\n]*)""",
    re.I | re.M)

# Audit events that change the file system, and which of their arguments are
# the paths changed. Copies list only their target: reading a real file is
# not a write.
_PATH_ARGS = {
    "os.mkdir": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.remove": (0,),
    "os.rmdir": (0,),
    "os.truncate": (0,),
    "os.link": (1,),
    "os.symlink": (1,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.chflags": (0,),
    "os.lchflags": (0,),
    "os.utime": (0,),
    "os.setxattr": (0,),
    "os.removexattr": (0,),
    "shutil.copyfile": (1,),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
    "shutil.copytree": (1,),
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
    "shutil.make_archive": (0,),
    "shutil.unpack_archive": (1,),
    "tempfile.mkstemp": (0,),
    "tempfile.mkdtemp": (0,),
    "sqlite3.connect": (0,),
}

_WRITE_FLAGS = (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)


# ---------------------------------------------------------------------------
# Spelling: pure string work (safe inside the audit hook)
# ---------------------------------------------------------------------------

def normalize(path):
    """One comparable spelling of `path`, or None when it is not a path.

    String work only: strips the \\\\?\\ and \\\\.\\ prefixes, makes the path
    absolute against the current folder (which also folds "..") and folds
    letter case and slashes the way the operating system compares them.
    """
    if isinstance(path, int) or path is None:
        return None
    try:
        text = os.fsdecode(os.fspath(path))
    except (TypeError, ValueError):
        return None
    if not text or "\x00" in text:
        return None
    if os.name == "nt":
        text = text.replace("/", "\\")
        upper = text[:8].upper()
        if upper.startswith("\\\\?\\UNC\\") or upper.startswith("\\\\.\\UNC\\"):
            text = "\\\\" + text[8:]
        elif text.startswith("\\\\?\\") or text.startswith("\\\\.\\"):
            text = text[4:]
    try:
        return os.path.normcase(os.path.abspath(text))
    except (TypeError, ValueError, OSError):
        return None


def is_under(path, root):
    """True when normalized `path` is `root` or lies inside it."""
    if path == root:
        return True
    prefix = root if root.endswith(os.sep) else root + os.sep
    return path.startswith(prefix)


# ---------------------------------------------------------------------------
# Spellings of one root (done once, at import; may touch the disk)
# ---------------------------------------------------------------------------

def drive_map():
    """{'X:': '\\\\server\\share' or 'C:\\folder'} for every mapped network
    drive and subst drive on this machine (Windows only; empty elsewhere)."""
    found = {}
    if os.name != "nt":
        return found
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return found
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = letter + ":"
        try:
            size = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.mpr.WNetGetConnectionW(drive, buf, ctypes.byref(size)) == 0 and buf.value:
                found[drive] = buf.value
                continue
        except Exception:
            pass
        try:
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.QueryDosDeviceW(drive, buf, 1024):
                target = buf.value
                if target.startswith("\\??\\"):  # a subst drive
                    found[drive] = target[4:]
        except Exception:
            pass
    return found


def _win_name(function_name, path):
    try:
        import ctypes
        fn = getattr(ctypes.windll.kernel32, function_name)
        buf = ctypes.create_unicode_buffer(32768)
        n = fn(str(path), buf, 32768)
        if 0 < n < 32768:
            return buf.value
    except Exception:
        pass
    return None


def _component_spellings(path):
    """Each spelling of `path` that mixes long and 8.3 short names folder by
    folder (Windows; the path or its nearest existing parent must exist)."""
    if os.name != "nt":
        return set()
    existing = path
    rest = []
    while existing and not os.path.exists(existing):
        parent, name = os.path.split(existing)
        if parent == existing:
            return set()
        rest.insert(0, name)
        existing = parent
    if not existing:
        return set()
    long_path = _win_name("GetLongPathNameW", existing) or existing
    drive, tail = os.path.splitdrive(long_path)
    parts = [p for p in tail.split(os.sep) if p]
    choices = []
    walked = drive + os.sep
    for part in parts:
        walked = os.path.join(walked, part)
        short = _win_name("GetShortPathNameW", walked)
        short_part = os.path.basename(short) if short else part
        choices.append({part, short_part})
    combos = set()
    for pick in itertools.islice(itertools.product(*choices), 512):
        combos.add(os.path.join(drive + os.sep, *pick, *rest))
    return combos


def spellings(root, drives=None):
    """Every normalized spelling of `root` (see the module docstring)."""
    base = normalize(root)
    if not base:
        return set()
    drives = drive_map() if drives is None else drives
    found = {base}
    try:
        found.add(normalize(os.path.realpath(base)))
    except (OSError, ValueError):
        pass
    for s in list(found):
        found.update(normalize(c) for c in _component_spellings(s))
    # Mapped/subst drive letter <-> its target, both ways.
    mapped = {normalize(d + os.sep).rstrip(os.sep): normalize(t) for d, t in drives.items()}
    for s in list(found):
        for drive, target in mapped.items():
            if not target:
                continue
            if s.startswith(drive + os.sep) or s == drive:
                found.add(normalize(target + s[len(drive):]))
            if is_under(s, target):
                found.add(normalize(drive + os.sep + s[len(target):].lstrip(os.sep)))
    found.discard(None)
    return found


# ---------------------------------------------------------------------------
# Where the real roots come from (done once, at import)
# ---------------------------------------------------------------------------

def _path_values(value):
    out = []
    for piece in str(value).split(os.pathsep):
        piece = piece.strip().strip('"').strip()
        if piece and os.path.isabs(piece):
            out.append(piece)
    return out


def roots_from_environment(environ):
    """Absolute paths held by lab variables (any letter case) in `environ`
    (a mapping, or (name, value) pairs)."""
    roots = []
    pairs = environ.items() if hasattr(environ, "items") else environ
    for name, value in pairs:
        if _ROOT_VARIABLE.match(name or ""):
            roots.extend(_path_values(value))
    return roots


def roots_from_activate_script(text):
    """Absolute paths a conda activate script assigns to lab variables."""
    roots = []
    for name, value in _ACTIVATE_ASSIGNMENT.findall(text or ""):
        if _ROOT_VARIABLE.match(name):
            roots.extend(_path_values(value))
    return roots


def _persistent_environment():
    """(name, value) pairs of the machine's stored user and system environment
    (Windows registry), which a shell that unset its variables still has on
    file. Pairs, not a dict: one name may be set in both places."""
    found = []
    if os.name != "nt":
        return found
    try:
        import winreg
    except Exception:
        return found
    places = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for hive, key_path in places:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                i = 0
                while True:
                    try:
                        name, value, _kind = winreg.EnumValue(key, i)
                    except OSError:
                        break
                    i += 1
                    if isinstance(value, str):
                        found.append((name, value))
        except OSError:
            continue
    return found


def _activate_scripts():
    prefixes = {sys.prefix, os.environ.get("CONDA_PREFIX") or ""}
    texts = []
    for prefix in prefixes:
        if not prefix:
            continue
        folder = os.path.join(prefix, "etc", "conda", "activate.d")
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            try:
                with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
            except OSError:
                continue
    return texts


def _roots_from_tool():
    """What mousebrain.config resolves on this machine, and what it derives."""
    roots = []
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mousebrain import config
            if config.is_configured():
                for name in ("ROOT_PATH", "PIPELINE_ROOT", "CLEARED_3D_DIR",
                             "DATA_SUMMARY_DIR", "CALIBRATION_RUNS_CSV",
                             "SLICES_2D_DIR"):
                    value = getattr(config, name, None)
                    if value:
                        roots.append(str(value))
            from mousebrain.analysis_registry import default_registry_root
            registry = default_registry_root()
            if registry:
                roots.append(str(registry))
    except Exception:
        pass
    return roots


def roots_from_file(path):
    roots = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    roots.extend(_path_values(line))
    except OSError:
        pass
    return roots


def collect_real_roots():
    """Every real lab root this machine knows of (see the module docstring)."""
    roots = []
    roots += roots_from_environment(os.environ)
    roots += roots_from_environment(_persistent_environment())
    for text in _activate_scripts():
        roots += roots_from_activate_script(text)
    roots += _roots_from_tool()
    listed = os.environ.get(ROOTS_FILE_VARIABLE)
    if listed:
        roots += roots_from_file(listed)
    unique = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

class RealRootRefused(PermissionError):
    """A test tried to write into a real lab folder."""


class Guard:
    """Decides, by string work only, whether an audited event writes into a
    real root; refuses it and records it."""

    def __init__(self, roots, exempt=(), drives=None):
        drives = drive_map() if drives is None else drives
        exempt_spellings = [spellings(e, drives) for e in exempt]
        self.roots = []      # [(root shown in messages, {spellings}, {exempt spellings inside it})]
        for root in roots:
            rs = spellings(root, drives)
            if not rs:
                continue
            inside = set()
            for es in exempt_spellings:
                if any(is_under(e, r) for e in es for r in rs):
                    inside |= es
            self.roots.append((root, rs, inside))
        self.hits = []       # [(event, normalized path)]

    def __bool__(self):
        return bool(self.roots)

    def refused_path(self, candidate):
        """The normalized path when writing `candidate` must be refused."""
        n = normalize(candidate)
        if not n:
            return None
        for _root, rs, inside in self.roots:
            if any(is_under(n, r) for r in rs):
                if any(is_under(n, e) for e in inside):
                    continue
                return n
        return None

    def candidates(self, event, args):
        if event == "open":
            # args: (path, mode, flags). builtin open passes a mode string;
            # os.open passes mode None and only the int flags. Read-only
            # opens (every import) must leave here as cheaply as possible.
            if len(args) < 3:
                return ()
            mode, flags = args[1], args[2]
            writes = isinstance(flags, int) and flags >= 0 and bool(flags & _WRITE_FLAGS)
            if not writes and isinstance(mode, str):
                writes = any(c in mode for c in "wax+")
            return (args[0],) if writes else ()
        where = _PATH_ARGS.get(event)
        if where is None:
            return ()
        return tuple(args[i] for i in where if i < len(args))

    def __call__(self, event, args):
        for candidate in self.candidates(event, args):
            hit = self.refused_path(candidate)
            if hit:
                self.hits.append((event, hit))
                raise RealRootRefused(
                    "TEST GUARD REFUSED %s on a real lab folder: %s "
                    "(point the test at tmp_path; see docs/pipeline/TEST_GUARD.md)"
                    % (event, hit))
