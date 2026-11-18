# The test guard: tests never write into your real data

MouseBrain's automated tests (the files in `tests/`) build their own small,
throwaway folders and point the tool at them. If one test forgets to do that,
it would act on your lab's real folders instead -- add a row to the real
`calibration_runs.csv`, drop a file into the real analysis registry, create a
folder on the shared drive -- and nothing would say so.

The **test guard** stops that. While the tests run, every attempt to write
into one of your real lab folders is **refused** before anything is written,
and the test that tried fails with a message naming the folder.

You do not need to switch it on. It is part of the test suite
(`tests/conftest.py` and `tests/real_root_guard.py`) and runs whenever the
tests run.

## What it refuses

Inside a real lab folder, during the whole test run:

- creating, writing, appending to or overwriting a file, by any route
  (`open(..., "w")`, low-level `os.open` with write flags, "touch", making a
  temporary file there);
- creating a folder (one level or many);
- renaming or moving a file into **or out of** a real folder;
- deleting a file or a folder;
- copying a file into a real folder (copying *from* one is a read, and allowed);
- changing a file's permissions or time stamp, linking to it;
- opening a database file there.

Reading is never refused.

## Which folders count as "real"

The guard works them out once, when the test run starts, from every place the
tool itself would learn them on a working machine:

1. **Environment variables**, in any mix of upper and lower case: every
   variable whose name begins with `MOUSE` (for example `MOUSEBRAIN_REGISTRY_ROOT`,
   `MOUSEBRAIN_APPROVED_METHOD`, or a sister tool's `MouseReach_PROCESSING_ROOT`)
   and `CONNECTOME_ROOT` / `SCI_CONNECTOME_ROOT`, when the value is a full path.
2. **The machine's saved environment** (on Windows: the user and system
   variables stored in the registry). So clearing the variables in the window
   you run the tests from does not switch the guard off.
3. **The conda environment's activate scripts** (`<env>\etc\conda\activate.d\`),
   which is where `CONNECTOME_ROOT` is normally set.
4. **What MouseBrain itself finds**: the lab root it detects from where it is
   installed, and the folders it derives from it -- the pipeline folder,
   `3D_Cleared` and its `2_Data_Summary` (which holds `calibration_runs.csv`),
   `2D_Slices`, and the analysis registry (`Registry`).
5. **A list you supply** (optional, see below).

Each folder is recognized however it is written: capital or small letters,
`/` or `\`, short "8.3" names such as `PROGRA~1`, the long-path prefix
`\\?\`, a mapped drive letter or the `\\server\share` name behind it (either
way round), and relative paths.

### Supplying your own list

If you want extra folders guarded -- or you run the tests in a window where
none of the sources above can be seen -- write the folders into a plain text
file, one full path per line (lines starting with `#` are ignored), and point
the variable `TEST_GUARD_ROOTS_FILE` at that file before running the tests:

```powershell
# PowerShell
$env:TEST_GUARD_ROOTS_FILE = "<a folder of your own>\guarded_folders.txt"
python -m pytest tests
```

```bash
# bash
export TEST_GUARD_ROOTS_FILE="<a folder of your own>/guarded_folders.txt"
python -m pytest tests
```

The paths go in a file, not in the variable itself, so they never show up in
the environment of the test run.

## What stays writable

- **The repository's own files** (its source, the tests, the docs, Python's
  `__pycache__` folders), even when the repository sits inside a real lab
  folder. Folders *beside* the repository stay guarded.
- **The system temp folder**, where the tests make their throwaway folders --
  only relevant if a real folder is a whole drive that also holds the temp
  folder.

## What a refusal looks like

The write itself fails with:

```
PermissionError: TEST GUARD REFUSED open on a real lab folder: <the path>
(point the test at tmp_path; see docs/pipeline/TEST_GUARD.md)
```

and the test is reported as failed (an `ERROR ... at teardown` line when the
test's own code caught the error), with:

```
TEST GUARD: this test tried to write into a REAL lab folder (the write was
refused, nothing was written). Point the test at tmp_path instead:
  open: <the path>
```

Nothing was written. The fix is always in the test: give it its own folder
(pytest's `tmp_path`) for the setting or variable it forgot. Never fix it by
emptying the list of real folders.

If an attempt happens outside any test (while the test files are being
loaded), the run still ends as failed, with a `TEST GUARD:` note at the end.

## What it cannot see

- A **separate program** a test starts (a subprocess) is not watched. Tests
  that start one must give it a throwaway folder themselves.
- A path given relative to an open folder handle (rare, low-level code).
- A write made **through a directory junction or symbolic link**. The guard
  compares how the path is written; it does not follow links inside it. So
  `open("<a junction pointing into a real folder>/x", "w")` is not refused.
  The tests never make such links; do not add one that points into real data.

## When the guard is off, and one thing that can trip it

- If none of the sources above yields a real folder (and no list is
  supplied), the guard is **off** for that run, because there is nothing to
  protect. Supply a list to be sure it is on.
- If the package is imported from a copy that sits inside a real folder other
  than the repository, Python's attempt to write its `.pyc` cache there is
  refused and counted as a hit, so the test fails. Set `PYTHONPYCACHEPREFIX`
  to a throwaway folder (the isolation block for test runs does) and it does
  not happen.

## pytest's cache never goes into the repository

pytest normally keeps a `.pytest_cache` folder in the repository, made by
renaming a temporary `pytest-cache-files-...` folder into place. On a network
drive that rename can fail and leave the temporary folder behind with
permissions only an administrator can clear. So:

- the repository's pytest settings (`pyproject.toml`) switch the cache off
  (`-p no:cacheprovider`);
- if a run switches it back on anyway (for example with `-o addopts=`), the
  test suite moves the cache into the system temp folder.

Leftover `pytest-cache-files-...` folders from before this was in place are
skipped by the test run; an administrator can delete them.
