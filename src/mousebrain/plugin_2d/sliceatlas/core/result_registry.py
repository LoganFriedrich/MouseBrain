"""Register what the 2D window and run_coloc save, in MouseBrain's own registry.

WHY: the 2D window (Save State, Count in ROI(s), the automatic save when
another image is opened, Export Batch Results) and ``run_coloc --save``
wrote their results only to files beside the images or to a path a person
chose. Nothing reached the analysis registry, so whatever pulls the
registry never saw them and nothing recorded which settings produced them.
Now each of those saves also registers what it wrote -- the files (copied
into the registry), the settings as ``method_params``, the image or folder
as ``source_files`` -- with the registry's usual provenance (time, machine,
method hash). WHERE the files themselves are written is unchanged.

WHICH ANALYSIS. Each producer has its own analysis, named after the project:

    <project>_Window_Results   the 2D window's saved results, one entry per
                               image (key: the image's file name without
                               extension)
    <project>_Window_Batch     Export Batch Results, one entry per batch
                               folder (key: the folder's name)
    <project>_Coloc_CLI        run_coloc --save, one entry per image

They are never registered under ``<project>_Detection`` or
``<project>_ROI_Analysis``. WHY: those hold the detection batch's results
and the ROI counts tied to them. A save from the window is often a trial
with other settings; registered there it would replace the batch's entry
for the image and retire the ROI counts resting on it. They record their
own settings and do not follow the shared approved method
(``follows_approved_method=False``): the window's particle settings are not
the batch's parameters, so comparing them would call every entry stale.
A lab can still give each of these analyses an approved method of its own
in approved_method.json.

THE PROJECT is the configured 2D project, ``MOUSEBRAIN_2D_PROJECT`` (the
same setting the detection batch uses when it is given no --project). It is
never guessed from a file or folder name. When it is not set, or no
registry root is configured, nothing is registered and the caller says so.
"""
import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ENV = "MOUSEBRAIN_2D_PROJECT"

# Settings that change on every save without changing how the results were
# made. They are left out of method_params, so an unchanged result saved
# again hashes the same and is not archived as a replacement.
VOLATILE_SETTINGS = ("saved_at", "layer_visibility")
# Tracker run ids: provenance (which run made the results), recorded with
# the source files rather than in the method.
RUN_ID_SETTINGS = ("particle_run_id", "detection_run_id", "coloc_run_id")

# The result tables of an image's saved state that are registered. The
# label images (.npz) are not: they are large, and the tables hold every
# number derived from them. Drawn regions are inputs, not results.
WINDOW_RESULT_TABLES = {
    "measurements": "measurements.csv",
    "particles": "particles.csv",
    "roi_detail": "roi_detail.csv",
}


def window_analysis_name(project: str) -> str:
    return f"{project}_Window_Results"


def window_batch_analysis_name(project: str) -> str:
    return f"{project}_Window_Batch"


def coloc_cli_analysis_name(project: str) -> str:
    return f"{project}_Coloc_CLI"


def configured_project() -> Optional[str]:
    value = os.environ.get(PROJECT_ENV, "").strip()
    return value or None


NOT_CONFIGURED_PROJECT = (
    f"not registered: no 2D project is configured (set {PROJECT_ENV} to the "
    f"project code)")
NOT_CONFIGURED_ROOT = (
    "not registered: no registry root is configured (set "
    "MOUSEBRAIN_REGISTRY_ROOT, or CONNECTOME_ROOT)")


def _open(analysis: str, lock_wait_seconds: Optional[float]):
    """The analysis's registry, or (None, why) when nothing is configured."""
    from mousebrain.analysis_registry import AnalysisRegistry, default_registry_root

    root = default_registry_root()
    if root is None:
        return None, NOT_CONFIGURED_ROOT
    reg = AnalysisRegistry(analysis_name=analysis, registry_root=root,
                           follows_approved_method=False)
    if lock_wait_seconds is not None:
        # The window must not freeze for the registry's usual five minutes
        # while another machine holds the lock (the tracker does the same).
        reg.LOCK_WAIT_SECONDS = lock_wait_seconds
    return reg, ""


def _has_manifest(analysis: str) -> bool:
    from mousebrain.analysis_registry import default_registry_root

    root = default_registry_root()
    return root is not None and (Path(root) / "exports" / analysis / "registry.json").is_file()


def _split_settings(settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    method = {k: v for k, v in (settings or {}).items()
              if k not in VOLATILE_SETTINGS and k not in RUN_ID_SETTINGS}
    runs = {k: str(settings[k]) for k in RUN_ID_SETTINGS
            if (settings or {}).get(k)}
    return method, runs


def _table_counts(path: Path) -> Dict[str, Any]:
    """Rows of a result table and, when it has an is_positive column, how
    many are positive."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(line for line in f if not line.startswith("#"))]
    counts: Dict[str, Any] = {"rows": len(rows)}
    if rows and "is_positive" in rows[0]:
        counts["positive"] = sum(1 for r in rows
                                 if str(r.get("is_positive", "")).strip().lower()
                                 in ("true", "1", "1.0", "yes"))
    return counts


def register_window_state(image_path, analysis_dir, settings: Dict[str, Any],
                          written, may_retire: bool = False,
                          lock_wait_seconds: Optional[float] = None) -> Tuple[Optional[bool], str]:
    """Register the result tables the window has just written for an image.

    *written* names the files of the saved state this save wrote; only the
    result tables among them are registered. When it wrote none and
    *may_retire* is True (the window owns the folder, so the folder holds no
    result either: a re-run that produced nothing, results cleared), the
    image's current entry is retired instead, so the registry never shows
    results the saved state no longer has.

    Returns (True, text) when registered or retired (text "" when there was
    nothing to do), (None, why) when nothing is configured to register
    into, and raises on a registry error (the caller reports it).
    """
    project = configured_project()
    if project is None:
        return None, NOT_CONFIGURED_PROJECT
    image_path, analysis_dir = Path(image_path), Path(analysis_dir)
    written = set(written or ())
    files = {kind: str(analysis_dir / name) for kind, name in WINDOW_RESULT_TABLES.items()
             if name in written and (analysis_dir / name).is_file()}
    if not files and not may_retire:
        return True, ""
    if not files and not _has_manifest(window_analysis_name(project)):
        # Nothing was ever registered there, so nothing to retire; opening
        # it would create an empty analysis folder, which whoever pulls the
        # registry would list as an analysis with no results.
        return True, ""
    reg, why = _open(window_analysis_name(project), lock_wait_seconds)
    if reg is None:
        return None, why
    sample = image_path.stem
    if not files:
        entry = reg.get_entry(sample)
        if entry and entry.get("is_current", True):
            if not reg.invalidate(sample=sample, include_own_folder=True):
                raise RuntimeError(
                    "the saved state holds no results any more, but its registry "
                    "entry could not be retired (a file would not move)")
            return True, f"registry entry of {sample} retired (no results saved)"
        return True, ""
    results = {kind: _table_counts(Path(path)) for kind, path in files.items()}
    method, runs = _split_settings(settings)
    source = {"image": str(image_path), "analysis_folder": str(analysis_dir)}
    source.update(runs)
    reg.register_output(sample=sample, category="window_results", files=files,
                        results=results, method_params=method, source_files=source)
    return True, f"registered in {reg.analysis_name}"


def register_window_batch(batch_folder, written: Dict[str, str], settings: Dict[str, Any],
                          results: Dict[str, Any],
                          lock_wait_seconds: Optional[float] = None) -> Tuple[Optional[bool], str]:
    """Register what Export Batch Results wrote (summary, detail, settings
    files, wherever the person chose to save them) under the batch folder's
    name. Returns as register_window_state."""
    project = configured_project()
    if project is None:
        return None, NOT_CONFIGURED_PROJECT
    reg, why = _open(window_batch_analysis_name(project), lock_wait_seconds)
    if reg is None:
        return None, why
    batch_folder = Path(batch_folder)
    method, runs = _split_settings(settings)
    source = {"batch_folder": str(batch_folder)}
    source.update(runs)
    reg.register_output(sample=batch_folder.name, category="window_batch",
                        files={k: str(v) for k, v in written.items()},
                        results=results, method_params=method, source_files=source)
    return True, f"registered in {reg.analysis_name}"


def register_coloc_cli(image_path, written: Dict[str, str], method_params: Dict[str, Any],
                       results: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """Register what run_coloc --save wrote. Returns as register_window_state."""
    project = configured_project()
    if project is None:
        return None, NOT_CONFIGURED_PROJECT
    reg, why = _open(coloc_cli_analysis_name(project), None)
    if reg is None:
        return None, why
    image_path = Path(image_path)
    reg.register_output(sample=image_path.stem, category="coloc_cli",
                        files={k: str(v) for k, v in written.items()},
                        results=results, method_params=method_params,
                        source_files={"image": str(image_path)})
    return True, f"registered in {reg.analysis_name}"
