#!/usr/bin/env python3
"""
analysis_registry.py - Registry for analysis outputs with provenance tracking.

Records the outputs of an analysis (figures, CSVs, counts) together with the
method parameters, source files and timestamp that produced them, so that a
result can always be traced back to how it was made and flagged as stale when
the method changes.

Why the registry lives inside the pipeline
------------------------------------------
This tool stands alone. It records its own outputs, with provenance, in a
folder it owns; it never writes into another tool's database. An integrator
(for example a lab database) may PULL from that folder whenever it likes --
the registry.json manifest and the audit log tell it what is there, how it was
produced, and whether it is still current. The registry knows nothing about
any consumer, so nothing here depends on one being installed.

Layout (rooted at the registry root, by default <pipeline root>/Registry/)
--------------------------------------------------------------------------
    Registry/
    +-- exports/{analysis_name}/       Data outputs for one analysis
    |   +-- registry.json              Manifest: one entry per registered output
    |   +-- {sample}/                  Detection outputs, per sample
    |   +-- {animal}/{region}/         ROI outputs, grouped by animal and region
    |   +-- roi_summary_{region}.csv   Per-region summary (ROI analyses)
    +-- figures/{analysis_name}/       Figures, organised by animal/region
    |   +-- {animal}/{region}/         (created on the first figure write)
    +-- logs/{analysis_name}.log       Audit trail, one line per event
    +-- logs/{analysis_name}.lock      The write lock, present only while a
    |                                  write is in progress. For that moment
    |                                  logs/ may also hold temporary files
    |                                  whose names start with "." and end in
    |                                  ".tmp" or ".part", and the lock's clock
    |                                  probes ".clock-<8 hex>"; while a lock
    |                                  left by a crashed writer is taken over,
    |                                  "{analysis_name}.lock.break" is held
    |                                  by the writer doing it (see
    |                                  mousebrain.file_lock). Any of these
    |                                  left behind by a crash is safe to
    |                                  delete when nothing is registering.
    +-- logs/{analysis_name}.registry.json.unreadable-{time}
    |                                  A damaged manifest, set aside for repair
    |                                  (only ever created when one was damaged)
    +-- approved_method.json           Optional: a lab's approved parameters,
                                       flat or per analysis (approved_method_for)

    Each manifest's header records the approved method its analysis is judged
    by, refreshed on every write; every write holds the analysis's lock; a
    manifest that cannot be parsed is set aside, never emptied, a new manifest
    is started in its place at once, and the set-aside file is listed in that
    manifest's header until a person deals with it. Operator
    guide: docs/pipeline/ANALYSIS_REGISTRY.md.

    WHY the lock, its probes, temporary files and set-aside manifests are
    kept OUT of exports/{analysis_name}/: an integrator mirrors every file
    under that folder. A lock caught there by a nightly pull would be copied
    as if it were a result (and archived as a vanished result the next
    night), and a probe or temporary file that vanished between the
    integrator listing the folder and reading it would stop its whole pull.
    So everything under exports/ and figures/ is a registered output or a
    manifest.

    WHY in logs/ and not in a folder of their own: logs/ already exists in
    every registry, sits on the same drive (so files can be renamed into
    place), and an integrator takes only the ``{analysis_name}.log`` files
    from it. No name used here ends in ".log". Adding new folders to a
    registry that people browse was avoided on purpose.

    Invalidated outputs are moved to <their folder>/_archived/<timestamp>/.
    Paths recorded in registry.json are RELATIVE to the registry root, so the
    whole folder can be moved or pulled elsewhere and still resolve.

Root resolution (see default_registry_root):
    1. MOUSEBRAIN_REGISTRY_ROOT environment variable
    2. <PIPELINE_ROOT>/Registry when mousebrain.config resolved a pipeline root
    3. nothing -- the registry refuses with a message rather than guessing

Usage:
    from mousebrain.analysis_registry import AnalysisRegistry, get_approved_method

    registry = AnalysisRegistry(analysis_name="ENCR_ROI_Analysis")

    # Register a processed sample
    out_paths = registry.register_output(
        sample="E01_01_S13_DCN",
        category="roi_analysis",
        files={"figure": "/path/to/fig.png", "roi_counts": "/path/to/counts.csv"},
        results={"n_nuclei": 13, "n_positive": 12, "positive_fraction": 0.923},
        method_params=get_approved_method(),
        source_files={"nd2": "/path/to/E01_01_S13_DCN.nd2"},
    )

    # Check which samples are stale after a method change
    stale = registry.get_stale_samples(new_method_params)

Inspect a registry from the command line:
    mousebrain-registry --name ENCR_ROI_Analysis --stale --summary
"""

import csv
import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from mousebrain.file_lock import exclusive_file_lock

__all__ = [
    "AnalysisRegistry",
    "RegistryUnreadableError",
    "DEFAULT_METHOD",
    "APPROVED_METHOD_FILENAME",
    "default_registry_root",
    "get_approved_method",
    "approved_method_for",
    "approved_method_source",
    "get_database_path",
    "parse_sample_id",
    "roi_results_from_rows",
    "main",
]


# =============================================================================
# METHOD DEFINITION
# =============================================================================

# Built-in method parameters used when a lab has not recorded its own (see
# get_approved_method). Each key is a parameter of the 2D slice analysis:
DEFAULT_METHOD = {
    "detection": "threshold+log",       # nucleus detection: intensity threshold + Laplacian-of-Gaussian
    "threshold_fraction": 0.20,         # threshold as a fraction of the channel's intensity range
    "log_threshold": 0.005,             # LoG response cut-off for a blob to count as a nucleus
    "log_decision_tree": True,          # resolve threshold/LoG disagreements with the decision tree
    "size_filter_um": [8, 25],          # accepted nucleus diameter range, micrometres
    "colocalization": "background_mean",  # positive if soma signal exceeds the background mean
    "sigma_threshold": 0,               # extra margin above background, in background std devs
    "soma_dilation": 6,                 # pixels to dilate each nucleus to sample its soma
    "background": "gmm",                # background model: gaussian mixture on tissue pixels
    "background_percentile": 10,        # percentile used when the GMM is not applicable
    "bg_exclusion_dilation": 50,        # dilation (iterations) around nuclei excluded from background
}

# Name of the optional per-installation override, looked up under the registry root.
APPROVED_METHOD_FILENAME = "approved_method.json"


def _load_method_file(path: Path, origin: str) -> Dict[str, Any]:
    """Read a method-parameter JSON file (a flat object of name -> value).

    WHY this raises instead of falling back to DEFAULT_METHOD: an override that
    is configured but unreadable would otherwise be ignored silently, every
    output would be hashed against the wrong method, and the staleness check
    would report "current" for results produced with parameters nobody chose.
    """
    try:
        # utf-8-sig: this file is written by hand, and an editor or shell
        # that adds a UTF-8 byte-order mark must not turn a valid file into
        # a "could not be read" error.
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"Approved-method file ({origin}) could not be read: {path}: {e}"
        ) from e
    if not isinstance(data, dict) or not data:
        raise RuntimeError(
            f"Approved-method file ({origin}) must hold a non-empty JSON object "
            f"of parameter name -> value: {path}"
        )
    return data


def approved_method_source(registry_root: Optional[Union[str, Path]] = None) -> str:
    """Where the approved method is read from on THIS machine: the
    MOUSEBRAIN_APPROVED_METHOD file, the registry's approved_method.json, or
    the built-in parameters. Recorded in each manifest header (see
    AnalysisRegistry._set_approved_header) because the environment variable
    wins and can differ from machine to machine."""
    env_path = os.environ.get("MOUSEBRAIN_APPROVED_METHOD")
    if env_path:
        return "MOUSEBRAIN_APPROVED_METHOD=%s" % env_path
    root = Path(registry_root) if registry_root else default_registry_root()
    if root is not None and (root / APPROVED_METHOD_FILENAME).is_file():
        return str(root / APPROVED_METHOD_FILENAME)
    return "built-in parameters"


def _approved_method_file(registry_root: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    """The parsed approved-method file, or None when none is configured.

    Resolution order:
      1. MOUSEBRAIN_APPROVED_METHOD -- path to a JSON file.
      2. ``<registry root>/approved_method.json`` when that file exists
         (*registry_root* if given, else default_registry_root()).
    """
    env_path = os.environ.get("MOUSEBRAIN_APPROVED_METHOD")
    if env_path:
        return _load_method_file(Path(env_path), "MOUSEBRAIN_APPROVED_METHOD")

    root = Path(registry_root) if registry_root else default_registry_root()
    if root is not None:
        candidate = root / APPROVED_METHOD_FILENAME
        if candidate.is_file():
            return _load_method_file(candidate, APPROVED_METHOD_FILENAME)
    return None


def _per_analysis_table(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The ``"analyses"`` table of a per-analysis approved-method file, or
    None for the plain (flat) form. Raises on a malformed table, for the
    same reason _load_method_file does: an approval that is configured but
    misread would silently compare every output with the wrong method.

    WHY either top-level key marks the per-analysis form: a file holding
    only ``{"default": {...}}`` read as the flat form would be a method
    with ONE parameter named "default", and every entry of every analysis
    following the shared method would then show as stale with no error. So "default" and "analyses" are reserved words: a file with
    either one is the per-analysis form (a missing "analyses" = an empty
    table), and neither can be the name of a parameter."""
    if "analyses" not in data and "default" not in data:
        return None
    table = data.get("analyses")
    if table is None:
        table = {}
    if not isinstance(table, dict):
        raise RuntimeError(
            "Approved-method file: 'analyses' must be an object of analysis "
            "name -> parameters (or null for 'no approved method').")
    for name, method in table.items():
        if method is not None and (not isinstance(method, dict) or not method):
            raise RuntimeError(
                "Approved-method file: the entry for analysis %r must be a "
                "non-empty object of parameter name -> value, or null." % name)
    default = data.get("default")
    if default is not None and (not isinstance(default, dict) or not default):
        raise RuntimeError(
            "Approved-method file: 'default' must be a non-empty object of "
            "parameter name -> value.")
    return table


def get_approved_method(registry_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Return the SHARED approved method: the parameters the slice-analysis
    tools run with and that analyses following the shared method are judged by.

    A lab records the parameters it has signed off on OUTSIDE the code, so the
    tool carries no project history. Resolution order:

      1. MOUSEBRAIN_APPROVED_METHOD -- path to a JSON file.
      2. ``<registry root>/approved_method.json`` when that file exists
         (*registry_root* if given, else default_registry_root()).
      3. DEFAULT_METHOD (the built-in parameters above).

    The file takes one of two forms:

    * flat -- the parameters themselves, with the same keys as
      DEFAULT_METHOD: ``{"detection": "threshold+log", ...}``;
    * per analysis -- ``{"default": {...}, "analyses": {"<analysis>": {...},
      "<other analysis>": null}}``. Here the shared method is ``"default"``
      (DEFAULT_METHOD when it is left out); per-analysis entries are read by
      approved_method_for().

    Whatever the source, the returned dict has the same flat shape, so method
    hashes computed from it compare with hashes already stored in a registry.

    Args:
        registry_root: Registry root to look under for approved_method.json.
            Defaults to the resolved default root.

    Returns:
        A fresh dict of parameter name -> value (safe to modify).
    """
    data = _approved_method_file(registry_root)
    if data is None:
        return dict(DEFAULT_METHOD)
    if _per_analysis_table(data) is not None:
        return dict(data.get("default") or DEFAULT_METHOD)
    return dict(data)


def approved_method_for(
    analysis_name: str,
    registry_root: Optional[Union[str, Path]] = None,
    follows_approved_method: bool = True,
) -> Optional[Dict[str, Any]]:
    """The approved method ONE analysis is judged by, or None if it has none.

    WHY per analysis: the shared approved method (get_approved_method) holds
    the slice-analysis parameters. Analyses that record their OWN parameters
    -- the 3D region counts and orientation checks register the atlas, ball
    sizes, model, check version... they actually ran with -- can never match
    it, so comparing them with it reported every one of their entries as
    "stale vs approved", which made the stale count meaningless. Now:

    1. If the approved-method file is in the per-analysis form and names
       *analysis_name*, that entry is its approved method (null = it has
       none).
    2. Otherwise an analysis that follows the shared approved method
       (*follows_approved_method*, declared by the code that registers it)
       is judged by get_approved_method().
    3. Otherwise it has no approved method: None. Its entries are never
       counted as stale; the manifest header says so (hash null).

    WHICH faults in the file stop WHICH analysis: every registration resolves
    this (it is recorded in the manifest header), so stopping on a fault
    anywhere would stop every registration of every analysis -- including the
    3D region counts, which use none of the file. So an analysis is stopped only by a
    fault that could change ITS answer: a file that cannot be read or is not
    a JSON object, an ``"analyses"`` that is not a table (it cannot be told
    whether it is named), its own entry, or -- when it follows the shared
    method and is not named -- anything get_approved_method() refuses. An
    analysis that does not follow the shared method and is not named carries
    on when another analysis's entry or ``"default"`` is malformed; those
    analyses still stop with the error.
    """
    data = _approved_method_file(registry_root)
    if data is not None and ("analyses" in data or "default" in data):
        table = data.get("analyses")
        if table is None:
            table = {}
        if not isinstance(table, dict):
            raise RuntimeError(
                "Approved-method file: 'analyses' must be an object of analysis "
                "name -> parameters (or null for 'no approved method').")
        if analysis_name in table:
            method = table[analysis_name]
            if method is None:
                return None
            if not isinstance(method, dict) or not method:
                raise RuntimeError(
                    "Approved-method file: the entry for analysis %r must be a "
                    "non-empty object of parameter name -> value, or null."
                    % analysis_name)
            return dict(method)
    if follows_approved_method:
        return get_approved_method(registry_root)
    return None


# =============================================================================
# SAMPLE ID PARSING
# =============================================================================

def parse_sample_id(sample: str) -> Tuple[str, str]:
    """Extract animal ID and region from a sample name.

    Handles formats like:
        "E01_01_S13_DCN"     -> ("E01_01", "DCN")
        "E01_01_S13_DCNv2"   -> ("E01_01", "DCNv2")
        "E01_01_S17_DCN001"  -> ("E01_01", "DCN001")
        "E01_01_S1_R3"       -> ("E01_01", "R3")

    The convention is: {animal_prefix}_{animal_num}_S{slice}_{region_suffix}
    where animal = first two underscore-separated parts, and region = everything
    after the S{N}_ token.

    Args:
        sample: Sample identifier string (typically the ND2 filename stem).

    Returns:
        Tuple of (animal_id, region). If parsing fails, returns (sample, "").
    """
    parts = sample.split("_")
    if len(parts) < 3:
        return (sample, "")

    # Animal is always the first two parts (e.g. E01_01)
    animal = f"{parts[0]}_{parts[1]}"

    # Find the slice token (S followed by digits)
    region_parts = []
    found_slice = False
    for part in parts[2:]:
        if not found_slice and part.startswith("S") and len(part) > 1 and part[1:].isdigit():
            found_slice = True
            continue
        if found_slice:
            region_parts.append(part)

    region = "_".join(region_parts) if region_parts else ""

    # Normalize region: strip trailing version digits for the base region name
    # but keep the full string for sample-level identification
    return (animal, region)


def _base_region(region: str) -> str:
    """Extract the base region name, stripping version suffixes.

    "DCN"      -> "DCN"
    "DCNv2"    -> "DCN"
    "DCN001"   -> "DCN"
    "DCNv2Z"   -> "DCN"

    Used for directory grouping -- all DCN variants go in the DCN folder.
    """
    import re
    # Strip trailing version/variant suffixes: v2, v2Z, 001, etc.
    match = re.match(r'^([A-Z]+)', region)
    return match.group(1) if match else region


# =============================================================================
# REGISTRY ROOT AND PATH COMPUTATION
# =============================================================================

def default_registry_root() -> Optional[Path]:
    """Where this tool records its analysis outputs when no root is passed.

    Resolution order:
      1. MOUSEBRAIN_REGISTRY_ROOT (explicit override; any folder).
      2. ``<PIPELINE_ROOT>/Registry`` when mousebrain.config resolved a
         pipeline root (mousebrain.config is imported lazily because it
         resolves the installation root on import and may warn or fail in a
         bare environment; the registry must stay usable with an explicit
         root regardless).
      3. None -- nothing is configured. Callers must fail loudly rather than
         invent a location.

    Returns:
        Path of the registry root, or None.
    """
    env = os.environ.get("MOUSEBRAIN_REGISTRY_ROOT")
    if env:
        return Path(env)

    try:
        from mousebrain.config import PIPELINE_ROOT
        if PIPELINE_ROOT:
            return Path(PIPELINE_ROOT) / "Registry"
    except Exception:
        pass

    return None


def get_database_path(
    registry_root: Path,
    category: str,
    sample: str,
    animal: str,
    region: str,
    analysis_name: str = "",
) -> Dict[str, Path]:
    """Compute the canonical output directories for a given output.

    Pure path computation -- nothing is created here. Layout under the root:
        exports/{analysis_name}/{animal}/{base_region}/   -- data files
        figures/{analysis_name}/{animal}/{base_region}/   -- figure files

    Args:
        registry_root: Root of the registry folder.
        category: "detection" or "roi_analysis".
        sample: Full sample ID (e.g. "E01_01_S13_DCN").
        animal: Animal ID (e.g. "E01_01").
        region: Region string (e.g. "DCN", "DCNv2").
        analysis_name: Name of the analysis (e.g. "ENCR_ROI_Analysis").

    Returns:
        Dict with keys 'export_dir' and 'figure_dir' mapping to Path objects.
    """
    registry_root = Path(registry_root)
    base_reg = _base_region(region) if region else ""

    if category == "detection":
        export_dir = registry_root / "exports" / analysis_name / sample
        figure_dir = export_dir  # detection figures live alongside data
    elif category == "roi_analysis":
        if animal and base_reg:
            export_dir = registry_root / "exports" / analysis_name / animal / base_reg
            figure_dir = registry_root / "figures" / analysis_name / animal / base_reg
        elif animal:
            export_dir = registry_root / "exports" / analysis_name / animal
            figure_dir = registry_root / "figures" / analysis_name / animal
        else:
            export_dir = registry_root / "exports" / analysis_name / sample
            figure_dir = registry_root / "figures" / analysis_name / sample
    else:
        export_dir = registry_root / "exports" / analysis_name / sample
        figure_dir = registry_root / "figures" / analysis_name / sample

    return {"export_dir": export_dir, "figure_dir": figure_dir}


class RegistryUnreadableError(RuntimeError):
    """registry.json exists but could not be read; see AnalysisRegistry._read_registry."""


# Time stamp at the end of a set-aside manifest's name.
_SET_ASIDE_TIME_FORMAT = "%Y%m%d_%H%M%S_%f"

# Output types that are routed to the figures/ tree instead of exports/.
_FIGURE_OUTPUT_TYPES = ("figure", "roi_figure", "qc_figure", "overlay")


def roi_results_from_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert a list of per-ROI count rows into the mapping register_roi_counts takes.

    ``count_cells_in_rois`` (plugin_2d.sliceatlas.core.roi) returns a LIST of
    row dicts, each carrying the ROI under a ``"name"`` key plus its counts,
    ending with an ``"Outside"`` row and a ``"TOTAL"`` row. The registry keys
    ROIs by name, so this turns

        [{"name": "Left", "total": 12, "positive": 11, ...}, ...]

    into

        {"Left": {"total": 12, "positive": 11, ...}, ...}

    Keys starting with "_" (private scratch values) are dropped. Dual-channel
    rows carry ``dual``/``red_only``/``green_only``/``neither`` instead of
    ``positive``/``negative``/``fraction``; for those, ``positive`` is filled
    from ``dual`` (the cells positive in both channels), ``negative`` with the
    rest and ``fraction`` with dual/total, because the per-region summary CSV
    has fixed columns and would otherwise record every dual-channel sample as
    zero positives. The original dual keys are kept alongside.

    Args:
        rows: Iterable of row dicts with a "name" key.

    Returns:
        Dict mapping ROI name -> count dict.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "name" not in row:
            continue
        name = str(row["name"])
        counts = {
            k: v for k, v in row.items()
            if k != "name" and not str(k).startswith("_")
        }
        if "positive" not in counts and "dual" in counts:
            total = int(counts.get("total", 0) or 0)
            pos = int(counts.get("dual", 0) or 0)
            counts["positive"] = pos
            counts["negative"] = total - pos
            counts["fraction"] = round(pos / total, 4) if total else 0.0
        out[name] = counts
    return out


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

class AnalysisRegistry:
    """Manages analysis outputs with provenance, file placement, and staleness detection.

    Each registry instance is tied to a single analysis_name (e.g.
    "ENCR_Detection" or "ENCR_ROI_Analysis") and maintains a registry.json
    manifest in the corresponding exports directory.

    The manifest tracks every registered output with:
      - Full method parameters and their deterministic hash
      - Source file paths (ND2, ROI JSON, etc.)
      - Key results (counts, fractions)
      - Output file locations (relative to the registry root)
      - Registration timestamp
      - Staleness flag (is_current)

    Concurrent writers: every change to the manifest is a read-change-write
    of the whole registry.json, done while holding the analysis's lock,
    ``logs/{analysis_name}.lock`` under the registry root
    (mousebrain.file_lock; one holder at a time across processes and
    machines), and written through a temporary file swapped in with
    os.replace. The lock prevents a LOST update (two registrations at once
    each writing back the manifest they read, so the second erased the
    first's entry); the temporary file prevents a TORN file. The per-region
    ROI summary CSV is rewritten under the same lock for the same reason.

    Args:
        analysis_name: Name of the analysis (e.g. "ENCR_Detection").
        registry_root: Root folder of the registry. Defaults to
            default_registry_root(); raises RuntimeError when nothing resolves.
        db_root: Deprecated alias of *registry_root*, kept so older callers
            keep working. Ignored when *registry_root* is given.
        follows_approved_method: Whether this analysis is judged by the
            SHARED approved method (get_approved_method). The code that
            registers an analysis declares it: True for analyses run with
            the shared parameters, False for analyses that record their own
            (see approved_method_for). The declaration is written into the
            manifest header, so a reader that does not declare it (None, as
            the CLI does) keeps whatever the producer declared; a manifest
            that never recorded one follows the shared method, as before.
    """

    # Current schema version for the registry JSON
    SCHEMA_VERSION = 1

    def __init__(
        self,
        analysis_name: str,
        registry_root: Optional[Union[str, Path]] = None,
        db_root: Optional[Union[str, Path]] = None,
        follows_approved_method: Optional[bool] = None,
    ):
        self.analysis_name = analysis_name
        self._follows_approved_method = follows_approved_method
        if registry_root is None and db_root is not None:
            registry_root = db_root
        resolved = Path(registry_root) if registry_root else default_registry_root()
        if resolved is None:
            raise RuntimeError(
                "AnalysisRegistry: no registry root configured. Set "
                "MOUSEBRAIN_REGISTRY_ROOT to the folder that should hold "
                "exports/, figures/ and logs/; or set CONNECTOME_ROOT so the "
                "pipeline root resolves (the registry then lives at "
                "<pipeline root>/Registry); or pass registry_root=."
            )
        self.registry_root = resolved
        self.exports_dir = self.registry_root / "exports" / analysis_name
        self.figures_dir = self.registry_root / "figures" / analysis_name
        self.logs_dir = self.registry_root / "logs"
        self.registry_path = self.exports_dir / "registry.json"
        # Outside the mirrored exports/ tree, in the existing logs/ folder
        # (see the module docstring for why there).
        self.work_dir = self.logs_dir
        self.lock_path = self.logs_dir / ("%s.lock" % analysis_name)
        self.set_aside_dir = self.logs_dir
        self.set_aside_prefix = "%s.%s.unreadable-" % (analysis_name, self.registry_path.name)
        # The [!] line about set-aside manifests is printed once per instance
        # (see _warn_about_set_aside), not on every read.
        self._set_aside_warned = False

        # exports/ and logs/ are needed by every registry (manifest + audit
        # log). figures/<analysis>/ is NOT created here: it is made on the
        # first figure write in register_output. WHY: analyses that never
        # produce figures (plain detection runs) used to leave empty
        # figures/<analysis>/ trees behind, and whoever pulls this folder
        # then sees phantom analyses with nothing in them.
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_root(self) -> Path:
        """Deprecated alias of registry_root (older callers read this name)."""
        return self.registry_root

    # -----------------------------------------------------------------
    # Registry I/O (atomic read/write)
    # -----------------------------------------------------------------

    # A lock untouched for this long (file server's clock) was left by a
    # crashed process and is taken over; a waiter gives up after
    # LOCK_WAIT_SECONDS with a message naming the holder and the lock file.
    LOCK_STALE_SECONDS = 120
    LOCK_WAIT_SECONDS = 300

    def _locked(self):
        """Hold this analysis's registry lock (see the class docstring).

        A live holder keeps its lock fresh by itself for as long as it holds
        it (mousebrain.file_lock refreshes it from a background thread), so a
        registration copying one large file over a slow share for longer
        than LOCK_STALE_SECONDS is never mistaken for a crashed writer.
        """
        return exclusive_file_lock(self.registry_path, "Registry",
                                   stale_seconds=self.LOCK_STALE_SECONDS,
                                   wait_seconds=self.LOCK_WAIT_SECONDS,
                                   lock_path=self.lock_path)

    def _read_registry(self, set_aside_unreadable: bool = False) -> Dict[str, Any]:
        """Read the registry manifest from disk.

        Returns a fresh skeleton ONLY when the file does not exist.

        WHY an unreadable manifest is never treated as empty: every write is a
        read-change-write of the whole manifest, so reading a damaged file as
        "no entries" and then writing one new entry over it erased the record
        of every output registered before. Instead:

        * a file that cannot be OPENED (share hiccup, file held by another
          program) raises RegistryUnreadableError and is left exactly where it
          is -- the next attempt will most likely read it fine;
        * a file that opens but is not a registry manifest (not JSON, or JSON
          without an "entries" object) raises RegistryUnreadableError too.
          When *set_aside_unreadable* is True (the two registration
          methods, which hold the registry lock) it is first MOVED to
          ``<registry root>/logs/<analysis>.registry.json.unreadable-<time>``
          -- its bytes are kept for a person to repair and merge back, outside
          the mirrored exports folder -- and a NEW manifest with no entries is
          written in its place at once, its header listing the set-aside file
          (``set_aside_manifests``), so later registrations carry on instead
          of every one failing until someone notices. Everything else (the
          query methods, the CLI including --refresh-approved, and
          invalidate) never moves anything.

        WHY a new manifest is written at once, not by the next registration:
        an integrator finds analyses by their ``exports/<analysis>/
        registry.json``. An analysis folder left without one -- for days, for
        an analysis registered once per brain -- is not listed at all, and an
        integrator that archives what it no longer lists would take that
        analysis's whole mirror away and drop it from its reports without a
        word. With a manifest always present the analysis stays listed, its
        files stay mirrored, and the header says what was set aside. If the
        new manifest cannot be written, the damaged file is put back where it
        was, so the folder is never left without one.

        WHY every read warns about set-aside manifests: once one has been set
        aside, the registry holds fewer entries than were registered, and
        nothing else says so. So a [!] line naming them is printed (once per
        instance), and every write lists them in the manifest header
        (``set_aside_manifests``) for an integrator to show, until a person
        repairs and removes them.
        """
        self._warn_about_set_aside()
        if not self.registry_path.exists():
            return self._empty_registry()

        try:
            # utf-8-sig, not utf-8: a person repairing a set-aside manifest by
            # hand (see the guide) may save it from an editor or shell that
            # writes a UTF-8 byte-order mark. Read as plain utf-8, that valid
            # file fails json.loads and would itself be set aside. utf-8-sig
            # reads BOM and no-BOM files alike; _write_registry still writes
            # without one, so the next write drops it.
            with open(self.registry_path, "r", encoding="utf-8-sig") as f:
                text = f.read()
        except OSError as e:
            raise RegistryUnreadableError(
                "The registry manifest %s could not be opened (%s). Nothing was "
                "changed; try again, and if it keeps failing check that the file "
                "is not held open by another program." % (self.registry_path, e)
            ) from e

        problem = None
        try:
            data = json.loads(text)
        except ValueError as e:
            problem = "it is not valid JSON (%s)" % e
        else:
            if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
                problem = "it has no 'entries' object"
        if problem is None:
            return data

        if not set_aside_unreadable:
            raise RegistryUnreadableError(
                "The registry manifest %s cannot be read: %s. Nothing was changed. "
                "Repair it by hand (or restore it from a backup) before relying on "
                "this registry; the next registration will move it into %s as "
                "%s<time>." % (
                    self.registry_path, problem, self.set_aside_dir, self.set_aside_prefix))

        self.set_aside_dir.mkdir(parents=True, exist_ok=True)
        aside = self.set_aside_dir / ("%s%s" % (
            self.set_aside_prefix, datetime.now().strftime(_SET_ASIDE_TIME_FORMAT)))
        # A rename only (logs/ is on the registry's drive): a copy-then-delete
        # fallback would, for a manifest held open by another program, leave
        # a copy here reported as set aside while the damaged file stayed in
        # place (see _rename_only). A refused rename raises and moves nothing.
        _rename_only(self.registry_path, aside)
        try:
            # A new, empty manifest in its place at once (see WHY above). Its
            # header lists the set-aside file (_set_approved_header).
            self._write_registry(self._empty_registry())
        except BaseException as e:
            # Never leave the folder without a manifest: put the damaged one
            # back where it was. Nothing else has changed.
            try:
                _rename_only(aside, self.registry_path)
                where = "it was left where it was, unchanged"
            except OSError as back:
                where = ("it is kept, unchanged, as %s and could not be put back "
                         "(%s); move it back to %s by hand" % (aside, back, self.registry_path))
            raise RegistryUnreadableError(
                "The registry manifest %s could not be read (%s), and no new manifest "
                "could be written in its place (%s); %s. THIS registration was not "
                "recorded." % (self.registry_path, problem, e, where)) from e
        self._log_event("set_aside_unreadable", "-", str(aside), problem)
        raise RegistryUnreadableError(
            "The registry manifest %s could not be read (%s). It has been kept, "
            "unchanged, as %s, a new manifest with no entries has been started in "
            "its place, and THIS registration was not recorded -- run it again. "
            "To recover the earlier entries, repair the kept file and merge its "
            "'entries' back into registry.json (see the analysis registry guide)."
            % (self.registry_path, problem, aside))

    def set_aside_files(self) -> List[Path]:
        """The damaged manifests of this analysis that were set aside and are
        still waiting for a person (oldest first)."""
        try:
            return sorted(p for p in self.set_aside_dir.iterdir()
                          if p.is_file() and p.name.startswith(self.set_aside_prefix))
        except OSError:
            return []

    def set_aside_records(self) -> List[Dict[str, Any]]:
        """One record per set-aside manifest: file name, when it was set
        aside, its size, and an ESTIMATE of how many entries it holds (the
        file is damaged, so its entries are counted by their
        "registered_at" fields, not parsed)."""
        records = []
        for path in self.set_aside_files():
            stamp = path.name[len(self.set_aside_prefix):]
            try:
                set_aside_at = datetime.strptime(stamp, _SET_ASIDE_TIME_FORMAT).isoformat()
            except ValueError:
                set_aside_at = None
            try:
                # utf-8-sig for the same reason as _read_registry.
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                size, estimate = path.stat().st_size, text.count('"registered_at"')
            except OSError:
                size, estimate = None, None
            records.append({"file": path.name, "set_aside_at": set_aside_at,
                            "size_bytes": size, "entries_estimate": estimate})
        return records

    def _record_set_aside(self, data: Dict[str, Any]) -> None:
        """List in the manifest header (``set_aside_manifests``) the damaged
        manifests still waiting for a person (see _read_registry); drop the
        field once there are none."""
        set_aside = self.set_aside_records()
        if set_aside:
            data["set_aside_manifests"] = set_aside
        else:
            data.pop("set_aside_manifests", None)

    def _warn_about_set_aside(self) -> None:
        """Print a [!] line (once per instance) when damaged manifests of this
        analysis are waiting in the set-aside place; see _read_registry."""
        if self._set_aside_warned:
            return
        files = self.set_aside_files()
        if not files:
            return
        self._set_aside_warned = True
        print("  [!] %s: %d damaged manifest%s set aside in %s (%s). Entries "
              "recorded only there are NOT in this registry until a person repairs "
              "the file and merges them back (see the analysis registry guide)." % (
                  self.analysis_name, len(files), "" if len(files) == 1 else "s",
                  self.set_aside_dir, ", ".join(p.name for p in files)))

    # How long a write keeps retrying while Windows refuses to replace
    # registry.json because a reader (the CLI, an integrator pulling this
    # folder) has it open for a moment -- the same rule the calibration
    # tracker uses for its CSV.
    REPLACE_RETRY_SECONDS = 10

    def _write_registry(self, data: Dict[str, Any], refresh_header: bool = True) -> None:
        """Write registry manifest atomically (write temp, then rename).

        This prevents corruption if two processes write simultaneously or
        the process is interrupted mid-write.

        Every write also refreshes the approved-method header
        (approved_method, approved_method_hash, approved_method_recorded_at).
        WHY: an integrator counts "stale vs approved" by comparing each
        entry's method_hash with the header's approved_method_hash. The
        header used to be written only when the manifest was first created,
        so after a lab changed its approved method every later count was
        still judged by the old one and showed nothing stale. A broken
        approved-method file therefore stops the write with its error rather
        than recording a header nobody chose.

        *refresh_header* False keeps the header the caller already set:
        invalidate() reads the approved method ONCE, before it moves any
        file, so that a broken approved-method file stops it before anything
        has moved rather than between a move and the write that records it.
        """
        if refresh_header:
            self._set_approved_header(data)
        data["last_updated"] = datetime.now().isoformat()
        content = json.dumps(data, indent=2, ensure_ascii=False, default=str)

        # Write to a temp file, then rename. os.replace is atomic on the same
        # filesystem on both Windows and POSIX. The temp file is in logs/
        # (same drive, outside the mirrored exports folder; see the module
        # docstring).
        self.work_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.work_dir), suffix=".tmp",
            prefix=".%s.registry_" % self.analysis_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            # Atomic replace; on Windows retried while a reader holds the file.
            deadline = time.monotonic() + self.REPLACE_RETRY_SECONDS
            while True:
                try:
                    os.replace(tmp_path, str(self.registry_path))
                    break
                except PermissionError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _empty_registry(self) -> Dict[str, Any]:
        """Return an empty registry skeleton."""
        data = {
            "analysis_name": self.analysis_name,
            "version": self.SCHEMA_VERSION,
            "entries": {},
            "last_updated": datetime.now().isoformat(),
        }
        self._set_approved_header(data)
        return data

    def follows_approved_method(self, data: Optional[Dict[str, Any]] = None) -> bool:
        """This analysis's declaration (see the constructor): the instance's
        own if it was given one, else the manifest's, else True."""
        if self._follows_approved_method is not None:
            return bool(self._follows_approved_method)
        if data is None:
            data = self._read_registry()
        return bool(data.get("follows_approved_method", True))

    def approved_method(self, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """The approved method this analysis is judged by, or None when it
        has none (see approved_method_for)."""
        return approved_method_for(self.analysis_name, self.registry_root,
                                   self.follows_approved_method(data))

    def _set_approved_header(self, data: Dict[str, Any]) -> None:
        """Write the approved-method fields of the manifest header.

        ``approved_method_hash`` is what an integrator compares each entry's
        ``method_hash`` with to count "stale vs approved". It is null when the
        analysis has no approved method: such an analysis has nothing to be
        stale against, and a reader must report "no approved method", never
        count its entries as stale.
        """
        follows = self.follows_approved_method(data)
        approved = approved_method_for(self.analysis_name, self.registry_root, follows)
        data["follows_approved_method"] = follows
        data["approved_method"] = approved
        data["approved_method_hash"] = (
            self.get_method_hash(approved) if approved is not None else None)
        # When, where and from what these fields were last read, so a reader
        # can say "approved method as recorded <time>" -- and so a header
        # that flips between two values because two machines resolve the
        # approved method differently (MOUSEBRAIN_APPROVED_METHOD set on one
        # of them) shows which machine wrote which.
        data["approved_method_recorded_at"] = datetime.now().isoformat()
        data["approved_method_recorded_on"] = _get_hostname()
        data["approved_method_source"] = approved_method_source(self.registry_root)
        # Refreshed here because every write passes through this method.
        self._record_set_aside(data)

    def refresh_approved_method(self) -> Optional[Dict[str, Any]]:
        """Re-read the approved method and rewrite the manifest header now,
        without registering anything (CLI: --refresh-approved).

        Every write refreshes the header anyway (see _write_registry); this
        is for a manifest that has not been written since the approved method
        changed. Returns the refreshed manifest, or None when this analysis
        has no manifest yet (nothing is created).

        A damaged manifest is NOT set aside here: this registers nothing, so
        it raises RegistryUnreadableError and leaves the file where it is (the
        CLI prints [FAIL]). Only a registration moves a damaged manifest away,
        because only a registration needs a manifest to write into.
        """
        with self._locked():
            if not self.registry_path.exists():
                return None
            data = self._read_registry(set_aside_unreadable=False)
            self._write_registry(data)
            return data

    # -----------------------------------------------------------------
    # Method hashing
    # -----------------------------------------------------------------

    @staticmethod
    def get_method_hash(method_params: Dict[str, Any]) -> str:
        """Compute a deterministic SHA-256 hash of method parameters.

        Uses JSON serialization with sorted keys so that logically identical
        parameter dicts always produce the same hash regardless of insertion
        order.

        Args:
            method_params: Dict of method parameter name -> value.

        Returns:
            Hex digest string (64 characters).
        """
        canonical = json.dumps(method_params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # -----------------------------------------------------------------
    # Output registration
    # -----------------------------------------------------------------

    def register_output(
        self,
        sample: str,
        category: str,
        files: Dict[str, str],
        results: Dict[str, Any],
        method_params: Dict[str, Any],
        source_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Path]:
        """Register an analysis output and copy its files into the registry.

        Copies each file in *files* to its canonical location under the
        registry root, records full provenance in the manifest, and returns
        the destination paths.

        Args:
            sample: Sample identifier (e.g. "E01_01_S13_DCN").
            category: "detection" or "roi_analysis".
            files: Dict mapping output type to source file path.
                   e.g. {"figure": "/path/to/fig.png",
                         "measurements": "/path/to/csv"}
            results: Dict of key results.
                     e.g. {"n_nuclei": 28, "n_positive": 15,
                           "positive_fraction": 0.536}
            method_params: Dict of method parameters used for this analysis.
            source_files: Optional dict of source file paths.
                          e.g. {"nd2": "/path/to.nd2",
                                "roi_json": "/path/to.rois.json"}

        Returns:
            Dict mapping output type -> destination Path where the file was
            placed under the registry root.
        """
        animal, region = parse_sample_id(sample)

        # Compute destination directories (created below, under the lock, only
        # when a file is actually placed there).
        paths = get_database_path(
            self.registry_root, category, sample, animal, region, self.analysis_name
        )
        export_dir = paths["export_dir"]
        figure_dir = paths["figure_dir"]

        # WHY everything happens under the lock, and in this order:
        # the files are placed only once the manifest has been read and the
        # approved method resolved -- the steps that can refuse. The files
        # used to be copied first, so a refusal (for example the lock not
        # coming free, a damaged manifest, a broken approved-method file) left the NEW
        # run's files in the registry while the manifest entry still
        # described the OLD run, and an integrator mirrored them as one.
        # A file being replaced is kept in logs/ until the manifest
        # write has succeeded, and put back if it fails (see _place).
        with self._locked() as lock:
            registry = self._read_registry(set_aside_unreadable=True)
            self._set_approved_header(registry)

            output_map: Dict[str, str] = {}
            dest_paths: Dict[str, Path] = {}
            placed: List[Tuple[Path, Optional[Path]]] = []
            try:
                for output_type, src_path_str in files.items():
                    src_path = Path(src_path_str)
                    if not src_path.exists():
                        print(f"  [!] WARNING: Source file does not exist: {src_path}")
                        continue

                    # Route figures to figure_dir, everything else to export_dir
                    if output_type in _FIGURE_OUTPUT_TYPES:
                        dest = figure_dir / src_path.name
                    else:
                        dest = export_dir / src_path.name

                    self._place(src_path, dest, placed)
                    lock.touch()
                    dest_paths[output_type] = dest

                    # Store path relative to the registry root for portability
                    try:
                        rel = dest.relative_to(self.registry_root)
                        output_map[output_type] = str(rel)
                    except ValueError:
                        output_map[output_type] = str(dest)

                # Build provenance entry
                method_hash = self.get_method_hash(method_params)
                entry = {
                    "sample": sample,
                    "animal": animal,
                    "region": region,
                    "category": category,
                    "results": results,
                    "method_params": method_params,
                    "method_hash": method_hash,
                    "source_files": {k: str(v) for k, v in (source_files or {}).items()},
                    "outputs": output_map,
                    "registered_at": datetime.now().isoformat(),
                    "hostname": _get_hostname(),
                    "is_current": True,
                }
                registry["entries"][sample] = entry
                self._write_registry(registry, refresh_header=False)
            except BaseException as e:
                self._undo_placed(placed, sample, e)
                raise
            self._discard_replaced(placed)

        # Write audit log
        self._log_event("register", sample, category, method_hash)

        return dest_paths

    def register_roi_counts(
        self,
        sample: str,
        region: str,
        roi_results: Union[Dict[str, Dict[str, Any]], Iterable[Dict[str, Any]]],
        method_params: Dict[str, Any],
        source_files: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Register ROI-level counts and update the per-region summary CSV.

        Args:
            sample: Sample identifier (e.g. "E01_01_S13_DCN").
            region: Brain region (e.g. "DCN"). Used for the summary CSV name.
            roi_results: Dict mapping ROI name -> count dict.
                         e.g. {"Left": {"total": 12, "positive": 11,
                                        "negative": 1, "fraction": 0.917},
                               "Right": {"total": 1, "positive": 1,
                                          "negative": 0, "fraction": 1.0},
                               "TOTAL": {"total": 13, "positive": 12,
                                          "negative": 1, "fraction": 0.923}}
                         A list of row dicts as returned by
                         count_cells_in_rois is accepted too and converted
                         with roi_results_from_rows.
            method_params: Dict of method parameters.
            source_files: Optional dict of source file paths.

        Returns:
            Path to the per-sample roi_counts CSV under the registry root.
        """
        if not isinstance(roi_results, dict):
            roi_results = roi_results_from_rows(roi_results)

        animal, _ = parse_sample_id(sample)
        base_reg = _base_region(region) if region else region

        export_dir = self.exports_dir / animal / base_reg
        counts_path = export_dir / f"{sample}_roi_counts.csv"
        summary_path = self.exports_dir / f"roi_summary_{base_reg}.csv"
        method_hash = self.get_method_hash(method_params)
        entry_key = f"{sample}__roi_counts"
        entry = {
            "sample": sample,
            "animal": animal,
            "region": region,
            "category": "roi_counts",
            "results": roi_results,
            "method_params": method_params,
            "method_hash": method_hash,
            "source_files": {k: str(v) for k, v in (source_files or {}).items()},
            "outputs": {
                "roi_counts": str(counts_path.relative_to(self.registry_root)),
            },
            "registered_at": datetime.now().isoformat(),
            "hostname": _get_hostname(),
            "is_current": True,
        }

        # The per-sample CSV, the region summary CSV and the manifest are all
        # written under the lock, AFTER the manifest has been read and the
        # approved method resolved, for the reasons given in register_output:
        # a refusal must leave the earlier counts in place, never new counts
        # beside an entry describing the old run. Both CSVs are written in
        # logs/ and swapped in; the files they replace are put back
        # if the manifest cannot be written.
        with self._locked() as lock:
            registry = self._read_registry(set_aside_unreadable=True)
            self._set_approved_header(registry)
            placed: List[Tuple[Path, Optional[Path]]] = []
            try:
                staged = self._staging_path(counts_path.name)
                try:
                    self._write_roi_counts_csv(staged, roi_results)
                    self._place(staged, counts_path, placed, move=True)
                finally:
                    _unlink_quietly(staged)
                lock.touch()

                staged = self._staging_path(summary_path.name)
                try:
                    self._update_region_summary(sample, base_reg, roi_results, out_path=staged)
                    self._place(staged, summary_path, placed, move=True)
                finally:
                    _unlink_quietly(staged)

                registry["entries"][entry_key] = entry
                self._write_registry(registry, refresh_header=False)
            except BaseException as e:
                self._undo_placed(placed, sample, e)
                raise
            self._discard_replaced(placed)

        self._log_event("register_roi_counts", sample, region, method_hash)
        return counts_path

    @staticmethod
    def _write_roi_counts_csv(path: Path, roi_results: Dict[str, Dict[str, Any]]) -> None:
        """Write one sample's per-ROI counts. Fixed columns first; any further
        count keys (e.g. the dual-channel dual/red_only/green_only/neither)
        are appended so no count is lost."""
        base_fields = ["roi", "total", "positive", "negative", "fraction"]
        extra_fields = sorted({
            str(k) for counts in roi_results.values() for k in counts
            if str(k) not in base_fields
        })
        fieldnames = base_fields + extra_fields
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for roi_name, counts in roi_results.items():
                row = {
                    "roi": roi_name,
                    "total": counts.get("total", 0),
                    "positive": counts.get("positive", 0),
                    "negative": counts.get("negative", 0),
                    "fraction": counts.get("fraction", 0.0),
                }
                for k in extra_fields:
                    if k in counts:
                        row[k] = counts[k]
                writer.writerow(row)

    # -----------------------------------------------------------------
    # Placing files so that a failed registration changes nothing
    # -----------------------------------------------------------------

    def _staging_path(self, name: str) -> Path:
        """A fresh path in logs/ (same drive as the registry, outside the
        mirrored folders) for a file about to be placed. The name starts with
        "." and ends in ".part", so it is never taken for an audit log."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir / (".%s.%s.%s.part" % (
            self.analysis_name, uuid.uuid4().hex[:8], name))

    def _place(self, src: Path, dest: Path, placed: List[Tuple[Path, Optional[Path]]],
               move: bool = False) -> None:
        """Put *src* at *dest* (copied, or moved when *move*), keeping the
        file it replaces in logs/ and recording (dest, kept copy) in
        *placed* so _undo_placed can put it back.

        The new file is first copied into logs/ and then renamed into
        place, so *dest* is never a half-copied file and is missing only for
        the instant between the two renames."""
        if not move and dest.exists() and _same_file(src, dest):
            return  # already where it belongs (re-registering the registry's own copy)
        staged = src
        if not move:
            staged = self._staging_path(dest.name)
            shutil.copy2(str(src), str(staged))
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            kept = None
            if dest.exists():
                kept = self._staging_path("replaced-" + dest.name)
                _move_file(dest, kept)
            placed.append((dest, kept))
            _move_file(staged, dest)
        finally:
            if not move:
                _unlink_quietly(staged)

    def _undo_placed(self, placed: List[Tuple[Path, Optional[Path]]], sample: str,
                     error: BaseException) -> None:
        """A registration failed after placing some files: put back what they
        replaced (newest first) and remove the ones that replaced nothing, so
        the registry holds exactly what the manifest describes. A file that
        cannot be put back is named in a [!] line and the audit log."""
        problem = ""
        for dest, kept in reversed(placed):
            try:
                if kept is not None:
                    _move_file(kept, dest)
                else:
                    dest.unlink()
            except OSError as e:
                problem += "; could not put back %s%s: %s" % (
                    dest, " (the earlier file is at %s)" % kept if kept else "", e)
        if problem:
            print("  [!] Registration of %s failed (%s: %s) and the earlier files "
                  "could not all be put back%s" % (sample, type(error).__name__, error,
                                                   problem))
            self._log_event("register_failed", sample, "files not restored", problem)

    @staticmethod
    def _discard_replaced(placed: List[Tuple[Path, Optional[Path]]]) -> None:
        """The manifest now describes the new files: drop the kept earlier ones."""
        for _dest, kept in placed:
            if kept is not None:
                _unlink_quietly(kept)

    def _update_region_summary(
        self,
        sample: str,
        base_region: str,
        roi_results: Dict[str, Dict[str, Any]],
        out_path: Optional[Path] = None,
    ) -> None:
        """Update (or create) the per-region summary CSV.

        The summary CSV (e.g. roi_summary_DCN.csv) aggregates TOTAL counts
        from every sample in that region. If the sample already has a row, it
        is replaced; otherwise a new row is appended. The result is written
        to *out_path* when given (register_roi_counts writes it in
        logs/ and swaps it in), else over the summary itself.

        Format:
            sample,roi,total,positive,negative,fraction
        """
        summary_path = self.exports_dir / f"roi_summary_{base_region}.csv"

        # Read existing rows (skip current sample if present)
        existing_rows = []
        if summary_path.exists():
            with open(summary_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("sample") != sample:
                        existing_rows.append(row)

        # Add this sample's TOTAL row
        total = roi_results.get("TOTAL", {})
        if total:
            existing_rows.append({
                "sample": sample,
                "roi": "TOTAL",
                "total": total.get("total", 0),
                "positive": total.get("positive", 0),
                "negative": total.get("negative", 0),
                "fraction": total.get("fraction", 0.0),
            })

        # Sort by sample name for consistent ordering
        existing_rows.sort(key=lambda r: r.get("sample", ""))

        # Write back
        fieldnames = ["sample", "roi", "total", "positive", "negative", "fraction"]
        with open(out_path or summary_path, "w", newline="", encoding="utf-8") as f:
            # extrasaction="ignore": per-sample rows may carry extra count columns
            # (dual-channel results); the summary keeps the fixed schema so every
            # region summary stays comparable, and the per-sample CSV keeps the rest.
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)

    # -----------------------------------------------------------------
    # Staleness detection
    # -----------------------------------------------------------------

    def check_staleness(
        self,
        current_method_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare each registered entry against the current method parameters.

        Returns a list of dicts describing stale entries, each containing:
          - sample: the sample name
          - registered_hash: the hash stored in the registry
          - current_hash: hash of current_method_params
          - registered_params: the method params that were used
          - current_params: the current method params
          - diff_keys: list of parameter keys that differ

        Args:
            current_method_params: The method parameters to compare against.

        Returns:
            List of stale-entry description dicts. Empty if everything is current.
        """
        current_hash = self.get_method_hash(current_method_params)
        registry = self._read_registry()
        stale = []

        for key, entry in registry.get("entries", {}).items():
            entry_hash = entry.get("method_hash", "")
            if entry_hash != current_hash and entry.get("is_current", True):
                registered_params = entry.get("method_params", {})
                diff_keys = _find_diff_keys(registered_params, current_method_params)
                stale.append({
                    "sample": entry.get("sample", key),
                    "entry_key": key,
                    "registered_hash": entry_hash,
                    "current_hash": current_hash,
                    "registered_params": registered_params,
                    "current_params": current_method_params,
                    "diff_keys": diff_keys,
                })

        return stale

    def get_stale_samples(
        self,
        current_method_params: Dict[str, Any],
    ) -> List[str]:
        """Return sample names that need reprocessing.

        Convenience wrapper around check_staleness that returns just the
        unique sample names.

        Args:
            current_method_params: The method parameters to compare against.

        Returns:
            Sorted list of sample name strings.
        """
        stale_entries = self.check_staleness(current_method_params)
        samples = sorted(set(e["sample"] for e in stale_entries))
        return samples

    # -----------------------------------------------------------------
    # Query / summary
    # -----------------------------------------------------------------

    def get_summary_df(self):
        """Return a pandas DataFrame of all registered entries.

        Each row is one registry entry with columns for sample, animal,
        region, category, method_hash, registered_at, is_current, plus
        all result keys flattened as result_{key}.

        Returns:
            pandas.DataFrame. Raises ImportError if pandas is unavailable.
        """
        import pandas as pd

        registry = self._read_registry()
        rows = []
        for key, entry in registry.get("entries", {}).items():
            row = {
                "entry_key": key,
                "sample": entry.get("sample", ""),
                "animal": entry.get("animal", ""),
                "region": entry.get("region", ""),
                "category": entry.get("category", ""),
                "method_hash": entry.get("method_hash", ""),
                "registered_at": entry.get("registered_at", ""),
                "is_current": entry.get("is_current", True),
                "hostname": entry.get("hostname", ""),
            }

            # Flatten results
            results = entry.get("results", {})
            if isinstance(results, dict):
                # Handle nested dicts (e.g. roi_results with TOTAL/Left/Right)
                if all(isinstance(v, dict) for v in results.values()):
                    # ROI counts: use TOTAL for summary
                    total = results.get("TOTAL", {})
                    for rk, rv in total.items():
                        row[f"result_{rk}"] = rv
                else:
                    for rk, rv in results.items():
                        row[f"result_{rk}"] = rv

            # Flatten source files
            for sk, sv in entry.get("source_files", {}).items():
                row[f"source_{sk}"] = sv

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("sample").reset_index(drop=True)
        return df

    def get_entry(self, sample: str) -> Optional[Dict[str, Any]]:
        """Get a single registry entry by its key (the sample name for
        register_output entries; ``{sample}__roi_counts`` for ROI counts).

        Args:
            sample: Entry key.

        Returns:
            The entry dict, or None if not found.
        """
        registry = self._read_registry()
        return registry.get("entries", {}).get(sample)

    def get_all_entries(self) -> Dict[str, Dict[str, Any]]:
        """Return all registry entries.

        Returns:
            Dict mapping entry key -> entry dict.
        """
        registry = self._read_registry()
        return dict(registry.get("entries", {}))

    # -----------------------------------------------------------------
    # Invalidation / archival
    # -----------------------------------------------------------------

    def own_folders(self, entry: Dict[str, Any]) -> List[Path]:
        """The folder(s) that belong to *entry*'s sample alone.

        An entry has its own folder when its category files it under a
        folder named after the sample alone -- ``exports/<analysis>/<sample>/``
        (a "detection" entry keeps its figures there too). Entries whose
        folder is shared with other samples (an ROI analysis files by animal
        and region) have none: what is in a shared folder cannot be told
        apart by sample, so nothing there is ever swept.
        """
        sample = entry.get("sample") or ""
        if not sample:
            return []
        paths = get_database_path(
            self.registry_root, entry.get("category", ""), sample,
            entry.get("animal", ""), entry.get("region", ""), self.analysis_name)
        folders: List[Path] = []
        for folder, top in ((paths["export_dir"], self.registry_root / "exports"),
                            (paths["figure_dir"], self.registry_root / "figures")):
            # Only a folder named after the sample, directly under the
            # analysis folder -- never the analysis folder itself (it holds
            # registry.json) nor a folder shared with other samples.
            if (folder.name == sample
                    and folder.parent == top / self.analysis_name
                    and folder not in folders and folder.is_dir()):
                folders.append(folder)
        return folders

    def own_folder_files(self, entry: Dict[str, Any]) -> List[Tuple[Path, Path]]:
        """Every file in *entry*'s own folder(s) (see own_folders), outside
        their _archived/ subfolder, as (folder, file) pairs.

        WHY this exists: such a folder can hold files of the same run that
        the entry does not list among its outputs (an earlier version of a
        producer copied more figures than it registered). They stay there
        when only the listed outputs are archived, and something that
        mirrors the folder then shows an earlier run's figures beside a
        newer entry, or beside no current entry at all.
        """
        found: List[Tuple[Path, Path]] = []
        for folder in self.own_folders(entry):
            for f in sorted(folder.rglob("*")):
                if f.relative_to(folder).parts[0] == "_archived":
                    continue
                if f.is_file():
                    found.append((folder, f))
        return found

    def invalidate(self, sample: Optional[str] = None,
                   include_own_folder: bool = False) -> List[str]:
        """Archive entries' output files and mark those entries invalidated.

        If *sample* is given, only that sample's entries are invalidated.
        If None, ALL current entries are invalidated.

        Archived files are moved to an ``_archived/{timestamp}/`` subdirectory
        beside where they were, so they are preserved but no longer in the
        active output path.

        With *include_own_folder*, every other file in the entry's own sample
        folder (see own_folder_files) is archived as well, into the same
        ``_archived/{timestamp}/`` (keeping its place below the folder), and
        under the same all-or-nothing rule: if any one of them will not move,
        everything already moved is put back and the entry stays current.
        WHY: retiring an entry must not leave files of an earlier run in the
        active folder that no current entry names.

        WHY one entry at a time, and only when every file moved:
        an entry marked invalidated whose file is still in the active folder
        tells every reader (and an integrator mirroring this folder) that a
        result is gone when it is not; and moving every file first and
        writing the manifest once at the end meant a crash part-way left files
        in _archived/ while the manifest still called their entries current.
        So:

        * the approved-method header is read and checked ONCE, before any
          file moves (a broken approved-method file stops the pass with
          nothing moved);
        * for each entry: move its files; if any move fails, put back the
          ones already moved, leave the entry CURRENT, print a [!] line and
          record it in the audit log, and go on to the next entry;
        * if all moved (or were already absent), mark it invalidated and
          write the manifest straight away. If THAT write fails (a share
          dropping out, the file held open past the retry time), the entry
          is marked current again, its files are put back, the failure is
          printed and logged, and the error is raised -- the pass stops,
          because the next write would most likely fail the same way.

        Anything else that stops the pass part-way through an entry's files
        (Ctrl+C, an unexpected error) also puts that entry's moved files back
        before it is raised, so the entry stays current with all its files in
        place. A damaged manifest stops the pass before anything moves; it is
        left where it is (only a registration sets one aside).

        The one window that remains is a hard stop of the process (power
        loss, the process killed) between an entry's last move and the write
        that records it: that entry's files are then in _archived/ while the
        manifest still calls it current, and the audit log has no
        "invalidate" line for it. The whole pass holds the registry lock,
        refreshed after each entry so a long pass is never mistaken for a
        crashed writer.

        Args:
            sample: Optional sample name to invalidate. None = invalidate all.

        Returns:
            List of entry keys that were invalidated. An entry whose files
            could not all be moved is NOT in it (see the printed warnings).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        invalidated: List[str] = []
        failed: List[str] = []

        with self._locked() as lock:
            # A damaged manifest is NOT set aside here: invalidate has nothing
            # of its own to record, and everything it would do concerns the
            # entries in that very file, so it raises RegistryUnreadableError
            # and moves nothing; a person repairs the file first.
            registry = self._read_registry(set_aside_unreadable=False)
            # Read (and check) the approved method before anything moves; the
            # per-entry writes below keep this header (refresh_header=False).
            self._set_approved_header(registry)

            for key, entry in list(registry.get("entries", {}).items()):
                if sample is not None and entry.get("sample") != sample:
                    continue
                if not entry.get("is_current", True):
                    continue  # already invalidated

                moved: List[Tuple[Path, Path]] = []
                problem = None
                try:
                    for _output_type, rel_path in entry.get("outputs", {}).items():
                        src = self.registry_root / rel_path
                        if not src.exists():
                            continue  # nothing left in the active path to archive
                        dest = src.parent / "_archived" / timestamp / src.name
                        try:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            _rename_only(src, dest)
                            moved.append((src, dest))
                        except OSError as e:
                            problem = "could not archive %s: %s" % (src, e)
                            break
                    if problem is None and include_own_folder:
                        for folder, src in self.own_folder_files(entry):
                            # Keeps its place below the sample folder, in the
                            # same _archived/<timestamp>/ as the listed outputs.
                            dest = (folder / "_archived" / timestamp
                                    / src.relative_to(folder))
                            try:
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                _rename_only(src, dest)
                                moved.append((src, dest))
                            except OSError as e:
                                problem = "could not archive %s: %s" % (src, e)
                                break
                except BaseException as e:
                    # Anything else stopping the pass between two files of one
                    # entry (Ctrl+C, an unexpected error): put back what this
                    # entry already moved, so the files match the manifest,
                    # which still calls the entry current; then stop.
                    problem = ("interrupted (%s: %s)" % (type(e).__name__, e)
                               + self._put_back(moved))
                    print("  [!] %s NOT invalidated, still current: %s" % (key, problem))
                    self._log_event("invalidate_failed", entry.get("sample", key),
                                    key, problem)
                    if invalidated:
                        self._log_event("invalidate", sample or "ALL",
                                        f"{len(invalidated)} entries", timestamp)
                    raise

                if problem is not None:
                    problem += self._put_back(moved)
                    print("  [!] %s NOT invalidated, still current: %s" % (key, problem))
                    self._log_event("invalidate_failed", entry.get("sample", key),
                                    key, problem)
                    failed.append(key)
                    continue

                entry["is_current"] = False
                entry["invalidated_at"] = datetime.now().isoformat()
                try:
                    self._write_registry(registry, refresh_header=False)
                except BaseException as e:
                    # The manifest on disk still calls this entry current:
                    # make the files and the in-memory entry say so too.
                    entry["is_current"] = True
                    entry.pop("invalidated_at", None)
                    problem = ("the manifest could not be written (%s: %s)"
                               % (type(e).__name__, e)) + self._put_back(moved)
                    print("  [!] %s NOT invalidated, still current: %s" % (key, problem))
                    print("  [!] Invalidation stopped; %d earlier entr%s invalidated."
                          % (len(invalidated), "y was" if len(invalidated) == 1 else "ies were"))
                    self._log_event("invalidate_failed", entry.get("sample", key),
                                    key, problem)
                    if invalidated:
                        self._log_event("invalidate", sample or "ALL",
                                        f"{len(invalidated)} entries", timestamp)
                    raise
                lock.touch()
                invalidated.append(key)

            if invalidated:
                self._log_event(
                    "invalidate",
                    sample or "ALL",
                    f"{len(invalidated)} entries",
                    timestamp,
                )

        if failed:
            print("  [!] %d entr%s could not be invalidated (files could not be moved); "
                  "they stay current. Free the files and run invalidate again."
                  % (len(failed), "y" if len(failed) == 1 else "ies"))
        return invalidated

    @staticmethod
    def _put_back(moved: List[Tuple[Path, Path]]) -> str:
        """Move archived files back to where they were (newest first).

        Returns "" when all went back, else text naming each file that could
        not be put back and where it is now, for the [!] line and the log."""
        problem = ""
        for src, dest in reversed(moved):
            try:
                _rename_only(dest, src)
            except OSError as e:
                problem += "; and could not put back %s (it is at %s): %s" % (src, dest, e)
        return problem

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    def _log_event(self, action: str, sample: str, detail: str, extra: str = "") -> None:
        """Append a line to the audit log.

        The log file is ``logs/{analysis_name}.log`` under the registry root.
        One line per event, tab-separated for easy parsing.
        """
        log_path = self.logs_dir / f"{self.analysis_name}.log"
        ts = datetime.now().isoformat()
        host = _get_hostname()
        line = f"{ts}\t{host}\t{action}\t{sample}\t{detail}\t{extra}\n"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass  # Logging failure should never break the pipeline


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _move_file(src: Path, dest: Path) -> None:
    """Rename *src* to *dest*, replacing *dest*; copy-and-delete when they are
    on different drives. On Windows a rename onto a file another program has
    open for a moment is retried for a few seconds, as the manifest write is."""
    deadline = time.monotonic() + AnalysisRegistry.REPLACE_RETRY_SECONDS
    while True:
        try:
            os.replace(str(src), str(dest))
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)
        except OSError:
            # Different drive (or another reason a rename cannot work).
            shutil.copy2(str(src), str(dest))
            src.unlink()
            return


def _rename_only(src: Path, dest: Path) -> None:
    """Rename *src* to *dest* (same drive), never copying; refuse if *dest*
    exists. A rename Windows refuses because another program has the file
    open is retried for REPLACE_RETRY_SECONDS, then raised.

    WHY never a copy: every move the registry makes within itself -- a
    file archived into _archived/ beside it by invalidate (and put back by
    _put_back), a damaged manifest set aside into logs/ -- stays on one
    drive. shutil.move falls back to copy-then-delete when the rename fails;
    for a file another program holds open the copy succeeds and the delete
    fails, so the original stays in place AND a stray copy is left behind:
    in _archived/, where nothing cleans it up and an integrator mirrors it
    as an archived result, or in logs/, where it is reported as a set-aside
    manifest while the damaged file is still in use. A failed rename leaves
    nothing behind.
    """
    deadline = time.monotonic() + AnalysisRegistry.REPLACE_RETRY_SECONDS
    while True:
        try:
            os.rename(str(src), str(dest))
            return
        except FileExistsError:
            raise
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(str(a), str(b))
    except OSError:
        return False


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _get_hostname() -> str:
    """Get the machine hostname, safely."""
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _find_diff_keys(
    registered: Dict[str, Any],
    current: Dict[str, Any],
) -> List[str]:
    """Find parameter keys that differ between two method-param dicts.

    Returns a list of keys where the value differs or a key exists in one
    dict but not the other.
    """
    all_keys = set(registered.keys()) | set(current.keys())
    diffs = []
    for k in sorted(all_keys):
        v_reg = registered.get(k)
        v_cur = current.get(k)
        # Normalize for comparison (JSON round-trip to handle int/float etc.)
        if json.dumps(v_reg, sort_keys=True, default=str) != json.dumps(v_cur, sort_keys=True, default=str):
            diffs.append(k)
    return diffs


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Minimal CLI for inspecting registry state (console script: mousebrain-registry)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mousebrain-registry",
        description="Analysis Registry inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mousebrain-registry --name ENCR_ROI_Analysis\n"
            "  mousebrain-registry --name ENCR_Detection --stale\n"
            "  mousebrain-registry --name ENCR_Detection --summary\n"
            "  mousebrain-registry --name ENCR_Detection --root /path/to/Registry\n"
            "  mousebrain-registry --name ENCR_Detection --refresh-approved --stale\n"
            "\n"
            "Guide: docs/pipeline/ANALYSIS_REGISTRY.md\n"
        ),
    )
    parser.add_argument(
        "--name", required=True, help="Analysis name (e.g. ENCR_Detection)"
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Registry root (default: MOUSEBRAIN_REGISTRY_ROOT, else <pipeline root>/Registry)",
    )
    # Deprecated spelling of --root, kept working but hidden from --help.
    parser.add_argument(
        "--db-root", dest="root", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--stale", action="store_true",
        help="Check for stale entries against the approved method"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary table"
    )
    parser.add_argument(
        "--refresh-approved", action="store_true",
        help=("Re-read the approved method now and rewrite it into the manifest "
              "header (the hash an integrator counts 'stale vs approved' against). "
              "Every registration does this anyway; use it after changing "
              "approved_method.json when nothing new has been registered since."),
    )
    # WHY these exist: the code that registers an analysis declares whether
    # it follows the shared approved method, and the declaration is kept in
    # the manifest. A manifest written before declarations existed carries
    # none (= follows the shared method) until its producer next registers;
    # these record the declaration now, without waiting for that.
    declare = parser.add_mutually_exclusive_group()
    declare.add_argument(
        "--no-approved-method", dest="follows", action="store_const", const=False,
        help=("Record in the manifest that this analysis has NO approved method "
              "unless approved_method.json names it (the 3D analyses), then "
              "refresh the header. Implies --refresh-approved."),
    )
    declare.add_argument(
        "--follows-approved-method", dest="follows", action="store_const", const=True,
        help=("Record in the manifest that this analysis is judged by the shared "
              "approved method, then refresh the header. Implies --refresh-approved."),
    )

    args = parser.parse_args(argv)
    if args.follows is not None:
        args.refresh_approved = True
    try:
        registry = AnalysisRegistry(analysis_name=args.name, registry_root=args.root,
                                    follows_approved_method=args.follows)
    except RuntimeError as e:  # unconfigured root: one line, exit code, no traceback
        print("[FAIL] %s" % e)
        return 1

    print("=" * 70)
    print(f"Analysis Registry: {args.name}")
    print(f"  Registry root: {registry.registry_root}")
    print(f"  Manifest:      {registry.registry_path}")
    print("=" * 70)

    # Listed here, before anything else, so a set-aside manifest is seen even
    # when the counts below look complete (or are zero). See _read_registry.
    registry._set_aside_warned = True  # listed in full here instead
    set_aside = registry.set_aside_records()
    if set_aside:
        print("\n  [!] %d damaged manifest%s of this analysis set aside, waiting for a "
              "person to repair and merge back:" % (len(set_aside),
                                                     "" if len(set_aside) == 1 else "s"))
        for rec in set_aside:
            print("      %s  (set aside %s, about %s entries)" % (
                registry.set_aside_dir / rec["file"], rec["set_aside_at"] or "?",
                "?" if rec["entries_estimate"] is None else rec["entries_estimate"]))
        print("      Entries recorded only there are NOT counted below. Guide: "
              "docs/pipeline/ANALYSIS_REGISTRY.md")

    try:
        if args.refresh_approved:
            refreshed = registry.refresh_approved_method()
            if refreshed is None:
                print("\n  [!] No manifest for this analysis yet; nothing to refresh.")
            else:
                print("\n  [OK] Approved-method header refreshed.")
        data = registry._read_registry()
        # Resolved here, inside the try, so a broken approved-method file is
        # one [FAIL] line, not a traceback.
        approved = registry.approved_method(data) if args.stale else None
    except (RuntimeError, TimeoutError) as e:  # unreadable manifest or approved-method file, lock held
        print("[FAIL] %s" % e)
        return 1
    entries = data.get("entries", {})
    n_current = sum(1 for e in entries.values() if e.get("is_current", True))
    n_stale = sum(1 for e in entries.values() if not e.get("is_current", True))

    print(f"\n  Total entries: {len(entries)}")
    print(f"  Current:       {n_current}")
    print(f"  Invalidated:   {n_stale}")
    print(f"  Last updated:  {data.get('last_updated', 'never')}")
    recorded_hash = data.get("approved_method_hash")
    print("  Approved method in the header: %s (recorded %s on %s, from %s)" % (
        recorded_hash[:12] if recorded_hash else "none declared",
        data.get("approved_method_recorded_at", "when the manifest was created"),
        data.get("approved_method_recorded_on", "?"),
        data.get("approved_method_source", "?")))

    if args.stale:
        print("\n--- Staleness check (vs approved method) ---")
        stale_list = [] if approved is None else registry.check_staleness(approved)
        if approved is None:
            print("  No approved method is declared for this analysis, so none of")
            print("  its entries is compared or counted as stale.")
        elif stale_list:
            print(f"  {len(stale_list)} stale entries:")
            for s in stale_list:
                diff = ", ".join(s["diff_keys"]) if s["diff_keys"] else "hash mismatch"
                print(f"    {s['sample']}: changed [{diff}]")
        else:
            print("  All entries are current.")

    if args.summary:
        print("\n--- Summary ---")
        try:
            df = registry.get_summary_df()
            if df.empty:
                print("  No entries.")
            else:
                # ASCII-safe table printing
                cols = ["sample", "category", "is_current", "registered_at"]
                result_cols = [c for c in df.columns if c.startswith("result_")]
                cols.extend(result_cols[:5])  # limit width
                display = df[cols] if all(c in df.columns for c in cols) else df
                print(display.to_string(index=False))
        except ImportError:
            print("  pandas not available; cannot print summary table.")
            # Fallback: just list samples
            for key, entry in sorted(entries.items()):
                status = "current" if entry.get("is_current", True) else "stale"
                print(f"    {entry.get('sample', key)}: {status}")

    print()
    return 0


# Older name of the entry point, kept as an alias.
_cli_main = main


if __name__ == "__main__":
    raise SystemExit(main())
