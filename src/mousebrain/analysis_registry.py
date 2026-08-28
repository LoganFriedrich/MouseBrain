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
    +-- approved_method.json           Optional: a lab's approved parameters

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
        sample="E02_01_S13_DCN",
        category="roi_analysis",
        files={"figure": "/path/to/fig.png", "roi_counts": "/path/to/counts.csv"},
        results={"n_nuclei": 13, "n_positive": 12, "positive_fraction": 0.923},
        method_params=get_approved_method(),
        source_files={"nd2": "/path/to/E02_01_S13_DCN.nd2"},
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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "AnalysisRegistry",
    "DEFAULT_METHOD",
    "APPROVED_METHOD_FILENAME",
    "default_registry_root",
    "get_approved_method",
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
        with open(path, "r", encoding="utf-8") as f:
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


def get_approved_method(registry_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Return the method parameters this installation treats as approved.

    A lab records the parameters it has signed off on OUTSIDE the code, so the
    tool carries no project history. Resolution order:

      1. MOUSEBRAIN_APPROVED_METHOD -- path to a JSON file holding the dict.
      2. ``<registry root>/approved_method.json`` when that file exists
         (*registry_root* if given, else default_registry_root()).
      3. DEFAULT_METHOD (the built-in parameters above).

    The JSON file is a flat object with the same keys as DEFAULT_METHOD, e.g.
    ``{"detection": "threshold+log", "threshold_fraction": 0.2, ...}``.
    Whatever the source, the returned dict has the same flat shape, so method
    hashes computed from it compare with hashes already stored in a registry.

    Args:
        registry_root: Registry root to look under for approved_method.json.
            Defaults to the resolved default root.

    Returns:
        A fresh dict of parameter name -> value (safe to modify).
    """
    env_path = os.environ.get("MOUSEBRAIN_APPROVED_METHOD")
    if env_path:
        return _load_method_file(Path(env_path), "MOUSEBRAIN_APPROVED_METHOD")

    root = Path(registry_root) if registry_root else default_registry_root()
    if root is not None:
        candidate = root / APPROVED_METHOD_FILENAME
        if candidate.is_file():
            return _load_method_file(candidate, APPROVED_METHOD_FILENAME)

    return dict(DEFAULT_METHOD)


# =============================================================================
# SAMPLE ID PARSING
# =============================================================================

def parse_sample_id(sample: str) -> Tuple[str, str]:
    """Extract animal ID and region from a sample name.

    Handles formats like:
        "E02_01_S13_DCN"     -> ("E02_01", "DCN")
        "E02_01_S13_DCNv2"   -> ("E02_01", "DCNv2")
        "E02_01_S17_DCN001"  -> ("E02_01", "DCN001")
        "E02_01_S1_R3"       -> ("E02_01", "R3")

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

    # Animal is always the first two parts (e.g. E02_01)
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
        sample: Full sample ID (e.g. "E02_01_S13_DCN").
        animal: Animal ID (e.g. "E02_01").
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

    Thread safety: writes use atomic temp-file-then-rename to avoid corruption
    when multiple processes register outputs concurrently.

    Args:
        analysis_name: Name of the analysis (e.g. "ENCR_Detection").
        registry_root: Root folder of the registry. Defaults to
            default_registry_root(); raises RuntimeError when nothing resolves.
        db_root: Deprecated alias of *registry_root*, kept so older callers
            keep working. Ignored when *registry_root* is given.
    """

    # Current schema version for the registry JSON
    SCHEMA_VERSION = 1

    def __init__(
        self,
        analysis_name: str,
        registry_root: Optional[Union[str, Path]] = None,
        db_root: Optional[Union[str, Path]] = None,
    ):
        self.analysis_name = analysis_name
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

    def _read_registry(self) -> Dict[str, Any]:
        """Read the registry manifest from disk.

        Returns a fresh skeleton if the file does not exist or is corrupted.
        """
        if not self.registry_path.exists():
            return self._empty_registry()

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation
            if not isinstance(data, dict) or "entries" not in data:
                return self._empty_registry()
            return data
        except (json.JSONDecodeError, OSError):
            return self._empty_registry()

    def _write_registry(self, data: Dict[str, Any]) -> None:
        """Write registry manifest atomically (write temp, then rename).

        This prevents corruption if two processes write simultaneously or
        the process is interrupted mid-write.
        """
        data["last_updated"] = datetime.now().isoformat()
        content = json.dumps(data, indent=2, ensure_ascii=False, default=str)

        # Write to a temp file in the same directory, then rename.
        # os.replace is atomic on the same filesystem on both Windows and POSIX.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.exports_dir), suffix=".tmp", prefix=".registry_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            # Atomic replace
            os.replace(tmp_path, str(self.registry_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _empty_registry(self) -> Dict[str, Any]:
        """Return an empty registry skeleton."""
        approved = get_approved_method(self.registry_root)
        return {
            "analysis_name": self.analysis_name,
            "version": self.SCHEMA_VERSION,
            "approved_method": approved,
            "approved_method_hash": self.get_method_hash(approved),
            "entries": {},
            "last_updated": datetime.now().isoformat(),
        }

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
            sample: Sample identifier (e.g. "E02_01_S13_DCN").
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

        # Compute destination directories
        paths = get_database_path(
            self.registry_root, category, sample, animal, region, self.analysis_name
        )
        export_dir = paths["export_dir"]
        figure_dir = paths["figure_dir"]
        export_dir.mkdir(parents=True, exist_ok=True)
        # figure_dir is created below, only when a figure is actually copied
        # (see the constructor for why figure folders are made lazily).

        # Copy files into the registry and build the relative output map
        output_map = {}
        dest_paths = {}
        for output_type, src_path_str in files.items():
            src_path = Path(src_path_str)
            if not src_path.exists():
                print(f"  [!] WARNING: Source file does not exist: {src_path}")
                continue

            # Route figures to figure_dir, everything else to export_dir
            if output_type in _FIGURE_OUTPUT_TYPES:
                figure_dir.mkdir(parents=True, exist_ok=True)
                dest = figure_dir / src_path.name
            else:
                dest = export_dir / src_path.name

            shutil.copy2(str(src_path), str(dest))
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

        # Update registry
        registry = self._read_registry()
        registry["entries"][sample] = entry
        self._write_registry(registry)

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
            sample: Sample identifier (e.g. "E02_01_S13_DCN").
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

        # Write per-sample ROI counts CSV
        export_dir = self.exports_dir / animal / base_reg
        export_dir.mkdir(parents=True, exist_ok=True)
        counts_path = export_dir / f"{sample}_roi_counts.csv"

        # Fixed columns first; any further count keys (e.g. the dual-channel
        # dual/red_only/green_only/neither) are appended so no count is lost.
        base_fields = ["roi", "total", "positive", "negative", "fraction"]
        extra_fields = sorted({
            str(k) for counts in roi_results.values() for k in counts
            if str(k) not in base_fields
        })
        fieldnames = base_fields + extra_fields
        with open(counts_path, "w", newline="", encoding="utf-8") as f:
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

        # Update the per-region summary CSV (roi_summary_{region}.csv)
        self._update_region_summary(sample, base_reg, roi_results)

        # Record in registry
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

        registry = self._read_registry()
        registry["entries"][entry_key] = entry
        self._write_registry(registry)

        self._log_event("register_roi_counts", sample, region, method_hash)
        return counts_path

    def _update_region_summary(
        self,
        sample: str,
        base_region: str,
        roi_results: Dict[str, Dict[str, Any]],
    ) -> None:
        """Update (or create) the per-region summary CSV.

        The summary CSV (e.g. roi_summary_DCN.csv) aggregates TOTAL counts
        from every sample in that region. If the sample already has a row, it
        is replaced; otherwise a new row is appended.

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
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
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

    def invalidate(self, sample: Optional[str] = None) -> List[str]:
        """Mark entries as stale and archive their output files.

        If *sample* is given, only that sample's entries are invalidated.
        If None, ALL entries are invalidated.

        Archived files are moved to an ``_archived/{timestamp}/`` subdirectory
        beside where they were, so they are preserved but no longer in the
        active output path.

        Args:
            sample: Optional sample name to invalidate. None = invalidate all.

        Returns:
            List of entry keys that were invalidated.
        """
        registry = self._read_registry()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        invalidated = []

        for key, entry in list(registry.get("entries", {}).items()):
            if sample is not None and entry.get("sample") != sample:
                continue
            if not entry.get("is_current", True):
                continue  # already stale

            # Archive output files
            for output_type, rel_path in entry.get("outputs", {}).items():
                src = self.registry_root / rel_path
                if src.exists():
                    archive_dir = src.parent / "_archived" / timestamp
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    dest = archive_dir / src.name
                    try:
                        shutil.move(str(src), str(dest))
                    except OSError as e:
                        print(f"  [!] WARNING: Could not archive {src}: {e}")

            entry["is_current"] = False
            entry["invalidated_at"] = datetime.now().isoformat()
            invalidated.append(key)

        if invalidated:
            self._write_registry(registry)
            self._log_event(
                "invalidate",
                sample or "ALL",
                f"{len(invalidated)} entries",
                timestamp,
            )

        return invalidated

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

    args = parser.parse_args(argv)
    try:
        registry = AnalysisRegistry(analysis_name=args.name, registry_root=args.root)
    except RuntimeError as e:  # unconfigured root: one line, exit code, no traceback
        print("[FAIL] %s" % e)
        return 1

    print("=" * 70)
    print(f"Analysis Registry: {args.name}")
    print(f"  Registry root: {registry.registry_root}")
    print(f"  Manifest:      {registry.registry_path}")
    print("=" * 70)

    data = registry._read_registry()
    entries = data.get("entries", {})
    n_current = sum(1 for e in entries.values() if e.get("is_current", True))
    n_stale = sum(1 for e in entries.values() if not e.get("is_current", True))

    print(f"\n  Total entries: {len(entries)}")
    print(f"  Current:       {n_current}")
    print(f"  Invalidated:   {n_stale}")
    print(f"  Last updated:  {data.get('last_updated', 'never')}")

    if args.stale:
        print("\n--- Staleness check (vs approved method) ---")
        approved = get_approved_method(registry.registry_root)
        stale_list = registry.check_staleness(approved)
        if stale_list:
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
