#!/usr/bin/env python3
"""
analysis_registry.py - Registry for analysis outputs with provenance tracking.

Manages analysis outputs (figures, CSVs, counts) with:
  - Automatic file copying to canonical database locations
  - Full provenance metadata (method params, source files, timestamps)
  - Deterministic method hashing for staleness detection
  - Atomic JSON writes for thread safety
  - Per-region summary CSV maintenance

The database location follows existing conventions:
    Databases/
    +-- exports/{analysis_name}/     Per-sample detection exports
    |   +-- {sample}/               Measurements, figures per sample
    |   +-- summary.csv             Batch summary
    +-- exports/{analysis_name}/     Per-sample ROI analysis
    |   +-- {animal}/               Per-animal grouping
    |   +-- roi_summary_{region}.csv
    +-- figures/{analysis_name}/     Figures organized by animal/region
    |   +-- {animal}/{region}/
    +-- logs/                        Audit trail

Usage:
    from mousebrain.analysis_registry import AnalysisRegistry, get_approved_method

    registry = AnalysisRegistry(analysis_name="ENCR_ROI_Analysis")

    # Register a processed sample
    db_paths = registry.register_output(
        sample="E02_01_S13_DCN",
        category="roi_analysis",
        files={"figure": "/path/to/fig.png", "roi_counts": "/path/to/counts.csv"},
        results={"n_nuclei": 13, "n_positive": 12, "positive_fraction": 0.923},
        method_params=get_approved_method(),
        source_files={"nd2": "/path/to/E02_01_S13_DCN.nd2"},
    )

    # Check which samples are stale after a method change
    stale = registry.get_stale_samples(new_method_params)
"""

import csv
import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "AnalysisRegistry",
    "parse_sample_id",
    "get_approved_method",
    "get_database_path",
]


# =============================================================================
# APPROVED METHOD DEFINITION
# =============================================================================

# This is the PI-approved analysis method for ENCR (as of 2026-02-23).
# Do NOT change without updating METHOD_LOG.md and getting PI sign-off.
_APPROVED_METHOD = {
    "detection": "threshold+log",
    "threshold_fraction": 0.20,
    "log_threshold": 0.005,
    "log_decision_tree": True,
    "size_filter_um": [8, 25],
    "colocalization": "background_mean",
    "sigma_threshold": 0,
    "soma_dilation": 6,
    "background": "gmm",
    "background_percentile": 10,
    "bg_exclusion_dilation": 50,
}


def get_approved_method() -> Dict[str, Any]:
    """Return the current PI-approved analysis method parameters.

    These match the APPROVED METHOD block in batch_encr.py and METHOD_LOG.md.

    Returns:
        Dict of method parameter name -> value.
    """
    return dict(_APPROVED_METHOD)


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
# DATABASE PATH COMPUTATION
# =============================================================================

_DEFAULT_DB_ROOT = Path(r"Y:\2_Connectome\Databases")


def get_database_path(
    db_root: Path,
    category: str,
    sample: str,
    animal: str,
    region: str,
    analysis_name: str = "",
) -> Dict[str, Path]:
    """Compute canonical database paths for a given output.

    Follows the existing directory conventions:
        exports/{analysis_name}/{animal}/{base_region}/   -- data files
        figures/{analysis_name}/{animal}/{base_region}/   -- figure files

    Args:
        db_root: Root of the Databases directory.
        category: "detection" or "roi_analysis".
        sample: Full sample ID (e.g. "E02_01_S13_DCN").
        animal: Animal ID (e.g. "E02_01").
        region: Region string (e.g. "DCN", "DCNv2").
        analysis_name: Name of the analysis (e.g. "ENCR_ROI_Analysis").

    Returns:
        Dict with keys 'export_dir' and 'figure_dir' mapping to Path objects.
    """
    base_reg = _base_region(region) if region else ""

    if category == "detection":
        export_dir = db_root / "exports" / analysis_name / sample
        figure_dir = export_dir  # detection figures live alongside data
    elif category == "roi_analysis":
        if animal and base_reg:
            export_dir = db_root / "exports" / analysis_name / animal / base_reg
            figure_dir = db_root / "figures" / analysis_name / animal / base_reg
        elif animal:
            export_dir = db_root / "exports" / analysis_name / animal
            figure_dir = db_root / "figures" / analysis_name / animal
        else:
            export_dir = db_root / "exports" / analysis_name / sample
            figure_dir = db_root / "figures" / analysis_name / sample
    else:
        export_dir = db_root / "exports" / analysis_name / sample
        figure_dir = db_root / "figures" / analysis_name / sample

    return {"export_dir": export_dir, "figure_dir": figure_dir}


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

class AnalysisRegistry:
    """Manages analysis outputs with provenance, auto-push, and staleness detection.

    Each registry instance is tied to a single analysis_name (e.g.
    "ENCR_Detection" or "ENCR_ROI_Analysis") and maintains a registry.json
    manifest in the corresponding exports directory.

    The manifest tracks every registered output with:
      - Full method parameters and their deterministic hash
      - Source file paths (ND2, ROI JSON, etc.)
      - Key results (counts, fractions)
      - Output file locations (relative to db_root)
      - Registration timestamp
      - Staleness flag (is_current)

    Thread safety: writes use atomic temp-file-then-rename to avoid corruption
    when multiple processes register outputs concurrently.

    Args:
        db_root: Path to the Databases directory. Defaults to Y:/2_Connectome/Databases.
        analysis_name: Name of the analysis (e.g. "ENCR_Detection").
    """

    # Current schema version for the registry JSON
    SCHEMA_VERSION = 1

    def __init__(
        self,
        analysis_name: str,
        db_root: Optional[Path] = None,
    ):
        self.analysis_name = analysis_name
        self.db_root = Path(db_root) if db_root else _DEFAULT_DB_ROOT
        self.exports_dir = self.db_root / "exports" / analysis_name
        self.figures_dir = self.db_root / "figures" / analysis_name
        self.logs_dir = self.db_root / "logs"
        self.registry_path = self.exports_dir / "registry.json"

        # Ensure base directories exist
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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
        approved = get_approved_method()
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
        """Register an analysis output and copy files to the database.

        Copies each file in *files* to the canonical database location, records
        full provenance in the registry manifest, and returns the destination
        paths.

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
            placed in the database.
        """
        animal, region = parse_sample_id(sample)

        # Compute destination directories
        paths = get_database_path(
            self.db_root, category, sample, animal, region, self.analysis_name
        )
        export_dir = paths["export_dir"]
        figure_dir = paths["figure_dir"]
        export_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        # Copy files to database and build relative output map
        output_map = {}
        dest_paths = {}
        for output_type, src_path_str in files.items():
            src_path = Path(src_path_str)
            if not src_path.exists():
                print(f"  [!] WARNING: Source file does not exist: {src_path}")
                continue

            # Route figures to figure_dir, everything else to export_dir
            if output_type in ("figure", "roi_figure", "qc_figure", "overlay"):
                dest = figure_dir / src_path.name
            else:
                dest = export_dir / src_path.name

            shutil.copy2(str(src_path), str(dest))
            dest_paths[output_type] = dest

            # Store path relative to db_root for portability
            try:
                rel = dest.relative_to(self.db_root)
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
        roi_results: Dict[str, Dict[str, Any]],
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
            method_params: Dict of method parameters.
            source_files: Optional dict of source file paths.

        Returns:
            Path to the per-sample roi_counts CSV in the database.
        """
        animal, _ = parse_sample_id(sample)
        base_reg = _base_region(region) if region else region

        # Write per-sample ROI counts CSV
        export_dir = self.exports_dir / animal / base_reg
        export_dir.mkdir(parents=True, exist_ok=True)
        counts_path = export_dir / f"{sample}_roi_counts.csv"

        fieldnames = ["roi", "total", "positive", "negative", "fraction"]
        with open(counts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for roi_name, counts in roi_results.items():
                writer.writerow({
                    "roi": roi_name,
                    "total": counts.get("total", 0),
                    "positive": counts.get("positive", 0),
                    "negative": counts.get("negative", 0),
                    "fraction": counts.get("fraction", 0.0),
                })

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
                "roi_counts": str(counts_path.relative_to(self.db_root)),
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

        Format matches existing convention:
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
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        """Get a single registry entry by sample name.

        Args:
            sample: Sample identifier.

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
        so they are preserved but no longer in the active output path.

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
                src = self.db_root / rel_path
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

        The log file is ``logs/{analysis_name}.log`` under db_root.
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

def _cli_main():
    """Minimal CLI for inspecting registry state."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analysis Registry inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m mousebrain.analysis_registry --name ENCR_ROI_Analysis\n"
            "  python -m mousebrain.analysis_registry --name ENCR_Detection --stale\n"
            "  python -m mousebrain.analysis_registry --name ENCR_Detection --summary\n"
        ),
    )
    parser.add_argument(
        "--name", required=True, help="Analysis name (e.g. ENCR_Detection)"
    )
    parser.add_argument(
        "--db-root", type=Path, default=None, help="Database root (default: auto-detect)"
    )
    parser.add_argument(
        "--stale", action="store_true",
        help="Check for stale entries against the approved method"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary table"
    )

    args = parser.parse_args()
    registry = AnalysisRegistry(analysis_name=args.name, db_root=args.db_root)

    print("=" * 70)
    print(f"Analysis Registry: {args.name}")
    print(f"  DB root:  {registry.db_root}")
    print(f"  Manifest: {registry.registry_path}")
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
        approved = get_approved_method()
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


if __name__ == "__main__":
    _cli_main()
