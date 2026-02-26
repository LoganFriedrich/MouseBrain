#!/usr/bin/env python3
"""
migrate_data_to_databases.py - Data migration for ENCR analysis compliance.

This script enforces the data handling philosophy by:
  1. Archiving old-method ENCR_Detection exports (no soma_dilation, sigma_threshold=1.5)
  2. Registering new batch results from Pipeline into Databases via AnalysisRegistry
  3. Registering ROI figures from Pipeline into Databases via AnalysisRegistry
  4. Cleaning up Pipeline copies that have been migrated to Databases

Run with: C:\\LAB_ROOT\\envs\\MouseBrain\\python.exe migrate_data_to_databases.py

Author: Data migration script (2026-02-22)
"""

import csv
import os
import shutil
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

DB_ROOT = Path(r"Y:\LAB_ROOT\Databases")
ENCR_DET_EXPORTS = DB_ROOT / "exports" / "ENCR_Detection"
ARCHIVE_DIR = ENCR_DET_EXPORTS / "_archived" / "20260222_pre_soma_dilation"

BATCH_RESULTS_ROOT = Path(
    r"Y:\LAB_ROOT\Tissue\MouseBrain_Pipeline\2D_Slices\2_Data_Summary\batch_results"
)
BATCH_SUMMARY_CSV = BATCH_RESULTS_ROOT / "summary.csv"

ROI_FIGURES_DIR = Path(
    r"Y:\LAB_ROOT\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ROI_Figures"
)

# HD_Regions directories where ND2 and ROI JSON source files live
HD_REGIONS_DIRS = [
    Path(r"Y:\LAB_ROOT\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_01_HD_Regions"),
    Path(r"Y:\LAB_ROOT\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions"),
    Path(r"Y:\LAB_ROOT\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_03_HD_Regions"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_soma_column(csv_path):
    """Check if a measurements CSV has the soma_mean_intensity column (new method)."""
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return False
            return "soma_mean_intensity" in header
    except (OSError, UnicodeDecodeError):
        return False


def find_measurements_csv(sample_dir):
    """Find the measurements CSV in a sample directory.

    Checks for both naming conventions:
      - measurements.csv (new style)
      - {sample}_measurements.csv (old style)
    """
    sample_name = sample_dir.name
    # Try generic name first
    generic = sample_dir / "measurements.csv"
    if generic.exists():
        return generic
    # Try named variant
    named = sample_dir / f"{sample_name}_measurements.csv"
    if named.exists():
        return named
    # Search for any *measurements*.csv
    for f in sample_dir.iterdir():
        if f.is_file() and "measurements" in f.name.lower() and f.suffix == ".csv":
            return f
    return None


def find_source_file(sample_stem, extension, search_dirs):
    """Search HD_Regions directories for a source file matching sample_stem.

    Searches both the root and Corrected/ subdirectory of each HD_Regions dir.
    """
    target = f"{sample_stem}{extension}"
    for hd_dir in search_dirs:
        # Check root
        candidate = hd_dir / target
        if candidate.exists():
            return candidate
        # Check Corrected subdirectory
        corrected = hd_dir / "Corrected" / target
        if corrected.exists():
            return corrected
    return None


def parse_batch_summary(csv_path):
    """Parse the batch summary CSV into a dict keyed by sample stem.

    Returns dict mapping sample_stem -> row dict.
    """
    results = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Sample stem is the filename without .nd2 extension
            filename = row.get("file", "")
            stem = filename.replace(".nd2", "") if filename else ""
            if stem:
                results[stem] = dict(row)
    return results


# ---------------------------------------------------------------------------
# Step 1: Archive old stale ENCR_Detection data
# ---------------------------------------------------------------------------

def step1_archive_old_data():
    """Scan exports/ENCR_Detection/ and archive samples without soma_mean_intensity."""
    print("=" * 70)
    print("STEP 1: Archive old-method ENCR_Detection data")
    print("=" * 70)

    if not ENCR_DET_EXPORTS.exists():
        print("  [!] ENCR_Detection exports directory not found. Skipping.")
        return 0, 0

    archived_count = 0
    kept_count = 0
    errors = []

    # Enumerate sample directories
    for item in sorted(ENCR_DET_EXPORTS.iterdir()):
        # Skip non-directories, _archived, and known non-sample files
        if not item.is_dir():
            continue
        if item.name.startswith("_"):
            continue

        measurements = find_measurements_csv(item)
        if measurements is None:
            # No measurements CSV at all -- treat as old/incomplete
            print(f"  ARCHIVE (no measurements): {item.name}")
            try:
                dest = ARCHIVE_DIR / item.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))
                archived_count += 1
            except OSError as e:
                err_msg = f"Failed to archive {item.name}: {e}"
                print(f"  [!] {err_msg}")
                errors.append(err_msg)
            continue

        if has_soma_column(measurements):
            # New method -- keep
            kept_count += 1
        else:
            # Old method -- archive
            print(f"  ARCHIVE (old method): {item.name}")
            try:
                dest = ARCHIVE_DIR / item.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))
                archived_count += 1
            except OSError as e:
                err_msg = f"Failed to archive {item.name}: {e}"
                print(f"  [!] {err_msg}")
                errors.append(err_msg)

    # Archive old summary.csv and batch_summary.csv
    for old_csv_name in ["summary.csv", "batch_summary.csv"]:
        old_csv = ENCR_DET_EXPORTS / old_csv_name
        if old_csv.exists():
            print(f"  ARCHIVE: {old_csv_name}")
            try:
                ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_csv), str(ARCHIVE_DIR / old_csv_name))
            except OSError as e:
                err_msg = f"Failed to archive {old_csv_name}: {e}"
                print(f"  [!] {err_msg}")
                errors.append(err_msg)

    print(f"\n  Summary: {archived_count} archived, {kept_count} kept")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    - {e}")

    return archived_count, kept_count


# ---------------------------------------------------------------------------
# Step 2: Register new batch results from Pipeline
# ---------------------------------------------------------------------------

def step2_register_batch_results():
    """Register each batch_results sample directory via AnalysisRegistry."""
    print("\n" + "=" * 70)
    print("STEP 2: Register new batch results from Pipeline")
    print("=" * 70)

    from mousebrain.analysis_registry import AnalysisRegistry, get_approved_method

    registry = AnalysisRegistry(analysis_name="ENCR_Detection")
    method_params = get_approved_method()

    # Parse batch summary for results metadata
    if not BATCH_SUMMARY_CSV.exists():
        print(f"  [!] Batch summary CSV not found: {BATCH_SUMMARY_CSV}")
        return 0, []

    summary_data = parse_batch_summary(BATCH_SUMMARY_CSV)
    print(f"  Found {len(summary_data)} samples in batch summary CSV")

    registered_count = 0
    errors = []

    for sample_stem in sorted(summary_data.keys()):
        row = summary_data[sample_stem]
        sample_dir = BATCH_RESULTS_ROOT / sample_stem
        status = row.get("status", "")

        # Build results dict from summary CSV
        results = {}
        for key in ["n_nuclei", "n_positive", "n_negative"]:
            val = row.get(key, "")
            if val != "":
                try:
                    results[key] = int(val)
                except (ValueError, TypeError):
                    pass
        for key in ["positive_fraction", "mean_fold_change", "median_fold_change", "background"]:
            val = row.get(key, "")
            if val != "":
                try:
                    results[key] = float(val)
                except (ValueError, TypeError):
                    pass
        results["status"] = status

        # Source ND2 path from summary
        nd2_path_str = row.get("path", "")
        source_files = {}
        if nd2_path_str:
            source_files["nd2"] = nd2_path_str

        # Find ROI JSON
        roi_json = find_source_file(sample_stem, ".rois.json", HD_REGIONS_DIRS)
        if roi_json:
            source_files["roi_json"] = str(roi_json)

        # Collect output files from the batch_results sample directory
        files = {}
        if sample_dir.is_dir():
            for f in sample_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix == ".csv" and "measurements" in f.name.lower():
                    files["measurements"] = str(f)
                elif f.suffix == ".csv":
                    files["data_csv"] = str(f)
                elif f.suffix == ".png" and "overlay" in f.name.lower():
                    files["overlay"] = str(f)
                elif f.suffix == ".png" and "annotated_overlay" in f.name.lower():
                    files["annotated_overlay"] = str(f)
                elif f.suffix == ".png" and "background" in f.name.lower():
                    files["qc_figure"] = str(f)
                elif f.suffix == ".png" and "fold_change" in f.name.lower():
                    files["fold_change_histogram"] = str(f)
                elif f.suffix == ".png" and "scatter" in f.name.lower():
                    files["intensity_scatter"] = str(f)
                elif f.suffix == ".png":
                    # Catch remaining PNGs by name
                    files[f.stem] = str(f)
        else:
            if status == "no_nuclei":
                print(f"  SKIP (no_nuclei, no dir): {sample_stem}")
                # Still register the metadata for completeness
                try:
                    # Register with no files -- just the metadata entry
                    registry.register_output(
                        sample=sample_stem,
                        category="detection",
                        files={},
                        results=results,
                        method_params=method_params,
                        source_files=source_files,
                    )
                    registered_count += 1
                    print(f"  REGISTERED (metadata only): {sample_stem}")
                except Exception as exc:
                    err_msg = f"{sample_stem}: {exc}"
                    print(f"  [!] ERROR: {err_msg}")
                    errors.append(err_msg)
                continue
            else:
                print(f"  [!] WARNING: No batch_results dir for {sample_stem} (status={status})")
                errors.append(f"{sample_stem}: no batch_results directory")
                continue

        # Register via AnalysisRegistry
        try:
            dest_paths = registry.register_output(
                sample=sample_stem,
                category="detection",
                files=files,
                results=results,
                method_params=method_params,
                source_files=source_files,
            )
            registered_count += 1
            n_files = len(dest_paths)
            print(f"  REGISTERED: {sample_stem} ({n_files} files)")
        except Exception as exc:
            err_msg = f"{sample_stem}: {exc}"
            print(f"  [!] ERROR registering {sample_stem}:")
            traceback.print_exc()
            errors.append(err_msg)

    print(f"\n  Summary: {registered_count} registered, {len(errors)} errors")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return registered_count, errors


# ---------------------------------------------------------------------------
# Step 3: Register ROI figures from Pipeline
# ---------------------------------------------------------------------------

def step3_register_roi_figures():
    """Register ROI coloc_result figures via AnalysisRegistry."""
    print("\n" + "=" * 70)
    print("STEP 3: Register ROI figures from Pipeline")
    print("=" * 70)

    from mousebrain.analysis_registry import (
        AnalysisRegistry,
        get_approved_method,
        parse_sample_id,
    )

    roi_registry = AnalysisRegistry(analysis_name="ENCR_ROI_Analysis")
    method_params = get_approved_method()

    if not ROI_FIGURES_DIR.exists():
        print(f"  [!] ROI_Figures directory not found: {ROI_FIGURES_DIR}")
        return 0, []

    registered_count = 0
    errors = []

    for fig in sorted(ROI_FIGURES_DIR.glob("*_coloc_result.png")):
        stem = fig.stem.replace("_coloc_result", "")
        animal, region = parse_sample_id(stem)

        # Find matching source files
        source_files = {}
        nd2_path = find_source_file(stem, ".nd2", HD_REGIONS_DIRS)
        if nd2_path:
            source_files["nd2"] = str(nd2_path)

        roi_json = find_source_file(stem, ".rois.json", HD_REGIONS_DIRS)
        if roi_json:
            source_files["roi_json"] = str(roi_json)

        # Basic results (empty -- the figure IS the result)
        results = {
            "animal": animal,
            "region": region,
        }

        try:
            dest_paths = roi_registry.register_output(
                sample=stem,
                category="roi_analysis",
                files={"figure": str(fig)},
                results=results,
                method_params=method_params,
                source_files=source_files,
            )
            registered_count += 1
            print(f"  REGISTERED: {stem} -> {animal}/{region}/")
        except Exception as exc:
            err_msg = f"{stem}: {exc}"
            print(f"  [!] ERROR registering {stem}:")
            traceback.print_exc()
            errors.append(err_msg)

    print(f"\n  Summary: {registered_count} registered, {len(errors)} errors")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return registered_count, errors


# ---------------------------------------------------------------------------
# Step 4: Clean up Pipeline copies
# ---------------------------------------------------------------------------

def step4_cleanup():
    """Remove ROI_Figures from Pipeline (now in Databases)."""
    print("\n" + "=" * 70)
    print("STEP 4: Clean up Pipeline copies")
    print("=" * 70)

    if ROI_FIGURES_DIR.exists():
        file_count = sum(1 for _ in ROI_FIGURES_DIR.iterdir() if _.is_file())
        print(f"  Removing {ROI_FIGURES_DIR} ({file_count} files)")
        try:
            shutil.rmtree(str(ROI_FIGURES_DIR))
            print(f"  [OK] Removed ROI_Figures directory")
        except OSError as e:
            print(f"  [!] Failed to remove ROI_Figures: {e}")
    else:
        print(f"  ROI_Figures directory already removed.")

    print("  NOTE: batch_results/ left in place (working/intermediate data)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ENCR Data Migration to Databases")
    print("Enforcing code/data separation and provenance tracking")
    print("Date: 2026-02-22")
    print("=" * 70)

    # Step 1: Archive old data
    archived, kept = step1_archive_old_data()

    # Step 2: Register batch results
    det_registered, det_errors = step2_register_batch_results()

    # Step 3: Register ROI figures
    roi_registered, roi_errors = step3_register_roi_figures()

    # Step 4: Cleanup
    step4_cleanup()

    # Final summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"  Step 1 - Archived old samples:   {archived}")
    print(f"  Step 1 - Kept (already new):     {kept}")
    print(f"  Step 2 - Detection registered:   {det_registered}")
    print(f"  Step 2 - Detection errors:       {len(det_errors)}")
    print(f"  Step 3 - ROI figures registered:  {roi_registered}")
    print(f"  Step 3 - ROI figure errors:       {len(roi_errors)}")
    print("=" * 70)

    total_errors = len(det_errors) + len(roi_errors)
    if total_errors > 0:
        print(f"\n  [!] {total_errors} total errors occurred. Review output above.")
        return 1
    else:
        print("\n  [OK] Migration completed successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
