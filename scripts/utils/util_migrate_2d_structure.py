"""
util_migrate_2d_structure.py - Reorganize 2D_Slices directory structure

Migrates from ad-hoc structure with scattered files to clean per-subject pipeline structure.

Usage:
    python util_migrate_2d_structure.py --dry-run   # Preview moves (default)
    python util_migrate_2d_structure.py --execute   # Actually perform migration

CRITICAL: This script uses shutil.move() to avoid doubling disk usage.
Backup your data before running --execute.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# Import canonical parser
from mousebrain.plugin_2d.sliceatlas.core.config import parse_sample_name, parse_filename


# =============================================================================
# CONSTANTS
# =============================================================================

ROOT = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices")

# Source directories
SOURCE_ENCR = ROOT / "ENCR"
SOURCE_SLICEATLAS = ROOT / "SliceAtlas_Data"
SOURCE_BATCH_RESULTS = SOURCE_ENCR / "batch_results"

# Target structure
TARGET_SUBJECTS = ROOT / "1_Subjects"
TARGET_DATA_SUMMARY = ROOT / "2_Data_Summary"


# =============================================================================
# FILE TYPE CLASSIFICATION
# =============================================================================

def classify_file(filepath: Path) -> str:
    """
    Classify a file by its type and determine target subdirectory.

    Returns:
        Subdirectory name (e.g., "0_Raw", "3_Detected", "4_Quantified")
    """
    name = filepath.name.lower()

    # Detection outputs
    if "_composite.tiff" in name or "_composite.tif" in name:
        return "3_Detected"
    if "_overlay.png" in name or "_annotated.png" in name:
        return "3_Detected"

    # Quantification outputs
    if "_measurements.csv" in name:
        return "4_Quantified"
    if ".coloc_result.png" in name:
        return "4_Quantified"
    if "dual_borders" in name and ".png" in name:
        return "4_Quantified"
    if "_histogram.png" in name or "_scatter.png" in name:
        return "4_Quantified"
    if "_background_mask.png" in name:
        return "4_Quantified"

    # Raw ND2 files
    if filepath.suffix.lower() == ".nd2":
        return "0_Raw"

    # Unknown - put in quantified as catch-all
    return "4_Quantified"


def extract_sample_name_from_filename(filepath: Path) -> str:
    """
    Extract the base sample name from a filename by removing suffixes.

    Examples:
        E02_01_S13_DCN.nd2 -> E02_01_S13_DCN
        E02_01_S13_DCN_composite.tiff -> E02_01_S13_DCN
        E02_01_S13_DCN.coloc_result.png -> E02_01_S13_DCN
    """
    name = filepath.stem

    # Remove known suffixes
    suffixes_to_remove = [
        "_composite", "_overlay", "_annotated", "_histogram", "_scatter",
        "_background_mask", "_measurements", ".coloc_result"
    ]

    for suffix in suffixes_to_remove:
        if suffix in name:
            name = name.split(suffix)[0]
            break

    return name


def get_subject_id_from_sample(sample_name: str) -> str:
    """
    Extract subject ID from sample name.

    Args:
        sample_name: Sample name like "E02_01_S13_DCN" or "ENCR_02_01"

    Returns:
        Subject ID like "ENCR_02_01"
    """
    parsed = parse_sample_name(sample_name)

    project = parsed.get("project", "")
    cohort = parsed.get("cohort", "")
    subject = parsed.get("subject", "")

    if project and cohort and subject:
        return f"{project}_{cohort}_{subject}"

    # Fallback: if already looks like subject format, return as-is
    if re.match(r'^[A-Z]+_\d{2}_\d{2}$', sample_name):
        return sample_name

    # Last resort: return the first part of underscore-split
    parts = sample_name.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"

    return sample_name


def get_subject_dir_type(source_dir: Path) -> Optional[str]:
    """
    Determine what kind of subject directory this is.

    Returns:
        "ATLAS" for atlas reference slices
        "HD" for HD region directories
        "STANDARD" for standard subject directories
        None if not a subject directory
    """
    name = source_dir.name

    if name.endswith("_ATLAS"):
        return "ATLAS"
    if "_HD_Regions" in name:
        return "HD"
    if re.match(r'^[A-Z]+_\d{2}_\d{2}$', name):
        return "STANDARD"

    return None


# =============================================================================
# MIGRATION PLANNING
# =============================================================================

class MigrationPlan:
    """Tracks all planned moves."""

    def __init__(self):
        self.moves: List[Tuple[Path, Path]] = []
        self.skips: List[Tuple[Path, str]] = []
        self.errors: List[Tuple[Path, str]] = []

    def add_move(self, src: Path, dst: Path):
        """Add a file move operation."""
        self.moves.append((src, dst))

    def add_skip(self, src: Path, reason: str):
        """Record a skipped file."""
        self.skips.append((src, reason))

    def add_error(self, src: Path, reason: str):
        """Record an error."""
        self.errors.append((src, reason))

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            "=" * 80,
            "MIGRATION SUMMARY",
            "=" * 80,
            f"Total moves planned: {len(self.moves)}",
            f"Total skips: {len(self.skips)}",
            f"Total errors: {len(self.errors)}",
            "",
        ]

        if self.errors:
            lines.append("ERRORS:")
            for src, reason in self.errors[:10]:
                lines.append(f"  {src.name}: {reason}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")
            lines.append("")

        if self.skips:
            lines.append("SKIPS:")
            for src, reason in self.skips[:10]:
                lines.append(f"  {src.name}: {reason}")
            if len(self.skips) > 10:
                lines.append(f"  ... and {len(self.skips) - 10} more")
            lines.append("")

        # Group moves by destination directory
        dst_groups: Dict[str, int] = {}
        for _, dst in self.moves:
            key = str(dst.parent.relative_to(ROOT))
            dst_groups[key] = dst_groups.get(key, 0) + 1

        lines.append("MOVES BY DESTINATION:")
        for dst_dir, count in sorted(dst_groups.items(), key=lambda x: x[0]):
            lines.append(f"  {dst_dir}: {count} files")

        return "\n".join(lines)


def plan_subject_directory_migration(
    source_dir: Path,
    plan: MigrationPlan,
    project: str = "ENCR"
) -> None:
    """
    Plan migration for a single subject directory.

    Args:
        source_dir: Source directory (e.g., ENCR_01_01 or ENCR_02_01_HD_Regions)
        plan: MigrationPlan to add operations to
        project: Project code (default "ENCR")
    """
    dir_type = get_subject_dir_type(source_dir)

    if not dir_type:
        plan.add_skip(source_dir, "Not a recognized subject directory")
        return

    # Determine subject ID
    if dir_type == "ATLAS":
        # ENCR_02_01_ATLAS -> ENCR_02_01
        subject_id = source_dir.name.replace("_ATLAS", "")
    elif dir_type == "HD":
        # ENCR_02_01_HD_Regions -> ENCR_02_01
        subject_id = source_dir.name.replace("_HD_Regions", "")
    else:
        subject_id = source_dir.name

    # Create target subject directory
    target_subject = TARGET_SUBJECTS / project / subject_id

    # Process all files in source directory
    for item in source_dir.rglob("*"):
        if not item.is_file():
            continue

        # Determine which subdirectory this file belongs to
        relative_to_source = item.relative_to(source_dir)

        # Special handling for Corrected/ subfolder
        is_corrected = "Corrected" in item.parts

        # Determine target subdirectory
        if dir_type == "ATLAS":
            target_subdir = "0_Raw_Atlas"
        elif dir_type == "HD":
            if item.suffix.lower() == ".nd2":
                target_subdir = "0_Raw_HD"
            else:
                file_type = classify_file(item)
                if is_corrected and file_type == "4_Quantified":
                    target_subdir = "4_Quantified_Corrected"
                else:
                    target_subdir = file_type
        else:  # STANDARD
            target_subdir = classify_file(item)

        target_path = target_subject / target_subdir / item.name

        # Check if destination already exists
        if target_path.exists():
            plan.add_skip(item, f"Destination already exists: {target_path}")
        else:
            plan.add_move(item, target_path)


def plan_batch_results_migration(plan: MigrationPlan, project: str = "ENCR") -> None:
    """
    Plan migration for batch_results directory.

    Args:
        plan: MigrationPlan to add operations to
        project: Project code (default "ENCR")
    """
    if not SOURCE_BATCH_RESULTS.exists():
        plan.add_skip(SOURCE_BATCH_RESULTS, "Batch results directory not found")
        return

    # Move batch_summary.csv to data summary
    batch_summary = SOURCE_BATCH_RESULTS / "batch_summary.csv"
    if batch_summary.exists():
        target = TARGET_DATA_SUMMARY / "batch_summary.csv"
        if target.exists():
            plan.add_skip(batch_summary, "Destination already exists")
        else:
            plan.add_move(batch_summary, target)

    # Process each subdirectory (per-sample results)
    for sample_dir in SOURCE_BATCH_RESULTS.iterdir():
        if not sample_dir.is_dir():
            continue

        # Extract subject ID from directory name
        sample_name = sample_dir.name

        try:
            subject_id = get_subject_id_from_sample(sample_name)
        except Exception as e:
            plan.add_error(sample_dir, f"Could not parse subject ID: {e}")
            continue

        target_subject = TARGET_SUBJECTS / project / subject_id

        # Process all files in this sample directory
        for item in sample_dir.rglob("*"):
            if not item.is_file():
                continue

            file_type = classify_file(item)
            target_path = target_subject / file_type / item.name

            if target_path.exists():
                plan.add_skip(item, f"Destination already exists: {target_path}")
            else:
                plan.add_move(item, target_path)


def plan_tracker_migration(plan: MigrationPlan) -> None:
    """
    Plan migration for tracker CSV.

    Args:
        plan: MigrationPlan to add operations to
    """
    source = SOURCE_SLICEATLAS / "calibration_runs.csv"
    target = TARGET_DATA_SUMMARY / "calibration_runs.csv"

    if not source.exists():
        plan.add_skip(source, "Source file not found")
        return

    if target.exists():
        plan.add_skip(source, "Destination already exists")
    else:
        plan.add_move(source, target)


def build_migration_plan() -> MigrationPlan:
    """
    Build complete migration plan.

    Returns:
        MigrationPlan with all planned operations
    """
    plan = MigrationPlan()

    # Migrate subject directories
    if SOURCE_ENCR.exists():
        for item in SOURCE_ENCR.iterdir():
            if item.is_dir() and item.name != "batch_results":
                plan_subject_directory_migration(item, plan, project="ENCR")

    # Migrate batch results
    plan_batch_results_migration(plan, project="ENCR")

    # Migrate tracker CSV
    plan_tracker_migration(plan)

    return plan


# =============================================================================
# EXECUTION
# =============================================================================

def execute_migration(plan: MigrationPlan, log_path: Path) -> None:
    """
    Execute the migration plan.

    Args:
        plan: MigrationPlan to execute
        log_path: Path to write log file
    """
    print("\n" + "=" * 80)
    print("EXECUTING MIGRATION")
    print("=" * 80)

    log_lines = [
        f"Migration executed: {datetime.now().isoformat()}",
        "",
        "MOVES:",
    ]

    success_count = 0
    error_count = 0

    for src, dst in plan.moves:
        try:
            # Create parent directory
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Move file
            shutil.move(str(src), str(dst))

            success_count += 1
            log_lines.append(f"OK: {src} -> {dst}")

            if success_count % 10 == 0:
                print(f"  Moved {success_count}/{len(plan.moves)} files...", end="\r")

        except Exception as e:
            error_count += 1
            error_msg = f"ERROR: {src} -> {dst}: {e}"
            log_lines.append(error_msg)
            print(f"\n{error_msg}")

    print(f"\n  Moved {success_count}/{len(plan.moves)} files successfully")
    print(f"  Errors: {error_count}")

    # Write log
    log_lines.append("")
    log_lines.append(plan.summary())

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines))

    print(f"\nMigration log written to: {log_path}")

    # Check for empty directories
    print("\n" + "=" * 80)
    print("CHECKING FOR EMPTY SOURCE DIRECTORIES")
    print("=" * 80)

    empty_dirs = []
    for root_dir in [SOURCE_ENCR, SOURCE_SLICEATLAS]:
        if not root_dir.exists():
            continue
        for dirpath in root_dir.rglob("*"):
            if dirpath.is_dir():
                try:
                    if not any(dirpath.iterdir()):
                        empty_dirs.append(dirpath)
                except PermissionError:
                    pass

    if empty_dirs:
        print(f"Found {len(empty_dirs)} empty directories:")
        for d in empty_dirs[:20]:
            print(f"  {d.relative_to(ROOT)}")
        if len(empty_dirs) > 20:
            print(f"  ... and {len(empty_dirs) - 20} more")
        print("\nThese directories were not automatically deleted.")
        print("Review and delete manually if appropriate.")
    else:
        print("No empty directories found.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate 2D_Slices directory to new structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview migration (default)
  python util_migrate_2d_structure.py

  # Preview with detailed output
  python util_migrate_2d_structure.py --dry-run

  # Execute migration (CAUTION: moves files)
  python util_migrate_2d_structure.py --execute

CRITICAL: This script uses shutil.move() to avoid doubling disk usage.
Backup your data before running --execute.
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview moves without executing (default)"
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform migration"
    )

    args = parser.parse_args()

    # Override dry-run if execute is specified
    if args.execute:
        args.dry_run = False

    # Verify root exists
    if not ROOT.exists():
        print(f"ERROR: Root directory not found: {ROOT}")
        return 1

    print("Building migration plan...")
    plan = build_migration_plan()

    print(plan.summary())

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files were moved")
        print("=" * 80)
        print("\nSample moves (first 20):")
        for src, dst in plan.moves[:20]:
            src_rel = src.relative_to(ROOT)
            dst_rel = dst.relative_to(ROOT)
            print(f"  {src_rel}")
            print(f"  -> {dst_rel}")
            print()

        if len(plan.moves) > 20:
            print(f"  ... and {len(plan.moves) - 20} more")

        print("\nRun with --execute to perform migration.")
    else:
        # Confirm execution
        print("\n" + "!" * 80)
        print("WARNING: About to move files")
        print("!" * 80)
        print(f"This will move {len(plan.moves)} files.")
        print("Files will be MOVED (not copied) to save disk space.")
        print("\nPress Ctrl+C now to cancel, or Enter to continue...")

        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return 0

        # Execute migration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = TARGET_DATA_SUMMARY / f"migration_log_{timestamp}.txt"

        execute_migration(plan, log_path)

    return 0


if __name__ == "__main__":
    exit(main())
