#!/usr/bin/env python3
"""
batch_process.py - Automated batch processing with QC triage.

Process thousands of brains with automatic quality control gates.
Brains that fail QC are flagged for manual review without stopping the batch.

Usage:
    # Process all .ims files in a folder
    python batch_process.py --input /path/to/ims/files

    # Process specific files
    python batch_process.py --input /path/to/brain1.ims /path/to/brain2.ims

    # Resume failed/pending brains
    python batch_process.py --resume

    # Show status of all brains
    python batch_process.py --status

QC Triage Categories:
    PASS   - Automatic QC passed, continue processing
    REVIEW - Completed but needs human verification
    FAIL   - Failed QC, needs manual intervention
    SKIP   - User marked to skip

The pipeline_status.csv file tracks all brains and their QC status.
"""

import argparse
import csv
import json
import os
import sys
import subprocess
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mousebrain.config import (
    BRAINS_ROOT, DATA_SUMMARY_DIR, find_script,
    parse_brain_name
)
from mousebrain.file_lock import exclusive_file_lock

SCRIPT_VERSION = "1.0.0"

# Pipeline status tracking file
PIPELINE_STATUS_CSV = DATA_SUMMARY_DIR / "pipeline_status.csv"

# How long the swap of the new status file into place is retried. On Windows a
# file that another program has open (a person looking at it, a second batch
# reading it) cannot be replaced until that program closes it; readers hold it
# for a moment only.
REPLACE_RETRY_SECONDS = 30

# QC thresholds
QC_THRESHOLDS = {
    'min_extracted_size_mb': 100,      # Minimum extracted data size
    'max_extracted_size_gb': 500,      # Maximum (sanity check)
    'min_crop_ratio': 0.3,             # Crop must keep at least 30% of volume
    'max_crop_ratio': 0.95,            # Crop should remove at least 5%
    'min_registration_score': 0.5,     # Registration quality threshold
    'min_cells_detected': 100,         # Minimum cells to find
    'max_cells_detected': 500000,      # Maximum (sanity check)
    'min_classification_keep': 0.1,    # Keep at least 10% of candidates
    'max_classification_keep': 0.95,   # Suspicious if keeping >95%
    'min_regions_with_cells': 10,      # Minimum regions with cells
}

# =============================================================================
# QC STATUS TRACKING
# =============================================================================

class QCStatus:
    """Track QC status for a brain."""
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    PENDING = "PENDING"
    SKIP = "SKIP"


def load_pipeline_status() -> Dict[str, dict]:
    """Load pipeline status from CSV."""
    status = {}
    if PIPELINE_STATUS_CSV.exists():
        with open(PIPELINE_STATUS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                status[row['brain_name']] = row
    return status


def save_pipeline_status(status: Dict[str, dict]):
    """Save pipeline status to CSV: a temporary file beside it, swapped in.

    Call it holding the status file's lock (update_brain_status does). The
    file was written in place, so a crash or a full disk part-way left it cut
    short and every brain after the cut lost its row.
    """
    if not status:
        return

    # Get all columns from all rows
    columns = set()
    for row in status.values():
        columns.update(row.keys())

    # Ensure key columns come first
    key_cols = ['brain_name', 'ims_path', 'status', 'current_step', 'last_updated',
                'qc_extract', 'qc_crop', 'qc_register', 'qc_detect', 'qc_classify', 'qc_count']
    ordered_cols = [c for c in key_cols if c in columns]
    ordered_cols.extend(sorted(c for c in columns if c not in key_cols))

    PIPELINE_STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = PIPELINE_STATUS_CSV.with_name("%s.tmp-%d-%s" % (
        PIPELINE_STATUS_CSV.name, os.getpid(), uuid.uuid4().hex[:6]))
    try:
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_cols)
            writer.writeheader()
            for row in status.values():
                writer.writerow({k: row.get(k, '') for k in ordered_cols})
        deadline = time.monotonic() + REPLACE_RETRY_SECONDS
        while True:
            try:
                os.replace(str(tmp), str(PIPELINE_STATUS_CSV))
                break
            except PermissionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
    finally:
        if tmp.exists():
            tmp.unlink()


def update_brain_status(brain_name: str, updates: dict):
    """Update status for a single brain.

    Read, change and write back under the status file's lock
    (mousebrain.file_lock). Without it, two batches -- or a batch and a second
    window -- each read the file, changed their own brain's row and wrote it
    back, and the second write silently dropped the first one's change. It is
    written before every step, so the window was wide.
    """
    PIPELINE_STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(PIPELINE_STATUS_CSV, "pipeline_status"):
        status = load_pipeline_status()
        if brain_name not in status:
            status[brain_name] = {'brain_name': brain_name}
        status[brain_name].update(updates)
        status[brain_name]['last_updated'] = datetime.now().isoformat()
        save_pipeline_status(status)


# =============================================================================
# QC CHECKS
# =============================================================================

def qc_check_extraction(brain_path: Path) -> Tuple[str, str]:
    """
    QC check after extraction.

    Returns (status, message)
    """
    extracted_dir = brain_path / "1_Extracted_Full"

    if not extracted_dir.exists():
        return QCStatus.FAIL, "Extraction folder not created"

    # Check metadata exists
    metadata_file = extracted_dir / "metadata.json"
    if not metadata_file.exists():
        return QCStatus.FAIL, "No metadata.json found"

    # Check for tiff files. Extraction writes one folder per channel
    # (ch0/Z0000.tif ...), never TIFFs at the top -- looking only at the top
    # failed every brain.
    tiff_files = [f for f in extracted_dir.glob("ch*/*.tif*") if f.is_file()]
    if not tiff_files:
        return QCStatus.FAIL, "No TIFF files extracted"

    # Check total size
    total_size = sum(f.stat().st_size for f in tiff_files) / (1024**2)  # MB

    if total_size < QC_THRESHOLDS['min_extracted_size_mb']:
        return QCStatus.FAIL, f"Extracted data too small ({total_size:.0f} MB)"

    if total_size > QC_THRESHOLDS['max_extracted_size_gb'] * 1024:
        return QCStatus.REVIEW, f"Extracted data unusually large ({total_size/1024:.1f} GB)"

    return QCStatus.PASS, f"OK ({total_size:.0f} MB, {len(tiff_files)} files)"


def qc_check_crop(brain_path: Path) -> Tuple[str, str]:
    """
    QC check after cropping.

    Returns (status, message)
    """
    # A crop is a ch0 folder of TIFFs plus metadata.json, by hand
    # (_Manual) or automatic -- what every crop writer produces. This used to
    # look for downsampled.zarr / downsampled.tiff, which no crop writer makes.
    crop_dir = None
    for name in ("2_Cropped_For_Registration_Manual", "2_Cropped_For_Registration"):
        d = brain_path / name
        if (d / "metadata.json").exists() and any((d / "ch0").glob("*.tif*")):
            crop_dir = d
            break
    if crop_dir is None:
        return QCStatus.FAIL, "No cropped output found (crop the brain by hand, then run it again)"

    # Check metadata for crop info
    metadata_file = crop_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                meta = json.load(f)

            # Check crop ratio if available
            if 'crop_ratio' in meta:
                ratio = meta['crop_ratio']
                if ratio < QC_THRESHOLDS['min_crop_ratio']:
                    return QCStatus.REVIEW, f"Crop removed too much ({ratio:.0%} kept)"
                if ratio > QC_THRESHOLDS['max_crop_ratio']:
                    return QCStatus.REVIEW, f"Crop removed very little ({ratio:.0%} kept)"
        except:
            pass

    return QCStatus.PASS, "OK"


def qc_check_registration(brain_path: Path) -> Tuple[str, str]:
    """
    QC check after registration.

    Returns (status, message)
    """
    reg_dir = brain_path / "3_Registered_Atlas"

    if not reg_dir.exists():
        return QCStatus.FAIL, "Registration folder not created"

    brainreg_json = reg_dir / "brainreg.json"
    if not brainreg_json.exists():
        return QCStatus.FAIL, "brainreg.json not found - registration failed"

    # Check for deformation fields (critical for cell mapping)
    deformation_field = reg_dir / "deformation_field_0.tiff"
    if not deformation_field.exists():
        return QCStatus.FAIL, "Deformation fields missing"

    # Check for boundaries (indicates successful segmentation)
    boundaries = reg_dir / "boundaries.tiff"
    if not boundaries.exists():
        return QCStatus.REVIEW, "boundaries.tiff missing - may affect visualization"

    # Registration always needs human review for approval -- of THIS
    # registration (mousebrain.registration_approval).
    from mousebrain.registration_approval import APPROVED, approval_state
    state, text = approval_state(reg_dir)
    if state != APPROVED:
        return QCStatus.REVIEW, f"Needs human approval (check QC images): {text}"

    return QCStatus.PASS, "OK (approved)"


def qc_check_detection(brain_path: Path) -> Tuple[str, str]:
    """
    QC check after cell detection.

    Returns (status, message)
    """
    detect_dir = brain_path / "4_Cell_Candidates"

    if not detect_dir.exists():
        return QCStatus.FAIL, "Detection folder not created"

    # The candidates step 5 will classify (pipeline_status.find_candidates),
    # the one rule every reader uses. This took the newest .xml at the top of
    # the folder, which need not be the file step 5 reads -- so the check could
    # pass or fail a brain on a different file from the one it then classified.
    from mousebrain.pipeline_status import find_candidates
    xml_file = find_candidates(brain_path)
    if xml_file is None:
        return QCStatus.FAIL, "No detection XML found"
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        cells = tree.findall('.//Marker') or tree.findall('.//marker')
        n_cells = len(cells)
    except:
        n_cells = 0

    if n_cells < QC_THRESHOLDS['min_cells_detected']:
        return QCStatus.FAIL, f"Too few cells detected ({n_cells})"

    if n_cells > QC_THRESHOLDS['max_cells_detected']:
        return QCStatus.REVIEW, f"Unusually many cells ({n_cells}) - check parameters"

    return QCStatus.PASS, f"OK ({n_cells} candidates)"


def qc_check_classification(brain_path: Path) -> Tuple[str, str]:
    """
    QC check after classification.

    Returns (status, message)
    """
    class_dir = brain_path / "5_Classified_Cells"

    if not class_dir.exists():
        return QCStatus.FAIL, "Classification folder not created"

    cells_xml = class_dir / "cells.xml"
    if not cells_xml.exists():
        return QCStatus.FAIL, "cells.xml not found"

    # Count classified cells
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(cells_xml)
        cells = tree.findall('.//Marker') or tree.findall('.//marker')
        n_cells = len(cells)
    except:
        n_cells = 0

    if n_cells < QC_THRESHOLDS['min_cells_detected']:
        return QCStatus.FAIL, f"Too few cells after classification ({n_cells})"

    return QCStatus.PASS, f"OK ({n_cells} cells)"


def qc_check_counting(brain_path: Path) -> Tuple[str, str]:
    """
    QC check after region counting.

    Returns (status, message)
    """
    count_dir = brain_path / "6_Region_Analysis"

    if not count_dir.exists():
        return QCStatus.FAIL, "Counting folder not created"

    # Check for output CSVs
    csv_files = list(count_dir.glob("*.csv"))
    if not csv_files:
        return QCStatus.FAIL, "No count CSVs found"

    # Check comparison report exists
    report = count_dir / "comparison_report.txt"
    if not report.exists():
        return QCStatus.REVIEW, "No comparison report generated"

    return QCStatus.PASS, f"OK ({len(csv_files)} files)"


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_step(script_name: str, args: List[str], timeout: int = 7200) -> Tuple[bool, str]:
    """
    Run a pipeline step script.

    Returns (success, message)
    """
    script_path = find_script(script_name)
    cmd = [sys.executable, str(script_path)] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script_path.parent),
        )

        if result.returncode == 0:
            return True, "OK"
        else:
            # Get last few lines of stderr
            error_lines = result.stderr.strip().split('\n')[-5:]
            return False, '\n'.join(error_lines)

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def setup_brain_from_ims(ims_path: Path) -> Tuple[Optional[Path], str]:
    """
    Create brain folder structure from .ims file.

    Returns (brain_path, message)
    """
    if not ims_path.exists():
        return None, f"IMS file not found: {ims_path}"

    # Parse brain name from filename
    # Expected format: 001_CNT_01_01_1p625x_z4.ims
    stem = ims_path.stem
    parsed = parse_brain_name(stem)

    if not parsed.get('brain_id'):
        return None, f"Could not parse brain name from: {stem}"

    # Create folder structure
    mouse_folder = BRAINS_ROOT / f"{parsed['brain_id']}_{parsed['subject_full']}"
    brain_folder = mouse_folder / stem
    raw_folder = brain_folder / "0_Raw_IMS"

    # Create folders
    raw_folder.mkdir(parents=True, exist_ok=True)

    # Copy or link IMS file
    target_ims = raw_folder / ims_path.name
    if not target_ims.exists():
        import shutil
        shutil.copy2(ims_path, target_ims)

    return brain_folder, f"Created {brain_folder.name}"


def mark_step_started(brain_name: str, brain_path: Path, step: str) -> None:
    """Save, BEFORE a step runs, that this brain is PENDING at that step.

    WHY: the status file used to be written only when a brain finished, failed
    or stopped for review. A batch killed mid-step (closed window, reboot,
    task ended) left its brain with no row -- or with a row from an earlier
    run -- and --resume, which looks for PENDING rows, could never find it.
    """
    update_brain_status(brain_name, {
        'status': QCStatus.PENDING,
        'current_step': step,
        'brain_path': str(brain_path),
    })


def process_brain(brain_path: Path, brain_name: str) -> dict:
    """
    Process a single brain through the pipeline with QC gates.

    Returns dict with results for each step.
    """
    results = {
        'brain_name': brain_name,
        'brain_path': str(brain_path),
        'started': datetime.now().isoformat(),
        'status': QCStatus.PENDING,
    }

    print(f"\n{'='*60}")
    print(f"Processing: {brain_name}")
    print(f"{'='*60}")

    # A new attempt: clear the previous attempt's verdict, which a merge into
    # the row would otherwise carry into this one (a PASS row still saying
    # "needs registration approval").
    update_brain_status(brain_name, {
        'failed_step': '', 'error': '', 'needs_registration_approval': '',
        'needs_manual_crop': '', 'completed': '', 'started': results['started'],
    })

    # Step 1: Extract
    print("\n[1/6] Extracting from IMS...")
    # --batch: no question to answer; --brain: this brain only. (This passed a
    # --yes that step 2 does not accept, so argparse stopped it and every
    # brain failed here.)
    # The brain FOLDER name: batch brain names are "<mouse>/<brain>", which
    # steps 2 and 3 do not match (steps 4-6 accept either form).
    mark_step_started(brain_name, brain_path, 'extract')
    success, msg = run_step('2_extract_and_analyze.py', ['--batch', '--brain', brain_path.name])
    if not success:
        results['qc_extract'] = f"FAIL: {msg}"
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'extract'
        update_brain_status(brain_name, results)
        return results

    qc_status, qc_msg = qc_check_extraction(brain_path)
    results['qc_extract'] = f"{qc_status}: {qc_msg}"
    if qc_status == QCStatus.FAIL:
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'extract'
        update_brain_status(brain_name, results)
        return results
    print(f"    QC: {qc_msg}")

    # Step 2: Crop (auto-crop, no manual)
    print("\n[2/6] Auto-cropping...")
    # Auto-crop is part of extract, check if manual crop is needed
    manual_crop = brain_path / "2_Cropped_For_Registration_Manual"
    if manual_crop.exists():
        print("    Using manual crop")

    mark_step_started(brain_name, brain_path, 'crop')
    qc_status, qc_msg = qc_check_crop(brain_path)
    results['qc_crop'] = f"{qc_status}: {qc_msg}"
    if qc_status == QCStatus.FAIL:
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'crop'
        results['needs_manual_crop'] = True
        update_brain_status(brain_name, results)
        return results
    print(f"    QC: {qc_msg}")

    # Step 3: Register
    print("\n[3/6] Registering to atlas...")
    # Same for step 3: --batch replaces the --yes it never accepted.
    mark_step_started(brain_name, brain_path, 'register')
    success, msg = run_step('3_register_to_atlas.py', ['--batch', '--brain', brain_path.name])
    if not success:
        results['qc_register'] = f"FAIL: {msg}"
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'register'
        update_brain_status(brain_name, results)
        return results

    qc_status, qc_msg = qc_check_registration(brain_path)
    results['qc_register'] = f"{qc_status}: {qc_msg}"
    print(f"    QC: {qc_msg}")

    # A registration that did not produce its files stops HERE, as a
    # registration failure. This case used to fall through to detection,
    # which then stopped on the missing approval, so the row said the brain
    # failed at 'detect' -- three steps from the cause. (Step 3 ending
    # without an error does not settle it: it also ends cleanly when it finds
    # nothing to register for the brain, e.g. a brainreg.json is there but the
    # deformation field is not.)
    if qc_status == QCStatus.FAIL:
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'register'
        update_brain_status(brain_name, results)
        return results

    # Registration needs a person's approval. A batch never approves: the
    # approval exists so that a person has looked at the registration before
    # hours of detection run on it. (--auto-approve used to try, with an
    # argument the approval script does not accept, so it never worked.)
    if qc_status == QCStatus.REVIEW:
        results['status'] = QCStatus.REVIEW
        results['needs_registration_approval'] = True
        update_brain_status(brain_name, results)
        print("    >>> Needs registration approval - stopping here")
        return results

    # Step 4: Detect cells
    print("\n[4/6] Detecting cells...")
    # Step 4 uses the settled method for the brain's imaging setup (the tracker's
    # Paradigm Best run) and fails, with the reason, when there is none.
    mark_step_started(brain_name, brain_path, 'detect')
    success, msg = run_step('4_detect_cells.py', ['--brain', brain_name])
    if not success:
        results['qc_detect'] = f"FAIL: {msg}"
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'detect'
        update_brain_status(brain_name, results)
        return results

    qc_status, qc_msg = qc_check_detection(brain_path)
    results['qc_detect'] = f"{qc_status}: {qc_msg}"
    if qc_status == QCStatus.FAIL:
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'detect'
        update_brain_status(brain_name, results)
        return results
    print(f"    QC: {qc_msg}")

    # Step 5: Classify cells
    print("\n[5/6] Classifying cells...")
    mark_step_started(brain_name, brain_path, 'classify')
    success, msg = run_step('5_classify_cells.py', ['--brain', brain_name])
    if not success:
        results['qc_classify'] = f"FAIL: {msg}"
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'classify'
        update_brain_status(brain_name, results)
        return results

    qc_status, qc_msg = qc_check_classification(brain_path)
    results['qc_classify'] = f"{qc_status}: {qc_msg}"
    if qc_status == QCStatus.FAIL:
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'classify'
        update_brain_status(brain_name, results)
        return results
    print(f"    QC: {qc_msg}")

    # Step 6: Count regions
    print("\n[6/6] Counting cells by region...")
    mark_step_started(brain_name, brain_path, 'count')
    success, msg = run_step('6_count_regions.py', ['--brain', brain_name])
    if not success:
        results['qc_count'] = f"FAIL: {msg}"
        results['status'] = QCStatus.FAIL
        results['failed_step'] = 'count'
        update_brain_status(brain_name, results)
        return results

    qc_status, qc_msg = qc_check_counting(brain_path)
    results['qc_count'] = f"{qc_status}: {qc_msg}"
    print(f"    QC: {qc_msg}")

    # Success!
    results['status'] = QCStatus.PASS
    results['completed'] = datetime.now().isoformat()
    update_brain_status(brain_name, results)

    print(f"\n    >>> COMPLETE!")
    return results


def show_status():
    """Show status of all tracked brains."""
    status = load_pipeline_status()

    if not status:
        print("\nNo brains tracked yet.")
        print(f"Status file: {PIPELINE_STATUS_CSV}")
        return

    print("\n" + "=" * 80)
    print("PIPELINE STATUS")
    print("=" * 80)

    # Group by status
    by_status = {}
    for brain_name, data in status.items():
        s = data.get('status', 'UNKNOWN')
        if s not in by_status:
            by_status[s] = []
        by_status[s].append((brain_name, data))

    # Show each group
    status_order = [QCStatus.PASS, QCStatus.REVIEW, QCStatus.FAIL, QCStatus.PENDING, QCStatus.SKIP]
    status_colors = {
        QCStatus.PASS: "COMPLETE",
        QCStatus.REVIEW: "NEEDS REVIEW",
        QCStatus.FAIL: "FAILED",
        QCStatus.PENDING: "PENDING",
        QCStatus.SKIP: "SKIPPED",
    }

    for s in status_order:
        if s not in by_status:
            continue

        print(f"\n[{status_colors.get(s, s)}] ({len(by_status[s])} brains)")
        print("-" * 60)

        for brain_name, data in by_status[s]:
            failed_step = data.get('failed_step', '')
            reason = data.get(f'qc_{failed_step}', '') if failed_step else ''

            if failed_step:
                print(f"  {brain_name}")
                print(f"      Failed at: {failed_step}")
                if reason:
                    print(f"      Reason: {reason[:60]}")
            elif data.get('needs_registration_approval'):
                print(f"  {brain_name}")
                print(f"      Needs registration approval")
            elif data.get('needs_manual_crop'):
                print(f"  {brain_name}")
                print(f"      Needs manual crop")
            else:
                print(f"  {brain_name}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total tracked: {len(status)}")
    for s in status_order:
        if s in by_status:
            print(f"  {status_colors.get(s, s)}: {len(by_status[s])}")

    print(f"\nStatus file: {PIPELINE_STATUS_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process brains with QC triage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', nargs='+', help='IMS files or folders to process')
    parser.add_argument('--resume', action='store_true', help='Resume pending/failed brains')
    parser.add_argument('--status', action='store_true', help='Show status of all brains')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry previously failed brains')

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.resume:
        # Find pending brains
        status = load_pipeline_status()
        to_process = []

        for brain_name, data in status.items():
            s = data.get('status')
            brain_path = Path(data.get('brain_path', ''))
            if s == QCStatus.PENDING:
                # Interrupted part-way: every step is saved as PENDING before
                # it starts (mark_step_started).
                print(f"  {brain_name}: interrupted at '{data.get('current_step', '?')}' -- resuming")
                to_process.append((brain_name, brain_path))
            elif s == QCStatus.REVIEW and data.get('needs_registration_approval'):
                # Resumed once a person has approved THIS registration.
                from mousebrain.registration_approval import is_approved
                if data.get('brain_path') and is_approved(brain_path / "3_Registered_Atlas"):
                    print(f"  {brain_name}: registration approved since -- resuming")
                    to_process.append((brain_name, brain_path))
                else:
                    print(f"  {brain_name}: Needs registration approval (use napari)")
            elif args.retry_failed and s == QCStatus.FAIL:
                to_process.append((brain_name, Path(data.get('brain_path', ''))))

        if not to_process:
            print("\nNo pending brains to resume.")
            show_status()
            return

        print(f"\nResuming {len(to_process)} brain(s)...")

    elif args.input:
        # Process input files/folders
        ims_files = []
        for path_str in args.input:
            path = Path(path_str)
            if path.is_dir():
                ims_files.extend(path.glob("*.ims"))
            elif path.suffix.lower() == '.ims':
                ims_files.append(path)

        if not ims_files:
            print("No .ims files found!")
            return

        print(f"\nFound {len(ims_files)} IMS files to process")

        # Setup each brain
        to_process = []
        for ims_file in ims_files:
            brain_path, msg = setup_brain_from_ims(ims_file)
            if brain_path:
                brain_name = f"{brain_path.parent.name}/{brain_path.name}"
                to_process.append((brain_name, brain_path))
                print(f"  {msg}")
            else:
                print(f"  SKIP: {msg}")

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        show_status()
        return

    # Process all brains
    results = []
    for brain_name, brain_path in to_process:
        try:
            result = process_brain(brain_path, brain_name)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            # Recording the failure can itself fail (the status file locked
            # past the wait, a full disk). Say so and go on to the next brain
            # rather than let it end the whole batch.
            try:
                update_brain_status(brain_name, {
                    'status': QCStatus.FAIL,
                    'error': str(e),
                })
            except Exception as e2:
                print(f"  [!] Could not record this failure in the status file: {e2}")

    # Final summary
    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)

    passed = sum(1 for r in results if r.get('status') == QCStatus.PASS)
    review = sum(1 for r in results if r.get('status') == QCStatus.REVIEW)
    failed = sum(1 for r in results if r.get('status') == QCStatus.FAIL)

    print(f"  Processed: {len(results)}")
    print(f"  PASSED: {passed}")
    print(f"  NEEDS REVIEW: {review}")
    print(f"  FAILED: {failed}")

    if review > 0:
        print("\nBrains needing review:")
        for r in results:
            if r.get('status') == QCStatus.REVIEW:
                print(f"  - {r['brain_name']}")

    if failed > 0:
        print("\nFailed brains:")
        for r in results:
            if r.get('status') == QCStatus.FAIL:
                print(f"  - {r['brain_name']}: {r.get('failed_step', 'unknown step')}")

    print(f"\nFull status: python batch_process.py --status")


if __name__ == '__main__':
    main()
