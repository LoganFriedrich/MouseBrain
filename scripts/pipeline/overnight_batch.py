#!/usr/bin/env python3
"""
overnight_batch.py - Overnight Batch Processing for SCI-Connectome Pipeline

Processes all cropped brains through the full pipeline (registration -> detection ->
classification -> counting) using brain 349's calibrated parameters, then generates
comparison reports and representative images.

Usage:
    python overnight_batch.py                  # Process all pending brains
    python overnight_batch.py --dry-run        # Show what would run
    python overnight_batch.py --brain 357      # Process specific brain only
    python overnight_batch.py --skip-complete  # Skip already completed steps

The script will:
1. Find all brains with completed cropping (2_Cropped_For_Registration/)
2. Use brain 349's calibrated detection parameters
3. Run each pipeline step with automatic retry on failure
4. Generate comparison report when complete
5. Generate representative images for each brain

Results saved to:
- 2_Data_Summary/reports/overnight_comparison_YYYYMMDD.md
- 2_Data_Summary/reports/images/{brain_name}/
- overnight_batch.log (execution log)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('overnight_batch.log', mode='a')
    ]
)
log = logging.getLogger(__name__)

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mousebrain.config import (
    BRAINS_ROOT, DATA_SUMMARY_DIR, SCRIPTS_DIR, MODELS_DIR,
    CALIBRATION_RUNS_CSV, REGION_COUNTS_CSV, parse_brain_name
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.0"

# Brain 349's calibrated detection parameters (from calibration_runs.csv)
REFERENCE_BRAIN = "349_CNT_01_02_1p625x_z4"
REFERENCE_PARAMS = {
    'ball_xy': 10.0,
    'ball_z': 10.0,
    'soma_diameter': 10.0,
    'threshold': 8.0,
    'preset': 'custom',
}

# Pipeline folder names
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_DETECTION = "4_Cell_Candidates"
FOLDER_CLASSIFICATION = "5_Classified_Cells"
FOLDER_ANALYSIS = "6_Region_Analysis"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds

# Output directories
REPORTS_DIR = DATA_SUMMARY_DIR / "reports"
IMAGES_DIR = REPORTS_DIR / "images"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp() -> str:
    """Get formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_cropped_brains() -> List[Tuple[Path, str]]:
    """Find all brains with completed cropping."""
    brains = []

    for mouse_dir in BRAINS_ROOT.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue

        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir() or pipeline_dir.name.startswith('.'):
                continue

            cropped = pipeline_dir / FOLDER_CROPPED
            if cropped.exists():
                # Check for actual data
                ch0 = cropped / "ch0"
                if ch0.exists() and list(ch0.glob("Z*.tif")):
                    brains.append((pipeline_dir, pipeline_dir.name))

    return sorted(brains, key=lambda x: x[1])


def get_brain_status(pipeline_dir: Path) -> Dict[str, bool]:
    """Get completion status of each pipeline step for a brain."""
    return {
        'cropped': (pipeline_dir / FOLDER_CROPPED / "ch0").exists(),
        'registered': (pipeline_dir / FOLDER_REGISTRATION / "brainreg.json").exists(),
        'detected': (pipeline_dir / FOLDER_DETECTION).exists() and
                    bool(list((pipeline_dir / FOLDER_DETECTION).glob("*.xml"))),
        'classified': (pipeline_dir / FOLDER_CLASSIFICATION / "cells.xml").exists(),
        'counted': (pipeline_dir / FOLDER_ANALYSIS / "cell_counts_by_region.csv").exists(),
    }


def run_with_retry(func, *args, max_retries: int = MAX_RETRIES, **kwargs) -> Tuple[bool, str]:
    """Run a function with automatic retry on failure."""
    last_error = ""

    for attempt in range(max_retries):
        try:
            success, result = func(*args, **kwargs)
            if success:
                return True, result

            last_error = result
            log.warning(f"Attempt {attempt + 1}/{max_retries} failed: {result}")

            if attempt < max_retries - 1:
                log.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)

        except Exception as e:
            last_error = str(e)
            log.error(f"Attempt {attempt + 1}/{max_retries} raised exception: {e}")
            traceback.print_exc()

            if attempt < max_retries - 1:
                log.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)

    return False, f"Failed after {max_retries} attempts. Last error: {last_error}"


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def run_registration(brain_name: str, pipeline_dir: Path) -> Tuple[bool, str]:
    """Run brainreg registration for a brain."""
    log.info(f"[{brain_name}] Running registration...")
    sys.stdout.flush()

    script = SCRIPTS_DIR / "3_register_to_atlas.py"

    # Use --brain to filter to just this specific brain
    # Stream output directly so we can see progress
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--batch", "--brain", brain_name],
            capture_output=False,  # Stream output to console
            timeout=7200,  # 2 hour timeout
            cwd=str(SCRIPTS_DIR)
        )

        # Check return code AND verify output files exist
        reg_folder = pipeline_dir / FOLDER_REGISTRATION
        brainreg_json = reg_folder / "brainreg.json"
        boundaries_tiff = reg_folder / "boundaries.tiff"

        if result.returncode != 0:
            return False, f"Registration failed with code {result.returncode}"

        # Verify registration actually produced output files
        if not brainreg_json.exists():
            return False, "Registration failed: brainreg.json not created"
        if not boundaries_tiff.exists():
            return False, "Registration failed: boundaries.tiff not created"

        # Auto-approve registration only if verification passed
        approval_marker = reg_folder / ".registration_approved"
        approval_marker.touch()
        log.info(f"[{brain_name}] Registration complete, auto-approved")
        return True, "Registration successful"

    except subprocess.TimeoutExpired:
        return False, "Registration timed out after 2 hours"
    except Exception as e:
        return False, f"Registration error: {e}"


def run_detection(brain_name: str, pipeline_dir: Path, params: Dict) -> Tuple[bool, str]:
    """Run cellfinder detection with specified parameters."""
    log.info(f"[{brain_name}] Running detection with params: {params}")
    sys.stdout.flush()

    script = SCRIPTS_DIR / "4_detect_cells.py"

    # Note: 4_detect_cells.py expects int values, not floats
    cmd = [
        sys.executable, str(script),
        "--brain", brain_name,
        "--ball-xy", str(int(params['ball_xy'])),
        "--ball-z", str(int(params['ball_z'])),
        "--threshold", str(int(params['threshold'])),
    ]

    if 'soma_diameter' in params:
        cmd.extend(["--soma-diameter", str(int(params['soma_diameter']))])

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Stream output to console
            timeout=14400,  # 4 hour timeout
            cwd=str(SCRIPTS_DIR)
        )

        if result.returncode == 0:
            log.info(f"[{brain_name}] Detection complete")
            return True, "Detection successful"
        else:
            return False, f"Detection failed with code {result.returncode}"

    except subprocess.TimeoutExpired:
        return False, "Detection timed out after 4 hours"
    except Exception as e:
        return False, f"Detection error: {e}"


def find_paradigm_model(imaging_paradigm: str) -> Optional[Path]:
    """
    Find the best model for an imaging paradigm.

    Priority order:
    1. Paradigm-specific model directory (e.g., "1p625x_z4_v1_*")
    2. Tracker's paradigm-best model
    3. Fallback to newest model

    Args:
        imaging_paradigm: e.g., "1p625x_z4"

    Returns:
        Path to model file, or None
    """
    if not MODELS_DIR.exists():
        return None

    # 1. Look for paradigm-specific model directory
    #    Pattern: {paradigm}_v{version}_{timestamp}
    paradigm_dirs = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir() and imaging_paradigm in model_dir.name:
            paradigm_dirs.append(model_dir)

    if paradigm_dirs:
        # Sort by name (newest timestamp last) and pick the newest
        paradigm_dirs.sort(key=lambda x: x.name)
        best_dir = paradigm_dirs[-1]
        log.info(f"Found paradigm-specific model directory: {best_dir.name}")

        # Find model.keras in this directory
        model_file = best_dir / "model.keras"
        if model_file.exists():
            return model_file
        # Try any .keras file
        keras_files = list(best_dir.glob("*.keras"))
        if keras_files:
            return next((f for f in keras_files if f.name == "model.keras"), keras_files[0])

    # 2. Try tracker's paradigm model lookup
    try:
        from mousebrain.tracker import ExperimentTracker
        tracker = ExperimentTracker()
        paradigm_model = tracker.get_paradigm_model(imaging_paradigm)
        if paradigm_model:
            model_path = Path(paradigm_model)
            if model_path.exists():
                log.info(f"Using tracker's paradigm model: {model_path}")
                return model_path
    except Exception as e:
        log.warning(f"Could not query tracker for paradigm model: {e}")

    # 3. Fallback to newest model directory with .keras files
    log.warning(f"No paradigm-specific model found for {imaging_paradigm}, using newest model")
    for model_dir in sorted(MODELS_DIR.iterdir(), reverse=True):
        if model_dir.is_dir():
            keras_files = list(model_dir.glob("*.keras"))
            if keras_files:
                return next((f for f in keras_files if f.name == "model.keras"), keras_files[0])
            h5_files = list(model_dir.glob("*.h5"))
            if h5_files:
                return h5_files[0]

    return None


def run_classification(brain_name: str, pipeline_dir: Path) -> Tuple[bool, str]:
    """Run cell classification using paradigm-specific trained model."""
    log.info(f"[{brain_name}] Running classification...")

    script = SCRIPTS_DIR / "5_classify_cells.py"

    # Extract imaging paradigm from brain name
    brain_info = parse_brain_name(brain_name)
    imaging_paradigm = brain_info.get('imaging_params', '')  # e.g., "1p625x_z4"
    log.info(f"[{brain_name}] Imaging paradigm: {imaging_paradigm}")

    # Find the best model for this paradigm
    model_path = find_paradigm_model(imaging_paradigm)

    if model_path:
        log.info(f"[{brain_name}] Using model: {model_path}")
    else:
        log.warning(f"[{brain_name}] No model found!")

    cmd = [sys.executable, str(script), "--brain", brain_name]
    if model_path:
        cmd.extend(["--model", str(model_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(SCRIPTS_DIR)
        )

        if result.returncode == 0:
            log.info(f"[{brain_name}] Classification complete")
            return True, "Classification successful"
        else:
            return False, f"Classification failed: {result.stderr[:500]}"

    except subprocess.TimeoutExpired:
        return False, "Classification timed out after 1 hour"
    except Exception as e:
        return False, f"Classification error: {e}"


def run_counting(brain_name: str, pipeline_dir: Path) -> Tuple[bool, str]:
    """Run regional cell counting."""
    log.info(f"[{brain_name}] Running region counting...")

    script = SCRIPTS_DIR / "6_count_regions.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script), "--brain", brain_name],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            cwd=str(SCRIPTS_DIR)
        )

        if result.returncode == 0:
            log.info(f"[{brain_name}] Counting complete")
            return True, "Counting successful"
        else:
            return False, f"Counting failed: {result.stderr[:500]}"

    except subprocess.TimeoutExpired:
        return False, "Counting timed out after 30 minutes"
    except Exception as e:
        return False, f"Counting error: {e}"


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================

def process_brain(brain_name: str, pipeline_dir: Path, skip_complete: bool = True) -> Dict:
    """Process a single brain through the full pipeline."""
    result = {
        'brain': brain_name,
        'status': 'pending',
        'steps': {},
        'total_cells': None,
        'error': None,
    }

    status = get_brain_status(pipeline_dir)

    # Step 1: Registration
    if status['registered'] and skip_complete:
        log.info(f"[{brain_name}] Registration already complete, skipping")
        result['steps']['registration'] = 'skipped'
    else:
        success, msg = run_with_retry(run_registration, brain_name, pipeline_dir)
        result['steps']['registration'] = 'success' if success else 'failed'
        if not success:
            result['status'] = 'failed'
            result['error'] = msg
            return result

    # Step 2: Detection
    if status['detected'] and skip_complete:
        log.info(f"[{brain_name}] Detection already complete, skipping")
        result['steps']['detection'] = 'skipped'
    else:
        success, msg = run_with_retry(run_detection, brain_name, pipeline_dir, REFERENCE_PARAMS)
        result['steps']['detection'] = 'success' if success else 'failed'
        if not success:
            result['status'] = 'failed'
            result['error'] = msg
            return result

    # Step 3: Classification
    if status['classified'] and skip_complete:
        log.info(f"[{brain_name}] Classification already complete, skipping")
        result['steps']['classification'] = 'skipped'
    else:
        success, msg = run_with_retry(run_classification, brain_name, pipeline_dir)
        result['steps']['classification'] = 'success' if success else 'failed'
        if not success:
            result['status'] = 'failed'
            result['error'] = msg
            return result

    # Step 4: Counting
    if status['counted'] and skip_complete:
        log.info(f"[{brain_name}] Counting already complete, skipping")
        result['steps']['counting'] = 'skipped'
    else:
        success, msg = run_with_retry(run_counting, brain_name, pipeline_dir)
        result['steps']['counting'] = 'success' if success else 'failed'
        if not success:
            result['status'] = 'failed'
            result['error'] = msg
            return result

    # Get total cells
    counts_csv = pipeline_dir / FOLDER_ANALYSIS / "cell_counts_by_region.csv"
    if counts_csv.exists():
        try:
            import csv
            with open(counts_csv, 'r') as f:
                reader = csv.DictReader(f)
                result['total_cells'] = sum(int(row.get('cell_count', 0)) for row in reader)
        except Exception as e:
            log.warning(f"[{brain_name}] Could not read cell counts: {e}")

    result['status'] = 'success'
    return result


# =============================================================================
# REPORTING
# =============================================================================

def generate_comparison_report(results: List[Dict]) -> Path:
    """Generate markdown comparison report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    report_path = REPORTS_DIR / f"overnight_comparison_{date_str}.md"

    # Get reference brain counts
    ref_counts = None
    ref_result = next((r for r in results if REFERENCE_BRAIN in r['brain']), None)
    if ref_result and ref_result['total_cells']:
        ref_counts = ref_result['total_cells']

    # Build report
    lines = [
        f"# Overnight Processing Report - {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Summary",
        f"- **Brains processed:** {len(results)}",
        f"- **Successful:** {sum(1 for r in results if r['status'] == 'success')}",
        f"- **Failed:** {sum(1 for r in results if r['status'] == 'failed')}",
        "",
        "## Detection Parameters (from brain 349)",
        f"- ball_xy: {REFERENCE_PARAMS['ball_xy']}",
        f"- ball_z: {REFERENCE_PARAMS['ball_z']}",
        f"- threshold: {REFERENCE_PARAMS['threshold']}",
        f"- soma_diameter: {REFERENCE_PARAMS['soma_diameter']}",
        "",
        "## Results by Brain",
        "",
        "| Brain | Total Cells | vs 349 | Status | Steps |",
        "|-------|-------------|--------|--------|-------|",
    ]

    for r in sorted(results, key=lambda x: x['brain']):
        cells = r['total_cells'] or 'N/A'

        if ref_counts and r['total_cells']:
            diff = ((r['total_cells'] - ref_counts) / ref_counts) * 100
            vs_ref = f"{diff:+.1f}%" if r['brain'] != REFERENCE_BRAIN else "(ref)"
        else:
            vs_ref = "N/A"

        status_icon = "+" if r['status'] == 'success' else "X"
        steps = ", ".join(f"{k}:{v[0]}" for k, v in r['steps'].items())

        lines.append(f"| {r['brain']} | {cells} | {vs_ref} | {status_icon} | {steps} |")

    # Add failed brain details
    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        lines.extend([
            "",
            "## Failed Brains",
            "",
        ])
        for r in failed:
            lines.append(f"### {r['brain']}")
            lines.append(f"- **Error:** {r['error']}")
            lines.append("")

    lines.extend([
        "",
        "---",
        f"*Generated by overnight_batch.py v{SCRIPT_VERSION}*",
    ])

    report_path.write_text('\n'.join(lines))
    log.info(f"Report saved to: {report_path}")

    return report_path


def generate_representative_images(results: List[Dict]) -> List[Path]:
    """Generate representative images for each successful brain."""
    generated = []

    for r in results:
        if r['status'] != 'success':
            continue

        brain_name = r['brain']

        # Find pipeline directory
        pipeline_dir = None
        for mouse_dir in BRAINS_ROOT.iterdir():
            for pd in mouse_dir.iterdir():
                if pd.name == brain_name:
                    pipeline_dir = pd
                    break
            if pipeline_dir:
                break

        if not pipeline_dir:
            continue

        # Create output directory
        img_dir = IMAGES_DIR / brain_name
        img_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Registration overlay
            reg_overlay = generate_registration_image(pipeline_dir, img_dir)
            if reg_overlay:
                generated.append(reg_overlay)

            # 2. Cell detection samples
            cell_imgs = generate_cell_detection_images(pipeline_dir, img_dir)
            generated.extend(cell_imgs)

            # 3. Regional heatmap
            heatmap = generate_regional_heatmap(pipeline_dir, img_dir)
            if heatmap:
                generated.append(heatmap)

        except Exception as e:
            log.warning(f"[{brain_name}] Error generating images: {e}")

    return generated


def generate_registration_image(pipeline_dir: Path, output_dir: Path) -> Optional[Path]:
    """Generate registration overlay image."""
    try:
        import numpy as np
        import tifffile
        from matplotlib import pyplot as plt

        reg_dir = pipeline_dir / FOLDER_REGISTRATION
        boundaries = reg_dir / "boundaries.tiff"

        if not boundaries.exists():
            return None

        # Load boundaries
        data = tifffile.imread(str(boundaries))
        mid_z = data.shape[0] // 2

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(data[mid_z], cmap='gray')
        ax.set_title(f"{pipeline_dir.name} - Registration (Z={mid_z})")
        ax.axis('off')

        output_path = output_dir / f"{pipeline_dir.name}_registration.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        log.info(f"Generated: {output_path.name}")
        return output_path

    except Exception as e:
        log.warning(f"Could not generate registration image: {e}")
        return None


def generate_cell_detection_images(pipeline_dir: Path, output_dir: Path) -> List[Path]:
    """Generate cell detection sample images at 3 Z levels."""
    generated = []

    try:
        import numpy as np
        import tifffile
        from matplotlib import pyplot as plt

        # Find signal channel
        cropped_dir = pipeline_dir / FOLDER_CROPPED / "ch0"
        if not cropped_dir.exists():
            cropped_dir = pipeline_dir / FOLDER_CROPPED / "ch1"

        if not cropped_dir.exists():
            return []

        tiff_files = sorted(cropped_dir.glob("Z*.tif"))
        if not tiff_files:
            return []

        # Load cells from XML
        cells = []
        det_xml = None
        det_dir = pipeline_dir / FOLDER_DETECTION
        if det_dir.exists():
            xml_files = list(det_dir.glob("*.xml"))
            if xml_files:
                det_xml = xml_files[0]
                cells = parse_cells_from_xml(det_xml)

        # Generate images at 25%, 50%, 75% Z
        total_z = len(tiff_files)
        z_positions = [int(total_z * 0.25), int(total_z * 0.5), int(total_z * 0.75)]

        for z_idx in z_positions:
            if z_idx >= total_z:
                continue

            img = tifffile.imread(str(tiff_files[z_idx]))

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img, cmap='gray', vmin=np.percentile(img, 5), vmax=np.percentile(img, 99))

            # Plot cells at this Z level (within +/- 5 slices)
            z_cells = [(c['x'], c['y']) for c in cells if abs(c['z'] - z_idx) < 5]
            if z_cells:
                xs, ys = zip(*z_cells)
                ax.scatter(xs, ys, c='red', s=10, alpha=0.7, marker='o')

            ax.set_title(f"{pipeline_dir.name} - Cells at Z={z_idx} ({len(z_cells)} cells)")
            ax.axis('off')

            output_path = output_dir / f"{pipeline_dir.name}_cells_Z{z_idx:04d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            generated.append(output_path)
            log.info(f"Generated: {output_path.name}")

    except Exception as e:
        log.warning(f"Could not generate cell detection images: {e}")

    return generated


def generate_regional_heatmap(pipeline_dir: Path, output_dir: Path) -> Optional[Path]:
    """Generate regional cell count heatmap."""
    try:
        import csv
        from matplotlib import pyplot as plt

        counts_csv = pipeline_dir / FOLDER_ANALYSIS / "cell_counts_by_region.csv"
        if not counts_csv.exists():
            return None

        # Read counts
        regions = []
        with open(counts_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get('cell_count', 0)) > 0:
                    regions.append({
                        'acronym': row.get('region_acronym', 'Unknown'),
                        'name': row.get('region_name', 'Unknown'),
                        'count': int(row['cell_count'])
                    })

        # Sort and take top 20
        regions.sort(key=lambda x: x['count'], reverse=True)
        top_regions = regions[:20]

        # Create bar chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        acronyms = [r['acronym'] for r in top_regions]
        counts = [r['count'] for r in top_regions]

        bars = ax.barh(acronyms[::-1], counts[::-1], color='steelblue')
        ax.set_xlabel('Cell Count')
        ax.set_title(f"{pipeline_dir.name} - Top 20 Regions by Cell Count")

        # Add count labels
        for bar, count in zip(bars, counts[::-1]):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=8)

        plt.tight_layout()

        output_path = output_dir / f"{pipeline_dir.name}_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        log.info(f"Generated: {output_path.name}")
        return output_path

    except Exception as e:
        log.warning(f"Could not generate regional heatmap: {e}")
        return None


def parse_cells_from_xml(xml_path: Path) -> List[Dict]:
    """Parse cell coordinates from XML file."""
    cells = []

    try:
        with open(xml_path, 'r') as f:
            content = f.read()

        # Simple regex-free parsing
        for line in content.split('\n'):
            if '<Marker>' in line:
                continue
            if '<MarkerX>' in line and '<MarkerY>' in line and '<MarkerZ>' in line:
                try:
                    # Extract coordinates (simplified parsing)
                    parts = line.split('>')
                    x = y = z = 0
                    for i, p in enumerate(parts):
                        if p.endswith('<MarkerX'):
                            x = float(parts[i+1].split('<')[0])
                        elif p.endswith('<MarkerY'):
                            y = float(parts[i+1].split('<')[0])
                        elif p.endswith('<MarkerZ'):
                            z = float(parts[i+1].split('<')[0])
                    cells.append({'x': x, 'y': y, 'z': int(z)})
                except:
                    pass
    except:
        pass

    return cells


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overnight batch processing for SCI-Connectome pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be processed without running')
    parser.add_argument('--brain', '-b', type=str,
                       help='Process specific brain only (partial name match)')
    parser.add_argument('--skip-complete', action='store_true', default=True,
                       help='Skip already completed steps (default: True)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Reprocess all steps even if complete')

    args = parser.parse_args()

    skip_complete = not args.force

    log.info("=" * 70)
    log.info(f"OVERNIGHT BATCH PROCESSING - v{SCRIPT_VERSION}")
    log.info(f"Started: {timestamp()}")
    log.info("=" * 70)

    # Find brains to process
    brains = find_cropped_brains()

    if args.brain:
        brains = [(p, n) for p, n in brains if args.brain in n]

    log.info(f"Found {len(brains)} brain(s) with cropping complete:")
    for pipeline_dir, name in brains:
        status = get_brain_status(pipeline_dir)
        status_str = " | ".join(f"{k}:{'+' if v else '-'}" for k, v in status.items())
        log.info(f"  - {name} [{status_str}]")

    if args.dry_run:
        log.info("\n[DRY RUN] Would process the above brains with these parameters:")
        log.info(f"  Reference brain: {REFERENCE_BRAIN}")
        log.info(f"  Detection params: {REFERENCE_PARAMS}")
        return 0

    if not brains:
        log.warning("No brains found to process!")
        return 1

    # Process each brain
    results = []

    for i, (pipeline_dir, brain_name) in enumerate(brains):
        log.info("")
        log.info("=" * 70)
        log.info(f"[{i+1}/{len(brains)}] Processing: {brain_name}")
        log.info("=" * 70)

        start_time = time.time()
        result = process_brain(brain_name, pipeline_dir, skip_complete=skip_complete)
        elapsed = time.time() - start_time

        result['elapsed_seconds'] = elapsed
        results.append(result)

        if result['status'] == 'success':
            log.info(f"[{brain_name}] SUCCESS - {result['total_cells']} cells ({elapsed/60:.1f} min)")
        else:
            log.error(f"[{brain_name}] FAILED - {result['error']}")

    # Generate reports
    log.info("")
    log.info("=" * 70)
    log.info("GENERATING REPORTS")
    log.info("=" * 70)

    report_path = generate_comparison_report(results)

    log.info("")
    log.info("=" * 70)
    log.info("GENERATING IMAGES")
    log.info("=" * 70)

    images = generate_representative_images(results)
    log.info(f"Generated {len(images)} images")

    # Final summary
    log.info("")
    log.info("=" * 70)
    log.info("BATCH PROCESSING COMPLETE")
    log.info("=" * 70)
    log.info(f"Finished: {timestamp()}")
    log.info(f"Total brains: {len(results)}")
    log.info(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    log.info(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    log.info(f"Report: {report_path}")
    log.info(f"Images: {IMAGES_DIR}")
    log.info("=" * 70)

    # Return non-zero if any failed
    return 0 if all(r['status'] == 'success' for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
