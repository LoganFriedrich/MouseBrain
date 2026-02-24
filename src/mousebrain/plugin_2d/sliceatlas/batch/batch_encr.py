"""
batch_encr.py - Batch colocalization analysis for ENCR HD region ND2 files.

Loads each ND2 file, runs threshold/Otsu nuclei detection on the red channel,
measures green signal colocalization, saves CSV results + QC overlay images.

Usage:
    python -m brainslice.batch.batch_encr
    python -m brainslice.batch.batch_encr --threshold 1.5 --dilation 50
    python -m brainslice.batch.batch_encr --output Y:/custom/output

------------------------------------------------------------------------
APPROVED METHOD FOR ENCR ANALYSIS (as of 2026-02-17, PI-approved)
------------------------------------------------------------------------
DETECTION:       threshold, method=otsu, physical size 8-25 um diameter
                 (auto-converted to pixel area per ND2 metadata)
COLOCALIZATION:  background_mean, sigma_threshold=0 (bg mean IS the threshold)
BACKGROUND:      gmm, percentile=10
SOMA DILATION:   6 px (measure cytoplasmic signal, not just nuclear)
SOMA EXCLUSION:  dilation >= 50 iterations for background estimation

How background_mean works:
  1. Detect nuclei in red channel (Otsu threshold, 8-25 um diameter).
  2. Dilate each nucleus by 6px to measure cytoplasmic green signal (soma_dilation).
  3. Dilate all nuclei generously (50 iter) to create background exclusion zones.
  4. Background = mean of green channel in tissue OUTSIDE exclusion zones.
  5. Cell is GFP-positive if: soma green > background_mean (sigma_threshold=0).

DO NOT change the colocalization method without:
  - Updating METHOD_LOG.md at:
    Y:\\2_Connectome\\Tissue\\MouseBrain_Pipeline\\2D_Slices\\ENCR\\METHOD_LOG.md
  - Getting PI sign-off

History: METHOD_LOG.md above shows all methods tried and which were rejected.
  fold_change was an earlier method -- it is NOT approved for ENCR.
  local_snr was also tried and superseded by background_mean.
------------------------------------------------------------------------
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.config import DATA_DIR, get_sample_dir, SampleDirs, parse_sample_name


# Default paths - resolve ENCR root from config
# DATA_DIR is now 1_Subjects/, so ENCR is a subdirectory
ENCR_ROOT = DATA_DIR / "ENCR" if DATA_DIR else Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\1_Subjects\ENCR")
DEFAULT_OUTPUT = ENCR_ROOT.parent.parent / "2_Data_Summary" / "batch_results"

# Default channel indices for ENCR ND2 files
# Channel 0 = 488nm (green/eYFP signal)
# Channel 1 = 561nm (red/nuclear)
RED_IDX = 1
GREEN_IDX = 0


def get_tuned_parameters() -> dict:
    """Look up tuned detection/colocalization parameters from the tracker.

    Searches for marked-best detection runs. If found, extracts the parameters
    that were used. This ensures interactive tuning in the napari widget
    automatically carries forward to batch processing.

    Returns dict with keys: det_model, min_area, max_area, bg_method,
    threshold, or empty dict if no best run found.
    """
    try:
        from ..tracker import SliceTracker
        tracker = SliceTracker()
    except Exception:
        return {}

    # Search all detection runs for any marked-best
    rows = tracker.search(run_type='detection', status='completed')
    best_rows = [r for r in rows if r.get('marked_best') == 'True']

    if not best_rows:
        # Fallback: use the most recent completed detection run
        if rows:
            best_rows = sorted(rows, key=lambda r: r.get('created_at', ''), reverse=True)[:1]
        else:
            return {}

    best = best_rows[0]
    params = {}

    model = best.get('det_model', '')
    if model:
        params['det_model'] = model

    min_a = best.get('det_min_area', '')
    if min_a:
        try:
            params['min_area'] = int(float(min_a))
        except (ValueError, TypeError):
            pass

    max_a = best.get('det_max_area', '')
    if max_a:
        try:
            params['max_area'] = int(float(max_a))
        except (ValueError, TypeError):
            pass

    # Check linked colocalization run for background method
    coloc_rows = tracker.search(run_type='colocalization')
    if coloc_rows:
        # Find colocalization linked to this detection or most recent
        linked = [r for r in coloc_rows if r.get('parent_run') == best.get('run_id')]
        if not linked:
            linked = sorted(coloc_rows, key=lambda r: r.get('created_at', ''), reverse=True)[:1]
        if linked:
            bg = linked[0].get('coloc_background_method', '')
            if bg:
                params['bg_method'] = bg
            thresh = linked[0].get('coloc_threshold_value', '')
            if thresh:
                try:
                    params['threshold'] = float(thresh)
                except (ValueError, TypeError):
                    pass

    return params


def find_nd2_files(root: Path) -> list:
    """Find all ND2 files in ENCR subject folders."""
    nd2_files = []

    # New structure: 1_Subjects/ENCR/ENCR_XX_XX/0_Raw_HD/ and 0_Raw/
    for subject_dir in sorted(root.glob("ENCR_*")):
        if not subject_dir.is_dir():
            continue
        # HD region files
        hd_dir = subject_dir / SampleDirs.RAW_HD
        if hd_dir.exists():
            nd2_files.extend(sorted(hd_dir.glob("*.nd2")))
        # Standard raw files
        raw_dir = subject_dir / SampleDirs.RAW
        if raw_dir.exists():
            nd2_files.extend(sorted(raw_dir.glob("*.nd2")))

    # Fallback: old structure (ENCR_02_XX_HD_Regions/)
    if not nd2_files:
        for hd_dir in sorted(root.glob("ENCR_02_*_HD_Regions")):
            # Check Corrected subfolder first (preferred)
            corrected = hd_dir / "Corrected"
            if corrected.exists():
                nd2_files.extend(sorted(corrected.glob("*.nd2")))
            # Also get files from base dir that aren't in Corrected
            corrected_stems = {f.stem for f in (corrected.glob("*.nd2") if corrected.exists() else [])}
            for f in sorted(hd_dir.glob("*.nd2")):
                if f.stem not in corrected_stems:
                    nd2_files.append(f)

    return nd2_files


def process_single(
    nd2_path: Path,
    output_dir: Path,
    threshold: float = 1.5,
    dilation: int = 50,
    min_area: int = 3,
    max_area: int = 114,
    bg_percentile: float = 10.0,
    soma_dilation: int = 6,
    sigma_threshold: float = 0,
) -> dict:
    """
    Process a single ND2 file through detection + colocalization.

    Detection uses threshold method (tuned for sparse retrograde-labeled
    neurons). Colocalization uses background_mean (PI's method) with
    soma dilation to measure cytoplasmic signal.

    Parameters:
        soma_dilation: Dilation radius (px) to create soma ROIs around nuclei.
            Retrograde tracer signal is cytoplasmic, so we must measure green
            in the dilated disk around each nucleus. Default 6 matches Feb 17
            validated runs.
        sigma_threshold: Number of std devs above background mean for positive
            classification. Default 0 = background mean IS the threshold.

    Returns dict with summary stats, or dict with 'error' key on failure.
    """
    from ..core.io import load_image, extract_channels
    from ..core.detection import detect_by_threshold, detect_with_log_augmentation
    from ..core.colocalization import ColocalizationAnalyzer
    from ..core.visualization import save_all_qc_figures

    # Initialize tracker
    try:
        from ..tracker import SliceTracker
        tracker = SliceTracker()
    except Exception:
        tracker = None

    result = {
        'file': nd2_path.name,
        'path': str(nd2_path),
        'animal': '',
        'region': '',
    }
    det_run_id = None
    coloc_run_id = None

    try:
        # Parse animal and region from filename (e.g., E02_01_S13_DCN.nd2)
        stem = nd2_path.stem
        parts = stem.split('_')
        if len(parts) >= 3:
            result['animal'] = f"{parts[0]}_{parts[1]}"
            result['region'] = '_'.join(parts[3:]) if len(parts) > 3 else parts[2]

        # Parse sample ID
        sample_id = nd2_path.stem

        # Load
        print(f"  Loading {nd2_path.name}...")
        data, metadata = load_image(nd2_path)
        red, green = extract_channels(data, red_idx=RED_IDX, green_idx=GREEN_IDX)

        # Physical size conversion
        voxel = metadata.get('voxel_size_um', {})
        pixel_um = voxel.get('x', 1.0) if isinstance(voxel, dict) else voxel[0]
        print(f"  Pixel size: {pixel_um:.2f} um/px")

        # Log detection run to tracker
        det_run_id = None
        if tracker:
            det_run_id = tracker.log_detection(
                sample_id=sample_id,
                channel="red",
                model="threshold+log",
                prob_thresh=0.0,
                nms_thresh=0.0,
                input_path=str(nd2_path),
                status="started",
            )

        # Detect using threshold + LoG augmentation with decision tree
        print(f"  Detecting nuclei (threshold+LoG, 8-25 um, {pixel_um:.2f} um/px)...")
        labels, details = detect_with_log_augmentation(
            red, pixel_um=pixel_um,
        )
        n_nuclei = details['filtered_count']
        result['n_nuclei'] = n_nuclei
        decision = details.get('decision', '?')
        n_thresh = details.get('n_threshold', 0)
        n_log = details.get('n_log_new', 0)
        print(f"  Found {n_nuclei} nuclei ({decision}: {n_thresh} threshold + {n_log} LoG)")

        # Update tracker with detection results
        if tracker and det_run_id:
            tracker.update_status(det_run_id, status="completed", det_nuclei_found=n_nuclei)

        if n_nuclei == 0:
            result['status'] = 'no_nuclei'
            return result

        # Log colocalization run to tracker
        coloc_run_id = None
        if tracker:
            coloc_run_id = tracker.log_colocalization(
                sample_id=sample_id,
                signal_channel="green",
                background_method="gmm",
                background_percentile=bg_percentile,
                threshold_method="background_mean",
                threshold_value=threshold,
                parent_run=det_run_id,
                input_path=str(nd2_path),
                status="started",
            )

        # Colocalize using background_mean (PI's method):
        # Mask out all possible somas from green channel, average remaining
        # tissue = background. Cell is positive if green > background mean.
        print(f"  Running colocalization (background_mean)...")
        analyzer = ColocalizationAnalyzer(
            background_method='gmm',
            background_percentile=bg_percentile,
        )
        background = analyzer.estimate_background(green, labels, dilation_iterations=dilation)
        tissue_mask = analyzer.estimate_tissue_mask(labels, dilation)
        # Measure cytoplasmic signal via Voronoi territories (non-overlapping)
        # This prevents neighboring cells' green signal from contaminating each other
        measurements = analyzer.measure_cytoplasm_intensities(
            green, labels, expansion_px=soma_dilation)
        classified = analyzer.classify_positive_negative(
            measurements, background,
            method='background_mean',
            signal_image=green,
            nuclei_labels=labels,
            sigma_threshold=sigma_threshold,
        )
        summary = analyzer.get_summary_statistics(classified)

        # Update tracker with colocalization results
        if tracker and coloc_run_id:
            tracker.update_status(
                coloc_run_id,
                status="completed",
                coloc_positive_cells=summary['positive_cells'],
                coloc_negative_cells=summary['negative_cells'],
                coloc_positive_fraction=summary['positive_fraction'],
                coloc_background_value=summary['background_used'],
            )

        result.update({
            'n_positive': summary['positive_cells'],
            'n_negative': summary['negative_cells'],
            'positive_fraction': round(summary['positive_fraction'], 4),
            'mean_fold_change': round(summary['mean_fold_change'], 2),
            'median_fold_change': round(summary['median_fold_change'], 2),
            'background': round(summary['background_used'], 2),
            'status': 'completed',
        })

        # Route output to per-subject pipeline folder
        parsed = parse_sample_name(sample_id)
        subject_id = f"{parsed['project']}_{parsed['cohort']}_{parsed['subject']}"
        subject_base = DATA_DIR / parsed['project'] / subject_id if DATA_DIR else output_dir
        quant_dir = subject_base / SampleDirs.QUANTIFIED
        quant_dir.mkdir(parents=True, exist_ok=True)
        detect_dir = subject_base / SampleDirs.DETECTED
        detect_dir.mkdir(parents=True, exist_ok=True)

        # Fallback sample_dir for backward compatibility
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save measurements CSV to quantification directory and export dir
        classified.to_csv(quant_dir / f"{sample_id}_measurements.csv", index=False)
        classified.to_csv(sample_dir / "measurements.csv", index=False)

        # Save QC figures to detection directory (primary) and fallback
        print(f"  Saving QC images...")
        save_all_qc_figures(
            output_dir=detect_dir,
            green_channel=green,
            nuclei_labels=labels,
            measurements_df=classified,
            tissue_mask=tissue_mask,
            threshold=threshold,
            background=background,
            prefix=sample_id,
        )
        # Also save to sample_dir for backward compatibility
        save_all_qc_figures(
            output_dir=sample_dir,
            green_channel=green,
            nuclei_labels=labels,
            measurements_df=classified,
            tissue_mask=tissue_mask,
            threshold=threshold,
            background=background,
            prefix=sample_id,
        )

        print(f"  Done: {summary['positive_cells']} pos, {summary['negative_cells']} neg ({summary['positive_fraction']*100:.1f}%)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"  ERROR: {e}")

        # Update tracker on error
        if tracker and det_run_id:
            tracker.update_status(det_run_id, status="error")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch colocalization analysis for ENCR ND2 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', type=Path, default=ENCR_ROOT,
                        help=f'ENCR root directory (default: {ENCR_ROOT})')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help=f'Output directory (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Sigma threshold: positive if green > bg_mean + N*std. 0 = bg mean IS the threshold (default: 0)')
    parser.add_argument('--soma-dilation', type=int, default=6,
                        help='Soma dilation radius (px) for cytoplasmic signal measurement (default: 6)')
    parser.add_argument('--dilation', type=int, default=50,
                        help='Nuclei exclusion dilation iterations for background (default: 50)')
    parser.add_argument('--bg-percentile', type=float, default=10.0,
                        help='Background percentile fallback (default: 10.0)')
    parser.add_argument('--min-area', type=int, default=3,
                        help='Minimum nucleus area in pixels (default: 3)')
    parser.add_argument('--max-area', type=int, default=114,
                        help='Maximum nucleus area in pixels (default: 114)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List files without processing')
    parser.add_argument('--ignore-tracker', action='store_true',
                        help='Ignore tracker parameters and use CLI defaults')
    args = parser.parse_args()

    # Consult tracker for tuned parameters (unless CLI explicitly overrode)
    param_source = "defaults"
    if not args.ignore_tracker:
        tuned = get_tuned_parameters()
        if tuned:
            param_source = "tracker (from interactive tuning)"
            # Apply tuned params as defaults (CLI args still override)
            if 'min_area' in tuned and args.min_area == 3:
                args.min_area = tuned['min_area']
            if 'max_area' in tuned and args.max_area == 114:
                args.max_area = tuned['max_area']
            if 'threshold' in tuned and args.threshold == 0:
                args.threshold = tuned['threshold']
        else:
            print("[!] WARNING: No tuned parameters found in tracker.")
            print("    Batch will use hardcoded defaults.")
            print("    To tune: run brainslice-detect interactively first.")
            print()

    print("=" * 60)
    print("ENCR Batch Colocalization Analysis")
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output}")
    print(f"Params:    {param_source}")
    print(f"Detection: threshold (otsu), area {args.min_area}-{args.max_area} px")
    print(f"Background: GMM (percentile={args.bg_percentile})")
    print(f"Coloc:     background_mean, sigma_threshold={args.threshold}")
    print(f"Soma dil:  {args.soma_dilation} px (cytoplasmic signal measurement)")
    print(f"BG excl:   {args.dilation} dilation iterations")
    print("=" * 60)

    # Find ND2 files
    nd2_files = find_nd2_files(args.input)
    print(f"\nFound {len(nd2_files)} ND2 files")

    if args.dry_run:
        for f in nd2_files:
            print(f"  {f.relative_to(args.input)}")
        return

    if not nd2_files:
        print("No ND2 files found!")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # Instantiate AnalysisRegistry once for the batch run
    try:
        from mousebrain.analysis_registry import AnalysisRegistry, get_approved_method
        registry = AnalysisRegistry(analysis_name="ENCR_Detection")
    except Exception as _reg_init_err:
        registry = None
        print(f"  Registry warning: {_reg_init_err}")

    # Process each file
    all_results = []
    start_time = time.time()

    for i, nd2_path in enumerate(nd2_files):
        print(f"\n[{i+1}/{len(nd2_files)}] {nd2_path.name}")
        result = process_single(
            nd2_path,
            args.output,
            threshold=args.threshold,
            dilation=args.dilation,
            min_area=args.min_area,
            max_area=args.max_area,
            bg_percentile=args.bg_percentile,
            soma_dilation=args.soma_dilation,
            sigma_threshold=args.threshold,
        )
        all_results.append(result)

        # Register output for successfully completed samples
        if result.get('status') == 'completed' and registry is not None:
            try:
                sample_id = nd2_path.stem
                sample_dir = args.output / sample_id
                output_files = {}
                if sample_dir.exists():
                    for f in sample_dir.iterdir():
                        if f.suffix == '.csv':
                            output_files['measurements'] = str(f)
                        elif f.suffix == '.png' and 'overlay' in f.name:
                            output_files['overlay'] = str(f)
                registry.register_output(
                    sample=sample_id,
                    category="detection",
                    files=output_files,
                    results={
                        'n_nuclei': result.get('n_nuclei', 0),
                        'n_positive': result.get('n_positive', 0),
                        'positive_fraction': result.get('positive_fraction', 0),
                        'background': result.get('background', 0),
                    },
                    method_params=get_approved_method(),
                    source_files={'nd2': str(nd2_path)},
                )
            except Exception as _reg_err:
                print(f"  Registry warning: {_reg_err}")

    # Write summary CSV
    summary_path = args.output / "summary.csv"
    fieldnames = [
        'file', 'animal', 'region', 'n_nuclei', 'n_positive', 'n_negative',
        'positive_fraction', 'mean_fold_change', 'median_fold_change',
        'background', 'status', 'error', 'path',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary
    elapsed = time.time() - start_time
    completed = sum(1 for r in all_results if r.get('status') == 'completed')
    errored = sum(1 for r in all_results if r.get('status') == 'error')
    no_nuclei = sum(1 for r in all_results if r.get('status') == 'no_nuclei')

    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE")
    print(f"  Processed: {len(all_results)} files in {elapsed/60:.1f} min")
    print(f"  Completed: {completed}")
    print(f"  No nuclei: {no_nuclei}")
    print(f"  Errors:    {errored}")
    print(f"  Summary:   {summary_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
