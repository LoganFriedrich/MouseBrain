"""
batch_encr.py - Batch colocalization analysis for ENCR HD region ND2 files.

Loads each ND2 file, runs StarDist nuclei detection on the nuclear channel,
measures green signal colocalization, saves CSV results + QC overlay images.

Usage:
    python -m brainslice.batch.batch_encr
    python -m brainslice.batch.batch_encr --threshold 3.0 --dilation 100
    python -m brainslice.batch.batch_encr --output Y:/custom/output
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# Default paths
ENCR_ROOT = Path(r"Y:\2_Connectome\Tissue\2D_Slices\ENCR")
DEFAULT_OUTPUT = ENCR_ROOT / "batch_results"

# Default channel indices for ENCR ND2 files
# Channel 0 = 488nm (green/eYFP signal)
# Channel 1 = 561nm (red/nuclear)
RED_IDX = 1
GREEN_IDX = 0


def find_nd2_files(root: Path) -> list:
    """Find all ND2 files in ENCR HD_Regions folders."""
    nd2_files = []
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
    threshold: float = 2.0,
    dilation: int = 50,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    bg_percentile: float = 10.0,
) -> dict:
    """
    Process a single ND2 file through detection + colocalization.

    Returns dict with summary stats, or dict with 'error' key on failure.
    """
    from ..core.io import load_image, extract_channels
    from ..core.detection import NucleiDetector
    from ..core.colocalization import ColocalizationAnalyzer
    from ..core.visualization import save_all_qc_figures

    result = {
        'file': nd2_path.name,
        'path': str(nd2_path),
        'animal': '',
        'region': '',
    }

    try:
        # Parse animal and region from filename (e.g., E02_01_S13_DCN.nd2)
        stem = nd2_path.stem
        parts = stem.split('_')
        if len(parts) >= 3:
            result['animal'] = f"{parts[0]}_{parts[1]}"
            result['region'] = '_'.join(parts[3:]) if len(parts) > 3 else parts[2]

        # Load
        print(f"  Loading {nd2_path.name}...")
        data, metadata = load_image(nd2_path)
        red, green = extract_channels(data, red_idx=RED_IDX, green_idx=GREEN_IDX)

        # Detect
        print(f"  Detecting nuclei...")
        detector = NucleiDetector(model_name='2D_versatile_fluo')
        labels, details = detector.detect(red, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        n_nuclei = len(np.unique(labels)) - 1
        result['n_nuclei'] = n_nuclei
        print(f"  Found {n_nuclei} nuclei")

        if n_nuclei == 0:
            result['status'] = 'no_nuclei'
            return result

        # Colocalize
        print(f"  Running colocalization...")
        analyzer = ColocalizationAnalyzer(
            background_method='percentile',
            background_percentile=bg_percentile,
        )
        background = analyzer.estimate_background(green, labels, dilation_iterations=dilation)
        tissue_mask = analyzer.estimate_tissue_mask(labels, dilation)
        measurements = analyzer.measure_nuclei_intensities(green, labels)
        classified = analyzer.classify_positive_negative(
            measurements, background, method='fold_change', threshold=threshold
        )
        summary = analyzer.get_summary_statistics(classified)

        result.update({
            'n_positive': summary['positive_cells'],
            'n_negative': summary['negative_cells'],
            'positive_fraction': round(summary['positive_fraction'], 4),
            'mean_fold_change': round(summary['mean_fold_change'], 2),
            'median_fold_change': round(summary['median_fold_change'], 2),
            'background': round(summary['background_used'], 2),
            'status': 'completed',
        })

        # Save outputs
        sample_dir = output_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save measurements CSV
        classified.to_csv(sample_dir / "measurements.csv", index=False)

        # Save QC figures
        print(f"  Saving QC images...")
        save_all_qc_figures(
            output_dir=sample_dir,
            green_channel=green,
            nuclei_labels=labels,
            measurements_df=classified,
            tissue_mask=tissue_mask,
            threshold=threshold,
            background=background,
            prefix=stem,
        )

        print(f"  Done: {summary['positive_cells']} pos, {summary['negative_cells']} neg ({summary['positive_fraction']*100:.1f}%)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"  ERROR: {e}")

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
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Fold-change threshold for positive classification (default: 2.0)')
    parser.add_argument('--dilation', type=int, default=50,
                        help='Nuclei exclusion dilation iterations (default: 50)')
    parser.add_argument('--bg-percentile', type=float, default=10.0,
                        help='Background percentile (default: 10.0)')
    parser.add_argument('--prob-thresh', type=float, default=0.5,
                        help='StarDist probability threshold (default: 0.5)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List files without processing')
    args = parser.parse_args()

    print("=" * 60)
    print("ENCR Batch Colocalization Analysis")
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output}")
    print(f"Threshold: {args.threshold}x fold change")
    print(f"Dilation:  {args.dilation}")
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
            bg_percentile=args.bg_percentile,
            prob_thresh=args.prob_thresh,
        )
        all_results.append(result)

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
