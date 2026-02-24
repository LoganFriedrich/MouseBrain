"""Worker: process a subset of ND2+ROI pairs. Called with start_idx and end_idx.

Usage:
    python _worker_batch.py <start_idx> <end_idx> [--skip-existing]

Processes pairs[start_idx:end_idx] from the full pair list.
"""
import sys
import time
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_encr_coloc_batch import find_nd2_roi_pairs, ENCR_ROOT_DEFAULT, OUTPUT_DEFAULT


def main():
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    skip_existing = '--skip-existing' in sys.argv

    pairs = find_nd2_roi_pairs(ENCR_ROOT_DEFAULT)
    my_pairs = pairs[start_idx:end_idx]

    print(f"Worker: processing pairs [{start_idx}:{end_idx}] = {len(my_pairs)} samples")
    print(f"Skip existing: {skip_existing}")

    from generate_coloc_roi_figure import generate_figure

    results = []
    errors = 0
    max_errors = 3

    for i, pair in enumerate(my_pairs):
        stem = pair['stem']
        out_dir = pair['output_dir']
        fig_path = out_dir / f"{stem}_coloc_result.png"

        if skip_existing and fig_path.exists():
            print(f"  [{start_idx+i+1}] SKIP {stem} (exists)")
            continue

        print(f"  [{start_idx+i+1}] {stem} ({pair['subject']}/{pair['region']})")
        t0 = time.time()

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            fp, roi_stats = generate_figure(
                nd2_path=str(pair['nd2']),
                roi_path=str(pair['roi']),
                out_dir=str(out_dir),
            )
            elapsed = time.time() - t0
            rs = roi_stats
            roi_parts = " | ".join(
                f"{rn}: {rd['n_positive']}/{rd['n_total']}"
                for rn, rd in rs.get('rois', {}).items()
            )
            print(f"    OK ({elapsed:.0f}s) {rs['n_positive']}/{rs['n_nuclei']} pos | {roi_parts}")
            results.append({'stem': stem, 'status': 'done', 'roi_stats': roi_stats})
            errors = 0
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAIL ({elapsed:.0f}s): {e}")
            traceback.print_exc()
            results.append({'stem': stem, 'status': 'error', 'error': str(e)})
            errors += 1
            if errors >= max_errors:
                print(f"  HALTED: {errors} consecutive errors")
                break

    # Save worker results
    results_path = OUTPUT_DEFAULT / f"worker_results_{start_idx}_{end_idx}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWorker done: {sum(1 for r in results if r['status']=='done')} ok, "
          f"{sum(1 for r in results if r['status']=='error')} errors")
    print(f"Results: {results_path}")


if __name__ == '__main__':
    main()
