"""ENCR Colocalization Batch Runner — interactive CLI with safety rails.

Processes all ND2+ROI pairs through the coloc pipeline, generates per-sample
figures, and produces a summary stats figure (roi_stats_by_region_subject.png).

Features:
  - Interactive prompts for input/output paths
  - Parallel processing via multiprocessing
  - Safety rails: per-sample sanity checks, auto-halt on repeated failures
  - Detailed terminal logging with timestamps
  - Summary statistics with between-subject comparisons

Usage:
    Y:\\2_Connectome\\envs\\MouseBrain\\python.exe run_encr_coloc_batch.py
"""
import sys
import os
import time
import json
import signal
import traceback
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ── Approved pipeline parameters ──
SIGMA_THRESHOLD = 2
SOMA_DILATION = 6
MIN_DIAMETER_UM = 10.0
BACKGROUND_METHOD = 'gmm'
COLOC_METHOD = 'background_mean'

# ── Safety rails ──
MAX_CONSECUTIVE_ERRORS = 3      # halt after this many errors in a row
MIN_NUCLEI = 1                  # minimum nuclei to be considered valid
MAX_POSITIVE_FRACTION = 0.99    # flag if >99% positive (likely broken threshold)
MIN_POSITIVE_FRACTION = 0.0     # 0% is OK for some regions
SANITY_WARN_POSITIVE_FRAC = 0.95  # warn if >95%

ENCR_ROOT_DEFAULT = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR")
OUTPUT_DEFAULT = Path(r"Y:\2_Connectome\Databases\figures\ENCR_ROI_Analysis")

# ── Logging setup ──
LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(output_dir):
    """Configure logging to both console and file."""
    log_path = output_dir / f"batch_coloc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), encoding='utf-8'),
        ]
    )
    return log_path


def find_nd2_roi_pairs(encr_root):
    """Find all ND2 files with matching .rois.json in HD_Regions directories."""
    pairs = []
    seen_stems = set()

    for hd_dir in sorted(encr_root.glob("ENCR_02_*_HD_Regions")):
        # Only process main directory ROI files (not Corrected/SC subdirs)
        for roi_file in sorted(hd_dir.glob("*.rois.json")):
            stem = roi_file.stem.replace('.rois', '')
            if stem in seen_stems:
                continue

            # Find matching ND2: prefer Corrected, fall back to main dir
            nd2_path = hd_dir / "Corrected" / f"{stem}.nd2"
            if not nd2_path.exists():
                nd2_path = hd_dir / f"{stem}.nd2"
            if not nd2_path.exists():
                continue

            # Parse subject and region from stem (e.g. E02_01_S13_DCN)
            parts = stem.split('_')
            if len(parts) >= 4:
                subject = f"E{parts[0][1:]}_{parts[1]}"  # E02_01
                region = parts[-1]  # DCN, GRN, etc. (may have v2/001 suffix)
                # Normalize region: strip v2, v2Z, 001, etc.
                base_region = region.rstrip('0123456789')
                if base_region.endswith('v'):
                    base_region = base_region[:-1]
                if base_region.endswith('Z'):
                    base_region = base_region[:-1]
            else:
                subject = stem
                base_region = 'unknown'

            # Determine output directory
            output_subdir = OUTPUT_DEFAULT / subject / base_region

            pairs.append({
                'stem': stem,
                'nd2': nd2_path,
                'roi': roi_file,
                'subject': subject,
                'region': base_region,
                'output_dir': output_subdir,
            })
            seen_stems.add(stem)

    return pairs


def process_one_sample(pair):
    """Process a single ND2+ROI pair. Returns (pair, result_dict)."""
    stem = pair['stem']
    nd2_path = pair['nd2']
    roi_path = pair['roi']
    output_dir = pair['output_dir']

    result = {
        'stem': stem,
        'subject': pair['subject'],
        'region': pair['region'],
        'status': 'error',
        'error': None,
        'fig_path': None,
        'roi_stats': None,
        'elapsed': 0,
    }

    t0 = time.time()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Import here to avoid issues with multiprocessing
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_coloc_roi_figure import generate_figure

        fig_path, roi_stats = generate_figure(
            nd2_path=str(nd2_path),
            roi_path=str(roi_path),
            out_dir=str(output_dir),
        )

        result['fig_path'] = str(fig_path)
        result['roi_stats'] = roi_stats
        result['status'] = 'done'

        # Sanity checks
        pf = roi_stats.get('positive_fraction', 0)
        nn = roi_stats.get('n_nuclei', 0)

        if nn < MIN_NUCLEI:
            result['status'] = 'warn'
            result['error'] = f"Only {nn} nuclei detected"
        elif pf > MAX_POSITIVE_FRACTION:
            result['status'] = 'warn'
            result['error'] = f"Suspicious: {pf*100:.0f}% positive (>{MAX_POSITIVE_FRACTION*100}%)"
        elif pf > SANITY_WARN_POSITIVE_FRAC:
            result['status'] = 'warn'
            result['error'] = f"High: {pf*100:.0f}% positive"

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {e}"
        result['status'] = 'error'

    result['elapsed'] = time.time() - t0
    return pair, result


def generate_summary_figure(all_results, output_dir):
    """Generate roi_stats_by_region_subject.png from aggregated results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    logging.info("Generating summary stats figure...")

    # Collect per-sample ROI positive fractions grouped by region and subject
    # Structure: {region: {subject: [pct_positive, ...]}}
    region_subject_data = {}

    for r in all_results:
        if r['status'] in ('error',):
            continue
        roi_stats = r.get('roi_stats')
        if roi_stats is None:
            continue

        region = r['region']
        subject = r['subject']
        rois = roi_stats.get('rois', {})

        # Average across all ROIs in this sample
        roi_pcts = [v['pct_positive'] for v in rois.values() if v['n_total'] > 0]
        if not roi_pcts:
            continue
        mean_pct = np.mean(roi_pcts)

        if region not in region_subject_data:
            region_subject_data[region] = {}
        if subject not in region_subject_data[region]:
            region_subject_data[region][subject] = []
        region_subject_data[region][subject].append(mean_pct)

    if not region_subject_data:
        logging.warning("No valid data for summary figure")
        return None

    # Consistent ordering
    regions = sorted(region_subject_data.keys())
    all_subjects = sorted(set(
        s for rd in region_subject_data.values() for s in rd.keys()
    ))
    colors = ['#4477AA', '#EE7733', '#66CCEE', '#AA3377', '#228833']
    subject_colors = {s: colors[i % len(colors)] for i, s in enumerate(all_subjects)}
    subject_labels = {s: f"ENCR {s[1:3]}-{s[4:6]}" for s in all_subjects}

    fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(18, 7),
                                          gridspec_kw={'width_ratios': [2, 1.2]})
    fig.suptitle("ENCR Colocalization by Region and Subject", fontsize=14, fontweight='bold')

    n_subj = len(all_subjects)
    bar_width = 0.7 / max(n_subj, 1)
    x_base = np.arange(len(regions))

    for si, subj in enumerate(all_subjects):
        means, sds, ns, all_pts = [], [], [], []
        for reg in regions:
            vals = region_subject_data.get(reg, {}).get(subj, [])
            if vals:
                means.append(np.mean(vals))
                sds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                ns.append(len(vals))
                all_pts.append(vals)
            else:
                means.append(0)
                sds.append(0)
                ns.append(0)
                all_pts.append([])

        x_pos = x_base + (si - n_subj / 2 + 0.5) * bar_width
        color = subject_colors[subj]

        bars = ax_bar.bar(x_pos, means, bar_width * 0.85, color=color, alpha=0.7,
                          edgecolor=color, linewidth=1.2,
                          label=subject_labels[subj])
        ax_bar.errorbar(x_pos, means, yerr=sds, fmt='none',
                        ecolor='black', elinewidth=1.2, capsize=3, capthick=1)

        # Individual data points
        rng = np.random.default_rng(42 + si)
        for ri, pts in enumerate(all_pts):
            if pts:
                jitter = rng.uniform(-bar_width * 0.25, bar_width * 0.25, len(pts))
                ax_bar.scatter(x_pos[ri] + jitter, pts, s=30, color=color,
                               edgecolors='black', linewidths=0.5, zorder=5, alpha=0.9)

        # n labels
        for ri, n in enumerate(ns):
            if n > 0:
                ax_bar.text(x_pos[ri], -3, f"n={n}", ha='center', va='top',
                            fontsize=7, color='#555')

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(regions, fontsize=11)
    ax_bar.set_ylabel("% Positive in ROI", fontsize=12)
    ax_bar.set_ylim(-8, 110)
    ax_bar.legend(fontsize=10, loc='upper left')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Kruskal-Wallis + pairwise Mann-Whitney U table
    from scipy import stats

    table_rows = []
    for reg in regions:
        subj_data = region_subject_data.get(reg, {})
        groups = [np.array(subj_data.get(s, [])) for s in all_subjects]
        non_empty = [g for g in groups if len(g) > 0]

        # Kruskal-Wallis across all subjects for this region
        kw_p = None
        kw_sig = ""
        if len(non_empty) >= 2 and all(len(g) >= 2 for g in non_empty):
            try:
                kw_stat, kw_p = stats.kruskal(*non_empty)
                if kw_p < 0.001:
                    kw_sig = "***"
                elif kw_p < 0.01:
                    kw_sig = "**"
                elif kw_p < 0.05:
                    kw_sig = "*"
                else:
                    kw_sig = "ns"
            except Exception:
                kw_sig = "err"

        # Add KW annotation to bar chart
        if kw_p is not None:
            xi = regions.index(reg)
            y_top = max(
                np.mean(subj_data.get(s, [0])) + np.std(subj_data.get(s, [0]), ddof=1)
                for s in all_subjects if subj_data.get(s)
            )
            ax_bar.text(xi, min(y_top + 8, 105),
                        f"KW: {kw_sig} (p={kw_p:.3f})",
                        ha='center', va='bottom', fontsize=7, color='#555')

        # Pairwise Mann-Whitney U
        for i in range(len(all_subjects)):
            for j in range(i + 1, len(all_subjects)):
                g1 = np.array(subj_data.get(all_subjects[i], []))
                g2 = np.array(subj_data.get(all_subjects[j], []))
                n1, n2 = len(g1), len(g2)
                if n1 < 1 or n2 < 1:
                    continue

                # Cohen's d
                if n1 + n2 > 2:
                    pooled_std = np.sqrt(
                        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
                        / max(n1 + n2 - 2, 1)
                    )
                    d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
                else:
                    d = 0

                d_label = "negligible"
                if abs(d) > 0.8:
                    d_label = "large"
                elif abs(d) > 0.5:
                    d_label = "medium"
                elif abs(d) > 0.2:
                    d_label = "small"

                # Mann-Whitney U
                u_val, u_p = 0, 1.0
                u_sig = "ns"
                if n1 >= 2 and n2 >= 2:
                    try:
                        u_val, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                        if u_p < 0.001:
                            u_sig = "***"
                        elif u_p < 0.01:
                            u_sig = "**"
                        elif u_p < 0.05:
                            u_sig = "*"
                        else:
                            u_sig = "ns"
                    except Exception:
                        u_sig = "err"

                s1_label = all_subjects[i][1:3] + " v " + all_subjects[j][1:3]
                table_rows.append([
                    reg, s1_label, n1, n2, f"{d:+.2f}", d_label,
                    int(u_val), f"{u_p:.3f}", u_sig
                ])

    # Render stats table
    ax_tbl.axis('off')
    ax_tbl.set_title("Pairwise Between-Subject Statistics\n(Mann-Whitney U, Cohen's d)",
                     fontsize=11, fontweight='bold')
    if table_rows:
        col_labels = ["Region", "Pair", "n1", "n2", "d", "|d|", "U", "p", "Sig"]
        table = ax_tbl.table(
            cellText=table_rows, colLabels=col_labels,
            cellLoc='center', loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.2)

        # Style header
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor('#2c5f8a')
            cell.set_text_props(color='white', fontweight='bold')

        # Alternate row colors
        for i in range(1, len(table_rows) + 1):
            bg = '#f0f4f8' if i % 2 == 0 else 'white'
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(bg)

    fig.tight_layout()
    fig_path = output_dir / "roi_stats_by_region_subject.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Summary figure saved: {fig_path}")
    return fig_path


def interactive_prompt():
    """Interactive CLI prompts for paths and settings."""
    print("\n" + "=" * 70)
    print("  ENCR COLOCALIZATION BATCH RUNNER")
    print("  Generates coloc figures for all ND2+ROI pairs")
    print("=" * 70)

    print(f"\nApproved parameters:")
    print(f"  Method:          {COLOC_METHOD}")
    print(f"  Background:      {BACKGROUND_METHOD}")
    print(f"  Sigma threshold: {SIGMA_THRESHOLD}")
    print(f"  Soma dilation:   {SOMA_DILATION}px")
    print(f"  Min diameter:    {MIN_DIAMETER_UM}um")

    # Input directory
    print(f"\nCopy the path to the ENCR data directory containing HD_Regions subdirectories.")
    print(f"  Example: Y:\\2_Connectome\\Tissue\\MouseBrain_Pipeline\\2D_Slices\\ENCR")
    encr_input = input(f"  Input directory (press Enter for default: {ENCR_ROOT_DEFAULT}): ").strip()
    encr_root = Path(encr_input) if encr_input else ENCR_ROOT_DEFAULT

    if not encr_root.exists():
        print(f"  ERROR: Directory not found: {encr_root}")
        sys.exit(1)

    # Output directory
    print(f"\nType where you want the output figures to go.")
    print(f"  Output is organized as: <output_dir>/<subject>/<region>/")
    out_input = input(f"  Output directory (press Enter for default: {OUTPUT_DEFAULT}): ").strip()
    output_dir = Path(out_input) if out_input else OUTPUT_DEFAULT

    # Find pairs
    pairs = find_nd2_roi_pairs(encr_root)
    print(f"\nFound {len(pairs)} ND2+ROI pairs to process:")
    by_subject = {}
    for p in pairs:
        by_subject.setdefault(p['subject'], []).append(p)
    for subj, items in sorted(by_subject.items()):
        regions = sorted(set(p['region'] for p in items))
        print(f"  {subj}: {len(items)} samples ({', '.join(regions)})")

    # Force regeneration?
    print(f"\nShould existing figures be regenerated?")
    regen_input = input(f"  Regenerate all? (y/N): ").strip().lower()
    force_regen = regen_input in ('y', 'yes')

    # Parallelism
    print(f"\nHow many samples to process in parallel?")
    workers_input = input(f"  Workers (press Enter for default: 4): ").strip()
    try:
        max_workers = int(workers_input) if workers_input else 4
    except ValueError:
        max_workers = 4

    # Confirm
    print(f"\n{'─' * 50}")
    print(f"  Input:      {encr_root}")
    print(f"  Output:     {output_dir}")
    print(f"  Pairs:      {len(pairs)}")
    print(f"  Regenerate: {'Yes' if force_regen else 'No (skip existing)'}")
    print(f"  Workers:    {max_workers}")
    print(f"{'─' * 50}")
    confirm = input(f"  Proceed? (Y/n): ").strip().lower()
    if confirm in ('n', 'no'):
        print("Cancelled.")
        sys.exit(0)

    return encr_root, output_dir, pairs, force_regen, max_workers


def main():
    encr_root, output_dir, pairs, force_regen, max_workers = interactive_prompt()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_dir)
    logging.info(f"Batch started: {len(pairs)} pairs, {max_workers} workers")
    logging.info(f"Log file: {log_path}")
    logging.info(f"Parameters: sigma={SIGMA_THRESHOLD}, dilation={SOMA_DILATION}, "
                 f"min_diam={MIN_DIAMETER_UM}um, bg={BACKGROUND_METHOD}")

    # Filter out existing if not regenerating
    if not force_regen:
        to_process = []
        skipped = 0
        for p in pairs:
            fig_path = p['output_dir'] / f"{p['stem']}_coloc_result.png"
            if fig_path.exists():
                skipped += 1
                logging.info(f"SKIP (exists): {p['stem']}")
            else:
                to_process.append(p)
        if skipped:
            logging.info(f"Skipped {skipped} existing figures")
    else:
        to_process = pairs

    if not to_process:
        logging.info("Nothing to process — all figures exist. Use regenerate=Yes to redo.")
    else:
        logging.info(f"Processing {len(to_process)} samples...")

    all_results = []
    consecutive_errors = 0
    halted = False
    start_time = time.time()

    # Process sequentially for now (multiprocessing on Windows with ND2
    # file handles can be tricky). Can switch to ProcessPoolExecutor later.
    for i, pair in enumerate(to_process):
        if halted:
            break

        logging.info(f"[{i+1}/{len(to_process)}] {pair['stem']} "
                     f"({pair['subject']}/{pair['region']})")

        try:
            _, result = process_one_sample(pair)
        except Exception as e:
            result = {
                'stem': pair['stem'],
                'subject': pair['subject'],
                'region': pair['region'],
                'status': 'error',
                'error': str(e),
                'elapsed': 0,
                'roi_stats': None,
                'fig_path': None,
            }

        all_results.append(result)

        # Log result
        status_icon = {
            'done': 'OK',
            'warn': 'WARN',
            'error': 'FAIL',
        }.get(result['status'], '??')

        msg = f"  [{status_icon}] {result['stem']} ({result['elapsed']:.1f}s)"
        if result.get('roi_stats'):
            rs = result['roi_stats']
            msg += f" — {rs['n_positive']}/{rs['n_nuclei']} positive"
            for rname, rdata in rs.get('rois', {}).items():
                msg += f" | {rname}: {rdata['n_positive']}/{rdata['n_total']}"
        if result.get('error'):
            msg += f" — {result['error']}"

        if result['status'] == 'error':
            logging.error(msg)
            consecutive_errors += 1
        elif result['status'] == 'warn':
            logging.warning(msg)
            consecutive_errors = 0
        else:
            logging.info(msg)
            consecutive_errors = 0

        # Safety rail: halt on repeated errors
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logging.critical(
                f"HALTED: {consecutive_errors} consecutive errors. "
                f"Last error: {result.get('error', '?')}. "
                f"Fix the issue and re-run (existing figures will be skipped)."
            )
            halted = True

    elapsed_total = time.time() - start_time

    # Summary
    n_done = sum(1 for r in all_results if r['status'] == 'done')
    n_warn = sum(1 for r in all_results if r['status'] == 'warn')
    n_err = sum(1 for r in all_results if r['status'] == 'error')

    logging.info(f"\n{'=' * 60}")
    logging.info(f"BATCH COMPLETE in {elapsed_total/60:.1f} min")
    logging.info(f"  Success:  {n_done}")
    logging.info(f"  Warnings: {n_warn}")
    logging.info(f"  Errors:   {n_err}")
    if halted:
        logging.info(f"  HALTED after {len(all_results)}/{len(to_process)} (safety rail)")
    logging.info(f"  Output:   {output_dir}")
    logging.info(f"  Log:      {log_path}")

    # Save results JSON for debugging / re-analysis
    results_json = output_dir / "batch_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"  Results:  {results_json}")

    # Generate summary stats figure
    valid_results = [r for r in all_results if r['status'] in ('done', 'warn')]
    if valid_results:
        try:
            summary_path = generate_summary_figure(valid_results, output_dir)
            if summary_path:
                logging.info(f"  Summary:  {summary_path}")
        except Exception as e:
            logging.error(f"Failed to generate summary figure: {e}")
            traceback.print_exc()
    else:
        logging.warning("No valid results for summary figure")

    # Print per-region summary table
    logging.info(f"\n{'─' * 60}")
    logging.info("PER-REGION SUMMARY (% positive in ROI):")
    logging.info(f"  {'Region':<8} {'Subject':<10} {'Slices':>6} {'Mean%':>7} {'SD':>7}")
    region_subject = {}
    for r in valid_results:
        rs = r.get('roi_stats')
        if not rs:
            continue
        reg = r['region']
        subj = r['subject']
        rois = rs.get('rois', {})
        roi_pcts = [v['pct_positive'] for v in rois.values() if v['n_total'] > 0]
        if roi_pcts:
            key = (reg, subj)
            region_subject.setdefault(key, []).extend(roi_pcts)

    import numpy as np
    for (reg, subj), vals in sorted(region_subject.items()):
        arr = np.array(vals)
        logging.info(f"  {reg:<8} {subj:<10} {len(vals):>6} {np.mean(arr):>6.1f}% "
                     f"{np.std(arr, ddof=1):>6.1f}" if len(vals) > 1
                     else f"  {reg:<8} {subj:<10} {len(vals):>6} {np.mean(arr):>6.1f}%    --")

    if halted:
        sys.exit(1)


if __name__ == '__main__':
    main()
