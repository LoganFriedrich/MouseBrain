#!/usr/bin/env python3
"""
util_hemisphere_laterality.py

Generate a hemisphere laterality analysis CSV and graphs showing:
- Left vs Right cell counts per eLife group per brain
- Suspicious regions excluded (descending tracing filter)
- Paired t-test for L vs R across brains
- eLife cervical (C4) uninjured reference comparison

Output:
    hemisphere_laterality_analysis.csv
    reports/laterality_butterfly.png
    reports/laterality_LR_ratio.png
    reports/laterality_heatmap.png

Usage:
    python util_hemisphere_laterality.py
    python util_hemisphere_laterality.py --no-graphs
    python util_hemisphere_laterality.py --output my_analysis.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, str(Path(__file__).parent))

from mousebrain.config import DATA_SUMMARY_DIR
from elife_region_mapping import (
    aggregate_to_elife, ELIFE_GROUPS, TracingType,
    is_suspicious_region,
)
from util_compare_to_published import CERVICAL_REFERENCE, KEY_RECOVERY_REGIONS


def main():
    parser = argparse.ArgumentParser(description="Hemisphere laterality analysis")
    laterality_dir = DATA_SUMMARY_DIR / "laterality"
    parser.add_argument('--output', '-o', type=Path,
                        default=laterality_dir / "hemisphere_laterality_analysis.csv")
    parser.add_argument('--input', '-i', type=Path,
                        default=DATA_SUMMARY_DIR / "region_counts.csv")
    parser.add_argument('--no-graphs', action='store_true',
                        help='Skip generating graphs (CSV only)')
    parser.add_argument('--no-raw', action='store_true',
                        help='Skip raw Allen region analysis (eLife-grouped only)')
    args = parser.parse_args()

    # Load region_counts.csv
    with open(args.input, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Filter to brains with hemisphere data
    brains = []
    for row in all_rows:
        if row.get('total_left') and row.get('total_right'):
            brain_id = row.get('brain_id', row['brain'].split('_')[0])

            # Extract region counts by hemisphere (exclude suspicious)
            total_counts = {}
            left_counts = {}
            right_counts = {}
            for col, val in row.items():
                if col.startswith('region_left_'):
                    acr = col[len('region_left_'):]
                    try:
                        c = int(val) if val else 0
                        if c > 0 and not is_suspicious_region(acr):
                            left_counts[acr] = c
                    except (ValueError, TypeError):
                        pass
                elif col.startswith('region_right_'):
                    acr = col[len('region_right_'):]
                    try:
                        c = int(val) if val else 0
                        if c > 0 and not is_suspicious_region(acr):
                            right_counts[acr] = c
                    except (ValueError, TypeError):
                        pass
                elif col.startswith('region_'):
                    acr = col[7:]
                    try:
                        c = int(val) if val else 0
                        if c > 0 and not is_suspicious_region(acr):
                            total_counts[acr] = c
                    except (ValueError, TypeError):
                        pass

            brains.append({
                'id': brain_id,
                'name': row['brain'],
                'cohort': row.get('cohort', ''),
                'elife_total': aggregate_to_elife(total_counts),
                'elife_left': aggregate_to_elife(left_counts),
                'elife_right': aggregate_to_elife(right_counts),
                'raw_total': total_counts,
                'raw_left': left_counts,
                'raw_right': right_counts,
                'total': int(row['total_cells']),
                'total_left': int(row['total_left']),
                'total_right': int(row['total_right']),
            })

    print(f"Brains with hemisphere data: {len(brains)}")
    for b in brains:
        print(f"  {b['id']}: {b['total']:,} total, {b['total_left']:,} L, {b['total_right']:,} R")

    if not brains:
        print("ERROR: No brains with hemisphere data found")
        sys.exit(1)

    # Get all eLife groups sorted by ID
    all_groups = sorted(ELIFE_GROUPS.keys(), key=lambda x: ELIFE_GROUPS[x]['id'])

    rows_out = []

    for group in all_groups:
        ref_mean, ref_std, ref_n = CERVICAL_REFERENCE.get(group, (0, 0, 0))
        is_key = any(k in group for k in KEY_RECOVERY_REGIONS)

        row = {
            'elife_group': group,
            'elife_id': ELIFE_GROUPS[group]['id'],
            'key_recovery': 'YES' if is_key else '',
            'elife_ref_mean': ref_mean,
            'elife_ref_std': ref_std,
            'elife_ref_n': ref_n,
        }

        brain_lefts = []
        brain_rights = []
        brain_totals = []

        for b in brains:
            bid = b['id']
            l = b['elife_left'].get(group, {}).get('count', 0)
            r = b['elife_right'].get(group, {}).get('count', 0)
            t = b['elife_total'].get(group, {}).get('count', 0)

            row[f'{bid}_total'] = t
            row[f'{bid}_left'] = l
            row[f'{bid}_right'] = r
            if r > 0:
                row[f'{bid}_LR_ratio'] = round(l / r, 3)
            elif l > 0:
                row[f'{bid}_LR_ratio'] = 'inf'
            else:
                row[f'{bid}_LR_ratio'] = 'NA'

            brain_lefts.append(l)
            brain_rights.append(r)
            brain_totals.append(t)

        arr_l = np.array(brain_lefts, dtype=float)
        arr_r = np.array(brain_rights, dtype=float)
        arr_t = np.array(brain_totals, dtype=float)

        row['mean_total'] = round(np.mean(arr_t), 1)
        row['mean_left'] = round(np.mean(arr_l), 1)
        row['mean_right'] = round(np.mean(arr_r), 1)
        if np.mean(arr_r) > 0:
            row['mean_LR_ratio'] = round(np.mean(arr_l) / np.mean(arr_r), 3)
        else:
            row['mean_LR_ratio'] = 'NA'

        # vs eLife reference
        if ref_mean > 0:
            pct_vs_elife = ((np.mean(arr_t) - ref_mean) / ref_mean) * 100
            row['pct_vs_elife'] = f"{pct_vs_elife:+.1f}%"
        else:
            row['pct_vs_elife'] = ''

        # Paired t-test: L vs R across brains
        _add_lr_stats(row, arr_l, arr_r)

        rows_out.append(row)

    # TOTALS row
    totals_row = {
        'elife_group': 'TOTAL (eLife-mapped, suspicious excluded)',
        'elife_id': 99,
        'key_recovery': '',
        'elife_ref_mean': sum(v[0] for v in CERVICAL_REFERENCE.values()),
        'elife_ref_std': '',
        'elife_ref_n': 4,
    }

    tl_arr, tr_arr, tt_arr = [], [], []
    for b in brains:
        bid = b['id']
        t = sum(d.get('count', 0) for g, d in b['elife_total'].items() if g != '[Unmapped]')
        l = sum(d.get('count', 0) for g, d in b['elife_left'].items() if g != '[Unmapped]')
        r = sum(d.get('count', 0) for g, d in b['elife_right'].items() if g != '[Unmapped]')
        totals_row[f'{bid}_total'] = t
        totals_row[f'{bid}_left'] = l
        totals_row[f'{bid}_right'] = r
        totals_row[f'{bid}_LR_ratio'] = round(l / r, 3) if r > 0 else 'NA'
        tl_arr.append(l)
        tr_arr.append(r)
        tt_arr.append(t)

    tl = np.array(tl_arr, dtype=float)
    tr = np.array(tr_arr, dtype=float)
    tt = np.array(tt_arr, dtype=float)
    totals_row['mean_total'] = round(np.mean(tt), 1)
    totals_row['mean_left'] = round(np.mean(tl), 1)
    totals_row['mean_right'] = round(np.mean(tr), 1)
    totals_row['mean_LR_ratio'] = round(np.mean(tl) / np.mean(tr), 3) if np.mean(tr) > 0 else 'NA'

    ref_total = sum(v[0] for v in CERVICAL_REFERENCE.values())
    pct = ((np.mean(tt) - ref_total) / ref_total) * 100
    totals_row['pct_vs_elife'] = f"{pct:+.1f}%"

    _add_lr_stats(totals_row, tl, tr)
    rows_out.append(totals_row)

    # Write CSV
    fieldnames = ['elife_group', 'elife_id', 'key_recovery',
                  'elife_ref_mean', 'elife_ref_std', 'elife_ref_n']
    for b in brains:
        bid = b['id']
        fieldnames.extend([f'{bid}_total', f'{bid}_left', f'{bid}_right', f'{bid}_LR_ratio'])
    fieldnames.extend(['mean_total', 'mean_left', 'mean_right', 'mean_LR_ratio', 'pct_vs_elife'])
    fieldnames.extend(['LR_ttest_t', 'LR_ttest_p', 'LR_sig', 'LR_direction',
                       'LR_cohens_d', 'LR_effect'])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\nSaved: {args.output}")
    print(f"Regions: {len(rows_out) - 1} eLife groups + 1 total row")

    # Print summary table
    print()
    hdr = f"{'eLife Group':<42} {'Mean L':>7} {'Mean R':>7} {'L/R':>6} {'p-val':>7} {'Sig':>4} {'Dir':>4} {'d':>7} {'Effect':>10} {'vs eLife':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        name = r['elife_group'][:40]
        if r.get('key_recovery') == 'YES':
            name += ' *'
        ml = r.get('mean_left', '')
        mr = r.get('mean_right', '')
        lr = r.get('mean_LR_ratio', '')
        p = r.get('LR_ttest_p', '')
        sig = r.get('LR_sig', '')
        dirn = r.get('LR_direction', '')
        cd = r.get('LR_cohens_d', '')
        eff = r.get('LR_effect', '')
        ve = r.get('pct_vs_elife', '')
        print(f"{name:<42} {ml:>7} {mr:>7} {lr:>6} {p:>7} {sig:>4} {dirn:>4} {cd:>7} {eff:>10} {ve:>9}")

    print()
    print("Key: * = key recovery region")
    print("Sig: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, ns = not significant")
    print("Cohen's d (paired): |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large")
    print("Suspicious regions (cerebellar cortex, white matter, olfactory, cortical L1-3) excluded")
    print(f"Reference: Wang et al. (2022) eLife, cervical C4, uninjured, n=4")
    print(f"NOTE: n={len(brains)} brains — limited statistical power for paired t-test")

    # Print interpretation section
    _print_interpretation(rows_out, brains, args.output)

    # Generate graphs (same directory as CSV output)
    out_dir = args.output.parent
    if not args.no_graphs:
        if not HAS_MATPLOTLIB:
            print("\nWARNING: matplotlib not available, skipping graphs")
        else:
            generate_graphs(brains, rows_out, all_groups, out_dir)

    # Raw Allen region analysis (ungrouped)
    if not args.no_raw:
        raw_csv = out_dir / "hemisphere_laterality_raw.csv"
        _run_raw_analysis(brains, raw_csv, out_dir, args.no_graphs)


# Regions classified by proximity to the brain surface.
# Surface-adjacent regions are more susceptible to meningeal false positives.
SURFACE_ADJACENT_REGIONS = {
    "Gigantocellular Reticular Nucleus",
    "Magnocellular Reticular Nucleus",
    "Lateral Reticular Nuclei",
    "Periaqueductal Gray",
    "Hypothalamic Periventricular Zone",
    "Parabrachial / Pedunculopontine",
    "Dorsal Reticular Nucleus",
    "Vestibular Nuclei",
    "Medullary Trigeminal Area",
    "Corticospinal",
}

DEEP_REGIONS = {
    "Solitariospinal Area",
    "Medullary Reticular Nuclei",
    "Perihypoglossal Area",
    "Raphe Nuclei",
    "Red Nucleus",
    "Pontine Central Gray Area",
    "Midbrain Midline Nuclei",
    "Cerebellospinal Nuclei",
}


def _print_interpretation(rows_out, brains, csv_path):
    """Print and save an interpretation section analyzing laterality patterns."""
    data_rows = [r for r in rows_out if r['elife_id'] != 99]

    # Classify regions
    surface_r_gt_l = []
    surface_symmetric = []
    deep_r_gt_l = []
    deep_symmetric = []

    for r in data_rows:
        group = r['elife_group']
        lr = r.get('mean_LR_ratio', 'NA')
        if not isinstance(lr, (int, float)):
            continue
        ml = float(r.get('mean_left', 0) or 0)
        mr = float(r.get('mean_right', 0) or 0)
        if ml + mr < 10:  # skip near-zero regions
            continue

        cd = r.get('LR_cohens_d', '')
        eff = r.get('LR_effect', '')
        entry = (group, lr, ml, mr, r.get('LR_sig', ''), cd, eff)

        if group in SURFACE_ADJACENT_REGIONS:
            if lr < 0.9:
                surface_r_gt_l.append(entry)
            else:
                surface_symmetric.append(entry)
        elif group in DEEP_REGIONS:
            if lr < 0.9:
                deep_r_gt_l.append(entry)
            else:
                deep_symmetric.append(entry)

    # Count overall direction
    r_gt_l_count = sum(1 for r in data_rows
                       if isinstance(r.get('mean_LR_ratio'), (int, float))
                       and r['mean_LR_ratio'] < 0.95
                       and (float(r.get('mean_left', 0) or 0) + float(r.get('mean_right', 0) or 0)) > 10)
    l_gt_r_count = sum(1 for r in data_rows
                       if isinstance(r.get('mean_LR_ratio'), (int, float))
                       and r['mean_LR_ratio'] > 1.05
                       and (float(r.get('mean_left', 0) or 0) + float(r.get('mean_right', 0) or 0)) > 10)

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("INTERPRETATION: HEMISPHERE LATERALITY PATTERNS")
    lines.append("=" * 80)

    # Overall summary
    lines.append("")
    lines.append(f"Overall: {r_gt_l_count} regions show R > L bias (ratio < 0.95), "
                 f"{l_gt_r_count} show L > R bias (ratio > 1.05)")

    # Surface vs deep pattern
    lines.append("")
    lines.append("SURFACE-ADJACENT vs DEEP REGION COMPARISON:")
    lines.append("-" * 60)

    if surface_r_gt_l:
        lines.append("")
        lines.append(f"  Surface-adjacent regions with R > L bias ({len(surface_r_gt_l)}):")
        for group, lr, ml, mr, sig, cd, eff in sorted(surface_r_gt_l, key=lambda x: x[1]):
            sig_str = f" ({sig})" if sig and sig != 'ns' else ""
            d_str = f"  d={cd:.2f} [{eff}]" if isinstance(cd, (int, float)) else ""
            lines.append(f"    {group:<40} L/R = {lr:.3f}  "
                         f"(L={ml:.0f}, R={mr:.0f}){sig_str}{d_str}")

    if surface_symmetric:
        lines.append(f"  Surface-adjacent regions near-symmetric ({len(surface_symmetric)}):")
        for group, lr, ml, mr, sig, cd, eff in sorted(surface_symmetric, key=lambda x: x[1]):
            d_str = f"  d={cd:.2f}" if isinstance(cd, (int, float)) else ""
            lines.append(f"    {group:<40} L/R = {lr:.3f}{d_str}")

    lines.append("")
    if deep_r_gt_l:
        lines.append(f"  Deep regions with R > L bias ({len(deep_r_gt_l)}):")
        for group, lr, ml, mr, sig, cd, eff in sorted(deep_r_gt_l, key=lambda x: x[1]):
            d_str = f"  d={cd:.2f}" if isinstance(cd, (int, float)) else ""
            lines.append(f"    {group:<40} L/R = {lr:.3f}{d_str}")
    else:
        lines.append("  Deep regions with R > L bias: NONE")

    if deep_symmetric:
        lines.append(f"  Deep regions near-symmetric or L > R ({len(deep_symmetric)}):")
        for group, lr, ml, mr, sig, cd, eff in sorted(deep_symmetric, key=lambda x: x[1]):
            d_str = f"  d={cd:.2f}" if isinstance(cd, (int, float)) else ""
            lines.append(f"    {group:<40} L/R = {lr:.3f}{d_str}")

    # Meningeal artifact caveat
    lines.append("")
    lines.append("POTENTIAL MENINGEAL ARTIFACT:")
    lines.append("-" * 60)
    lines.append("  The R > L bias is concentrated in surface-adjacent brainstem")
    lines.append("  regions (ventral and lateral surfaces), while deep/midline")
    lines.append("  structures are near-symmetric. This spatial pattern is consistent")
    lines.append("  with residual meningeal tissue on the right brain surface causing")
    lines.append("  false-positive cell detections that get atlas-assigned to the")
    lines.append("  nearest legitimate brain region.")
    lines.append("")
    lines.append("  WHAT THE CURRENT FILTER CATCHES:")
    lines.append("    - Entire region types that shouldn't contain spinal-projecting")
    lines.append("      neurons (cerebellar cortex, white matter, olfactory, cortical")
    lines.append("      layers 1-3, hippocampus, basal ganglia)")
    lines.append("")
    lines.append("  WHAT IT CANNOT CATCH:")
    lines.append("    - Meningeal false positives that are spatially near legitimate")
    lines.append("      brainstem regions. These get atlas-registered to the nearest")
    lines.append("      brain region (e.g., GRN, IRN, MARN) and pass all filters.")
    lines.append("    - The filter works by region IDENTITY, not spatial LOCATION")
    lines.append("      (surface vs. deep within a region).")
    lines.append("")
    lines.append("  POSSIBLE NEXT STEPS:")
    lines.append("    1. Visual QC of right-hemisphere detections in napari, focusing")
    lines.append("       on brain surface areas in brainstem")
    lines.append("    2. Spatial analysis of per-cell coordinates from")
    lines.append("       all_points_information.csv — check if right-hemisphere cells")
    lines.append("       in surface-adjacent regions cluster near the brain boundary")
    lines.append("    3. Compare meningeal removal quality across brains")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    print(report_text)

    # Save as companion text file alongside CSV
    txt_path = csv_path.with_suffix('.txt')
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Hemisphere Laterality Analysis Report\n")
        f.write(f"n = {len(brains)} brains\n")
        f.write(f"Suspicious regions excluded (descending tracing filter)\n")
        f.write(f"Reference: Wang et al. (2022) eLife, cervical C4, uninjured, n=4\n\n")
        f.write(report_text)
    print(f"\nSaved interpretation: {txt_path}")


def _run_raw_analysis(brains, output_csv, reports_dir, no_graphs):
    """Run laterality analysis on individual Allen regions (no eLife grouping).

    Same statistics as the eLife-grouped analysis but at per-region granularity.
    Only includes regions with mean total >= 5 cells across brains.
    """
    print("\n" + "=" * 80)
    print("RAW ALLEN REGION LATERALITY ANALYSIS (ungrouped)")
    print("=" * 80)

    # Collect all regions across brains
    all_regions = set()
    for b in brains:
        all_regions.update(b['raw_left'].keys())
        all_regions.update(b['raw_right'].keys())
        all_regions.update(b['raw_total'].keys())

    all_regions = sorted(all_regions)
    print(f"Total Allen regions (suspicious excluded): {len(all_regions)}")

    # Build rows for each region
    rows_out = []
    min_count = 5  # minimum mean total to include

    for region in all_regions:
        brain_lefts = []
        brain_rights = []
        brain_totals = []

        row = {
            'region': region,
            'elife_group': '',
        }

        # Look up eLife group for context
        from elife_region_mapping import get_elife_group
        elife_grp = get_elife_group(region)
        if elife_grp:
            row['elife_group'] = elife_grp

        for b in brains:
            bid = b['id']
            l = b['raw_left'].get(region, 0)
            r = b['raw_right'].get(region, 0)
            t = b['raw_total'].get(region, 0)

            row[f'{bid}_total'] = t
            row[f'{bid}_left'] = l
            row[f'{bid}_right'] = r
            if r > 0:
                row[f'{bid}_LR_ratio'] = round(l / r, 3)
            elif l > 0:
                row[f'{bid}_LR_ratio'] = 'inf'
            else:
                row[f'{bid}_LR_ratio'] = 'NA'

            brain_lefts.append(l)
            brain_rights.append(r)
            brain_totals.append(t)

        arr_l = np.array(brain_lefts, dtype=float)
        arr_r = np.array(brain_rights, dtype=float)
        arr_t = np.array(brain_totals, dtype=float)

        mean_total = float(np.mean(arr_t))
        if mean_total < min_count:
            continue  # skip low-count regions

        row['mean_total'] = round(mean_total, 1)
        row['mean_left'] = round(np.mean(arr_l), 1)
        row['mean_right'] = round(np.mean(arr_r), 1)
        if np.mean(arr_r) > 0:
            row['mean_LR_ratio'] = round(np.mean(arr_l) / np.mean(arr_r), 3)
        else:
            row['mean_LR_ratio'] = 'NA'

        _add_lr_stats(row, arr_l, arr_r)
        rows_out.append(row)

    # Sort by mean total descending
    rows_out.sort(key=lambda r: float(r.get('mean_total', 0) or 0), reverse=True)

    print(f"Regions with mean total >= {min_count}: {len(rows_out)}")

    # Write CSV
    fieldnames = ['region', 'elife_group']
    for b in brains:
        bid = b['id']
        fieldnames.extend([f'{bid}_total', f'{bid}_left', f'{bid}_right', f'{bid}_LR_ratio'])
    fieldnames.extend(['mean_total', 'mean_left', 'mean_right', 'mean_LR_ratio'])
    fieldnames.extend(['LR_ttest_t', 'LR_ttest_p', 'LR_sig', 'LR_direction',
                       'LR_cohens_d', 'LR_effect'])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"Saved: {output_csv}")

    # Print summary table (top regions by effect size)
    sig_rows = [r for r in rows_out
                if isinstance(r.get('LR_cohens_d'), (int, float))
                and abs(r['LR_cohens_d']) >= 0.5]
    sig_rows.sort(key=lambda r: abs(r['LR_cohens_d']), reverse=True)

    print(f"\nAllen regions with |Cohen's d| >= 0.5 (medium+ effect): {len(sig_rows)}")
    print()
    hdr = f"{'Region':<20} {'eLife Group':<30} {'Mean L':>7} {'Mean R':>7} {'L/R':>6} {'d':>7} {'Effect':>10} {'Dir':>4} {'Sig':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in sig_rows[:40]:  # top 40
        name = r['region'][:18]
        grp = r.get('elife_group', '')[:28]
        ml = r.get('mean_left', '')
        mr = r.get('mean_right', '')
        lr = r.get('mean_LR_ratio', '')
        cd = r.get('LR_cohens_d', '')
        eff = r.get('LR_effect', '')
        dirn = r.get('LR_direction', '')
        sig = r.get('LR_sig', '')
        print(f"{name:<20} {grp:<30} {ml:>7} {mr:>7} {lr:>6} {cd:>7} {eff:>10} {dirn:>4} {sig:>4}")

    # Count R>L vs L>R
    r_gt_l = sum(1 for r in rows_out
                 if isinstance(r.get('mean_LR_ratio'), (int, float)) and r['mean_LR_ratio'] < 0.9)
    l_gt_r = sum(1 for r in rows_out
                 if isinstance(r.get('mean_LR_ratio'), (int, float)) and r['mean_LR_ratio'] > 1.1)
    sym = len(rows_out) - r_gt_l - l_gt_r
    print(f"\nOverall: {r_gt_l} R>L (ratio<0.9), {sym} near-symmetric, {l_gt_r} L>R (ratio>1.1)")

    # Graphs (if enabled and matplotlib available)
    if not no_graphs and HAS_MATPLOTLIB:
        reports_dir.mkdir(parents=True, exist_ok=True)
        # Filter to top regions for readable graphs
        top_n = 35
        top_rows = rows_out[:top_n]
        group_names = [r['region'] for r in top_rows]
        _graph_raw_butterfly(group_names, top_rows, brains, reports_dir)
        _graph_raw_heatmap(group_names, top_rows, brains, reports_dir)

    print("=" * 80)


def _graph_raw_butterfly(group_names, active_rows, brains, reports_dir):
    """Butterfly chart for raw Allen regions."""
    n = len(group_names)
    mean_lefts = [float(r.get('mean_left', 0) or 0) for r in active_rows]
    mean_rights = [float(r.get('mean_right', 0) or 0) for r in active_rows]

    fig, ax = plt.subplots(figsize=(14, max(8, n * 0.4)))
    y = np.arange(n)

    # Color bars by eLife group membership
    colors_l = []
    colors_r = []
    for r in active_rows:
        grp = r.get('elife_group', '')
        if grp:
            colors_l.append('#4A90D9')
            colors_r.append('#D94A4A')
        else:
            colors_l.append('#8CBAE8')  # lighter for unmapped
            colors_r.append('#E89090')

    ax.barh(y, [-v for v in mean_lefts], height=0.7,
            color=colors_l, edgecolor='#333333', linewidth=0.3,
            label='Left Hemisphere', zorder=3)
    ax.barh(y, mean_rights, height=0.7,
            color=colors_r, edgecolor='#333333', linewidth=0.3,
            label='Right Hemisphere', zorder=3)

    # Individual brain points
    brain_colors = ['#1a5276', '#7d3c98', '#117a65']
    for bi, b in enumerate(brains):
        for gi, row in enumerate(active_rows):
            region = row['region']
            l_val = b['raw_left'].get(region, 0)
            r_val = b['raw_right'].get(region, 0)
            color = brain_colors[bi % len(brain_colors)]
            ax.plot(-l_val, gi, 'o', color=color, markersize=3, alpha=0.4, zorder=4)
            ax.plot(r_val, gi, 'o', color=color, markersize=3, alpha=0.4, zorder=4)

    # Labels with eLife group annotation
    labels = []
    for r in active_rows:
        name = r['region']
        grp = r.get('elife_group', '')
        if grp:
            labels.append(f"{name}  ({grp[:20]})")
        else:
            labels.append(f"{name}  [unmapped]")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=1, zorder=5)
    ax.set_xlabel('Mean Cell Count', fontsize=11)
    ax.set_title('Raw Allen Region Laterality: Left vs Right (top {} by count)\n'
                 '(suspicious regions excluded, n={} brains)'.format(n, len(brains)),
                 fontsize=13, fontweight='bold')

    xlim = ax.get_xlim()
    ax.text(xlim[0] * 0.5, -1.2, 'LEFT', ha='center', fontsize=12,
            fontweight='bold', color='#4A90D9')
    ax.text(xlim[1] * 0.5, -1.2, 'RIGHT', ha='center', fontsize=12,
            fontweight='bold', color='#D94A4A')

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.02, 0.01,
             'Lighter bars = regions not mapped to eLife groups | Small dots = individual brains\n'
             'CAVEAT: R > L bias in surface-adjacent regions may reflect meningeal artifact',
             fontsize=7, color='#666666', va='bottom')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = reports_dir / 'laterality_raw_butterfly.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def _graph_raw_heatmap(group_names, active_rows, brains, reports_dir):
    """Heatmap of L/R ratios for raw Allen regions."""
    n_groups = len(group_names)
    brain_ids = [b['id'] for b in brains]
    col_labels = brain_ids + ['Mean']
    n_cols = len(col_labels)

    ratio_matrix = np.full((n_groups, n_cols), np.nan)
    for gi, row in enumerate(active_rows):
        region = row['region']
        for bi, b in enumerate(brains):
            l = b['raw_left'].get(region, 0)
            r = b['raw_right'].get(region, 0)
            if r > 0:
                ratio_matrix[gi, bi] = l / r
            elif l > 0:
                ratio_matrix[gi, bi] = 2.5
        mean_lr = row.get('mean_LR_ratio', 'NA')
        if isinstance(mean_lr, (int, float)):
            ratio_matrix[gi, -1] = mean_lr

    # Labels with eLife group
    labels = []
    for r in active_rows:
        grp = r.get('elife_group', '')
        if grp:
            labels.append(f"{r['region']}  ({grp[:15]})")
        else:
            labels.append(r['region'])

    vmin, vmax = 0.3, 2.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdBu

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.5 + 4), max(8, n_groups * 0.38)))

    im = ax.imshow(ratio_matrix, cmap=cmap, norm=norm, aspect='auto',
                   interpolation='nearest')

    for gi in range(n_groups):
        for ci in range(n_cols):
            val = ratio_matrix[gi, ci]
            if np.isnan(val):
                ax.text(ci, gi, '-', ha='center', va='center',
                        fontsize=7, color='#aaaaaa')
            else:
                text_color = 'white' if abs(val - 1.0) > 0.5 else 'black'
                fontweight = 'bold' if ci == n_cols - 1 else 'normal'
                ax.text(ci, gi, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=text_color, fontweight=fontweight)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_groups + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.axvline(n_cols - 1.5, color='black', linewidth=2)

    ax.set_title('Raw Allen Region L/R Ratio Heatmap (top {} by count)\n'
                 '(Blue = L > R, Red = R > L, White = symmetric)'.format(n_groups),
                 fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Left / Right Ratio', fontsize=10)
    cbar.set_ticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75])

    fig.text(0.02, 0.01,
             'Parenthesized = eLife group | Bold column = mean across brains\n'
             'CAVEAT: R > L bias in surface-adjacent regions may reflect meningeal artifact',
             fontsize=7, color='#666666', va='bottom')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = reports_dir / 'laterality_raw_heatmap.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_graphs(brains, rows_out, all_groups, reports_dir):
    """Generate laterality analysis graphs.

    Creates three figures:
    1. Butterfly chart: mean L/R counts as diverging horizontal bars
    2. L/R ratio dot plot: per-group ratios with individual brain points
    3. Heatmap: L/R ratios across groups and brains
    """
    # Filter to data rows only (exclude TOTAL row) and groups with nonzero counts
    data_rows = [r for r in rows_out if r['elife_id'] != 99]
    active_rows = [r for r in data_rows
                   if (r.get('mean_left', 0) or 0) > 0 or (r.get('mean_right', 0) or 0) > 0]

    # Sort by total count descending for butterfly and heatmap
    active_rows.sort(key=lambda r: float(r.get('mean_total', 0) or 0), reverse=True)

    group_names = [r['elife_group'] for r in active_rows]
    mean_lefts = [float(r.get('mean_left', 0) or 0) for r in active_rows]
    mean_rights = [float(r.get('mean_right', 0) or 0) for r in active_rows]
    sig_markers = [r.get('LR_sig', '') for r in active_rows]
    directions = [r.get('LR_direction', '') for r in active_rows]
    is_key = [r.get('key_recovery', '') == 'YES' for r in active_rows]

    # =========================================================================
    # Graph 1: Butterfly Chart (Diverging Horizontal Bar)
    # =========================================================================
    _graph_butterfly(group_names, mean_lefts, mean_rights, sig_markers,
                     directions, is_key, brains, active_rows, reports_dir)

    # =========================================================================
    # Graph 2: L/R Ratio Dot Plot (Forest Plot Style)
    # =========================================================================
    _graph_lr_ratio(group_names, active_rows, brains, reports_dir)

    # =========================================================================
    # Graph 3: Heatmap of L/R Ratios
    # =========================================================================
    _graph_heatmap(group_names, active_rows, brains, reports_dir)


def _graph_butterfly(group_names, mean_lefts, mean_rights, sig_markers,
                     directions, is_key, brains, active_rows, reports_dir):
    """Butterfly chart: left hemisphere extends left, right extends right."""
    n = len(group_names)
    fig, ax = plt.subplots(figsize=(14, max(8, n * 0.45)))

    y = np.arange(n)

    # Shorten long names
    short_names = []
    for i, name in enumerate(group_names):
        label = name[:35]
        if is_key[i]:
            label += ' *'
        short_names.append(label)

    # Left hemisphere: negative direction (extends left)
    bars_l = ax.barh(y, [-v for v in mean_lefts], height=0.7,
                     color='#4A90D9', edgecolor='#2C5F8A', linewidth=0.5,
                     label='Left Hemisphere', zorder=3)

    # Right hemisphere: positive direction (extends right)
    bars_r = ax.barh(y, mean_rights, height=0.7,
                     color='#D94A4A', edgecolor='#8A2C2C', linewidth=0.5,
                     label='Right Hemisphere', zorder=3)

    # Individual brain data points
    brain_colors = ['#1a5276', '#7d3c98', '#117a65']
    for bi, b in enumerate(brains):
        for gi, row in enumerate(active_rows):
            group = row['elife_group']
            l_val = b['elife_left'].get(group, {}).get('count', 0)
            r_val = b['elife_right'].get(group, {}).get('count', 0)
            color = brain_colors[bi % len(brain_colors)]
            ax.plot(-l_val, gi, 'o', color=color, markersize=3, alpha=0.5, zorder=4)
            ax.plot(r_val, gi, 'o', color=color, markersize=3, alpha=0.5, zorder=4)

    # Significance markers
    max_val = max(max(mean_lefts), max(mean_rights)) if mean_lefts else 1
    for i, sig in enumerate(sig_markers):
        if sig and sig not in ('', 'ns'):
            # Place marker at the end of whichever side is larger
            if directions[i] == 'L>R':
                x_pos = -(mean_lefts[i] + max_val * 0.03)
            else:
                x_pos = mean_rights[i] + max_val * 0.03
            ax.text(x_pos, i, sig, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#333333')

    # Styling
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=1, zorder=5)
    ax.set_xlabel('Mean Cell Count', fontsize=11)
    ax.set_title('Hemisphere Laterality: Left vs Right Cell Counts by eLife Region\n'
                 '(suspicious regions excluded, n={} brains)'.format(len(brains)),
                 fontsize=13, fontweight='bold')

    # Add L/R labels at top
    xlim = ax.get_xlim()
    ax.text(xlim[0] * 0.5, -1.2, 'LEFT', ha='center', fontsize=12,
            fontweight='bold', color='#4A90D9')
    ax.text(xlim[1] * 0.5, -1.2, 'RIGHT', ha='center', fontsize=12,
            fontweight='bold', color='#D94A4A')

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Footnotes
    fig.text(0.02, 0.01,
             '* = Key recovery region | Sig: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1\n'
             'Small dots = individual brains | Reference: Wang et al. (2022) eLife, C4 uninjured\n'
             'CAVEAT: R > L bias in surface-adjacent regions may reflect meningeal artifact (see report)',
             fontsize=7, color='#666666', va='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = reports_dir / 'laterality_butterfly.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def _graph_lr_ratio(group_names, active_rows, brains, reports_dir):
    """Forest-plot-style L/R ratio chart with individual brain dots."""
    n = len(group_names)
    fig, ax = plt.subplots(figsize=(10, max(8, n * 0.45)))

    y = np.arange(n)
    is_key = [r.get('key_recovery', '') == 'YES' for r in active_rows]

    # X-axis cap for readability (outliers annotated with arrows)
    x_cap = 3.0

    # Compute per-brain ratios for each group
    for gi, row in enumerate(active_rows):
        group = row['elife_group']
        brain_ratios = []
        for b in brains:
            l = b['elife_left'].get(group, {}).get('count', 0)
            r = b['elife_right'].get(group, {}).get('count', 0)
            if r > 0:
                brain_ratios.append(l / r)
            elif l > 0:
                brain_ratios.append(999)  # placeholder for inf

        # Individual brain points
        for bi, ratio in enumerate(brain_ratios):
            color = '#555555'
            if ratio > x_cap:
                # Outlier: draw at cap with arrow indicator
                ax.plot(x_cap - 0.05, gi, '>', color='#D94A4A', markersize=6,
                        alpha=0.7, zorder=4)
                ax.text(x_cap + 0.02, gi, f'{ratio:.1f}', fontsize=6,
                        color='#D94A4A', va='center', zorder=4)
            elif ratio < 1 / x_cap:
                ax.plot(1 / x_cap + 0.05, gi, '<', color='#4A90D9', markersize=6,
                        alpha=0.7, zorder=4)
            else:
                ax.plot(ratio, gi, 'o', color=color, markersize=5, alpha=0.5, zorder=3)

        # Mean ratio (larger marker)
        mean_lr = row.get('mean_LR_ratio', 'NA')
        if isinstance(mean_lr, (int, float)):
            face_color = '#D94A4A' if mean_lr < 1.0 else '#4A90D9' if mean_lr > 1.0 else '#888888'
            edge_color = '#8A2C2C' if mean_lr < 1.0 else '#2C5F8A' if mean_lr > 1.0 else '#555555'
            marker_size = 10 if is_key[gi] else 8
            plot_x = min(mean_lr, x_cap - 0.1)
            ax.plot(plot_x, gi, 'D', color=face_color, markeredgecolor=edge_color,
                    markersize=marker_size, markeredgewidth=1.2, zorder=5)

            # Significance annotation
            sig = row.get('LR_sig', '')
            if sig and sig not in ('', 'ns'):
                ax.text(plot_x + 0.08, gi - 0.15, sig,
                        fontsize=8, fontweight='bold', color='#333333', zorder=6)

    # Reference line at 1.0 (perfect symmetry)
    ax.axvline(1.0, color='#333333', linewidth=1.5, linestyle='--', zorder=2, alpha=0.8)
    ax.set_xlim(0, x_cap)

    # Labels
    short_names = []
    for i, name in enumerate(group_names):
        label = name[:35]
        if is_key[i]:
            label += ' *'
        short_names.append(label)

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Left / Right Ratio', fontsize=11)
    ax.set_title('Hemisphere Laterality Index (L/R Ratio) by eLife Region\n'
                 '(ratio > 1 = more Left, ratio < 1 = more Right)',
                 fontsize=13, fontweight='bold')

    # Color zones
    ax.axvspan(0, 1.0, alpha=0.05, color='#D94A4A', zorder=0)
    ax.axvspan(1.0, x_cap, alpha=0.05, color='#4A90D9', zorder=0)
    ax.text(0.97, -0.8, 'R > L', ha='right', fontsize=10, color='#D94A4A',
            fontweight='bold')
    ax.text(1.03, -0.8, 'L > R', ha='left', fontsize=10, color='#4A90D9',
            fontweight='bold')

    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.02, 0.01,
             '* = Key recovery region | Diamonds = mean across brains | Circles = individual brains\n'
             'Dashed line = perfect symmetry (L/R = 1.0) | Sig: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1\n'
             'CAVEAT: R > L bias in surface-adjacent regions may reflect meningeal artifact (see report)',
             fontsize=7, color='#666666', va='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = reports_dir / 'laterality_LR_ratio.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def _graph_heatmap(group_names, active_rows, brains, reports_dir):
    """Heatmap of L/R ratios: rows=regions, columns=brains + mean."""
    n_groups = len(group_names)
    brain_ids = [b['id'] for b in brains]
    col_labels = brain_ids + ['Mean']
    n_cols = len(col_labels)

    # Build ratio matrix
    ratio_matrix = np.full((n_groups, n_cols), np.nan)
    for gi, row in enumerate(active_rows):
        group = row['elife_group']
        for bi, b in enumerate(brains):
            l = b['elife_left'].get(group, {}).get('count', 0)
            r = b['elife_right'].get(group, {}).get('count', 0)
            if r > 0:
                ratio_matrix[gi, bi] = l / r
            elif l > 0:
                ratio_matrix[gi, bi] = 2.5  # cap
            # else stays NaN

        mean_lr = row.get('mean_LR_ratio', 'NA')
        if isinstance(mean_lr, (int, float)):
            ratio_matrix[gi, -1] = mean_lr

    # Short names for y-axis
    is_key = [r.get('key_recovery', '') == 'YES' for r in active_rows]
    short_names = []
    for i, name in enumerate(group_names):
        label = name[:35]
        if is_key[i]:
            label += ' *'
        short_names.append(label)

    # Diverging colormap centered at 1.0
    vmin, vmax = 0.4, 1.8
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdBu  # Red = R>L (ratio<1), Blue = L>R (ratio>1)

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.5 + 3), max(8, n_groups * 0.42)))

    im = ax.imshow(ratio_matrix, cmap=cmap, norm=norm, aspect='auto',
                   interpolation='nearest')

    # Annotate cells
    for gi in range(n_groups):
        for ci in range(n_cols):
            val = ratio_matrix[gi, ci]
            if np.isnan(val):
                ax.text(ci, gi, '-', ha='center', va='center',
                        fontsize=8, color='#aaaaaa')
            else:
                # Text color: dark on light backgrounds, light on dark
                text_color = 'white' if abs(val - 1.0) > 0.4 else 'black'
                fontweight = 'bold' if ci == n_cols - 1 else 'normal'
                ax.text(ci, gi, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight=fontweight)

    # Significance column (rightmost extra column as text)
    sig_labels = [r.get('LR_sig', '') for r in active_rows]

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels(short_names, fontsize=9)

    # Add significance as text to the right of the heatmap
    for gi, sig in enumerate(sig_labels):
        if sig and sig not in ('', 'ns'):
            ax.text(n_cols - 0.3, gi, sig, ha='left', va='center',
                    fontsize=8, fontweight='bold', color='#333333')

    # Gridlines between cells
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_groups + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    # Separate mean column with thicker line
    ax.axvline(n_cols - 1.5, color='black', linewidth=2)

    ax.set_title('L/R Ratio Heatmap by eLife Region and Brain\n'
                 '(Blue = L > R, Red = R > L, White = symmetric)',
                 fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Left / Right Ratio', fontsize=10)
    cbar.set_ticks([0.5, 0.75, 1.0, 1.25, 1.5])

    fig.text(0.02, 0.01,
             '* = Key recovery region | Bold column = mean across brains\n'
             'Sig: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1 (paired t-test)\n'
             'CAVEAT: R > L bias in surface-adjacent regions may reflect meningeal artifact (see report)',
             fontsize=7, color='#666666', va='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = reports_dir / 'laterality_heatmap.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def _add_lr_stats(row, arr_l, arr_r):
    """Add L vs R statistical test results to a row dict.

    Includes paired t-test and Cohen's d (paired/dz):
        d = mean(L - R) / std(L - R)
    Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large
    """
    if len(arr_l) >= 2 and (np.sum(arr_l) + np.sum(arr_r)) > 0:
        try:
            t_stat, p_val = sp_stats.ttest_rel(arr_l, arr_r)
            row['LR_ttest_t'] = round(t_stat, 3)
            row['LR_ttest_p'] = round(p_val, 4)
            if p_val < 0.001:
                row['LR_sig'] = '***'
            elif p_val < 0.01:
                row['LR_sig'] = '**'
            elif p_val < 0.05:
                row['LR_sig'] = '*'
            elif p_val < 0.1:
                row['LR_sig'] = '.'
            else:
                row['LR_sig'] = 'ns'

            if np.mean(arr_l) > np.mean(arr_r):
                row['LR_direction'] = 'L>R'
            elif np.mean(arr_r) > np.mean(arr_l):
                row['LR_direction'] = 'R>L'
            else:
                row['LR_direction'] = 'L=R'

            # Cohen's d (paired / dz)
            diff = arr_l - arr_r
            sd_diff = np.std(diff, ddof=1)
            if sd_diff > 0:
                d = float(np.mean(diff) / sd_diff)
                row['LR_cohens_d'] = round(d, 3)
                if abs(d) >= 0.8:
                    row['LR_effect'] = 'large'
                elif abs(d) >= 0.5:
                    row['LR_effect'] = 'medium'
                elif abs(d) >= 0.2:
                    row['LR_effect'] = 'small'
                else:
                    row['LR_effect'] = 'negligible'
            else:
                row['LR_cohens_d'] = 0.0
                row['LR_effect'] = 'negligible'
        except Exception:
            row['LR_ttest_t'] = ''
            row['LR_ttest_p'] = ''
            row['LR_sig'] = ''
            row['LR_direction'] = ''
            row['LR_cohens_d'] = ''
            row['LR_effect'] = ''
    else:
        row['LR_ttest_t'] = ''
        row['LR_ttest_p'] = ''
        row['LR_sig'] = ''
        row['LR_direction'] = ''
        row['LR_cohens_d'] = ''
        row['LR_effect'] = ''


if __name__ == '__main__':
    main()
