#!/usr/bin/env python3
"""
util_generate_elife_report.py

Generate eLife comparison reports with suspicious region filtering.

This script:
1. Loads brain counts from region_counts.csv
2. Groups by eLife 25 categories
3. Applies suspicious region filtering (for descending tracing)
4. Compares to eLife cervical reference
5. Outputs reports with both raw and corrected totals

Usage:
    # Generate report for all brains in region_counts.csv
    python util_generate_elife_report.py

    # Generate for specific brains
    python util_generate_elife_report.py --brains 349 357 367 368

    # For ATLAS tracing (no filtering)
    python util_generate_elife_report.py --ascending
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from mousebrain.config import DATA_SUMMARY_DIR
from elife_region_mapping import (
    TracingType,
    ELIFE_GROUPS,
    aggregate_to_elife,
    get_suspicious_count,
    get_corrected_total,
    DESCENDING_SUSPICIOUS_REGIONS,
)
from util_compare_to_published import CERVICAL_REFERENCE, KEY_RECOVERY_REGIONS


def load_region_counts_csv(csv_path: Path) -> List[dict]:
    """Load all brain rows from region_counts.csv."""
    brains = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            brains.append(row)
    return brains


def extract_allen_counts(row: dict) -> Dict[str, int]:
    """Extract Allen region counts from a CSV row.

    Skips hemisphere-specific columns (region_left_*, region_right_*)
    to avoid creating phantom regions like 'left_GRN'.
    """
    counts = {}
    for key, value in row.items():
        if key.startswith('region_left_') or key.startswith('region_right_'):
            continue
        if key.startswith('region_'):
            acronym = key.replace('region_', '')
            try:
                count = int(value) if value else 0
                if count > 0:
                    counts[acronym] = count
            except (ValueError, TypeError):
                pass
    return counts


def generate_elife_comparison_csv(
    brains: List[dict],
    output_path: Path,
    tracing_type: TracingType = TracingType.DESCENDING
):
    """Generate the main eLife comparison CSV."""

    # Process each brain
    brain_data = []
    for row in brains:
        brain_name = row.get('brain', '')
        brain_id = row.get('brain_id', brain_name.split('_')[0] if brain_name else '')

        allen_counts = extract_allen_counts(row)
        elife_grouped = aggregate_to_elife(allen_counts)
        suspicious = get_suspicious_count(allen_counts, tracing_type)

        brain_data.append({
            'name': brain_name,
            'id': brain_id,
            'allen_counts': allen_counts,
            'elife_grouped': elife_grouped,
            'suspicious': suspicious,
            'raw_total': suspicious['total'],
            'corrected_total': suspicious['total_legitimate'],
        })

    # Build comparison CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Build header
        fieldnames = ['Region_Group', 'eLife_Mean', 'eLife_StdDev']
        for bd in brain_data:
            fieldnames.append(f"Brain_{bd['id']}")
        for bd in brain_data:
            fieldnames.append(f"{bd['id']}_vs_eLife")
        fieldnames.append('Notes')

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Sort regions: key recovery first, then alphabetically
        all_regions = list(CERVICAL_REFERENCE.keys())
        key_first = [r for r in all_regions if any(k in r for k in KEY_RECOVERY_REGIONS)]
        other = [r for r in all_regions if r not in key_first]
        sorted_regions = key_first + sorted(other)

        # Add notes for key regions (names match elife_region_mapping.py)
        notes_map = {
            "Red Nucleus": "Key for recovery",
            "Gigantocellular Reticular Nucleus": "Key for recovery",
            "Parabrachial / Pedunculopontine": "Key for recovery",
            "Pontine Central Gray Area": "Key for recovery",
            "Corticospinal": "Key for recovery",
            "Lateral Reticular Nuclei": "IRN+LRNm+PARN etc",
            "Medullary Reticular Nuclei": "MDRNd+MDRNv",
            "Pontine Reticular Nuclei": "PRNr+PRNc",
            "Periaqueductal Gray": "PAG",
            "Midbrain Reticular Nuclei": "MRN",
            "Magnocellular Reticular Nucleus": "MARN",
        }

        for region in sorted_regions:
            ref_mean, ref_std, n = CERVICAL_REFERENCE.get(region, (0, 0, 0))

            row = {
                'Region_Group': region + ('*' if any(k in region for k in KEY_RECOVERY_REGIONS) else ''),
                'eLife_Mean': ref_mean,
                'eLife_StdDev': ref_std,
                'Notes': notes_map.get(region, ''),
            }

            # Add brain counts
            for bd in brain_data:
                elife_count = bd['elife_grouped'].get(region, {}).get('count', 0)
                row[f"Brain_{bd['id']}"] = elife_count

                # Calculate vs eLife
                if ref_mean > 0:
                    pct_diff = ((elife_count - ref_mean) / ref_mean) * 100
                    row[f"{bd['id']}_vs_eLife"] = f"{pct_diff:+.0f}%"
                else:
                    row[f"{bd['id']}_vs_eLife"] = ""

            writer.writerow(row)

        # Add totals row
        ref_total = sum(v[0] for v in CERVICAL_REFERENCE.values())
        totals_row = {
            'Region_Group': 'TOTAL (eLife-mapped)',
            'eLife_Mean': ref_total,
            'eLife_StdDev': '',
            'Notes': '',
        }
        for bd in brain_data:
            elife_total = sum(
                d.get('count', 0) for g, d in bd['elife_grouped'].items()
                if g != '[Unmapped]'
            )
            totals_row[f"Brain_{bd['id']}"] = elife_total
            if ref_total > 0:
                pct_diff = ((elife_total - ref_total) / ref_total) * 100
                totals_row[f"{bd['id']}_vs_eLife"] = f"{pct_diff:+.0f}%"
        writer.writerow(totals_row)

        # Add raw totals row
        raw_row = {
            'Region_Group': 'RAW TOTAL (all detections)',
            'eLife_Mean': '',
            'eLife_StdDev': '',
            'Notes': 'Before suspicious filtering',
        }
        for bd in brain_data:
            raw_row[f"Brain_{bd['id']}"] = bd['raw_total']
            raw_row[f"{bd['id']}_vs_eLife"] = ''
        writer.writerow(raw_row)

        # Add corrected totals row (descending tracing only)
        if tracing_type == TracingType.DESCENDING:
            corrected_row = {
                'Region_Group': 'CORRECTED TOTAL (excluding suspicious)',
                'eLife_Mean': '',
                'eLife_StdDev': '',
                'Notes': 'Excludes cerebellar cortex, white matter, olfactory, cortical L1-3',
            }
            for bd in brain_data:
                corrected_row[f"Brain_{bd['id']}"] = bd['corrected_total']
                corrected_row[f"{bd['id']}_vs_eLife"] = ''
            writer.writerow(corrected_row)

            # Add suspicious breakdown
            f.write('\n# SUSPICIOUS REGION BREAKDOWN (likely false positives)\n')
            for cat, cat_data in DESCENDING_SUSPICIOUS_REGIONS.items():
                sus_row = {
                    'Region_Group': f'Suspicious: {cat}',
                    'eLife_Mean': '',
                    'eLife_StdDev': '',
                    'Notes': cat_data['description'],
                }
                for bd in brain_data:
                    sus_count = bd['suspicious']['suspicious'].get(cat, {}).get('count', 0)
                    sus_row[f"Brain_{bd['id']}"] = sus_count
                    sus_row[f"{bd['id']}_vs_eLife"] = ''
                writer.writerow(sus_row)

    print(f"  Saved: {output_path}")
    return brain_data


def generate_summary_text(brain_data: List[dict], tracing_type: TracingType) -> str:
    """Generate a text summary of the comparison."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"ELIFE COMPARISON SUMMARY - {tracing_type.value.upper()} TRACING")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)
    lines.append("")

    ref_total = sum(v[0] for v in CERVICAL_REFERENCE.values())
    lines.append(f"eLife Reference (cervical C4, uninjured, n=4): {ref_total:,} cells")
    lines.append("")

    lines.append("BRAIN COUNTS:")
    lines.append(f"{'Brain':<20} {'Raw':>10} {'Corrected':>10} {'eLife-mapped':>12} {'vs eLife':>10}")
    lines.append("-" * 65)

    for bd in brain_data:
        elife_total = sum(
            d.get('count', 0) for g, d in bd['elife_grouped'].items()
            if g != '[Unmapped]'
        )
        pct = ((elife_total - ref_total) / ref_total) * 100 if ref_total else 0

        lines.append(
            f"{bd['id']:<20} {bd['raw_total']:>10,} {bd['corrected_total']:>10,} "
            f"{elife_total:>12,} {pct:>+9.0f}%"
        )

    lines.append("")

    if tracing_type == TracingType.DESCENDING:
        lines.append("SUSPICIOUS REGION SUMMARY:")
        for bd in brain_data:
            sus = bd['suspicious']
            sus_pct = (sus['total_suspicious'] / sus['total'] * 100) if sus['total'] else 0
            lines.append(f"  {bd['id']}: {sus['total_suspicious']:,} suspicious ({sus_pct:.1f}% of raw)")

        lines.append("")
        lines.append("NOTE: Suspicious regions include cerebellar cortex, white matter,")
        lines.append("      olfactory, cortical L1-3, and other regions without spinal projections.")
        lines.append("      These are flagged for DESCENDING (retrograde spinal) tracing only.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate eLife comparison reports with suspicious region filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--brains', nargs='+', help='Specific brain IDs to include')
    parser.add_argument('--output', '-o', type=Path,
                        default=DATA_SUMMARY_DIR / "reports" / "elife_comparison_corrected.csv",
                        help='Output CSV path')
    parser.add_argument('--ascending', action='store_true',
                        help='ATLAS/ascending tracing (no suspicious filtering)')
    parser.add_argument('--input', '-i', type=Path,
                        default=DATA_SUMMARY_DIR / "region_counts.csv",
                        help='Input region_counts.csv path')

    args = parser.parse_args()

    tracing_type = TracingType.ASCENDING if args.ascending else TracingType.DESCENDING

    print(f"\n{'='*60}")
    print("eLife Comparison Report Generator")
    print(f"{'='*60}")
    print(f"Tracing type: {tracing_type.value.upper()}")

    # Load data
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading: {args.input}")
    all_brains = load_region_counts_csv(args.input)
    print(f"  Found {len(all_brains)} brains")

    # Filter if specific brains requested
    if args.brains:
        filtered = []
        for row in all_brains:
            brain_id = row.get('brain_id', '')
            brain_name = row.get('brain', '')
            if any(b in brain_id or b in brain_name for b in args.brains):
                filtered.append(row)
        all_brains = filtered
        print(f"  Filtered to {len(all_brains)} brains: {args.brains}")

    if not all_brains:
        print("ERROR: No brains to process")
        sys.exit(1)

    # Generate report
    print(f"\nGenerating report...")
    brain_data = generate_elife_comparison_csv(all_brains, args.output, tracing_type)

    # Generate and print summary
    summary = generate_summary_text(brain_data, tracing_type)
    print(f"\n{summary}")

    # Save summary as text file too
    summary_path = args.output.with_suffix('.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n  Saved summary: {summary_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
