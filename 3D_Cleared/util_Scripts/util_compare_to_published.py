#!/usr/bin/env python3
"""
util_compare_to_published.py

Compare your cell counts to the published eLife data (Wang et al., 2022).

This tool loads the reference data from the eLife paper and compares your
region counts against the published averages for **CERVICAL (C4) injected**
uninjured animals - the appropriate reference for cervical spinal cord studies.

================================================================================
HOW TO USE
================================================================================
    # Compare a single brain's counts
    python util_compare_to_published.py --brain 349_CNT_01_02_1p625x_z4

    # Compare a specific CSV file
    python util_compare_to_published.py --csv path/to/cell_counts_by_region.csv

    # Show reference data only
    python util_compare_to_published.py --show-reference

    # Compare to lumbar (L1) reference instead of cervical
    python util_compare_to_published.py --brain 349_CNT_01_02_1p625x_z4 --lumbar

================================================================================
REFERENCE DATA
================================================================================
Source: Wang et al. (2022) eLife
"Brain-wide analysis of the supraspinal connectome reveals anatomical
correlates to functional recovery after spinal injury"
DOI: 10.7554/eLife.76254

DEFAULT: Cervical (C4) injection reference from the "Cervical / lumbar co-injected"
section, cervical-injected animals (173, 174, 175, 176), n=4.

KEY DIFFERENCE from lumbar:
  - Cerebellospinal Nuclei: 310 (cervical) vs 0 (lumbar) - validates pipeline!
  - Red Nucleus: 610 (cervical) vs 3397 (lumbar)
  - Gigantocellular: 5015 (cervical) vs 9824 (lumbar)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import BRAINS_ROOT, DATA_SUMMARY_DIR
from elife_region_mapping import (
    TracingType, get_suspicious_count, get_corrected_total,
    format_corrected_summary, is_suspicious_region, DESCENDING_SUSPICIOUS_REGIONS,
    get_elife_group, aggregate_to_elife
)

# =============================================================================
# REFERENCE DATA (from eLife Source Data 1)
# =============================================================================
# Format: {region_name: (average_count, std_dev, n_animals)}

# CERVICAL (C4) REFERENCE - Default for cervical spinal cord studies
# From "Cervical / lumbar co-injected" section, cervical-injected animals 173-176
# NOTE: Names must match elife_region_mapping.py ELIFE_GROUPS keys exactly
CERVICAL_REFERENCE = {
    # Summary Region: (mean, std, n=4) from C4-injected uninjured animals
    "Solitariospinal Area": (500, 301, 4),
    "Medullary Reticular Nuclei": (1069, 795, 4),
    "Magnocellular Reticular Nucleus": (272, 128, 4),
    "Gigantocellular Reticular Nucleus": (5015, 2688, 4),  # Key for recovery!
    "Lateral Reticular Nuclei": (1928, 1259, 4),
    "Dorsal Reticular Nucleus": (137, 77, 4),
    "Raphe Nuclei": (145, 55, 4),
    "Perihypoglossal Area": (235, 208, 4),
    "Medullary Trigeminal Area": (180, 198, 4),
    "Vestibular Nuclei": (838, 516, 4),
    "Pontine Reticular Nuclei": (1377, 509, 4),
    "Superior Olivary Complex": (28, 21, 4),
    "Pontine Trigeminal Area": (44, 25, 4),
    "Pontine Central Gray Area": (572, 167, 4),  # Key for recovery!
    "Parabrachial / Pedunculopontine": (364, 216, 4),  # Key for recovery!
    "Cerebellospinal Nuclei": (310, 193, 4),  # KEY: non-zero for cervical!
    "Red Nucleus": (610, 86, 4),  # Key for recovery!
    "Midbrain Midline Nuclei": (484, 392, 4),
    "Midbrain Reticular Nuclei": (772, 554, 4),
    "Periaqueductal Gray": (326, 324, 4),
    "Hypothalamic Periventricular Zone": (26, 23, 4),
    "Hypothalamic Lateral Area": (473, 374, 4),
    "Hypothalamic Medial Area": (86, 103, 4),
    "Thalamus": (96, 151, 4),
    "Corticospinal": (6500, 1500, 4),  # Key for recovery!
}

# LUMBAR (L1) REFERENCE - For comparison only
# From L1-injected uninjured animals (159, 162, 166, 167, 168)
# NOTE: Names must match elife_region_mapping.py ELIFE_GROUPS keys exactly
LUMBAR_REFERENCE = {
    # Summary Region: (mean, std, n=5) from L1-injected uninjured animals
    "Solitariospinal Area": (878, 252, 5),
    "Medullary Reticular Nuclei": (635, 139, 5),
    "Magnocellular Reticular Nucleus": (1097, 528, 5),
    "Gigantocellular Reticular Nucleus": (7997, 1520, 5),
    "Lateral Reticular Nuclei": (1441, 568, 5),
    "Dorsal Reticular Nucleus": (14, 8, 5),
    "Raphe Nuclei": (535, 258, 5),
    "Perihypoglossal Area": (116, 49, 5),
    "Medullary Trigeminal Area": (11, 7, 5),
    "Vestibular Nuclei": (166, 47, 5),
    "Pontine Reticular Nuclei": (2185, 941, 5),
    "Superior Olivary Complex": (25, 11, 5),
    "Pontine Trigeminal Area": (22, 9, 5),
    "Pontine Central Gray Area": (1528, 471, 5),
    "Parabrachial / Pedunculopontine": (300, 96, 5),
    "Cerebellospinal Nuclei": (0, 0, 5),  # KEY: zero for lumbar!
    "Red Nucleus": (1860, 1050, 5),
    "Midbrain Midline Nuclei": (271, 148, 5),
    "Midbrain Reticular Nuclei": (646, 219, 5),
    "Periaqueductal Gray": (232, 244, 5),
    "Hypothalamic Periventricular Zone": (473, 291, 5),
    "Hypothalamic Lateral Area": (1170, 419, 5),
    "Hypothalamic Medial Area": (56, 38, 5),
    "Thalamus": (167, 246, 5),
    "Corticospinal": (8713, 433, 5),
}

# Default to cervical reference
PUBLISHED_REFERENCE = CERVICAL_REFERENCE

# Regions that are particularly important for functional recovery
KEY_RECOVERY_REGIONS = [
    "Pedunculopontine",  # PPN - significant predictor
    "Parabrachial / Pedunculopontine",
    "Red Nucleus",
    "Gigantocellular Reticular Nucleus",
    "Pontine Central Gray Area",
    "Corticospinal",
]

# Mapping from Allen Atlas region names to eLife summary regions
ATLAS_TO_ELIFE_MAP = {
    # Exact matches or close matches
    "Dorsal motor nucleus of vagus nerve": "Solitariospinal Area",
    "Cuneate nucleus": "Solitariospinal Area",
    "Gracile nucleus": "Solitariospinal Area",
    "Nucleus of the solitary tract": "Solitariospinal Area",
    "Medullary reticular nucleus": "Medullary Reticular Nuclei",
    "Magnocellular reticular nucleus": "Magnocellular Ret. Nuc.",
    "Gigantocellular reticular nucleus": "Gigantocellular reticular nucleus",
    "Lateral reticular nucleus": "Lateral Reticular Nuclei",
    "Paragigantocellular reticular nucleus": "Lateral Reticular Nuclei",
    "Intermediate reticular nucleus": "Lateral Reticular Nuclei",
    "Parvicellular reticular nucleus": "Lateral Reticular Nuclei",
    "Nucleus raphe magnus": "Raphe Nuclei",
    "Nucleus raphe obscurus": "Raphe Nuclei",
    "Hypoglossal nucleus": "Perihypoglossal Area",
    "Nucleus prepositus": "Perihypoglossal Area",
    "Spinal nucleus of the trigeminal": "Medullary Trigeminal Area",
    "Vestibular nuclei": "Vestibular Nuclei",
    "Spinal vestibular nucleus": "Vestibular Nuclei",
    "Medial vestibular nucleus": "Vestibular Nuclei",
    "Lateral vestibular nucleus": "Vestibular Nuclei",
    "Pontine reticular nucleus": "Pontine Reticular Nuclei",
    "Superior olivary complex": "Superior Olivary Complex",
    "Pontine central gray": "Pontine Central Gray Area",
    "Locus ceruleus": "Pontine Central Gray Area",
    "Barrington's nucleus": "Pontine Central Gray Area",
    "Parabrachial nucleus": "Parabrachial / Pedunculopontine",
    "Pedunculopontine nucleus": "Parabrachial / Pedunculopontine",
    "Fastigial nucleus": "Cerebellospinal Nuclei",
    "Interposed nucleus": "Cerebellospinal Nuclei",
    "Red nucleus": "Red Nucleus",
    "Edinger-Westphal nucleus": "Midbrain Midline Nuclei",
    "Midbrain reticular nucleus": "Midbrain Reticular Nuclei",
    "Periaqueductal gray": "Periaqueductal Gray",
    "Paraventricular hypothalamic nucleus": "Hypothalamic Periventricular Zone",
    "Dorsomedial nucleus of the hypothalamus": "Hypothalamic Lateral Area",
    "Lateral hypothalamic area": "Hypothalamic Lateral Area",
    "Zona incerta": "Hypothalamic Lateral Area",
    "Anterior hypothalamic nucleus": "Hypothalamic Medial Area",
    "Posterior hypothalamic nucleus": "Hypothalamic Medial Area",
    "Ventromedial hypothalamic nucleus": "Hypothalamic Medial Area",
    "Thalamus": "Thalamus",
    # Add corticospinal mapping
    "Primary motor area": "Corticospinal",
    "Secondary motor area": "Corticospinal",
    "Motor cortex": "Corticospinal",
}


def load_your_counts(csv_path: Path) -> dict:
    """Load your cell counts from a CSV file.

    Returns dict with Allen acronyms as keys (preferred for suspicious region filtering).
    """
    import csv

    counts = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Prefer region_acronym for Allen atlas compatibility
            # Fall back to other column names for legacy CSVs
            region = (row.get('region_acronym') or row.get('region') or
                      row.get('Region') or row.get('region_name'))
            count = row.get('cell_count') or row.get('count') or row.get('cells') or row.get('Count')

            if region and count:
                try:
                    counts[region] = int(count)
                except ValueError:
                    pass

    return counts


def map_to_elife_regions(your_counts: dict) -> dict:
    """Map your Allen Atlas regions to eLife summary regions.

    Uses the comprehensive elife_region_mapping module which handles both
    Allen acronyms and full names.
    """
    mapped = {}

    for region, count in your_counts.items():
        # First try the elife_region_mapping module (handles acronyms properly)
        elife_region = get_elife_group(region)

        # Fall back to local mapping for legacy full names
        if not elife_region:
            elife_region = ATLAS_TO_ELIFE_MAP.get(region)

        if not elife_region:
            # Try partial match for full names
            region_lower = region.lower()
            for atlas_name, elife_name in ATLAS_TO_ELIFE_MAP.items():
                if atlas_name.lower() in region_lower or region_lower in atlas_name.lower():
                    elife_region = elife_name
                    break

        if elife_region and elife_region != "[Unmapped]":
            mapped[elife_region] = mapped.get(elife_region, 0) + count
        else:
            # Keep unmapped regions with original name
            mapped[f"[unmapped] {region}"] = count

    return mapped


def compare_counts(your_counts: dict, reference: dict, ref_name: str,
                   show_all: bool = False, raw_allen_counts: dict = None,
                   tracing_type: TracingType = TracingType.DESCENDING):
    """Compare your counts to published reference.

    Args:
        your_counts: Your eLife-mapped counts
        reference: Published reference data
        ref_name: Reference name for display
        show_all: Show all regions including zeros
        raw_allen_counts: Optional raw Allen counts for suspicious region analysis
        tracing_type: Type of tracing for suspicious region filtering
    """
    print("\n" + "=" * 80)
    print(f"COMPARISON: Your Counts vs {ref_name} Reference (Wang et al., 2022 eLife)")
    print("=" * 80)

    # Show suspicious region summary if we have raw counts
    if raw_allen_counts and tracing_type == TracingType.DESCENDING:
        suspicious = get_suspicious_count(raw_allen_counts, tracing_type)
        raw_total = suspicious["total"]
        legit_total = suspicious["total_legitimate"]
        suspicious_total = suspicious["total_suspicious"]
        print(f"\nSUSPICIOUS REGION FILTERING (for descending/retrograde tracing):")
        print(f"  Raw detection total:  {raw_total:>8,} cells")
        print(f"  Legitimate regions:   {legit_total:>8,} cells ({legit_total/raw_total*100:.1f}%)")
        print(f"  Suspicious excluded:  {suspicious_total:>8,} cells ({suspicious_total/raw_total*100:.1f}%)")
        print(f"  (Suspicious = cerebellar cortex, white matter, olfactory, cortical L1-3, etc.)")

    # Header
    print(f"\n{'Region':<45} {'Yours':>8} {'Ref':>8} {'Diff':>8} {'%Match':>8}")
    print("-" * 80)

    total_yours = 0
    total_ref = 0
    matched_regions = 0

    # Sort by importance (key regions first) then alphabetically
    regions = list(reference.keys())
    key_first = [r for r in regions if any(k in r for k in KEY_RECOVERY_REGIONS)]
    other = [r for r in regions if r not in key_first]
    sorted_regions = key_first + sorted(other)

    for region in sorted_regions:
        ref_mean, ref_std, n = reference[region]
        your_count = your_counts.get(region, 0)

        diff = your_count - ref_mean
        pct = (your_count / ref_mean * 100) if ref_mean > 0 else 0

        # Highlight key regions
        marker = " *" if any(k in region for k in KEY_RECOVERY_REGIONS) else ""

        # Color coding hint (for terminal)
        if pct >= 80 and pct <= 120:
            status = "OK"
        elif pct >= 50 and pct <= 150:
            status = "~"
        elif your_count == 0 and ref_mean > 0:
            status = "MISS"
        else:
            status = "!!"

        if your_count > 0 or show_all:
            print(f"{region:<43}{marker} {your_count:>8} {ref_mean:>8} {diff:>+8} {pct:>7.1f}%  {status}")

        total_yours += your_count
        total_ref += ref_mean
        if your_count > 0:
            matched_regions += 1

    # Summary
    print("-" * 80)
    print(f"{'TOTAL (eLife-mapped)':<45} {total_yours:>8} {total_ref:>8}")
    print(f"\nRegions with counts: {matched_regions}/{len(reference)}")
    print(f"Overall ratio: {total_yours/total_ref*100:.1f}% of published average")

    # Key regions summary
    print("\n" + "=" * 80)
    print("KEY RECOVERY REGIONS (most important for functional outcomes)")
    print("=" * 80)
    for region in sorted_regions:
        if any(k in region for k in KEY_RECOVERY_REGIONS):
            ref_mean, ref_std, n = reference[region]
            your_count = your_counts.get(region, 0)
            pct = (your_count / ref_mean * 100) if ref_mean > 0 else 0
            print(f"  {region}: {your_count} / {ref_mean} ({pct:.0f}%)")


def show_reference(reference: dict = None, ref_name: str = "CERVICAL (C4)"):
    """Display the published reference data."""
    if reference is None:
        reference = CERVICAL_REFERENCE

    # Get n from first entry
    first_entry = list(reference.values())[0]
    n_animals = first_entry[2] if len(first_entry) > 2 else "?"

    print("\n" + "=" * 80)
    print(f"PUBLISHED REFERENCE DATA - {ref_name} (Wang et al., 2022 eLife)")
    print(f"Uninjured animals, n={n_animals}")
    print("=" * 80)

    print(f"\n{'Region':<45} {'Mean':>8} {'StdDev':>8}")
    print("-" * 80)

    total = 0
    for region, (mean, std, n) in sorted(reference.items()):
        marker = " *" if any(k in region for k in KEY_RECOVERY_REGIONS) else ""
        print(f"{region:<43}{marker} {mean:>8} {std:>8}")
        total += mean

    print("-" * 80)
    print(f"{'TOTAL':<45} {total:>8}")
    print("\n* = Key region for functional recovery")

    # Show key differences between cervical and lumbar
    if ref_name == "CERVICAL (C4)":
        print("\n" + "-" * 80)
        print("KEY CERVICAL vs LUMBAR DIFFERENCES:")
        print("  Cerebellospinal Nuclei: 310 (cervical) vs 0 (lumbar)")
        print("  Red Nucleus: 610 (cervical) vs 1860 (lumbar)")
        print("  Gigantocellular: 5015 (cervical) vs 7997 (lumbar)")


def find_brain_counts(brain_name: str) -> Path:
    """Find the counts CSV for a brain."""
    for mouse_dir in BRAINS_ROOT.iterdir():
        if not mouse_dir.is_dir():
            continue
        for pipeline_dir in mouse_dir.iterdir():
            if brain_name in pipeline_dir.name:
                csv_path = pipeline_dir / "6_Region_Analysis" / "cell_counts_by_region.csv"
                if csv_path.exists():
                    return csv_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare your counts to published eLife data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare retrograde cervical tracing (default)
    python util_compare_to_published.py --brain 349_CNT_01_02_1p625x_z4

    # Compare ATLAS (ascending) tracing - no suspicious filtering
    python util_compare_to_published.py --brain 349_CNT_01_02_1p625x_z4 --ascending

    # Use lumbar reference instead of cervical
    python util_compare_to_published.py --brain 349_CNT_01_02_1p625x_z4 --lumbar
        """
    )
    parser.add_argument("--brain", help="Brain name to compare")
    parser.add_argument("--csv", help="Path to your counts CSV")
    parser.add_argument("--show-reference", action="store_true", help="Show reference data only")
    parser.add_argument("--all", action="store_true", help="Show all regions including zeros")
    parser.add_argument("--lumbar", action="store_true",
                        help="Use lumbar (L1) reference instead of cervical (C4)")
    parser.add_argument("--ascending", action="store_true",
                        help="ATLAS/ascending tracing - disables suspicious region filtering")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable suspicious region filtering (same as --ascending)")

    args = parser.parse_args()

    # Determine tracing type
    if args.ascending or args.no_filter:
        tracing_type = TracingType.ASCENDING
        print("NOTE: ATLAS/ascending tracing mode - no suspicious region filtering")
    else:
        tracing_type = TracingType.DESCENDING

    # Select reference dataset
    if args.lumbar:
        reference = LUMBAR_REFERENCE
        ref_name = "LUMBAR (L1)"
    else:
        reference = CERVICAL_REFERENCE
        ref_name = "CERVICAL (C4)"

    if args.show_reference:
        show_reference(reference, ref_name)
        return

    # Find counts file
    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
    elif args.brain:
        csv_path = find_brain_counts(args.brain)
        if not csv_path:
            print(f"ERROR: Could not find counts for brain '{args.brain}'")
            print("Run 6_count_regions.py first to generate counts.")
            sys.exit(1)
    else:
        print("Usage: python util_compare_to_published.py --brain <name> or --csv <path>")
        print("       python util_compare_to_published.py --show-reference")
        print("       python util_compare_to_published.py --show-reference --lumbar")
        print("       python util_compare_to_published.py --brain <name> --ascending  # For ATLAS tracing")
        sys.exit(1)

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print(f"Loading counts from: {csv_path}")
    print(f"Using {ref_name} reference data")
    print(f"Tracing type: {tracing_type.value.upper()}")

    # Load and compare
    your_counts = load_your_counts(csv_path)  # Raw Allen counts
    mapped_counts = map_to_elife_regions(your_counts)  # eLife-grouped counts
    compare_counts(mapped_counts, reference, ref_name,
                   show_all=args.all, raw_allen_counts=your_counts,
                   tracing_type=tracing_type)


if __name__ == "__main__":
    main()
