#!/usr/bin/env python3
"""
util_pipeline_comparison.py

Generate comprehensive comparison reports showing how different pipeline
configurations affect region counting results.

This answers the question: "When I ran detection with X settings, classification
with Y network, and counted regions, I got Z results. How does that compare to
other runs and to the eLife reference?"

Usage:
    python util_pipeline_comparison.py --brain 349_CNT_01_02_1p625x_z4
    python util_pipeline_comparison.py --brain 349_CNT_01_02_1p625x_z4 --output report.md
    python util_pipeline_comparison.py --brain 349_CNT_01_02_1p625x_z4 --format csv

Output:
    - Markdown report with full pipeline comparison
    - CSV with regional counts across runs
    - Comparison to eLife reference data
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent))

from mousebrain.config import BRAINS_ROOT, DATA_SUMMARY_DIR
from mousebrain.tracker import ExperimentTracker
from util_compare_to_published import (
    PUBLISHED_REFERENCE,
    ATLAS_TO_ELIFE_MAP,
    KEY_RECOVERY_REGIONS,
    map_to_elife_regions,
    load_your_counts
)


def find_brain_path(brain_name: str) -> Optional[Path]:
    """Find the full path to a brain's pipeline folder."""
    for mouse_dir in BRAINS_ROOT.iterdir():
        if not mouse_dir.is_dir():
            continue
        for pipeline_dir in mouse_dir.iterdir():
            if brain_name in pipeline_dir.name:
                return pipeline_dir
    return None


def get_pipeline_runs(tracker: ExperimentTracker, brain_name: str) -> Dict[str, List[dict]]:
    """
    Get all pipeline runs for a brain, organized by type.

    Returns:
        {
            'detection': [...],
            'classification': [...],
            'region_counting': [...],
            'registration': [...]
        }
    """
    runs = {
        'detection': [],
        'classification': [],
        'region_counting': [],
        'registration': []
    }

    # Map display names to tracker exp_types
    tracker_types = {
        'detection': 'detection',
        'classification': 'classification',
        'region_counting': 'counts',  # Tracker uses 'counts', not 'region_counting'
        'registration': 'registration'
    }

    for display_type in list(runs.keys()):
        exp_type = tracker_types.get(display_type, display_type)
        results = tracker.search(brain=brain_name, exp_type=exp_type)
        # Filter to completed/approved only (registration can be 'approved')
        valid_statuses = ['completed', 'approved']
        completed = [r for r in results if r.get('status') in valid_statuses]
        # Sort by date (newest first)
        completed.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        runs[display_type] = completed

    return runs


def link_pipeline_runs(runs: Dict[str, List[dict]]) -> List[dict]:
    """
    Link detection -> classification -> region_counting into complete pipelines.

    Returns list of "pipeline configurations" where each has:
        {
            'id': unique identifier,
            'detection': {...detection run...},
            'classification': {...classification run...},
            'region_counting': {...region counting run...},
            'registration': {...registration run...}
        }
    """
    pipelines = []

    # Start from region counting (the end result) and work backwards
    for rc_run in runs.get('region_counting', []):
        pipeline = {
            'id': rc_run.get('exp_id', 'unknown'),
            'date': rc_run.get('created_at', '')[:10],
            'region_counting': rc_run,
            'classification': None,
            'detection': None,
            'registration': None
        }

        # Find linked classification
        class_id = rc_run.get('parent_exp_id') or rc_run.get('class_exp_id')
        if class_id:
            for class_run in runs.get('classification', []):
                if class_run.get('exp_id') == class_id:
                    pipeline['classification'] = class_run
                    break

        # If no explicit link, use most recent classification before this count
        if not pipeline['classification'] and runs.get('classification'):
            rc_date = rc_run.get('created_at', '')
            for class_run in runs['classification']:
                if class_run.get('created_at', '') <= rc_date:
                    pipeline['classification'] = class_run
                    break

        # Find linked detection
        if pipeline['classification']:
            det_id = pipeline['classification'].get('parent_exp_id') or pipeline['classification'].get('det_exp_id')
            if det_id:
                for det_run in runs.get('detection', []):
                    if det_run.get('exp_id') == det_id:
                        pipeline['detection'] = det_run
                        break

        # Find registration (usually just one per brain)
        if runs.get('registration'):
            pipeline['registration'] = runs['registration'][0]

        pipelines.append(pipeline)

    # If no region counting runs, try to build from classifications
    if not pipelines and runs.get('classification'):
        # Get registration run safely (may be empty list)
        reg_runs = runs.get('registration', [])
        reg_run = reg_runs[0] if reg_runs else None

        for class_run in runs['classification']:
            pipeline = {
                'id': class_run.get('exp_id', 'unknown'),
                'date': class_run.get('created_at', '')[:10],
                'region_counting': None,
                'classification': class_run,
                'detection': None,
                'registration': reg_run
            }
            pipelines.append(pipeline)

    return pipelines


def load_region_counts_for_run(brain_path: Path, run: dict) -> Dict[str, int]:
    """Load region counts CSV for a given run."""
    # Try output_path from run first
    output_path = run.get('output_path')
    if output_path:
        csv_path = Path(output_path)
        if csv_path.is_dir():
            csv_path = csv_path / "cell_counts_by_region.csv"
        if csv_path.exists():
            return load_your_counts(csv_path)

    # Fallback to standard location
    csv_path = brain_path / "6_Region_Analysis" / "cell_counts_by_region.csv"
    if csv_path.exists():
        return load_your_counts(csv_path)

    return {}


def format_detection_params(det_run: Optional[dict]) -> str:
    """Format detection parameters as a string."""
    if not det_run:
        return "N/A"

    params = []
    if det_run.get('det_ball_xy'):
        params.append(f"ball_xy={det_run['det_ball_xy']}")
    if det_run.get('det_ball_z'):
        params.append(f"ball_z={det_run['det_ball_z']}")
    if det_run.get('det_threshold'):
        params.append(f"threshold={det_run['det_threshold']}")
    if det_run.get('det_soma_diameter'):
        params.append(f"soma={det_run['det_soma_diameter']}")
    if det_run.get('det_preset'):
        params.append(f"preset={det_run['det_preset']}")

    return ", ".join(params) if params else "default"


def format_classification_params(class_run: Optional[dict]) -> str:
    """Format classification parameters as a string."""
    if not class_run:
        return "N/A"

    model = class_run.get('class_model', 'default')
    if model:
        model = Path(model).stem  # Just the filename without path

    cells = class_run.get('class_cells_kept', '?')
    rejected = class_run.get('class_cells_rejected', '?')

    return f"model={model}, kept={cells}, rejected={rejected}"


def format_registration_params(reg_run: Optional[dict]) -> str:
    """Format registration parameters as a string."""
    if not reg_run:
        return "N/A"

    atlas = reg_run.get('reg_atlas', 'allen_mouse_10um')
    orientation = reg_run.get('reg_orientation', '?')

    return f"atlas={atlas}, orientation={orientation}"


def generate_markdown_report(
    brain_name: str,
    brain_path: Path,
    pipelines: List[dict],
    output_path: Optional[Path] = None
) -> str:
    """Generate a comprehensive markdown comparison report."""

    lines = []
    lines.append(f"# Pipeline Comparison Report")
    lines.append(f"")
    lines.append(f"**Brain:** {brain_name}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Path:** {brain_path}")
    lines.append(f"")

    # ==========================================================================
    # PIPELINE CONFIGURATIONS
    # ==========================================================================
    lines.append(f"## Pipeline Configurations")
    lines.append(f"")

    if not pipelines:
        lines.append("*No completed pipeline runs found in tracker.*")
        lines.append("")
    else:
        for i, pipeline in enumerate(pipelines, 1):
            lines.append(f"### Run {i} ({pipeline['date']})")
            lines.append(f"")
            lines.append(f"| Stage | Parameters |")
            lines.append(f"|-------|------------|")
            lines.append(f"| **Detection** | {format_detection_params(pipeline.get('detection'))} |")
            lines.append(f"| **Classification** | {format_classification_params(pipeline.get('classification'))} |")
            lines.append(f"| **Registration** | {format_registration_params(pipeline.get('registration'))} |")

            if pipeline.get('region_counting'):
                rc = pipeline['region_counting']
                total = rc.get('rc_total_cells', '?')
                lines.append(f"| **Region Counting** | total_cells={total} |")
            else:
                lines.append(f"| **Region Counting** | *not run* |")

            lines.append(f"")

    # ==========================================================================
    # REGIONAL COUNTS COMPARISON TABLE
    # ==========================================================================
    lines.append(f"## Regional Counts Comparison")
    lines.append(f"")

    # Load counts for each pipeline
    pipeline_counts = []
    for pipeline in pipelines:
        if pipeline.get('region_counting'):
            counts = load_region_counts_for_run(brain_path, pipeline['region_counting'])
            mapped = map_to_elife_regions(counts)
            pipeline_counts.append({
                'id': pipeline['id'][:8],
                'date': pipeline['date'],
                'raw': counts,
                'mapped': mapped,
                'total': sum(counts.values())
            })

    # Also try loading from filesystem if no tracker runs
    if not pipeline_counts:
        csv_path = brain_path / "6_Region_Analysis" / "cell_counts_by_region.csv"
        if csv_path.exists():
            counts = load_your_counts(csv_path)
            mapped = map_to_elife_regions(counts)
            pipeline_counts.append({
                'id': 'filesystem',
                'date': 'current',
                'raw': counts,
                'mapped': mapped,
                'total': sum(counts.values())
            })

    if not pipeline_counts:
        lines.append("*No region counts available. Run region counting first.*")
        lines.append("")
    else:
        # Build comparison table header
        header = "| Region |"
        separator = "|--------|"
        for pc in pipeline_counts:
            header += f" Run {pc['date']} |"
            separator += "----------:|"
        header += " eLife Ref | vs eLife |"
        separator += "----------:|----------:|"

        lines.append(header)
        lines.append(separator)

        # Get all eLife regions, key regions first
        all_regions = list(PUBLISHED_REFERENCE.keys())
        key_first = [r for r in all_regions if any(k in r for k in KEY_RECOVERY_REGIONS)]
        other = [r for r in all_regions if r not in key_first]
        sorted_regions = key_first + sorted(other)

        for region in sorted_regions:
            ref_mean, ref_std, n = PUBLISHED_REFERENCE[region]

            row = f"| {region} |"

            # Add count from each pipeline
            last_count = 0
            for pc in pipeline_counts:
                count = pc['mapped'].get(region, 0)
                row += f" {count:,} |"
                last_count = count

            # Add eLife reference
            row += f" {ref_mean:,} |"

            # Add % vs eLife (using last pipeline)
            if ref_mean > 0:
                pct = (last_count / ref_mean) * 100
                row += f" {pct:.0f}% |"
            else:
                row += " - |"

            # Mark key regions
            if any(k in region for k in KEY_RECOVERY_REGIONS):
                row = row.replace(f"| {region} |", f"| **{region}** * |")

            lines.append(row)

        # Totals row
        ref_total = sum(v[0] for v in PUBLISHED_REFERENCE.values())
        row = "| **TOTAL** |"
        for pc in pipeline_counts:
            row += f" **{pc['total']:,}** |"
        row += f" **{ref_total:,}** |"
        if pipeline_counts:
            pct = (pipeline_counts[-1]['total'] / ref_total) * 100
            row += f" **{pct:.0f}%** |"
        lines.append(row)

        lines.append(f"")
        lines.append(f"\\* Key recovery regions (predictive of functional outcomes)")
        lines.append(f"")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    lines.append(f"## Summary")
    lines.append(f"")

    if pipeline_counts:
        ref_total = sum(v[0] for v in PUBLISHED_REFERENCE.values())

        for pc in pipeline_counts:
            pct = (pc['total'] / ref_total) * 100
            lines.append(f"- **Run {pc['date']}**: {pc['total']:,} total cells ({pct:.1f}% of eLife reference)")

        lines.append(f"- **eLife Reference**: {ref_total:,} total cells (L1-injected uninjured, n=5)")
        lines.append(f"")

        # Note about expected differences for injured cords
        lines.append(f"### Interpretation Notes")
        lines.append(f"")
        lines.append(f"- eLife reference is from **uninjured** animals")
        lines.append(f"- Injured spinal cords will have **reduced counts** in regions below injury level")
        lines.append(f"- Key recovery regions (Red Nucleus, PPN, Gigantocellular) are most important")
        lines.append(f"- Compare across your own runs to assess parameter optimization")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*Report generated by util_pipeline_comparison.py*")

    report = "\n".join(lines)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def generate_csv_report(
    brain_name: str,
    brain_path: Path,
    pipelines: List[dict],
    output_path: Path
) -> None:
    """Generate CSV comparison of regional counts across runs."""

    # Load counts for each pipeline
    pipeline_counts = []
    for pipeline in pipelines:
        if pipeline.get('region_counting'):
            counts = load_region_counts_for_run(brain_path, pipeline['region_counting'])
            mapped = map_to_elife_regions(counts)
            pipeline_counts.append({
                'id': pipeline['id'][:8],
                'date': pipeline['date'],
                'detection': format_detection_params(pipeline.get('detection')),
                'classification': format_classification_params(pipeline.get('classification')),
                'mapped': mapped,
                'total': sum(counts.values())
            })

    # Fallback to filesystem
    if not pipeline_counts:
        csv_path = brain_path / "6_Region_Analysis" / "cell_counts_by_region.csv"
        if csv_path.exists():
            counts = load_your_counts(csv_path)
            mapped = map_to_elife_regions(counts)
            pipeline_counts.append({
                'id': 'current',
                'date': 'filesystem',
                'detection': 'N/A',
                'classification': 'N/A',
                'mapped': mapped,
                'total': sum(counts.values())
            })

    # Build CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        # Header row with run info
        fieldnames = ['region', 'elife_mean', 'elife_std']
        for i, pc in enumerate(pipeline_counts):
            fieldnames.append(f'run{i+1}_count')
            fieldnames.append(f'run{i+1}_vs_elife_pct')
        fieldnames.append('is_key_region')

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write metadata rows as comments (CSV readers will handle these)
        for i, pc in enumerate(pipeline_counts):
            meta_row = {fn: '' for fn in fieldnames}
            meta_row['region'] = f'# Run {i+1}: {pc["date"]}'
            meta_row['elife_mean'] = pc['detection']
            meta_row['elife_std'] = pc['classification']
            writer.writerow(meta_row)

        # Data rows
        for region in PUBLISHED_REFERENCE.keys():
            ref_mean, ref_std, n = PUBLISHED_REFERENCE[region]

            row = {
                'region': region,
                'elife_mean': ref_mean,
                'elife_std': ref_std,
                'is_key_region': 'Y' if any(k in region for k in KEY_RECOVERY_REGIONS) else ''
            }

            for i, pc in enumerate(pipeline_counts):
                count = pc['mapped'].get(region, 0)
                row[f'run{i+1}_count'] = count
                if ref_mean > 0:
                    row[f'run{i+1}_vs_elife_pct'] = round((count / ref_mean) * 100, 1)
                else:
                    row[f'run{i+1}_vs_elife_pct'] = ''

            writer.writerow(row)

    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate pipeline comparison reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python util_pipeline_comparison.py --brain 349_CNT_01_02_1p625x_z4
    python util_pipeline_comparison.py --brain 349_CNT_01_02_1p625x_z4 --output report.md
    python util_pipeline_comparison.py --brain 349_CNT_01_02_1p625x_z4 --format csv
        """
    )

    parser.add_argument('--brain', '-b', required=True,
                       help='Brain name to analyze')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--format', '-f', choices=['markdown', 'csv', 'both'],
                       default='markdown', help='Output format')

    args = parser.parse_args()

    # Find brain
    brain_path = find_brain_path(args.brain)
    if not brain_path:
        print(f"Error: Brain not found: {args.brain}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Pipeline Comparison Report")
    print(f"{'='*70}")
    print(f"Brain: {args.brain}")
    print(f"Path: {brain_path}")

    # Load tracker data
    tracker = ExperimentTracker()
    runs = get_pipeline_runs(tracker, args.brain)

    print(f"\nFound runs:")
    print(f"  Detection: {len(runs['detection'])}")
    print(f"  Classification: {len(runs['classification'])}")
    print(f"  Region Counting: {len(runs['region_counting'])}")
    print(f"  Registration: {len(runs['registration'])}")

    # Link into pipelines
    pipelines = link_pipeline_runs(runs)
    print(f"\nLinked pipelines: {len(pipelines)}")

    # Generate output
    if args.output:
        output_base = args.output
    else:
        output_base = DATA_SUMMARY_DIR / "reports" / f"{args.brain}_comparison"

    if args.format in ['markdown', 'both']:
        md_path = output_base.with_suffix('.md') if args.output else output_base.parent / f"{output_base.name}.md"
        report = generate_markdown_report(args.brain, brain_path, pipelines, md_path)
        print(f"\n{report}")

    if args.format in ['csv', 'both']:
        csv_path = output_base.with_suffix('.csv') if args.output else output_base.parent / f"{output_base.name}.csv"
        generate_csv_report(args.brain, brain_path, pipelines, csv_path)

    print(f"\n{'='*70}")
    print(f"[OK] Report generation complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
