#!/usr/bin/env python3
"""
experiments.py

Browse, search, and annotate experiment history.

This is your lab notebook for the BrainGlobe pipeline - quickly see what
you've tried, what worked, what didn't, and find that run from last week
where the detection looked pretty good.

Usage:
    python experiments.py                    # Show recent experiments
    python experiments.py --type detection   # Show detection runs only
    python experiments.py --type crop        # Show crop optimization runs
    python experiments.py --brain 349*       # Filter by brain (wildcards ok)
    python experiments.py --show det_20241218_abc123  # Show details
    python experiments.py --rate det_20241218_abc123  # Rate an experiment
    python experiments.py --compare det_123 det_456   # Compare two runs
    python experiments.py --best detection 5          # Top 5 rated detections
    python experiments.py --best crop 3               # Top 3 crop optimizations
    python experiments.py --unrated                   # Show unrated experiments
    python experiments.py --export results.csv        # Export filtered results
    python experiments.py --crop-history 349_CNT_01_02  # Show all crops for a brain
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from experiment_tracker import ExperimentTracker, EXPERIMENT_TYPES

# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def format_duration(seconds):
    """Format seconds as human-readable duration."""
    if not seconds:
        return "-"
    try:
        s = float(seconds)
        if s < 60:
            return f"{s:.0f}s"
        elif s < 3600:
            return f"{s/60:.1f}m"
        else:
            return f"{s/3600:.1f}h"
    except:
        return str(seconds)


def format_timestamp(iso_timestamp):
    """Format ISO timestamp as readable date/time."""
    if not iso_timestamp:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return iso_timestamp[:16]


def format_score(score):
    """Format a score value (0-1) as percentage."""
    if not score:
        return "-"
    try:
        return f"{float(score)*100:.1f}%"
    except:
        return str(score)


def print_table(experiments, columns=None):
    """Print experiments as a formatted table."""
    if not experiments:
        print("No experiments found.")
        return
    
    if columns is None:
        columns = ['experiment_id', 'timestamp', 'brain', 'status', 'rating']
    
    # Calculate column widths
    widths = {}
    for col in columns:
        header_width = len(col)
        max_val_width = max(len(str(exp.get(col, ''))[:40]) for exp in experiments)
        widths[col] = max(header_width, max_val_width, 5)
    
    # Print header
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(f"\n{header}")
    print("-" * len(header))
    
    # Print rows
    for exp in experiments:
        row_values = []
        for col in columns:
            val = exp.get(col, '')
            if col == 'timestamp':
                val = format_timestamp(val)
            elif col == 'duration_seconds':
                val = format_duration(val)
            elif col == 'rating':
                val = f"{val}/5" if val else "-"
            elif col in ['crop_quality_score', 'crop_combined_score']:
                val = format_score(val)
            row_values.append(str(val)[:widths[col]].ljust(widths[col]))
        print(" | ".join(row_values))
    
    print(f"\nTotal: {len(experiments)} experiments")


def print_comparison(tracker, exp_ids):
    """Print side-by-side comparison of experiments."""
    experiments = [tracker.get_experiment(eid) for eid in exp_ids]
    experiments = [e for e in experiments if e]  # Remove None
    
    if len(experiments) < 2:
        print("Need at least 2 valid experiment IDs to compare.")
        return
    
    # Determine what type of experiments these are
    types = set(e.get('experiment_type') for e in experiments)
    if len(types) > 1:
        print(f"Warning: Comparing different experiment types: {types}")
    
    exp_type = experiments[0].get('experiment_type')
    
    # Fields to compare based on type
    common_fields = ['experiment_id', 'timestamp', 'brain', 'status', 'duration_seconds', 'rating', 'notes']
    
    type_fields = {
        'detection': ['det_soma_diameter', 'det_ball_xy_size', 'det_ball_z_size', 
                     'det_ball_overlap_fraction', 'det_candidates_found'],
        'training': ['train_network_depth', 'train_epochs', 'train_num_positive',
                    'train_num_negative', 'train_final_accuracy'],
        'classification': ['class_model_path', 'class_cells_found', 'class_rejected'],
        'registration': ['reg_atlas', 'reg_orientation'],
        'counts': ['counts_total_cells'],
        'crop': ['crop_start_y', 'crop_optimal_y', 'crop_quality_score', 
                'crop_combined_score', 'crop_iterations', 'crop_algorithm'],
    }
    
    fields = common_fields + type_fields.get(exp_type, [])
    
    # Print comparison table
    print(f"\n{'='*60}")
    print(f"Comparing {len(experiments)} {exp_type} experiments")
    print(f"{'='*60}\n")
    
    # Calculate field name width
    field_width = max(len(f) for f in fields)
    val_width = 25
    
    # Header
    header = "Field".ljust(field_width) + " | " + " | ".join(
        f"Exp {i+1}".center(val_width) for i in range(len(experiments))
    )
    print(header)
    print("-" * len(header))
    
    # Rows
    for field in fields:
        values = []
        for exp in experiments:
            val = exp.get(field, '-')
            if field == 'timestamp':
                val = format_timestamp(val)
            elif field == 'duration_seconds':
                val = format_duration(val)
            elif field == 'rating':
                val = f"{val}/5" if val else "-"
            elif field in ['crop_quality_score', 'crop_combined_score']:
                val = format_score(val)
            values.append(str(val)[:val_width].center(val_width))
        
        # Highlight differences
        unique_vals = set(str(exp.get(field, '')) for exp in experiments)
        prefix = "* " if len(unique_vals) > 1 and field not in ['experiment_id', 'timestamp'] else "  "
        
        print(f"{prefix}{field.ljust(field_width-2)} | " + " | ".join(values))
    
    print(f"\n* = values differ between experiments")


def print_crop_history(tracker, brain_pattern):
    """Print crop optimization history for a brain."""
    experiments = tracker.get_experiments(
        experiment_type="crop",
        brain=brain_pattern,
    )
    
    if not experiments:
        print(f"No crop optimization runs found for: {brain_pattern}")
        return
    
    print(f"\n{'='*70}")
    print(f"Crop Optimization History: {brain_pattern}")
    print(f"{'='*70}")
    
    # Get best crop
    best = None
    best_score = -1
    for exp in experiments:
        if exp.get('status') == 'completed':
            try:
                score = float(exp.get('crop_combined_score', 0) or 0)
                if score > best_score:
                    best_score = score
                    best = exp
            except:
                pass
    
    # Print table
    cols = ['experiment_id', 'timestamp', 'crop_start_y', 'crop_optimal_y', 
            'crop_quality_score', 'crop_combined_score', 'crop_iterations', 'rating']
    
    # Custom header names
    header_names = {
        'crop_start_y': 'Start Y',
        'crop_optimal_y': 'Optimal Y',
        'crop_quality_score': 'Quality',
        'crop_combined_score': 'Combined',
        'crop_iterations': 'Iters',
    }
    
    # Calculate widths
    widths = {}
    for col in cols:
        display_name = header_names.get(col, col)
        header_width = len(display_name)
        max_val_width = max(len(str(exp.get(col, ''))[:20]) for exp in experiments)
        widths[col] = max(header_width, max_val_width, 5)
    
    # Print header
    header = " | ".join(header_names.get(col, col).ljust(widths[col]) for col in cols)
    print(f"\n{header}")
    print("-" * len(header))
    
    # Print rows
    for exp in experiments:
        row_values = []
        is_best = exp == best
        
        for col in cols:
            val = exp.get(col, '')
            if col == 'timestamp':
                val = format_timestamp(val)
            elif col == 'rating':
                val = f"{val}/5" if val else "-"
            elif col in ['crop_quality_score', 'crop_combined_score']:
                val = format_score(val)
            
            cell = str(val)[:widths[col]].ljust(widths[col])
            row_values.append(cell)
        
        row_str = " | ".join(row_values)
        if is_best:
            print(f"{row_str}  ★ BEST")
        else:
            print(row_str)
    
    print(f"\nTotal: {len(experiments)} crop optimization runs")
    
    if best:
        print(f"\n★ Best result: Y={best.get('crop_optimal_y')} "
              f"(combined score: {format_score(best.get('crop_combined_score'))})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Browse and manage experiment history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments.py                      # Recent experiments
  python experiments.py -t detection         # Detection runs only
  python experiments.py -t crop              # Crop optimization runs
  python experiments.py -b "349*"            # Filter by brain
  python experiments.py --show det_202412... # Full details
  python experiments.py --rate det_202412... # Add rating
  python experiments.py --compare det_1 det_2
  python experiments.py --compare cro_1 cro_2  # Compare crop runs
  python experiments.py --best detection 5   # Top 5 rated
  python experiments.py --best crop 3        # Top 3 crop optimizations
  python experiments.py --crop-history 349*  # All crops for a brain
  python experiments.py --unrated            # Needs review
  python experiments.py --stats              # Summary statistics
        """
    )
    
    # Filters
    parser.add_argument('--type', '-t', choices=EXPERIMENT_TYPES,
                        help='Filter by experiment type')
    parser.add_argument('--brain', '-b', help='Filter by brain (supports * wildcard)')
    parser.add_argument('--status', '-s', choices=['started', 'completed', 'failed'],
                        help='Filter by status')
    parser.add_argument('--recent', '-n', type=int, default=20,
                        help='Show N most recent (default: 20)')
    parser.add_argument('--unrated', '-u', action='store_true',
                        help='Show only unrated experiments')
    parser.add_argument('--rated', action='store_true',
                        help='Show only rated experiments')
    parser.add_argument('--min-rating', type=int, choices=[1,2,3,4,5],
                        help='Minimum rating to show')
    
    # Actions
    parser.add_argument('--show', metavar='ID',
                        help='Show full details of an experiment')
    parser.add_argument('--rate', metavar='ID',
                        help='Rate an experiment (interactive)')
    parser.add_argument('--note', nargs=2, metavar=('ID', 'NOTE'),
                        help='Add a note to an experiment')
    parser.add_argument('--compare', nargs='+', metavar='ID',
                        help='Compare multiple experiments')
    parser.add_argument('--best', nargs=2, metavar=('TYPE', 'N'),
                        help='Show top N rated of a type')
    parser.add_argument('--crop-history', metavar='BRAIN',
                        help='Show crop optimization history for a brain')
    
    # Output
    parser.add_argument('--stats', action='store_true',
                        help='Show summary statistics')
    parser.add_argument('--export', metavar='FILE',
                        help='Export filtered results to CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show more columns')
    
    # Config
    parser.add_argument('--log', help='Path to experiment log CSV')
    
    args = parser.parse_args()
    
    tracker = ExperimentTracker(log_path=args.log)
    
    # Handle specific actions
    if args.show:
        tracker.print_experiment(args.show)
        return
    
    if args.rate:
        exp = tracker.get_experiment(args.rate)
        if not exp:
            print(f"Experiment not found: {args.rate}")
            return
        
        # Show current state
        print(f"\nExperiment: {args.rate}")
        print(f"Type: {exp.get('experiment_type')}")
        print(f"Brain: {exp.get('brain')}")
        
        # Type-specific info
        exp_type = exp.get('experiment_type')
        if exp_type == 'crop':
            print(f"Optimal Y: {exp.get('crop_optimal_y')}")
            print(f"Quality Score: {format_score(exp.get('crop_quality_score'))}")
            print(f"Combined Score: {format_score(exp.get('crop_combined_score'))}")
        elif exp_type == 'detection':
            print(f"Candidates found: {exp.get('det_candidates_found')}")
        elif exp_type == 'classification':
            print(f"Cells found: {exp.get('class_cells_found')}")
        
        print(f"Current rating: {exp.get('rating') or 'Unrated'}")
        print(f"Current notes: {exp.get('notes') or 'None'}")
        
        try:
            rating = input("\nNew rating (1-5): ").strip()
            if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                note = input("Add note (or Enter to skip): ").strip()
                tracker.rate_experiment(args.rate, int(rating), note if note else None)
            else:
                print("Invalid rating. Use 1-5.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
        return
    
    if args.note:
        exp_id, note = args.note
        tracker.add_notes(exp_id, note)
        return
    
    if args.compare:
        print_comparison(tracker, args.compare)
        return
    
    if args.crop_history:
        print_crop_history(tracker, args.crop_history)
        return
    
    if args.best:
        exp_type, n = args.best
        if exp_type not in EXPERIMENT_TYPES:
            print(f"Invalid type. Choose from: {', '.join(EXPERIMENT_TYPES)}")
            return
        experiments = tracker.get_best(exp_type, int(n))
        print(f"\nTop {n} rated {exp_type} experiments:")
        
        # Type-specific columns
        if exp_type == 'detection':
            cols = ['experiment_id', 'brain', 'rating', 'det_soma_diameter', 
                   'det_ball_xy_size', 'det_candidates_found']
        elif exp_type == 'training':
            cols = ['experiment_id', 'brain', 'rating', 'train_network_depth',
                   'train_epochs', 'train_final_accuracy']
        elif exp_type == 'crop':
            cols = ['experiment_id', 'brain', 'rating', 'crop_optimal_y',
                   'crop_quality_score', 'crop_combined_score', 'crop_iterations']
        else:
            cols = ['experiment_id', 'brain', 'rating', 'status']
        
        print_table(experiments, columns=cols)
        return
    
    if args.stats:
        all_exps = tracker.get_experiments()
        
        print(f"\n{'='*40}")
        print("Experiment Statistics")
        print(f"{'='*40}")
        print(f"\nTotal experiments: {len(all_exps)}")
        
        # By type
        print(f"\nBy type:")
        for exp_type in EXPERIMENT_TYPES:
            type_exps = [e for e in all_exps if e.get('experiment_type') == exp_type]
            if type_exps:
                rated = len([e for e in type_exps if e.get('rating')])
                completed = len([e for e in type_exps if e.get('status') == 'completed'])
                print(f"  {exp_type}: {len(type_exps)} ({completed} completed, {rated} rated)")
        
        # By status
        print(f"\nBy status:")
        for status in ['completed', 'started', 'failed']:
            status_exps = [e for e in all_exps if e.get('status') == status]
            if status_exps:
                print(f"  {status}: {len(status_exps)}")
        
        # Rating distribution
        rated_exps = [e for e in all_exps if e.get('rating')]
        if rated_exps:
            print(f"\nRating distribution:")
            for r in range(5, 0, -1):
                count = len([e for e in rated_exps if e.get('rating') == str(r)])
                bar = '█' * count
                print(f"  {r}: {bar} ({count})")
        
        # Crop-specific stats
        crop_exps = [e for e in all_exps if e.get('experiment_type') == 'crop' 
                     and e.get('status') == 'completed']
        if crop_exps:
            print(f"\nCrop optimization stats:")
            
            # Average quality score
            quality_scores = []
            for exp in crop_exps:
                try:
                    quality_scores.append(float(exp.get('crop_quality_score', 0) or 0))
                except:
                    pass
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"  Average quality score: {avg_quality*100:.1f}%")
            
            # Average iterations
            iterations = []
            for exp in crop_exps:
                try:
                    iterations.append(int(exp.get('crop_iterations', 0) or 0))
                except:
                    pass
            if iterations:
                avg_iters = sum(iterations) / len(iterations)
                print(f"  Average iterations: {avg_iters:.1f}")
            
            # Unique brains optimized
            unique_brains = set(exp.get('brain') for exp in crop_exps if exp.get('brain'))
            print(f"  Unique brains optimized: {len(unique_brains)}")
        
        # Unrated
        unrated = [e for e in all_exps if not e.get('rating') and e.get('status') == 'completed']
        if unrated:
            print(f"\nUnrated completed experiments: {len(unrated)}")
        
        return
    
    # Default: list experiments with filters
    experiments = tracker.get_experiments(
        experiment_type=args.type,
        brain=args.brain,
        status=args.status,
        rated_only=args.rated,
        unrated_only=args.unrated,
        min_rating=args.min_rating,
        limit=args.recent,
    )
    
    if args.export:
        # Export to CSV
        import csv
        with open(args.export, 'w', newline='', encoding='utf-8') as f:
            if experiments:
                writer = csv.DictWriter(f, fieldnames=experiments[0].keys())
                writer.writeheader()
                writer.writerows(experiments)
        print(f"Exported {len(experiments)} experiments to {args.export}")
        return
    
    # Choose columns based on type filter and verbosity
    if args.type == 'detection':
        cols = ['experiment_id', 'timestamp', 'brain', 'det_soma_diameter', 
               'det_candidates_found', 'rating']
    elif args.type == 'training':
        cols = ['experiment_id', 'timestamp', 'train_network_depth',
               'train_epochs', 'train_final_accuracy', 'rating']
    elif args.type == 'crop':
        cols = ['experiment_id', 'timestamp', 'brain', 'crop_optimal_y',
               'crop_quality_score', 'crop_combined_score', 'rating']
    elif args.verbose:
        cols = ['experiment_id', 'experiment_type', 'timestamp', 'brain', 
               'status', 'duration_seconds', 'rating']
    else:
        cols = ['experiment_id', 'experiment_type', 'brain', 'status', 'rating']
    
    print_table(experiments, columns=cols)
    
    # Hint about unrated
    if not args.unrated and experiments:
        unrated_count = len([e for e in experiments if not e.get('rating') and e.get('status') == 'completed'])
        if unrated_count > 0:
            print(f"\nTip: {unrated_count} experiments need rating. Use --unrated to see them.")


if __name__ == '__main__':
    main()
