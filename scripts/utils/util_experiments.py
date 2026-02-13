#!/usr/bin/env python3
"""
util_experiments.py

Utility: Interactive CLI for browsing, searching, comparing, and rating experiments.

================================================================================
HOW TO USE
================================================================================
    python util_experiments.py                   # Interactive mode
    python util_experiments.py search "349"      # Search by brain
    python util_experiments.py recent            # Show recent experiments
    python util_experiments.py best detection    # Best detection runs
    python util_experiments.py compare det_xxx det_yyy  # Compare experiments
    python util_experiments.py rate det_xxx 4    # Rate an experiment
    python util_experiments.py stats             # Show statistics

================================================================================
INTERACTIVE MODE
================================================================================
When run without arguments, provides an interactive menu:
    
    EXPERIMENT BROWSER
    ==================
    1. Recent experiments
    2. Search experiments
    3. Best rated
    4. Compare runs
    5. Rate an experiment
    6. Statistics
    q. Quit
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mousebrain.tracker import ExperimentTracker, EXP_TYPES
except ImportError:
    print("ERROR: mousebrain.tracker not found!")
    print("Make sure mousebrain package is installed.")
    sys.exit(1)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def format_duration(seconds):
    """Format duration nicely."""
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
        return "-"


def format_rating(rating):
    """Format rating as stars."""
    if not rating:
        return "-----"
    try:
        r = int(rating)
        return "★" * r + "☆" * (5 - r)
    except:
        return "-----"


def status_symbol(status):
    """Get symbol for status."""
    symbols = {
        'completed': '✓',
        'failed': '✗',
        'started': '○',
        'running': '►',
        'cancelled': '⊘',
    }
    return symbols.get(status, '?')


def print_experiment_table(rows: List[Dict], title: str = "Experiments", verbose: bool = False):
    """Print experiments in a table format."""
    if not rows:
        print(f"\n{title}: None found")
        return
    
    print(f"\n{title} ({len(rows)} results)")
    print("=" * 90)
    print(f"{'Status':<8} {'Type':<5} {'ID':<25} {'Brain':<30} {'Rating':<7} {'Time':<8}")
    print("-" * 90)
    
    for row in rows:
        status = status_symbol(row.get('status', '?'))
        exp_type = row.get('exp_type', '?')[:4].upper()
        exp_id = row.get('exp_id', '?')[:25]
        brain = row.get('brain', '?')[:30]
        rating = format_rating(row.get('rating'))
        duration = format_duration(row.get('duration_seconds'))
        
        print(f"{status:<8} {exp_type:<5} {exp_id:<25} {brain:<30} {rating:<7} {duration:<8}")
        
        if verbose:
            # Show extra details
            cells = row.get('det_cells_found') or row.get('class_cells_found') or row.get('count_total_cells')
            if cells:
                print(f"         Cells: {cells}")
            notes = row.get('notes')
            if notes:
                print(f"         Notes: {notes[:60]}...")
    
    print("=" * 90)


def print_experiment_detail(row: Dict):
    """Print detailed view of a single experiment."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {row.get('exp_id', '?')}")
    print("=" * 60)
    
    fields = [
        ('Type', 'exp_type'),
        ('Brain', 'brain'),
        ('Status', 'status'),
        ('Created', 'created_at'),
        ('Duration', 'duration_seconds'),
        ('Rating', 'rating'),
    ]
    
    for label, key in fields:
        val = row.get(key, '')
        if key == 'duration_seconds':
            val = format_duration(val)
        elif key == 'rating':
            val = format_rating(val)
        if val:
            print(f"  {label:<15}: {val}")
    
    # Type-specific fields
    exp_type = row.get('exp_type')
    
    if exp_type == 'detection':
        print("\n  Detection Parameters:")
        print(f"    Preset: {row.get('det_preset', '-')}")
        print(f"    ball_xy: {row.get('det_ball_xy', '-')}")
        print(f"    ball_z: {row.get('det_ball_z', '-')}")
        print(f"    soma: {row.get('det_soma_diameter', '-')}")
        print(f"    threshold: {row.get('det_threshold', '-')}")
        print(f"    Cells found: {row.get('det_cells_found', '-')}")
    
    elif exp_type == 'training':
        print("\n  Training Parameters:")
        print(f"    Epochs: {row.get('train_epochs', '-')}")
        print(f"    Learning rate: {row.get('train_learning_rate', '-')}")
        print(f"    Augment: {row.get('train_augment', '-')}")
        print(f"    Loss: {row.get('train_loss', '-')}")
        print(f"    Accuracy: {row.get('train_accuracy', '-')}")
    
    elif exp_type == 'classification':
        print("\n  Classification Parameters:")
        print(f"    Model: {row.get('class_model_path', '-')}")
        print(f"    Cube size: {row.get('class_cube_size', '-')}")
        print(f"    Cells found: {row.get('class_cells_found', '-')}")
        print(f"    Rejected: {row.get('class_rejected', '-')}")
    
    elif exp_type == 'counts':
        print("\n  Counting Results:")
        print(f"    Total cells: {row.get('count_total_cells', '-')}")
        print(f"    Output CSV: {row.get('count_output_csv', '-')}")
    
    # Paths
    print("\n  Paths:")
    for key in ['input_path', 'output_path', 'model_path']:
        val = row.get(key)
        if val:
            print(f"    {key}: {val}")
    
    # Notes
    notes = row.get('notes')
    if notes:
        print(f"\n  Notes: {notes}")
    
    # Tags
    tags = row.get('tags')
    if tags:
        print(f"  Tags: {tags}")
    
    print("=" * 60)


def print_comparison(rows: List[Dict]):
    """Print side-by-side comparison of experiments."""
    if len(rows) < 2:
        print("Need at least 2 experiments to compare")
        return
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    
    # Collect all unique keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    
    # Filter to interesting keys
    skip_keys = {'hostname', 'script_version', 'created_at'}
    compare_keys = [k for k in sorted(all_keys) if k not in skip_keys and any(row.get(k) for row in rows)]
    
    # Header
    col_width = max(20, 80 // (len(rows) + 1))
    header = f"{'Field':<20}"
    for row in rows:
        header += f" | {row.get('exp_id', '?')[:col_width-3]:<{col_width-3}}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for key in compare_keys:
        line = f"{key:<20}"
        for row in rows:
            val = str(row.get(key, ''))[:col_width-3]
            line += f" | {val:<{col_width-3}}"
        print(line)
    
    print("=" * 80)


def print_statistics(tracker: ExperimentTracker):
    """Print statistics summary."""
    stats = tracker.get_statistics()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal experiments: {stats['total']}")
    print(f"Rated: {stats['rated']}")
    if stats['avg_rating'] > 0:
        print(f"Average rating: {stats['avg_rating']:.1f}★")
    
    print("\nBy Type:")
    for t in sorted(stats['by_type'].keys()):
        print(f"  {t:<15}: {stats['by_type'][t]}")
    
    print("\nBy Status:")
    for s in sorted(stats['by_status'].keys()):
        print(f"  {s:<15}: {stats['by_status'][s]}")
    
    print("\nTop Brains (by # experiments):")
    sorted_brains = sorted(stats['by_brain'].items(), key=lambda x: x[1], reverse=True)[:10]
    for b, count in sorted_brains:
        print(f"  {b:<40}: {count}")
    
    print("=" * 60)


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(tracker: ExperimentTracker):
    """Run interactive experiment browser."""
    while True:
        print("\n" + "=" * 60)
        print("EXPERIMENT BROWSER")
        print("=" * 60)
        print("\n  1. Recent experiments")
        print("  2. Search experiments")
        print("  3. Best rated")
        print("  4. Compare runs")
        print("  5. Rate an experiment")
        print("  6. View experiment details")
        print("  7. Statistics")
        print("  q. Quit")
        print("-" * 60)
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice == 'q':
            break
        
        elif choice == '1':
            # Recent
            limit = input("How many? [20]: ").strip() or "20"
            rows = tracker.get_recent(limit=int(limit))
            print_experiment_table(rows, "Recent Experiments")
        
        elif choice == '2':
            # Search
            term = input("Search term (brain name, exp_id, etc.): ").strip()
            if not term:
                continue
            
            exp_type = input("Filter by type? [detection/training/classification/counts/all]: ").strip().lower()
            if exp_type == 'all' or not exp_type:
                exp_type = None
            
            rows = tracker.search(brain=term, exp_type=exp_type, limit=50)
            print_experiment_table(rows, f"Search: '{term}'")
        
        elif choice == '3':
            # Best rated
            exp_type = input("Type [detection/training/classification/counts]: ").strip().lower()
            if exp_type not in EXP_TYPES:
                print(f"Invalid type. Choose from: {', '.join(EXP_TYPES)}")
                continue
            
            rows = tracker.get_best(exp_type, limit=10)
            print_experiment_table(rows, f"Best {exp_type} runs")
        
        elif choice == '4':
            # Compare
            ids_input = input("Enter experiment IDs to compare (comma-separated): ").strip()
            if not ids_input:
                continue
            
            exp_ids = [x.strip() for x in ids_input.split(',')]
            rows = tracker.compare_experiments(exp_ids)
            
            if len(rows) < 2:
                print("Could not find enough experiments to compare")
                continue
            
            print_comparison(rows)
        
        elif choice == '5':
            # Rate
            exp_id = input("Experiment ID to rate: ").strip()
            if not exp_id:
                continue
            
            exp = tracker.get_experiment(exp_id)
            if not exp:
                print(f"Experiment not found: {exp_id}")
                continue
            
            print(f"\nExperiment: {exp_id}")
            print(f"  Type: {exp.get('exp_type')}")
            print(f"  Brain: {exp.get('brain')}")
            print(f"  Current rating: {format_rating(exp.get('rating'))}")
            
            rating = input("\nNew rating (1-5): ").strip()
            if not rating.isdigit() or not 1 <= int(rating) <= 5:
                print("Invalid rating")
                continue
            
            notes = input("Add a note (optional): ").strip()
            tracker.rate_experiment(exp_id, int(rating), notes if notes else None)
            print("Rating saved!")
        
        elif choice == '6':
            # View details
            exp_id = input("Experiment ID: ").strip()
            if not exp_id:
                continue
            
            exp = tracker.get_experiment(exp_id)
            if not exp:
                print(f"Experiment not found: {exp_id}")
                continue
            
            print_experiment_detail(exp)
        
        elif choice == '7':
            # Statistics
            print_statistics(tracker)
        
        else:
            print("Invalid choice")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Browse and manage BrainGlobe experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    python experiments.py                        # Interactive mode
    python experiments.py recent [N]             # Show N recent experiments
    python experiments.py search TERM            # Search experiments
    python experiments.py best TYPE              # Best rated of a type
    python experiments.py compare ID1 ID2 ...    # Compare experiments
    python experiments.py rate ID RATING [NOTE]  # Rate an experiment
    python experiments.py view ID                # View experiment details
    python experiments.py stats                  # Show statistics

Types: detection, training, classification, counts
        """
    )
    
    parser.add_argument('command', nargs='?', default='interactive',
                        choices=['interactive', 'recent', 'search', 'best', 
                                'compare', 'rate', 'view', 'stats'])
    parser.add_argument('args', nargs='*', help='Command arguments')
    parser.add_argument('--csv', help='Path to experiments CSV')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ExperimentTracker(args.csv)
    
    if args.command == 'interactive' or (args.command == 'interactive' and not args.args):
        interactive_mode(tracker)
    
    elif args.command == 'recent':
        limit = int(args.args[0]) if args.args else 20
        rows = tracker.get_recent(limit=limit)
        print_experiment_table(rows, "Recent Experiments", verbose=args.verbose)
    
    elif args.command == 'search':
        if not args.args:
            print("Usage: experiments.py search TERM")
            sys.exit(1)
        term = ' '.join(args.args)
        rows = tracker.search(brain=term, limit=50)
        print_experiment_table(rows, f"Search: '{term}'", verbose=args.verbose)
    
    elif args.command == 'best':
        if not args.args or args.args[0] not in EXP_TYPES:
            print(f"Usage: experiments.py best TYPE")
            print(f"Types: {', '.join(EXP_TYPES)}")
            sys.exit(1)
        exp_type = args.args[0]
        rows = tracker.get_best(exp_type, limit=10)
        print_experiment_table(rows, f"Best {exp_type} runs", verbose=args.verbose)
    
    elif args.command == 'compare':
        if len(args.args) < 2:
            print("Usage: experiments.py compare ID1 ID2 [ID3 ...]")
            sys.exit(1)
        rows = tracker.compare_experiments(args.args)
        print_comparison(rows)
    
    elif args.command == 'rate':
        if len(args.args) < 2:
            print("Usage: experiments.py rate ID RATING [NOTE]")
            sys.exit(1)
        exp_id = args.args[0]
        rating = int(args.args[1])
        notes = ' '.join(args.args[2:]) if len(args.args) > 2 else None
        
        if tracker.rate_experiment(exp_id, rating, notes):
            print(f"Rated {exp_id}: {format_rating(str(rating))}")
        else:
            print(f"Experiment not found: {exp_id}")
    
    elif args.command == 'view':
        if not args.args:
            print("Usage: experiments.py view ID")
            sys.exit(1)
        exp = tracker.get_experiment(args.args[0])
        if exp:
            print_experiment_detail(exp)
        else:
            print(f"Experiment not found: {args.args[0]}")
    
    elif args.command == 'stats':
        print_statistics(tracker)


if __name__ == '__main__':
    main()
