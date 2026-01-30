#!/usr/bin/env python3
"""
run_counts.py

Wrapper for brainglobe-segmentation that auto-logs to experiment tracker.

This counts classified cells by brain region using the registered atlas,
producing a CSV of cell counts per anatomical structure.

================================================================================
WHAT IT DOES
================================================================================
1. Takes classified cells (from 5_Classified_Cells/)
2. Uses the registered atlas (from 3_Registered_Atlas/)
3. Assigns each cell to a brain region
4. Outputs counts per region as CSV

================================================================================
USAGE
================================================================================
Basic (auto-finds paths):
    python run_counts.py --brain 349_CNT_01_02_1p625x_z4

With specific paths:
    python run_counts.py --brain 349_CNT_01_02_1p625x_z4 \\
        --cells path/to/cells.xml \\
        --registration path/to/registration

Dry run:
    python run_counts.py --brain 349_CNT_01_02_1p625x_z4 --dry-run

================================================================================
OUTPUT
================================================================================
Results go to: {pipeline}/6_Region_Analysis/
    - cell_counts.csv        : Counts per brain region
    - cell_positions.csv     : All cells with their region assignments
    - summary.json           : Quick summary statistics

================================================================================
REQUIREMENTS
================================================================================
    conda activate brainglobe-env
    pip install brainglobe-segmentation
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from experiment_tracker import ExperimentTracker

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Pipeline folder names
FOLDER_REGISTRATION = "3_Registered_Atlas"
FOLDER_CLASSIFICATION = "5_Classified_Cells"
FOLDER_ANALYSIS = "6_Region_Analysis"


# =============================================================================
# BRAIN DISCOVERY
# =============================================================================

def find_brain(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT) -> Optional[Path]:
    """Find a brain's pipeline folder."""
    root = Path(root)
    
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name == pipeline_folder.name:
                return pipeline_folder
            if brain_name == f"{mouse_folder.name}/{pipeline_folder.name}":
                return pipeline_folder
    
    # Partial match
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name in pipeline_folder.name:
                return pipeline_folder
    
    return None


def find_cells_file(pipeline_folder: Path) -> Optional[Path]:
    """Find classified cells file."""
    classification_folder = pipeline_folder / FOLDER_CLASSIFICATION
    
    # Try common names
    for name in ["cells.xml", "cell_classification.xml", "classified_cells.xml"]:
        path = classification_folder / name
        if path.exists():
            return path
    
    # Try any XML file
    xml_files = list(classification_folder.glob("*.xml"))
    if xml_files:
        return xml_files[0]
    
    # Try NPY format
    npy_files = list(classification_folder.glob("*.npy"))
    if npy_files:
        return npy_files[0]
    
    return None


def find_registration(pipeline_folder: Path) -> Optional[Path]:
    """Find registration folder."""
    reg_folder = pipeline_folder / FOLDER_REGISTRATION
    
    if reg_folder.exists():
        # Verify it has required files
        if (reg_folder / "registered_atlas.tiff").exists():
            return reg_folder
        if (reg_folder / "registered_annotation.tiff").exists():
            return reg_folder
    
    return None


# =============================================================================
# CELL COUNTING
# =============================================================================

def count_cells_in_file(cells_path: Path) -> int:
    """Count cells in a cells file (XML or NPY)."""
    if cells_path.suffix.lower() == '.xml':
        try:
            tree = ET.parse(cells_path)
            root = tree.getroot()
            return len(root.findall('.//Cell'))
        except Exception:
            return 0
    
    elif cells_path.suffix.lower() == '.npy':
        try:
            import numpy as np
            cells = np.load(str(cells_path))
            return len(cells)
        except Exception:
            return 0
    
    return 0


def run_brainglobe_segmentation(
    cells_path: Path,
    registration_path: Path,
    output_path: Path,
    atlas: str = None,
) -> Tuple[bool, float, int, Path]:
    """
    Run brainglobe-segmentation to assign cells to regions.
    
    Returns:
        (success, duration_seconds, total_cells, regions_file_path)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # brainglobe-segmentation command
    cmd = [
        "brainglobe-segmentation",
        "-c", str(cells_path),
        "-r", str(registration_path),
        "-o", str(output_path),
    ]
    
    if atlas:
        cmd.extend(["--atlas", atlas])
    
    print(f"\nRunning: brainglobe-segmentation")
    print(f"  Cells: {cells_path}")
    print(f"  Registration: {registration_path}")
    print(f"  Output: {output_path}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Find output file
            regions_file = output_path / "cell_counts.csv"
            if not regions_file.exists():
                # Try alternative names
                for name in ["summary.csv", "regions.csv", "counts.csv"]:
                    alt = output_path / name
                    if alt.exists():
                        regions_file = alt
                        break
            
            # Count total cells from output
            total_cells = 0
            if regions_file.exists():
                try:
                    with open(regions_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Sum up cell counts
                            for key in ['count', 'cells', 'n_cells', 'cell_count']:
                                if key in row:
                                    try:
                                        total_cells += int(row[key])
                                    except ValueError:
                                        pass
                                    break
                except Exception:
                    pass
            
            return True, duration, total_cells, regions_file
        else:
            return False, duration, 0, None
    
    except FileNotFoundError:
        # brainglobe-segmentation not installed, try manual approach
        print("  brainglobe-segmentation not found, using manual counting...")
        return run_manual_counting(cells_path, registration_path, output_path)
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error: {e}")
        return False, duration, 0, None


def run_manual_counting(
    cells_path: Path,
    registration_path: Path,
    output_path: Path,
) -> Tuple[bool, float, int, Path]:
    """
    Manual cell counting using brainglobe-atlasapi.
    
    Fallback if brainglobe-segmentation isn't installed.
    """
    start_time = time.time()
    
    try:
        import numpy as np
        import tifffile
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        return False, 0, 0, None
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load cells
    if cells_path.suffix.lower() == '.xml':
        cells = parse_cells_xml(cells_path)
    else:
        cells = np.load(str(cells_path))
    
    if len(cells) == 0:
        print("No cells found in input file")
        return False, time.time() - start_time, 0, None
    
    print(f"  Loaded {len(cells)} cells")
    
    # Load registered atlas
    atlas_path = registration_path / "registered_atlas.tiff"
    if not atlas_path.exists():
        atlas_path = registration_path / "registered_annotation.tiff"
    
    if not atlas_path.exists():
        print(f"Error: No registered atlas found in {registration_path}")
        return False, time.time() - start_time, 0, None
    
    registered = tifffile.imread(str(atlas_path))
    print(f"  Loaded atlas: {registered.shape}")
    
    # Load atlas for region names
    # Try to determine atlas from brainreg.json
    atlas_name = "allen_mouse_25um"
    brainreg_json = registration_path / "brainreg.json"
    if brainreg_json.exists():
        try:
            with open(brainreg_json, 'r') as f:
                config = json.load(f)
            atlas_name = config.get('atlas', atlas_name)
        except Exception:
            pass
    
    try:
        atlas = BrainGlobeAtlas(atlas_name)
    except Exception as e:
        print(f"Warning: Could not load atlas {atlas_name}: {e}")
        atlas = None
    
    # Count cells per region
    region_counts = {}
    cell_assignments = []
    
    for cell in cells:
        # Get cell position (z, y, x)
        if isinstance(cell, dict):
            z, y, x = int(cell['z']), int(cell['y']), int(cell['x'])
        else:
            z, y, x = int(cell[0]), int(cell[1]), int(cell[2])
        
        # Clamp to image bounds
        z = max(0, min(z, registered.shape[0] - 1))
        y = max(0, min(y, registered.shape[1] - 1))
        x = max(0, min(x, registered.shape[2] - 1))
        
        # Get region ID at this position
        region_id = int(registered[z, y, x])
        
        # Get region name
        region_name = f"Region_{region_id}"
        if atlas and region_id > 0:
            try:
                structure = atlas.structures[region_id]
                region_name = structure.get('name', region_name)
            except (KeyError, TypeError):
                pass
        
        # Count
        if region_name not in region_counts:
            region_counts[region_name] = {'id': region_id, 'count': 0}
        region_counts[region_name]['count'] += 1
        
        cell_assignments.append({
            'z': z, 'y': y, 'x': x,
            'region_id': region_id,
            'region_name': region_name,
        })
    
    duration = time.time() - start_time
    
    # Save results
    regions_file = output_path / "cell_counts.csv"
    with open(regions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_name', 'region_id', 'cell_count'])
        for name, data in sorted(region_counts.items(), key=lambda x: -x[1]['count']):
            writer.writerow([name, data['id'], data['count']])
    
    positions_file = output_path / "cell_positions.csv"
    with open(positions_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['z', 'y', 'x', 'region_id', 'region_name'])
        writer.writeheader()
        writer.writerows(cell_assignments)
    
    # Save summary
    summary = {
        'total_cells': len(cells),
        'regions_with_cells': len(region_counts),
        'top_regions': [
            {'name': name, 'count': data['count']}
            for name, data in sorted(region_counts.items(), key=lambda x: -x[1]['count'])[:10]
        ],
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Assigned {len(cells)} cells to {len(region_counts)} regions")
    
    return True, duration, len(cells), regions_file


def parse_cells_xml(xml_path: Path) -> List[Dict]:
    """Parse cells from XML file."""
    cells = []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for cell in root.findall('.//Cell'):
            z = float(cell.get('z', 0))
            y = float(cell.get('y', 0))
            x = float(cell.get('x', 0))
            cells.append({'z': z, 'y': y, 'x': x})
    
    except Exception as e:
        print(f"Error parsing XML: {e}")
    
    return cells


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Count cells by brain region with auto-logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic (auto-finds cells and registration)
  python run_counts.py --brain 349_CNT_01_02_1p625x_z4

  # Specify paths explicitly
  python run_counts.py --brain 349_CNT_01_02_1p625x_z4 \\
      --cells path/to/cells.xml \\
      --registration path/to/registration

  # Link to classification experiment
  python run_counts.py --brain 349_CNT_01_02_1p625x_z4 \\
      --parent cla_20241218_abc123

  # Dry run
  python run_counts.py --brain 349_CNT_01_02_1p625x_z4 --dry-run
        """
    )
    
    # Required
    parser.add_argument('--brain', '-b', required=True,
                        help='Brain/pipeline to process')
    
    # Optional paths (auto-discovered if not provided)
    parser.add_argument('--cells', '-c', type=Path,
                        help='Path to classified cells file')
    parser.add_argument('--registration', '-r', type=Path,
                        help='Path to registration folder')
    
    # Options
    parser.add_argument('--atlas', '-a',
                        help='Atlas name (default: from registration)')
    parser.add_argument('--parent', help='Parent experiment ID to link to')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Re-run even if counts exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without running')
    parser.add_argument('--notes', help='Notes to add to experiment log')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT,
                        help='Root folder for brains')
    
    args = parser.parse_args()
    
    # Find brain
    pipeline_folder = find_brain(args.brain, args.root)
    if not pipeline_folder:
        print(f"Error: Brain not found: {args.brain}")
        print(f"Searched in: {args.root}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Regional Cell Counting")
    print(f"{'='*60}")
    print(f"Brain: {pipeline_folder.name}")
    print(f"Pipeline: {pipeline_folder}")
    
    # Find cells file
    if args.cells:
        cells_path = args.cells
    else:
        cells_path = find_cells_file(pipeline_folder)
    
    if not cells_path or not cells_path.exists():
        print(f"\nError: No classified cells found")
        print(f"  Looked in: {pipeline_folder / FOLDER_CLASSIFICATION}")
        print(f"\nRun classification first with run_classification.py")
        sys.exit(1)
    
    print(f"Cells: {cells_path}")
    
    # Count input cells
    input_cell_count = count_cells_in_file(cells_path)
    print(f"Input cells: {input_cell_count}")
    
    # Find registration
    if args.registration:
        registration_path = args.registration
    else:
        registration_path = find_registration(pipeline_folder)
    
    if not registration_path or not registration_path.exists():
        print(f"\nError: No registration found")
        print(f"  Looked in: {pipeline_folder / FOLDER_REGISTRATION}")
        print(f"\nRun registration first with run_registration.py")
        sys.exit(1)
    
    print(f"Registration: {registration_path}")
    
    # Output folder
    output_folder = pipeline_folder / FOLDER_ANALYSIS
    
    # Check if already done
    if output_folder.exists() and (output_folder / "cell_counts.csv").exists():
        if not args.force:
            print(f"\nCounts already exist: {output_folder}")
            print("Use --force to re-run")
            sys.exit(0)
        else:
            print(f"\nRe-running counts (--force)")
    
    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Would count {input_cell_count} cells by region")
        print(f"Output would go to: {output_folder}")
        sys.exit(0)
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Log experiment start
    exp_id = tracker.log_counts(
        brain=pipeline_folder.name,
        registration_path=str(registration_path),
        cells_path=str(cells_path),
        atlas=args.atlas,
        output_path=str(output_folder),
        parent_experiment=args.parent,
        status="started",
        notes=args.notes,
    )
    
    print(f"\nExperiment: {exp_id}")
    
    # Run counting
    success, duration, total_cells, regions_file = run_brainglobe_segmentation(
        cells_path=cells_path,
        registration_path=registration_path,
        output_path=output_folder,
        atlas=args.atlas,
    )
    
    # Update experiment status
    if success:
        tracker.update_status(
            exp_id,
            status="completed",
            duration_seconds=round(duration, 1),
            counts_total_cells=total_cells,
            counts_regions_file=str(regions_file) if regions_file else "",
        )
        
        print(f"\n{'='*60}")
        print(f"COUNTING COMPLETE")
        print(f"{'='*60}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total cells counted: {total_cells}")
        print(f"Results: {regions_file}")
        print(f"Experiment: {exp_id}")
        
        # Show top regions
        if regions_file and regions_file.exists():
            print(f"\nTop regions:")
            try:
                with open(regions_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)[:10]
                    for row in rows:
                        name = row.get('region_name', row.get('name', 'Unknown'))
                        count = row.get('cell_count', row.get('count', '?'))
                        print(f"  {name}: {count}")
            except Exception:
                pass
        
        # Prompt for rating
        print()
        try:
            rating = input("Rate this counting run (1-5, or Enter to skip): ").strip()
            if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                note = input("Add a note (or Enter to skip): ").strip()
                tracker.rate_experiment(exp_id, int(rating), note if note else None)
        except (EOFError, KeyboardInterrupt):
            pass
    
    else:
        tracker.update_status(
            exp_id,
            status="failed",
            duration_seconds=round(duration, 1),
        )
        
        print(f"\n{'='*60}")
        print(f"COUNTING FAILED")
        print(f"{'='*60}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Experiment: {exp_id}")
        sys.exit(1)


if __name__ == '__main__':
    main()
