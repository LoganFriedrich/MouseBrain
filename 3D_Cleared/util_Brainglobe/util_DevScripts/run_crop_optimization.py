#!/usr/bin/env python3
"""
run_crop_optimization.py

Find the optimal Y-axis crop position for brain registration.

This uses a hill-climbing algorithm to iteratively find the best position
to crop the spinal cord/lower tissue while preserving essential brain regions.
The algorithm balances registration quality against preserving tissue.

================================================================================
THE PROBLEM
================================================================================
Lightsheet brain samples often include spinal cord tissue at the bottom that
confuses atlas registration. We need to remove enough to get good registration,
but not so much that we lose brainstem regions we care about.

================================================================================
THE SOLUTION
================================================================================
This script:
1. Starts at a position you specify (or click interactively)
2. Tests registration quality at that position
3. Tests positions ABOVE and BELOW
4. Moves toward better scores
5. Shrinks step size and repeats
6. Stops when converged

The scoring penalizes removing tissue, so if quality is similar,
it prefers keeping more brain.

================================================================================
USAGE
================================================================================
Interactive mode (with napari):
    python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --interactive

CLI mode (start at specific Y):
    python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --start-y 6000

CLI mode (start at percentage):
    python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --start-pct 60

With custom parameters:
    python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --start-pct 60 \\
        --penalty 0.15 --step 15 --min-step 3

Dry run (see what would happen):
    python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --dry-run

================================================================================
ALGORITHM
================================================================================
Hill-climbing with adaptive step size:

    1. Start at initial Y position
    2. Test current position → get quality score
    3. Test Y + step and Y - step
    4. If better position found: move there, repeat from step 2
    5. If no better position: shrink step size by half
    6. Stop when step size < minimum threshold
    
    Combined Score = quality_score - (penalty × crop_fraction)
    
    This means:
    - If Y=8000 gives quality 0.85 and Y=7000 gives 0.84
      → Prefer Y=8000 (more brain kept, similar quality)
    - If Y=8000 gives quality 0.85 and Y=7000 gives 0.65
      → Prefer Y=7000 (quality difference too big)

================================================================================
QUALITY ASSESSMENT
================================================================================
Registration quality is measured by comparing brainstem region volumes
in the registered sample against atlas reference values:

    - Medulla (354): Should be ~100% of atlas
    - Pons (771): Should be ~100% of atlas
    - Midbrain (313): Should be ~100% of atlas
    - Cerebellum (512): Should be ~100% of atlas
    - Hindbrain (1065): Should be ~100% of atlas

Regions at 70-130% of atlas volume score 1.0 (good)
Regions at 50-150% of atlas volume score 0.5 (ok)
Regions outside this range score 0.0 (bad)

================================================================================
OUTPUT
================================================================================
Results are logged to the experiment tracker (experiment_log.csv) and
the optimal crop is saved to the pipeline's 2_Cropped_For_Registration folder.

================================================================================
REQUIREMENTS
================================================================================
    pip install napari brainglobe-atlasapi brainreg tifffile numpy
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from experiment_tracker import ExperimentTracker

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Pipeline folder names
FOLDER_EXTRACTED = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"

# Hill-climbing parameters
DEFAULT_INITIAL_STEP_PCT = 10   # Start with ±10% steps
DEFAULT_MIN_STEP_PCT = 2        # Stop when step < 2%
DEFAULT_STEP_SHRINK = 0.5       # Halve step on each iteration
DEFAULT_MAX_ITERATIONS = 20     # Safety limit
DEFAULT_CROP_PENALTY = 0.1      # Penalty weight for removing tissue

# Fast atlas for optimization (higher resolution for final)
OPTIMIZATION_ATLAS = "allen_mouse_50um"
FINAL_ATLAS = "allen_mouse_25um"

# Brainstem regions to evaluate (Allen Mouse Brain Atlas IDs)
EVALUATION_REGIONS = {
    'Medulla': 354,
    'Pons': 771,
    'Midbrain': 313,
    'Cerebellum': 512,
    'Hindbrain': 1065,
}

# Quality thresholds (fraction of atlas reference volume)
GOOD_RANGE = (0.70, 1.30)   # 70-130% = full score
OK_RANGE = (0.50, 1.50)     # 50-150% = partial score


# =============================================================================
# BRAIN DISCOVERY
# =============================================================================

def find_brain(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT) -> Optional[Path]:
    """
    Find a brain's pipeline folder.
    
    Args:
        brain_name: Brain identifier (can be partial)
        root: Root folder to search
    
    Returns:
        Path to pipeline folder, or None if not found
    """
    root = Path(root)
    
    # Direct match: root/mouse/pipeline
    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name == pipeline_folder.name:
                return pipeline_folder
            # Also try mouse/pipeline combined name
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


def get_extracted_channel(pipeline_folder: Path, channel: int = 0) -> Optional[Path]:
    """Get path to extracted channel folder."""
    ch_folder = pipeline_folder / FOLDER_EXTRACTED / f"ch{channel}"
    if ch_folder.exists() and len(list(ch_folder.glob("Z*.tif"))) > 0:
        return ch_folder
    return None


def get_metadata(pipeline_folder: Path) -> Dict:
    """Load metadata from pipeline folder."""
    meta_path = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


def get_image_shape(channel_folder: Path) -> Tuple[int, int, int]:
    """Get Z, Y, X shape from TIFF stack folder."""
    import tifffile
    
    tiff_files = sorted(channel_folder.glob("Z*.tif"))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {channel_folder}")
    
    # Get Z from count
    z = len(tiff_files)
    
    # Get Y, X from first file
    first_img = tifffile.imread(str(tiff_files[0]))
    y, x = first_img.shape
    
    return (z, y, x)


# =============================================================================
# CROPPING
# =============================================================================

def crop_channel_to_temp(
    source_folder: Path,
    y_start: int,
    temp_dir: Path,
) -> Path:
    """
    Copy a channel folder with Y-axis cropping to temp directory.
    
    Args:
        source_folder: Path to source channel (with Z*.tif files)
        y_start: Y position to start crop (removes rows 0 to y_start-1)
        temp_dir: Temporary directory to write to
    
    Returns:
        Path to cropped channel folder
    """
    import tifffile
    
    output_folder = temp_dir / "cropped_ch0"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    tiff_files = sorted(source_folder.glob("Z*.tif"))
    
    for tiff_path in tiff_files:
        img = tifffile.imread(str(tiff_path))
        
        # Crop Y axis (rows 0 to y_start-1 are removed)
        cropped = img[y_start:, :]
        
        # Save with same name
        output_path = output_folder / tiff_path.name
        tifffile.imwrite(str(output_path), cropped)
    
    return output_folder


def apply_final_crop(
    source_folder: Path,
    output_folder: Path,
    y_start: int,
    metadata: Dict,
) -> Path:
    """
    Apply final crop and save to pipeline folder.
    
    Args:
        source_folder: Path to source channel folder
        output_folder: Path to output folder (2_Cropped_For_Registration)
        y_start: Y position to start crop
        metadata: Original metadata dict
    
    Returns:
        Path to cropped channel folder
    """
    import tifffile
    
    # Create output structure
    output_folder.mkdir(parents=True, exist_ok=True)
    ch_folder = output_folder / "ch0"
    ch_folder.mkdir(exist_ok=True)
    
    # Get source shape for metadata
    source_shape = get_image_shape(source_folder)
    
    tiff_files = sorted(source_folder.glob("Z*.tif"))
    
    print(f"    Cropping {len(tiff_files)} slices at Y={y_start}...")
    for i, tiff_path in enumerate(tiff_files):
        img = tifffile.imread(str(tiff_path))
        cropped = img[y_start:, :]
        
        output_path = ch_folder / tiff_path.name
        tifffile.imwrite(str(output_path), cropped)
        
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(tiff_files)}...")
    
    # Save metadata
    crop_meta = metadata.copy()
    crop_meta['crop_y_start'] = y_start
    crop_meta['crop_y_original'] = source_shape[1]
    crop_meta['crop_y_new'] = source_shape[1] - y_start
    crop_meta['dimensions_original'] = {
        'z': source_shape[0],
        'y': source_shape[1],
        'x': source_shape[2],
    }
    crop_meta['dimensions_cropped'] = {
        'z': source_shape[0],
        'y': source_shape[1] - y_start,
        'x': source_shape[2],
    }
    
    meta_path = output_folder / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(crop_meta, f, indent=2)
    
    print(f"    Saved cropped data to {output_folder}")
    return ch_folder


# =============================================================================
# REGISTRATION & QUALITY EVALUATION
# =============================================================================

def run_fast_registration(
    input_folder: Path,
    output_folder: Path,
    voxel_sizes: Tuple[float, float, float],
    orientation: str = "iar",
    atlas: str = OPTIMIZATION_ATLAS,
) -> bool:
    """
    Run brainreg with fast settings for optimization.
    
    Returns:
        True if successful, False otherwise
    """
    vz, vy, vx = voxel_sizes
    
    cmd = [
        "brainreg",
        str(input_folder),
        str(output_folder),
        "-v", str(vz), str(vy), str(vx),
        "--orientation", orientation,
        "--atlas", atlas,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("      Warning: Registration timed out")
        return False
    except Exception as e:
        print(f"      Warning: Registration failed: {e}")
        return False


def evaluate_registration_quality(
    registration_folder: Path,
    atlas_name: str = OPTIMIZATION_ATLAS,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate registration quality by comparing brainstem region volumes
    to atlas reference values.
    
    Args:
        registration_folder: Path to brainreg output folder
        atlas_name: Name of atlas used
    
    Returns:
        (overall_score, region_scores_dict)
        overall_score is 0-1, region_scores is dict of region_name -> score
    """
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
        import tifffile
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        return 0.0, {}
    
    # Load atlas
    try:
        atlas = BrainGlobeAtlas(atlas_name)
    except Exception as e:
        print(f"      Warning: Could not load atlas: {e}")
        return 0.0, {}
    
    # Load registered annotation
    annotation_path = registration_folder / "registered_atlas.tiff"
    if not annotation_path.exists():
        annotation_path = registration_folder / "registered_annotation.tiff"
    
    if not annotation_path.exists():
        print(f"      Warning: No registered atlas found in {registration_folder}")
        return 0.0, {}
    
    try:
        registered = tifffile.imread(str(annotation_path))
    except Exception as e:
        print(f"      Warning: Could not load registered atlas: {e}")
        return 0.0, {}
    
    # Get voxel volume (for converting to physical units)
    voxel_volume = np.prod(atlas.resolution) / 1e9  # Convert to mm³
    
    region_scores = {}
    
    for region_name, region_id in EVALUATION_REGIONS.items():
        # Get atlas reference volume for this region
        try:
            atlas_mask = atlas.annotation == region_id
            # Include child regions
            for child_id in atlas.get_structure_descendants(region_id):
                atlas_mask |= (atlas.annotation == child_id)
            atlas_volume = np.sum(atlas_mask) * voxel_volume
        except Exception:
            atlas_volume = 0
        
        if atlas_volume == 0:
            region_scores[region_name] = 0.0
            continue
        
        # Get sample volume for this region
        sample_mask = registered == region_id
        # Include child regions
        try:
            for child_id in atlas.get_structure_descendants(region_id):
                sample_mask |= (registered == child_id)
        except Exception:
            pass
        
        sample_volume = np.sum(sample_mask) * voxel_volume
        
        # Calculate ratio
        ratio = sample_volume / atlas_volume if atlas_volume > 0 else 0
        
        # Score based on how close to 100%
        if GOOD_RANGE[0] <= ratio <= GOOD_RANGE[1]:
            score = 1.0
        elif OK_RANGE[0] <= ratio <= OK_RANGE[1]:
            # Linear interpolation between OK and GOOD
            if ratio < GOOD_RANGE[0]:
                score = 0.5 + 0.5 * (ratio - OK_RANGE[0]) / (GOOD_RANGE[0] - OK_RANGE[0])
            else:
                score = 0.5 + 0.5 * (OK_RANGE[1] - ratio) / (OK_RANGE[1] - GOOD_RANGE[1])
        else:
            score = 0.0
        
        region_scores[region_name] = round(score, 3)
    
    # Overall score is average of region scores
    if region_scores:
        overall_score = sum(region_scores.values()) / len(region_scores)
    else:
        overall_score = 0.0
    
    return round(overall_score, 4), region_scores


def test_crop_position(
    source_folder: Path,
    y_position: int,
    voxel_sizes: Tuple[float, float, float],
    orientation: str = "iar",
    atlas: str = OPTIMIZATION_ATLAS,
) -> Tuple[float, Dict[str, float]]:
    """
    Test a crop position by:
    1. Cropping to temp folder
    2. Running registration
    3. Evaluating quality
    
    Returns:
        (quality_score, region_scores)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Crop
        cropped_folder = crop_channel_to_temp(source_folder, y_position, temp_path)
        
        # Register
        reg_folder = temp_path / "registration"
        success = run_fast_registration(
            cropped_folder,
            reg_folder,
            voxel_sizes,
            orientation,
            atlas,
        )
        
        if not success:
            return 0.0, {}
        
        # Evaluate
        quality, region_scores = evaluate_registration_quality(reg_folder, atlas)
        
        return quality, region_scores


# =============================================================================
# HILL-CLIMBING OPTIMIZER
# =============================================================================

class HillClimbingOptimizer:
    """
    Hill-climbing optimizer for finding optimal crop position.
    """
    
    def __init__(
        self,
        source_folder: Path,
        image_shape: Tuple[int, int, int],
        voxel_sizes: Tuple[float, float, float],
        orientation: str = "iar",
        initial_step_pct: float = DEFAULT_INITIAL_STEP_PCT,
        min_step_pct: float = DEFAULT_MIN_STEP_PCT,
        step_shrink: float = DEFAULT_STEP_SHRINK,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        crop_penalty: float = DEFAULT_CROP_PENALTY,
        atlas: str = OPTIMIZATION_ATLAS,
        verbose: bool = True,
    ):
        self.source_folder = source_folder
        self.z, self.y, self.x = image_shape
        self.voxel_sizes = voxel_sizes
        self.orientation = orientation
        self.initial_step_pct = initial_step_pct
        self.min_step_pct = min_step_pct
        self.step_shrink = step_shrink
        self.max_iterations = max_iterations
        self.crop_penalty = crop_penalty
        self.atlas = atlas
        self.verbose = verbose
        
        # History for logging
        self.step_history = []
        self.region_history = []
    
    def y_to_pct(self, y: int) -> float:
        """Convert Y position to percentage from top."""
        return (y / self.y) * 100
    
    def pct_to_y(self, pct: float) -> int:
        """Convert percentage to Y position."""
        return int((pct / 100) * self.y)
    
    def combined_score(self, quality: float, y_position: int) -> float:
        """
        Calculate combined score that balances quality vs tissue preservation.
        
        combined = quality - (penalty × crop_fraction)
        
        Higher Y = less cropping = higher combined score for same quality
        """
        crop_fraction = y_position / self.y
        return quality - (self.crop_penalty * crop_fraction)
    
    def test_position(self, y: int) -> Tuple[float, float, Dict[str, float]]:
        """
        Test a position and return (quality, combined_score, region_scores).
        """
        if self.verbose:
            pct = self.y_to_pct(y)
            print(f"    Testing Y={y} ({pct:.1f}%)...")
        
        quality, region_scores = test_crop_position(
            self.source_folder,
            y,
            self.voxel_sizes,
            self.orientation,
            self.atlas,
        )
        
        combined = self.combined_score(quality, y)
        
        if self.verbose:
            print(f"      Quality: {quality:.3f}, Combined: {combined:.3f}")
        
        # Record history
        self.step_history.append({
            'y': y,
            'pct': round(self.y_to_pct(y), 2),
            'quality': round(quality, 4),
            'combined': round(combined, 4),
        })
        self.region_history.append(region_scores)
        
        return quality, combined, region_scores
    
    def optimize(self, start_y: int) -> Tuple[int, float, float, Dict[str, float]]:
        """
        Run hill-climbing optimization.
        
        Args:
            start_y: Initial Y position to test
        
        Returns:
            (optimal_y, quality_score, combined_score, region_scores)
        """
        current_y = start_y
        step = self.pct_to_y(self.initial_step_pct)
        min_step = self.pct_to_y(self.min_step_pct)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Hill-Climbing Optimization")
            print(f"{'='*60}")
            print(f"Image size: {self.z} × {self.y} × {self.x}")
            print(f"Starting at Y={start_y} ({self.y_to_pct(start_y):.1f}%)")
            print(f"Initial step: {step} pixels ({self.initial_step_pct:.1f}%)")
            print(f"Min step: {min_step} pixels ({self.min_step_pct:.1f}%)")
            print(f"Crop penalty: {self.crop_penalty}")
            print()
        
        # Test initial position
        quality, combined, region_scores = self.test_position(current_y)
        best_y = current_y
        best_quality = quality
        best_combined = combined
        best_regions = region_scores
        
        iteration = 0
        
        while step >= min_step and iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"\n  Iteration {iteration} (step={step} pixels)")
            
            # Test positions above and below
            candidates = []
            
            # Test Y + step (more cropping, less brain)
            y_more = min(current_y + step, self.y - 100)  # Keep at least 100 rows
            if y_more != current_y:
                q, c, r = self.test_position(y_more)
                candidates.append((y_more, q, c, r))
            
            # Test Y - step (less cropping, more brain)
            y_less = max(current_y - step, 0)
            if y_less != current_y:
                q, c, r = self.test_position(y_less)
                candidates.append((y_less, q, c, r))
            
            # Find best candidate
            best_candidate = None
            for y, q, c, r in candidates:
                if c > best_combined:
                    best_candidate = (y, q, c, r)
                    best_combined = c
            
            if best_candidate:
                # Move to better position
                current_y, best_quality, best_combined, best_regions = best_candidate
                best_y = current_y
                
                if self.verbose:
                    print(f"  → Moving to Y={current_y} (combined={best_combined:.3f})")
            else:
                # No improvement - shrink step
                step = int(step * self.step_shrink)
                
                if self.verbose:
                    print(f"  → No improvement, shrinking step to {step}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Complete")
            print(f"{'='*60}")
            print(f"Optimal Y: {best_y} ({self.y_to_pct(best_y):.1f}%)")
            print(f"Quality Score: {best_quality:.3f}")
            print(f"Combined Score: {best_combined:.3f}")
            print(f"Iterations: {iteration}")
            print(f"\nRegion Scores:")
            for region, score in best_regions.items():
                status = "✓" if score >= 0.7 else "△" if score >= 0.5 else "✗"
                print(f"  {status} {region}: {score:.2f}")
            print()
        
        return best_y, best_quality, best_combined, best_regions


# =============================================================================
# INTERACTIVE MODE (NAPARI)
# =============================================================================

def run_interactive(
    source_folder: Path,
    image_shape: Tuple[int, int, int],
    voxel_sizes: Tuple[float, float, float],
    optimizer_kwargs: Dict,
) -> Optional[int]:
    """
    Run interactive mode with napari to select starting position.
    
    Returns selected Y position, or None if cancelled.
    """
    try:
        import napari
        import tifffile
    except ImportError:
        print("Error: napari not installed. Use --start-y or --start-pct instead.")
        return None
    
    print("\nLoading image for visualization...")
    
    # Load a subset of slices for preview
    tiff_files = sorted(source_folder.glob("Z*.tif"))
    step = max(1, len(tiff_files) // 50)  # Show ~50 slices
    
    sample_files = tiff_files[::step]
    sample = np.stack([tifffile.imread(str(f)) for f in sample_files])
    
    z, y, x = sample.shape
    
    print(f"Loaded preview: {z} slices")
    print("\nInstructions:")
    print("  1. Click on the image to place a horizontal line")
    print("  2. Drag the line to adjust crop position")
    print("  3. Everything ABOVE the line will be kept")
    print("  4. Press 'Enter' when ready to optimize")
    print("  5. Press 'Escape' to cancel")
    
    selected_y = [None]  # Use list to allow modification in callback
    
    viewer = napari.Viewer(title="Select Crop Position")
    
    # Add image
    viewer.add_image(
        sample,
        name="Brain Preview",
        scale=(step * voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]),
    )
    
    # Add horizontal line (shapes layer)
    initial_y = y // 2
    line_data = np.array([[[0, initial_y, 0], [0, initial_y, x]]])
    
    shapes_layer = viewer.add_shapes(
        line_data,
        shape_type='line',
        edge_color='yellow',
        edge_width=3,
        name="Crop Line",
    )
    
    # Update callback
    @shapes_layer.events.data.connect
    def on_line_moved(event):
        if len(shapes_layer.data) > 0:
            line = shapes_layer.data[0]
            y_pos = int(line[0, 1])
            selected_y[0] = y_pos
            print(f"\r  Crop position: Y={y_pos} ({100*y_pos/image_shape[1]:.1f}% from top)    ", end='')
    
    # Key bindings
    @viewer.bind_key('Enter')
    def confirm(viewer):
        if selected_y[0] is not None:
            viewer.close()
    
    @viewer.bind_key('Escape')
    def cancel(viewer):
        selected_y[0] = None
        viewer.close()
    
    napari.run()
    
    if selected_y[0] is None:
        print("\nCancelled.")
        return None
    
    # Scale Y back to full image coordinates
    final_y = selected_y[0]
    print(f"\n\nSelected: Y={final_y}")
    
    return final_y


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find optimal Y-crop position using hill-climbing optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with napari
  python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --interactive

  # Start at specific Y position
  python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --start-y 6000

  # Start at percentage from top
  python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --start-pct 60

  # Custom optimization parameters
  python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --start-pct 60 \\
      --penalty 0.15 --step 15 --min-step 3

  # Dry run
  python run_crop_optimization.py --brain 349_CNT_01_02_1p625x_z4 --dry-run
        """
    )
    
    # Required
    parser.add_argument('--brain', '-b', required=True,
                        help='Brain/pipeline to process')
    
    # Starting position (one required unless interactive)
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument('--start-y', type=int,
                             help='Starting Y position (pixels)')
    start_group.add_argument('--start-pct', type=float,
                             help='Starting position as %% from top (0-100)')
    start_group.add_argument('--interactive', '-i', action='store_true',
                             help='Select starting position with napari')
    
    # Optimization parameters
    parser.add_argument('--step', type=float, default=DEFAULT_INITIAL_STEP_PCT,
                        help=f'Initial step size in %% (default: {DEFAULT_INITIAL_STEP_PCT})')
    parser.add_argument('--min-step', type=float, default=DEFAULT_MIN_STEP_PCT,
                        help=f'Minimum step size in %% (default: {DEFAULT_MIN_STEP_PCT})')
    parser.add_argument('--penalty', type=float, default=DEFAULT_CROP_PENALTY,
                        help=f'Crop penalty weight (default: {DEFAULT_CROP_PENALTY})')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f'Maximum iterations (default: {DEFAULT_MAX_ITERATIONS})')
    
    # Options
    parser.add_argument('--apply', '-a', action='store_true',
                        help='Apply the optimal crop to pipeline folder')
    parser.add_argument('--channel', '-c', type=int, default=0,
                        help='Channel to use (default: 0)')
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
    print(f"Crop Optimization")
    print(f"{'='*60}")
    print(f"Brain: {pipeline_folder.name}")
    print(f"Pipeline: {pipeline_folder}")
    
    # Get source channel
    source_folder = get_extracted_channel(pipeline_folder, args.channel)
    if not source_folder:
        print(f"Error: No extracted channel found at {pipeline_folder / FOLDER_EXTRACTED}")
        sys.exit(1)
    
    # Get metadata
    metadata = get_metadata(pipeline_folder)
    voxel = metadata.get('voxel_size_um', {})
    voxel_sizes = (
        voxel.get('z', 4.0),
        voxel.get('y', 4.0),
        voxel.get('x', 4.0),
    )
    orientation = metadata.get('orientation', 'iar')
    
    print(f"Voxel sizes: {voxel_sizes}")
    print(f"Orientation: {orientation}")
    
    # Get image shape
    image_shape = get_image_shape(source_folder)
    print(f"Image shape: Z={image_shape[0]}, Y={image_shape[1]}, X={image_shape[2]}")
    
    # Determine starting Y
    if args.interactive:
        start_y = run_interactive(
            source_folder,
            image_shape,
            voxel_sizes,
            {},
        )
        if start_y is None:
            sys.exit(0)
    elif args.start_y:
        start_y = args.start_y
    elif args.start_pct:
        start_y = int((args.start_pct / 100) * image_shape[1])
    else:
        # Default to 50%
        start_y = image_shape[1] // 2
        print(f"No starting position specified, using 50%: Y={start_y}")
    
    start_pct = (start_y / image_shape[1]) * 100
    print(f"Starting position: Y={start_y} ({start_pct:.1f}%)")
    
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would optimize crop position starting at Y={start_y}")
        print(f"Parameters: step={args.step}%, min_step={args.min_step}%, penalty={args.penalty}")
        sys.exit(0)
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Log experiment start
    exp_id = tracker.log_crop(
        brain=pipeline_folder.name,
        start_y=start_y,
        start_pct=round(start_pct, 2),
        penalty_weight=args.penalty,
        algorithm="hill_climbing",
        atlas=OPTIMIZATION_ATLAS,
        regions_evaluated=",".join(EVALUATION_REGIONS.keys()),
        output_path=str(pipeline_folder / FOLDER_CROPPED),
        status="started",
        notes=args.notes,
    )
    
    print(f"\nExperiment: {exp_id}")
    
    # Run optimization
    start_time = time.time()
    
    try:
        optimizer = HillClimbingOptimizer(
            source_folder=source_folder,
            image_shape=image_shape,
            voxel_sizes=voxel_sizes,
            orientation=orientation,
            initial_step_pct=args.step,
            min_step_pct=args.min_step,
            crop_penalty=args.penalty,
            max_iterations=args.max_iter,
            atlas=OPTIMIZATION_ATLAS,
        )
        
        optimal_y, quality, combined, region_scores = optimizer.optimize(start_y)
        optimal_pct = (optimal_y / image_shape[1]) * 100
        
        duration = time.time() - start_time
        
        # Update experiment with results
        tracker.update_status(
            exp_id,
            status="completed",
            duration_seconds=round(duration, 1),
            crop_optimal_y=optimal_y,
            crop_optimal_pct=round(optimal_pct, 2),
            crop_quality_score=round(quality, 4),
            crop_combined_score=round(combined, 4),
            crop_iterations=len(optimizer.step_history),
            crop_region_scores=json.dumps(region_scores),
            crop_step_history=json.dumps(optimizer.step_history),
        )
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Optimal crop: Y={optimal_y} ({optimal_pct:.1f}%)")
        print(f"Quality score: {quality:.3f}")
        print(f"Combined score: {combined:.3f}")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Experiment: {exp_id}")
        
        # Apply crop if requested
        if args.apply:
            print(f"\nApplying optimal crop...")
            output_folder = pipeline_folder / FOLDER_CROPPED
            apply_final_crop(source_folder, output_folder, optimal_y, metadata)
            print(f"Saved to: {output_folder}")
        else:
            print(f"\nTo apply this crop, run:")
            print(f"  python run_crop_optimization.py --brain {pipeline_folder.name} "
                  f"--start-y {optimal_y} --apply")
        
        # Prompt for rating
        print()
        try:
            rating = input("Rate this optimization (1-5, or Enter to skip): ").strip()
            if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                note = input("Add a note (or Enter to skip): ").strip()
                tracker.rate_experiment(exp_id, int(rating), note if note else None)
        except (EOFError, KeyboardInterrupt):
            pass
        
    except Exception as e:
        duration = time.time() - start_time
        tracker.update_status(
            exp_id,
            status="failed",
            duration_seconds=round(duration, 1),
            notes=f"Error: {str(e)}",
        )
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
