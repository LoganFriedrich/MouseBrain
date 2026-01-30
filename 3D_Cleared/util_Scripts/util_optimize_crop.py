#!/usr/bin/env python3
"""
util_optimize_crop.py

Utility: Find the optimal Y-crop for atlas registration using iterative testing.

================================================================================
THE PROBLEM
================================================================================
The gradient-based crop detection in Script 2 sometimes removes too much tissue,
cutting into the brainstem and cerebellum. This causes registration failures
because these regions are distorted or missing.

================================================================================
THE SOLUTION
================================================================================
This script takes a brute-force optimization approach:

1. Try multiple crop fractions (0%, 10%, 20%, 30%, 40%, 50%)
2. For each crop, run brainreg registration
3. Evaluate registration quality by checking if key brainstem regions
   have reasonable 3D shapes compared to the atlas
4. Find the MINIMUM crop that still produces good registration

================================================================================
HOW IT WORKS
================================================================================
For each crop level:
  1. Copy the full extraction and crop to that Y level
  2. Run brainreg with the cropped data
  3. Load the registered atlas and check these brainstem regions:
     - Medulla (ID: 354)
     - Pons (ID: 771)  
     - Midbrain (ID: 313)
     - Cerebellum (ID: 512)
     - Hindbrain (ID: 1065)
  4. Calculate 3D shape metrics:
     - Volume ratio (vs atlas reference)
     - Compactness (surface^3 / (36*pi * volume^2), sphere = 1)
     - Aspect ratios
  5. Compare to atlas reference values
  6. Score as GOOD, MARGINAL, or BAD

The optimal crop is the minimum that produces GOOD shapes for all key regions.

================================================================================
HOW TO USE
================================================================================
    python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4
    python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4 --crops 0,10,20,30
    python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4 --quick  # Only 0,25,50%

================================================================================
OUTPUT
================================================================================
Creates in the pipeline folder:
    _crop_optimization/
    ├── optimization_results.json   # Summary of all tests
    ├── crop_0pct/                  # Registration with 0% crop
    │   └── brainreg output...
    ├── crop_10pct/
    ├── ...
    └── recommended_crop.txt        # The winning crop percentage

================================================================================
REQUIREMENTS
================================================================================
    pip install brainreg brainglobe-atlasapi numpy scipy scikit-image tifffile
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "1.0.0"

from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT

# Pipeline folders
FOLDER_EXTRACTED_FULL = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"

# Optimization output folder
FOLDER_OPTIMIZATION = "_crop_optimization"

# Default crop percentages to test (percentage of Y removed from posterior)
DEFAULT_CROPS = [0, 10, 20, 30, 40, 50]
QUICK_CROPS = [0, 25, 50]

# Atlas and registration settings
DEFAULT_ATLAS = "allen_mouse_10um"
DEFAULT_N_FREE_CPUS = 106

# Brainstem regions to evaluate (Allen Mouse Brain Atlas structure IDs)
EVALUATION_REGIONS = {
    'Medulla': 354,
    'Pons': 771,
    'Midbrain': 313,
    'Cerebellum': 512,
    'Hindbrain': 1065,
}

# Quality thresholds
VOLUME_RATIO_MIN = 0.5   # Region should be at least 50% of atlas volume
VOLUME_RATIO_MAX = 1.5   # Region should be at most 150% of atlas volume
COMPACTNESS_MAX = 3.0    # Compactness should be < 3x atlas value
ASPECT_RATIO_MAX = 2.0   # Aspect ratios should be < 2x atlas value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def find_pipeline(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find pipeline folder for a brain."""
    root = Path(root)
    
    for mouse_dir in root.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            full_name = f"{mouse_dir.name}/{pipeline_dir.name}"
            if brain_name in [pipeline_dir.name, full_name]:
                return pipeline_dir, mouse_dir
    
    return None, None


# =============================================================================
# CROPPING
# =============================================================================

def create_cropped_copy(
    full_folder: Path,
    output_folder: Path,
    crop_fraction: float,
    metadata: dict,
) -> Tuple[int, int]:
    """
    Create a cropped copy of the extracted data.
    
    Args:
        full_folder: Path to 1_Extracted_Full
        output_folder: Where to put the cropped data
        crop_fraction: Fraction of Y to remove (0.0-1.0)
        metadata: Metadata dict from full extraction
    
    Returns:
        (original_y, cropped_y)
    """
    import tifffile
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get dimensions
    original_y = metadata.get('dimensions', {}).get('y', 0)
    if not original_y:
        # Measure from first slice
        ch0 = full_folder / "ch0"
        first_tif = next(ch0.glob("Z*.tif"))
        img = tifffile.imread(str(first_tif))
        original_y = img.shape[0]  # Y is first dimension in ZYX
    
    # Calculate crop
    y_to_keep = int(original_y * (1 - crop_fraction))
    
    print(f"    [{timestamp()}] Creating {crop_fraction*100:.0f}% crop: Y={original_y} -> {y_to_keep}")
    
    # Process each channel
    channels = metadata.get('channels', {})
    num_channels = channels.get('count', 2)
    
    for ch_idx in range(num_channels):
        ch_in = full_folder / f"ch{ch_idx}"
        ch_out = output_folder / f"ch{ch_idx}"
        
        if not ch_in.exists():
            continue
        
        ch_out.mkdir(exist_ok=True)
        
        tif_files = sorted(ch_in.glob("Z*.tif"))
        
        for tif_path in tif_files:
            img = tifffile.imread(str(tif_path))
            
            # Crop in Y (first dimension)
            cropped = img[:y_to_keep, :]
            
            tifffile.imwrite(str(ch_out / tif_path.name), cropped)
    
    # Create metadata for cropped data
    crop_metadata = metadata.copy()
    if 'dimensions' not in crop_metadata:
        crop_metadata['dimensions'] = {}
    crop_metadata['dimensions']['y'] = y_to_keep
    crop_metadata['crop'] = {
        'crop_axis': 'Y',
        'original_y': original_y,
        'cropped_y': y_to_keep,
        'crop_fraction': crop_fraction,
        'removed_rows': original_y - y_to_keep,
    }
    
    with open(output_folder / "metadata.json", 'w') as f:
        json.dump(crop_metadata, f, indent=2)
    
    return original_y, y_to_keep


# =============================================================================
# REGISTRATION
# =============================================================================

def run_registration(
    data_folder: Path,
    output_folder: Path,
    metadata: dict,
    atlas: str = DEFAULT_ATLAS,
    n_free_cpus: int = DEFAULT_N_FREE_CPUS,
) -> Tuple[bool, float]:
    """
    Run brainreg registration.
    
    Returns:
        (success, duration_seconds)
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get voxel sizes
    voxel = metadata.get('voxel_size_um', {})
    voxel_z = voxel.get('z', 4)
    voxel_xy = voxel.get('x', 4)
    
    # Get channel info
    channels = metadata.get('channels', {})
    background_ch = channels.get('background_channel', 1)
    
    # Input is the background channel (for registration)
    input_path = data_folder / f"ch{background_ch}"
    if not input_path.exists():
        input_path = data_folder / "ch0"
    
    orientation = metadata.get('orientation', 'iar')
    
    cmd = [
        "brainreg",
        str(input_path),
        str(output_folder),
        "-v", str(voxel_z), str(voxel_xy), str(voxel_xy),
        "--orientation", orientation,
        "--atlas", atlas,
    ]
    
    if n_free_cpus:
        cmd.extend(["--n-free-cpus", str(n_free_cpus)])
    
    print(f"    [{timestamp()}] Running brainreg...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        success = result.returncode == 0 and (output_folder / "brainreg.json").exists()
        return success, duration
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return False, time.time() - start_time


# =============================================================================
# QUALITY EVALUATION
# =============================================================================

def get_atlas_reference_metrics(atlas_name: str = DEFAULT_ATLAS) -> Dict:
    """
    Get reference shape metrics from the atlas.
    """
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
        
        atlas = BrainGlobeAtlas(atlas_name)
        annotation = atlas.annotation
        
        reference = {}
        
        for region_name, region_id in EVALUATION_REGIONS.items():
            mask = annotation == region_id
            
            if not mask.any():
                continue
            
            metrics = calculate_shape_metrics(mask)
            reference[region_name] = {
                'id': region_id,
                'volume': metrics['volume'],
                'compactness': metrics['compactness'],
                'aspect_ratio_xy': metrics['aspect_ratio_xy'],
                'aspect_ratio_z': metrics['aspect_ratio_z'],
            }
        
        return reference
        
    except Exception as e:
        print(f"    Warning: Could not get atlas reference: {e}")
        return {}


def calculate_shape_metrics(mask: np.ndarray) -> Dict:
    """
    Calculate 3D shape metrics for a binary mask.
    
    Returns:
        Dictionary with volume, surface_area, compactness, bounding_box, aspect_ratios
    """
    from scipy import ndimage
    
    # Volume = count of voxels
    volume = np.sum(mask)
    
    if volume == 0:
        return {
            'volume': 0,
            'surface_area': 0,
            'compactness': float('inf'),
            'bbox': (0, 0, 0, 0, 0, 0),
            'aspect_ratio_xy': 0,
            'aspect_ratio_z': 0,
        }
    
    # Surface area approximation using erosion
    # Count boundary voxels (voxels adjacent to background)
    eroded = ndimage.binary_erosion(mask)
    surface_voxels = np.sum(mask & ~eroded)
    surface_area = surface_voxels * 6  # Rough approximation
    
    # Compactness: surface^3 / (36*pi * volume^2)
    # For a sphere, this equals 1. Higher values = less compact
    if volume > 0:
        compactness = (surface_area ** 3) / (36 * np.pi * volume ** 2)
    else:
        compactness = float('inf')
    
    # Bounding box
    coords = np.where(mask)
    if len(coords[0]) > 0:
        bbox = (
            coords[0].min(), coords[0].max(),
            coords[1].min(), coords[1].max(),
            coords[2].min(), coords[2].max(),
        )
        z_size = bbox[1] - bbox[0] + 1
        y_size = bbox[3] - bbox[2] + 1
        x_size = bbox[5] - bbox[4] + 1
        
        aspect_ratio_xy = max(x_size, y_size) / max(min(x_size, y_size), 1)
        aspect_ratio_z = max(z_size, max(x_size, y_size)) / max(min(z_size, min(x_size, y_size)), 1)
    else:
        bbox = (0, 0, 0, 0, 0, 0)
        aspect_ratio_xy = 0
        aspect_ratio_z = 0
    
    return {
        'volume': int(volume),
        'surface_area': int(surface_area),
        'compactness': float(compactness),
        'bbox': bbox,
        'aspect_ratio_xy': float(aspect_ratio_xy),
        'aspect_ratio_z': float(aspect_ratio_z),
    }


def evaluate_registration(
    registration_folder: Path,
    reference_metrics: Dict,
    atlas_name: str = DEFAULT_ATLAS,
) -> Dict:
    """
    Evaluate registration quality by checking brainstem region shapes.
    
    Returns:
        Dictionary with per-region scores and overall quality
    """
    try:
        import tifffile
        
        # Load registered atlas
        registered_atlas = registration_folder / "registered_atlas.tiff"
        if not registered_atlas.exists():
            return {'status': 'failed', 'reason': 'no registered atlas'}
        
        annotation = tifffile.imread(str(registered_atlas))
        
        results = {
            'status': 'evaluated',
            'regions': {},
            'scores': {},
            'overall': 'UNKNOWN',
        }
        
        good_count = 0
        bad_count = 0
        
        for region_name, region_id in EVALUATION_REGIONS.items():
            mask = annotation == region_id
            
            if not mask.any():
                results['regions'][region_name] = {
                    'status': 'missing',
                    'score': 'BAD',
                }
                bad_count += 1
                continue
            
            # Calculate metrics
            metrics = calculate_shape_metrics(mask)
            
            # Compare to reference
            ref = reference_metrics.get(region_name, {})
            
            # Volume ratio
            ref_vol = ref.get('volume', 1)
            vol_ratio = metrics['volume'] / ref_vol if ref_vol > 0 else 0
            
            # Compactness ratio
            ref_compact = ref.get('compactness', 1)
            compact_ratio = metrics['compactness'] / ref_compact if ref_compact > 0 else float('inf')
            
            # Aspect ratio comparison
            ref_ar_xy = ref.get('aspect_ratio_xy', 1)
            ar_xy_ratio = metrics['aspect_ratio_xy'] / ref_ar_xy if ref_ar_xy > 0 else 0
            
            # Score
            issues = []
            
            if vol_ratio < VOLUME_RATIO_MIN:
                issues.append(f"volume too small ({vol_ratio:.2f}x)")
            elif vol_ratio > VOLUME_RATIO_MAX:
                issues.append(f"volume too large ({vol_ratio:.2f}x)")
            
            if compact_ratio > COMPACTNESS_MAX:
                issues.append(f"not compact enough ({compact_ratio:.2f}x)")
            
            if ar_xy_ratio > ASPECT_RATIO_MAX:
                issues.append(f"distorted aspect ({ar_xy_ratio:.2f}x)")
            
            if not issues:
                score = 'GOOD'
                good_count += 1
            elif len(issues) == 1:
                score = 'MARGINAL'
            else:
                score = 'BAD'
                bad_count += 1
            
            results['regions'][region_name] = {
                'volume': metrics['volume'],
                'volume_ratio': vol_ratio,
                'compactness': metrics['compactness'],
                'compactness_ratio': compact_ratio,
                'aspect_ratio_xy': metrics['aspect_ratio_xy'],
                'score': score,
                'issues': issues,
            }
            results['scores'][region_name] = score
        
        # Overall assessment
        if bad_count == 0 and good_count >= len(EVALUATION_REGIONS) * 0.8:
            results['overall'] = 'GOOD'
        elif bad_count <= 1:
            results['overall'] = 'MARGINAL'
        else:
            results['overall'] = 'BAD'
        
        return results
        
    except Exception as e:
        return {'status': 'error', 'reason': str(e)}


# =============================================================================
# OPTIMIZATION
# =============================================================================

def run_optimization(
    pipeline_folder: Path,
    crop_percentages: List[int],
    atlas: str = DEFAULT_ATLAS,
    n_free_cpus: int = DEFAULT_N_FREE_CPUS,
) -> Dict:
    """
    Run the full optimization loop.
    
    Returns:
        Dictionary with results for each crop level and recommendation
    """
    full_folder = pipeline_folder / FOLDER_EXTRACTED_FULL
    opt_folder = pipeline_folder / FOLDER_OPTIMIZATION
    
    # Load metadata
    metadata_path = full_folder / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get atlas reference metrics
    print(f"\n[{timestamp()}] Loading atlas reference metrics...")
    reference_metrics = get_atlas_reference_metrics(atlas)
    
    if not reference_metrics:
        print("    Warning: Could not load reference metrics, will skip quality check")
    
    # Clean up old optimization if present
    if opt_folder.exists():
        print(f"[{timestamp()}] Removing previous optimization folder...")
        shutil.rmtree(opt_folder)
    
    opt_folder.mkdir(parents=True)
    
    results = {
        'brain': pipeline_folder.name,
        'timestamp': datetime.now().isoformat(),
        'crops_tested': [],
        'best_crop': None,
        'recommendation': None,
    }
    
    best_crop = None
    best_score = None
    
    total_start = time.time()
    
    for i, crop_pct in enumerate(crop_percentages):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(crop_percentages)}] Testing {crop_pct}% crop")
        print(f"{'='*60}")
        
        crop_fraction = crop_pct / 100.0
        crop_folder = opt_folder / f"crop_{crop_pct}pct"
        data_folder = crop_folder / "data"
        reg_folder = crop_folder / "registration"
        
        # Create cropped copy
        try:
            original_y, cropped_y = create_cropped_copy(
                full_folder, data_folder, crop_fraction, metadata
            )
        except Exception as e:
            print(f"    ERROR creating crop: {e}")
            continue
        
        # Run registration
        reg_success, reg_duration = run_registration(
            data_folder, reg_folder, metadata, atlas, n_free_cpus
        )
        
        print(f"    [{timestamp()}] Registration: {'SUCCESS' if reg_success else 'FAILED'} ({format_duration(reg_duration)})")
        
        if not reg_success:
            results['crops_tested'].append({
                'crop_pct': crop_pct,
                'original_y': original_y,
                'cropped_y': cropped_y,
                'registration': 'failed',
                'duration': reg_duration,
            })
            continue
        
        # Evaluate quality
        if reference_metrics:
            print(f"    [{timestamp()}] Evaluating registration quality...")
            quality = evaluate_registration(reg_folder, reference_metrics, atlas)
            
            overall = quality.get('overall', 'UNKNOWN')
            print(f"    [{timestamp()}] Quality: {overall}")
            
            for region, score in quality.get('scores', {}).items():
                print(f"        {region}: {score}")
        else:
            quality = {'overall': 'UNKNOWN', 'reason': 'no reference metrics'}
            overall = 'UNKNOWN'
        
        # Record result
        crop_result = {
            'crop_pct': crop_pct,
            'original_y': original_y,
            'cropped_y': cropped_y,
            'registration': 'success',
            'duration': reg_duration,
            'quality': quality,
        }
        results['crops_tested'].append(crop_result)
        
        # Track best
        if overall == 'GOOD' and (best_crop is None or crop_pct < best_crop):
            best_crop = crop_pct
            best_score = 'GOOD'
        elif overall == 'MARGINAL' and best_score != 'GOOD' and (best_crop is None or crop_pct < best_crop):
            best_crop = crop_pct
            best_score = 'MARGINAL'
    
    total_duration = time.time() - total_start
    
    # Determine recommendation
    if best_crop is not None:
        results['best_crop'] = best_crop
        results['recommendation'] = f"Use {best_crop}% crop (quality: {best_score})"
    else:
        results['recommendation'] = "No good crop found - manual inspection recommended"
    
    results['total_duration'] = total_duration
    
    # Save results
    with open(opt_folder / "optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if best_crop is not None:
        with open(opt_folder / "recommended_crop.txt", 'w') as f:
            f.write(f"Recommended crop: {best_crop}%\n")
            f.write(f"Quality: {best_score}\n")
            f.write(f"\nTo apply, update Script 2's crop detection or\n")
            f.write(f"manually set crop_fraction = {best_crop/100:.2f}\n")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optimize Y-crop for atlas registration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests multiple crop levels to find the minimum crop that
produces good registration quality for brainstem regions.

Examples:
    python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4
    python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4 --quick
    python util_optimize_crop.py --brain 349_CNT_01_02_1p625x_z4 --crops 0,15,30,45

Output:
    Creates _crop_optimization/ folder with all test registrations
    and a recommendation for the best crop percentage.
        """
    )
    
    parser.add_argument('--brain', '-b', required=True,
                        help='Brain/pipeline to optimize')
    parser.add_argument('--crops', '-c',
                        help='Comma-separated crop percentages to test (default: 0,10,20,30,40,50)')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick mode - only test 0, 25, 50%%')
    parser.add_argument('--atlas', default=DEFAULT_ATLAS,
                        help=f'Atlas to use (default: {DEFAULT_ATLAS})')
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS)
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BrainGlobe Crop Optimization")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)
    
    # Find pipeline
    pipeline_folder, mouse_folder = find_pipeline(args.brain, args.root)
    if not pipeline_folder:
        print(f"ERROR: Brain not found: {args.brain}")
        sys.exit(1)
    
    print(f"\nBrain: {mouse_folder.name}/{pipeline_folder.name}")
    
    # Check prerequisites
    full_folder = pipeline_folder / FOLDER_EXTRACTED_FULL
    if not full_folder.exists():
        print(f"ERROR: No extracted data found at {full_folder}")
        print("Run Script 2 (2_extract_and_analyze.py) first!")
        sys.exit(1)
    
    # Determine crop percentages
    if args.crops:
        crop_percentages = [int(x.strip()) for x in args.crops.split(',')]
    elif args.quick:
        crop_percentages = QUICK_CROPS
    else:
        crop_percentages = DEFAULT_CROPS
    
    print(f"Crop levels to test: {crop_percentages}")
    print(f"Atlas: {args.atlas}")
    
    # Confirm
    response = input(f"\nThis will take a while. Continue? [Enter/q]: ").strip()
    if response.lower() == 'q':
        print("Cancelled.")
        return
    
    # Run optimization
    results = run_optimization(
        pipeline_folder,
        crop_percentages,
        atlas=args.atlas,
        n_free_cpus=args.n_free_cpus,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal time: {format_duration(results['total_duration'])}")
    print(f"\nResults:")
    
    for crop_result in results['crops_tested']:
        pct = crop_result['crop_pct']
        reg = crop_result['registration']
        quality = crop_result.get('quality', {}).get('overall', '-')
        print(f"  {pct:2d}% crop: registration={reg}, quality={quality}")
    
    print(f"\nRecommendation: {results['recommendation']}")
    print(f"\nFull results saved to: {pipeline_folder / FOLDER_OPTIMIZATION}")


if __name__ == '__main__':
    main()
