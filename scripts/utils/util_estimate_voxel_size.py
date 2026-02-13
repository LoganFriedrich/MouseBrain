#!/usr/bin/env python3
"""
util_estimate_voxel_size.py

Utility: Estimate voxel size by measuring the brain in the image.

A mouse brain is approximately 10-11 mm wide. This script:
1. Loads a middle slice from your extracted TIFFs
2. Finds the brain tissue using thresholding
3. Measures the brain width in pixels
4. Calculates estimated voxel size

Just run it and pick a brain to analyze.
"""

import sys
from pathlib import Path
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT

# Expected mouse brain width in mm (adult C57BL/6)
# Allen CCF says ~11.4 mm left-right, but actual brains vary
# Using a range for sanity checking
EXPECTED_BRAIN_WIDTH_MM = 10.5  # Conservative estimate
EXPECTED_BRAIN_MIN_MM = 8.0
EXPECTED_BRAIN_MAX_MM = 13.0


def find_extracted_channels(root_path):
    """Find all extracted channel folders."""
    root_path = Path(root_path)
    results = []
    
    for mouse_folder in root_path.iterdir():
        if not mouse_folder.is_dir() or mouse_folder.name.startswith('.'):
            continue
        
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            
            # Look for extracted channels
            for folder_name in ['1_Extracted_Channels_from_1_ims_to_brainglobe', 
                                '1_Extracted_Full',
                                '1_Channels']:
                channels_folder = pipeline_folder / folder_name
                if channels_folder.exists():
                    ch0 = channels_folder / 'ch0'
                    ch1 = channels_folder / 'ch1'
                    
                    # Prefer ch1 (usually autofluorescence, better for brain outline)
                    if ch1.exists() and len(list(ch1.glob('Z*.tif'))) > 0:
                        results.append((pipeline_folder.name, ch1))
                    elif ch0.exists() and len(list(ch0.glob('Z*.tif'))) > 0:
                        results.append((pipeline_folder.name, ch0))
                    break
    
    return results


def load_middle_slice(channel_folder):
    """Load the middle Z slice from a channel folder."""
    import tifffile
    
    tif_files = sorted(channel_folder.glob('Z*.tif'))
    if not tif_files:
        return None, 0
    
    middle_idx = len(tif_files) // 2
    middle_file = tif_files[middle_idx]
    
    img = tifffile.imread(str(middle_file))
    return img, middle_idx


def measure_brain_width(img, show_plot=True):
    """
    Measure the brain width in pixels using simple thresholding.
    
    Returns (width_pixels, height_pixels, debug_info)
    """
    from scipy import ndimage
    
    # Normalize to 0-1
    img_float = img.astype(np.float32)
    img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)
    
    # Try multiple threshold levels and pick the one that gives a reasonable brain shape
    best_width = 0
    best_height = 0
    best_mask = None
    best_threshold = 0
    
    for threshold in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        # Threshold
        mask = img_norm > threshold
        
        # Clean up with morphological operations
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_opening(mask, iterations=3)
        mask = ndimage.binary_closing(mask, iterations=3)
        
        # Find largest connected component (the brain)
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            continue
        
        # Get size of each component
        component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_idx = np.argmax(component_sizes) + 1
        brain_mask = labeled == largest_idx
        
        # Measure bounding box
        rows = np.any(brain_mask, axis=1)
        cols = np.any(brain_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            continue
        
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        width = col_max - col_min
        height = row_max - row_min
        
        # Check if this looks like a reasonable brain (aspect ratio ~1.0-1.5)
        aspect = max(width, height) / (min(width, height) + 1)
        
        # Also check that it's not too small (< 20% of image) or too big (> 90%)
        area_fraction = np.sum(brain_mask) / (img.shape[0] * img.shape[1])
        
        if 0.15 < area_fraction < 0.85 and 0.7 < aspect < 2.0:
            if width > best_width:  # Prefer larger measurements
                best_width = width
                best_height = height
                best_mask = brain_mask
                best_threshold = threshold
    
    debug_info = {
        'threshold': best_threshold,
        'mask': best_mask,
        'img_norm': img_norm,
    }
    
    return best_width, best_height, debug_info


def estimate_voxel_size(width_pixels, height_pixels):
    """
    Estimate voxel size from measured brain dimensions.
    
    Returns (voxel_xy_um, confidence)
    """
    # Use width (left-right) as primary measurement
    # Mouse brain is ~10-11 mm wide
    
    voxel_from_width = (EXPECTED_BRAIN_WIDTH_MM * 1000) / width_pixels
    
    # Sanity check with height (anterior-posterior is ~13mm, but we're looking at a slice)
    # The visible height depends on where in the brain we are
    
    return voxel_from_width, 'medium'


def main():
    print("=" * 60)
    print("Voxel Size Estimator")
    print("=" * 60)
    print()
    print("This tool estimates XY voxel size by measuring the brain")
    print(f"and comparing to expected mouse brain width (~{EXPECTED_BRAIN_WIDTH_MM} mm)")
    print()
    
    # Check for required packages
    try:
        import tifffile
        from scipy import ndimage
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install tifffile scipy")
        return
    
    # Find extracted channels
    root = DEFAULT_BRAINGLOBE_ROOT
    if not root.exists():
        print(f"Default path not found: {root}")
        root = Path(input("Enter path to search: ").strip().strip('"'))
    
    print(f"Scanning {root}...")
    channels = find_extracted_channels(root)
    
    if not channels:
        print("No extracted channels found!")
        print("Run 1_ims_to_brainglobe.py first to extract TIFFs.")
        return
    
    print(f"\nFound {len(channels)} brain(s) with extracted channels:\n")
    for i, (name, path) in enumerate(channels, 1):
        print(f"  {i}. {name}")
    
    print()
    choice = input("Enter number to analyze (or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        return
    
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(channels)):
            print("Invalid selection")
            return
    except ValueError:
        print("Invalid input")
        return
    
    name, channel_path = channels[idx]
    
    print(f"\nAnalyzing: {name}")
    print(f"Channel: {channel_path.name}")
    
    # Load middle slice
    print("Loading middle slice...")
    img, slice_idx = load_middle_slice(channel_path)
    
    if img is None:
        print("Could not load image!")
        return
    
    print(f"  Loaded slice {slice_idx}, shape: {img.shape}")
    
    # Measure brain
    print("Measuring brain...")
    width_px, height_px, debug = measure_brain_width(img)
    
    if width_px == 0:
        print("  Could not detect brain in image!")
        print("  The image might need different thresholding.")
        return
    
    print(f"  Brain dimensions: {width_px} x {height_px} pixels")
    
    # Estimate voxel size
    voxel_xy, confidence = estimate_voxel_size(width_px, height_px)
    
    print()
    print("=" * 60)
    print("ESTIMATED VOXEL SIZE")
    print("=" * 60)
    print()
    print(f"  Measured brain width: {width_px} pixels")
    print(f"  Expected brain width: ~{EXPECTED_BRAIN_WIDTH_MM} mm")
    print()
    print(f"  Estimated XY voxel: {voxel_xy:.2f} µm")
    print()
    
    # Compare to filename z-step if available
    if '_z' in name:
        try:
            z_part = name.split('_z')[-1].split('_')[0].replace('p', '.')
            z_step = float(z_part)
            print(f"  Z voxel from filename: {z_step:.2f} µm")
            print()
            
            if abs(voxel_xy - z_step) < 1.0:
                print("  ✓ XY and Z are similar - likely isotropic voxels")
            else:
                print(f"  Note: XY ({voxel_xy:.2f}) differs from Z ({z_step:.2f})")
                print("        This could be anisotropic voxels, or measurement error")
        except:
            pass
    
    print()
    print("-" * 60)
    print("SUGGESTED VALUES FOR BRAINREG:")
    print("-" * 60)
    
    # Round to reasonable precision
    voxel_rounded = round(voxel_xy, 1)
    
    # Get Z from filename if possible
    z_voxel = voxel_rounded
    if '_z' in name:
        try:
            z_part = name.split('_z')[-1].split('_')[0].replace('p', '.')
            z_voxel = float(z_part)
        except:
            pass
    
    print()
    print(f"  brainreg ... -v {z_voxel:.1f} {voxel_rounded:.1f} {voxel_rounded:.1f}")
    print()
    
    # Offer to show the detection
    try:
        show = input("Show detection overlay? (y/n): ").strip().lower()
        if show == 'y':
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f'Original (slice {slice_idx})')
            axes[0].axis('off')
            
            # With detection overlay
            axes[1].imshow(img, cmap='gray')
            if debug['mask'] is not None:
                # Draw contour
                from scipy import ndimage
                edges = ndimage.binary_dilation(debug['mask']) ^ debug['mask']
                axes[1].imshow(np.ma.masked_where(~edges, edges), cmap='Reds', alpha=0.8)
            axes[1].set_title(f'Detected brain (width={width_px}px)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Could not show plot: {e}")


if __name__ == '__main__':
    main()
    print()
    input("Press Enter to close...")
