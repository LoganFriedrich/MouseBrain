#!/usr/bin/env python3
"""
util_registration_qc.py

Creates detailed quality control visualizations for brain registration.

Shows your registered brain slices next to atlas reference slices at the same
dorsal-ventral positions, with overlays and region labels.

Usage:
    python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4
    python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4 --levels 5
    python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4 --output custom_qc.png

Output:
    Creates QC_registration_detailed.png in the registration folder showing:
    - Your brain slice at each level
    - Corresponding atlas reference slice
    - Overlay with region boundaries
    - Labels and scale bars in micrometers
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile

# =============================================================================
# CONFIGURATION
# =============================================================================

from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
FOLDER_REGISTERED = "3_Registered_Atlas"
DEFAULT_NUM_LEVELS = 5  # Number of dorsal-ventral levels to show

# =============================================================================
# BRAIN DISCOVERY
# =============================================================================

def find_brain(brain_name: str, root: Path = DEFAULT_BRAINGLOBE_ROOT):
    """Find a brain's pipeline folder."""
    root = Path(root)

    for mouse_folder in root.iterdir():
        if not mouse_folder.is_dir():
            continue
        for pipeline_folder in mouse_folder.iterdir():
            if not pipeline_folder.is_dir():
                continue
            if brain_name == pipeline_folder.name or brain_name in pipeline_folder.name:
                return pipeline_folder

    return None


# =============================================================================
# LOAD REGISTRATION DATA
# =============================================================================

def load_registration_data(pipeline_folder: Path):
    """Load registered brain and atlas data."""
    reg_folder = pipeline_folder / FOLDER_REGISTERED

    if not reg_folder.exists():
        print(f"Error: Registration folder not found: {reg_folder}")
        return None

    # Load registered brain (downsampled to atlas resolution)
    registered_brain_path = reg_folder / "downsampled.tiff"
    if not registered_brain_path.exists():
        registered_brain_path = reg_folder / "downsampled_standard.tiff"

    if not registered_brain_path.exists():
        print(f"Error: Registered brain not found in {reg_folder}")
        return None

    print(f"Loading registered brain: {registered_brain_path}")
    registered_brain = tifffile.imread(str(registered_brain_path))

    # Load registered atlas annotations
    atlas_annotations_path = reg_folder / "registered_atlas.tiff"
    if not atlas_annotations_path.exists():
        atlas_annotations_path = reg_folder / "annotation.tiff"

    if atlas_annotations_path.exists():
        print(f"Loading atlas annotations: {atlas_annotations_path}")
        atlas_annotations = tifffile.imread(str(atlas_annotations_path))
    else:
        atlas_annotations = None
        print("Warning: Atlas annotations not found")

    # Load metadata to get voxel sizes
    metadata_path = pipeline_folder / "1_Extracted_Full" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Get atlas resolution from brainreg metadata
    brainreg_json = reg_folder / "brainreg.json"
    if brainreg_json.exists():
        with open(brainreg_json, 'r') as f:
            brainreg_meta = json.load(f)
            atlas_name = brainreg_meta.get('atlas', 'allen_mouse_25um')
            # Extract resolution from atlas name (e.g., allen_mouse_10um -> 10.0)
            if 'um' in atlas_name:
                atlas_resolution = float(atlas_name.split('_')[-1].replace('um', ''))
            else:
                atlas_resolution = 25.0
    else:
        atlas_resolution = 25.0  # Default fallback

    return {
        'registered_brain': registered_brain,
        'atlas_annotations': atlas_annotations,
        'metadata': metadata,
        'atlas_resolution': atlas_resolution,
        'reg_folder': reg_folder
    }


# =============================================================================
# CREATE QC VISUALIZATION
# =============================================================================

def create_qc_visualization(data, num_levels=DEFAULT_NUM_LEVELS, output_path=None):
    """
    Create detailed QC visualization showing brain slices at multiple levels.

    Args:
        data: Dictionary with registration data
        num_levels: Number of dorsal-ventral levels to show
        output_path: Custom output path (default: reg_folder/QC_registration_detailed.png)
    """
    registered_brain = data['registered_brain']
    atlas_annotations = data['atlas_annotations']
    atlas_resolution = data['atlas_resolution']
    reg_folder = data['reg_folder']

    # Determine slice positions (evenly spaced through dorsal-ventral axis)
    z_max = registered_brain.shape[0]
    slice_indices = np.linspace(int(z_max * 0.1), int(z_max * 0.9), num_levels, dtype=int)

    # Create figure with subplots: 3 columns (brain, atlas, overlay) x num_levels rows
    fig, axes = plt.subplots(num_levels, 3, figsize=(15, 5 * num_levels))
    if num_levels == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Registration Quality Control - Dorsal to Ventral',
                 fontsize=16, fontweight='bold', y=0.995)

    for row, z_idx in enumerate(slice_indices):
        # Calculate position in micrometers
        position_um = z_idx * atlas_resolution

        # Get brain slice
        brain_slice = registered_brain[z_idx, :, :]

        # Get atlas slice if available
        if atlas_annotations is not None:
            atlas_slice = atlas_annotations[z_idx, :, :]
            # Create boundaries for visualization
            atlas_boundaries = create_boundary_image(atlas_slice)
        else:
            atlas_slice = np.zeros_like(brain_slice)
            atlas_boundaries = atlas_slice

        # Column 1: Your brain
        ax = axes[row, 0]
        im = ax.imshow(brain_slice, cmap='gray', interpolation='nearest')
        ax.set_title(f'Your Brain\nZ={z_idx} ({position_um:.0f}μm from dorsal)',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        add_scale_bar(ax, atlas_resolution, brain_slice.shape[1])

        # Column 2: Atlas reference
        ax = axes[row, 1]
        ax.imshow(atlas_slice, cmap='nipy_spectral', interpolation='nearest', alpha=0.7)
        ax.set_title(f'Atlas Reference\nAllen Mouse Brain Atlas',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        add_scale_bar(ax, atlas_resolution, brain_slice.shape[1])

        # Column 3: Overlay
        ax = axes[row, 2]
        ax.imshow(brain_slice, cmap='gray', interpolation='nearest')
        ax.imshow(atlas_boundaries, cmap='hot', interpolation='nearest', alpha=0.3)
        ax.set_title(f'Overlay\nBrain + Atlas Boundaries',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        add_scale_bar(ax, atlas_resolution, brain_slice.shape[1])

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    if output_path is None:
        output_path = reg_folder / "QC_registration_detailed.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nQC visualization saved to: {output_path}")
    plt.close()

    return output_path


def create_boundary_image(annotation_slice):
    """Create an image showing region boundaries."""
    from scipy import ndimage

    # Find edges using Sobel filter
    edges_x = ndimage.sobel(annotation_slice.astype(float), axis=0)
    edges_y = ndimage.sobel(annotation_slice.astype(float), axis=1)
    edges = np.hypot(edges_x, edges_y)

    # Threshold to get boundaries
    boundaries = edges > 0

    return boundaries.astype(float)


def generate_slice_qc_images(data, slice_indices=None, output_folder=None):
    """
    Generate individual QC images for each slice showing brain + boundaries overlay.

    These match the format of manual QC screenshots - dark background with
    green brain tissue and cyan/green atlas boundary contours overlaid.

    Args:
        data: Dictionary with registration data
        slice_indices: List of Z indices, or None for auto-selection
        output_folder: Where to save images (default: reg_folder)

    Returns:
        List of saved image paths
    """
    registered_brain = data['registered_brain']
    atlas_annotations = data['atlas_annotations']
    reg_folder = data['reg_folder']

    if output_folder is None:
        output_folder = reg_folder
    output_folder = Path(output_folder)

    z_max = registered_brain.shape[0]

    # Auto-select strategic slice positions if not specified
    if slice_indices is None:
        # Choose ~8 slices at strategic positions through the brain
        # Similar to user's examples: anterior, several mid-brain, posterior
        slice_indices = [
            int(z_max * 0.10),  # Anterior (olfactory)
            int(z_max * 0.20),  #
            int(z_max * 0.30),  # Mid-anterior
            int(z_max * 0.40),  #
            int(z_max * 0.50),  # Mid-brain
            int(z_max * 0.60),  #
            int(z_max * 0.70),  # Mid-posterior
            int(z_max * 0.85),  # Posterior (brainstem)
        ]

    saved_paths = []

    print(f"\nGenerating {len(slice_indices)} QC slice images...")

    for z_idx in slice_indices:
        if z_idx < 0 or z_idx >= z_max:
            print(f"  Skipping invalid slice index: {z_idx}")
            continue

        # Get brain slice - normalize to 0-1
        brain_slice = registered_brain[z_idx, :, :].astype(float)
        if brain_slice.max() > 0:
            brain_slice = brain_slice / brain_slice.max()

        # Get boundaries from atlas
        if atlas_annotations is not None:
            atlas_slice = atlas_annotations[z_idx, :, :]
            boundaries = create_boundary_image(atlas_slice)
        else:
            boundaries = np.zeros_like(brain_slice)

        # Create RGB image: green brain tissue + cyan boundaries
        h, w = brain_slice.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        # Green channel: brain tissue (with some in R and B for visibility)
        rgb[:, :, 0] = brain_slice * 0.2  # Slight red
        rgb[:, :, 1] = brain_slice * 0.8  # Strong green
        rgb[:, :, 2] = brain_slice * 0.2  # Slight blue

        # Overlay boundaries in bright green/cyan
        boundary_mask = boundaries > 0
        rgb[boundary_mask, 0] = 0.2   # Some red
        rgb[boundary_mask, 1] = 0.9   # Strong green
        rgb[boundary_mask, 2] = 0.3   # Some blue/cyan

        # Clip to valid range
        rgb = np.clip(rgb, 0, 1)

        # Save as PNG
        output_path = output_folder / f"QC_slice_{z_idx:04d}.png"

        # Use matplotlib to save (handles float arrays well)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none', pad_inches=0.1)
        plt.close()

        saved_paths.append(output_path)
        print(f"  Saved: {output_path.name}")

    print(f"\nGenerated {len(saved_paths)} QC slice images in {output_folder}")
    return saved_paths


def add_scale_bar(ax, resolution_um, image_width, bar_length_um=1000):
    """Add a scale bar to the image."""
    # Calculate bar length in pixels
    bar_length_px = bar_length_um / resolution_um

    # Position in bottom-right corner
    x_start = image_width - bar_length_px - 20
    y_pos = ax.get_ylim()[0] - 20  # Near bottom

    # Draw scale bar
    rect = Rectangle((x_start, y_pos), bar_length_px, 5,
                     facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    # Add label
    ax.text(x_start + bar_length_px/2, y_pos - 15, f'{bar_length_um}μm',
           ha='center', va='top', color='white', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create detailed registration QC visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4
  python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4 --levels 7
  python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4 --slices
  python util_registration_qc.py --brain 349_CNT_01_02_1p625x_z4 --slices --slice-indices 100,250,400,600
        """
    )

    parser.add_argument('--brain', '-b', required=True,
                       help='Brain/pipeline to analyze')
    parser.add_argument('--levels', '-l', type=int, default=DEFAULT_NUM_LEVELS,
                       help=f'Number of dorsal-ventral levels to show (default: {DEFAULT_NUM_LEVELS})')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Custom output path (default: registration folder)')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT,
                       help='Root folder for brains')
    parser.add_argument('--slices', '-s', action='store_true',
                       help='Generate individual slice images (QC_slice_NNNN.png)')
    parser.add_argument('--slice-indices', type=str, default=None,
                       help='Comma-separated slice indices for --slices mode (default: auto)')

    args = parser.parse_args()

    # Find brain
    pipeline_folder = find_brain(args.brain, args.root)
    if not pipeline_folder:
        print(f"Error: Brain not found: {args.brain}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Registration QC Visualization")
    print(f"{'='*60}")
    print(f"Brain: {pipeline_folder.name}")
    print(f"Path: {pipeline_folder}")

    # Load registration data
    data = load_registration_data(pipeline_folder)
    if data is None:
        sys.exit(1)

    print(f"\nRegistered brain shape: {data['registered_brain'].shape}")
    print(f"Atlas resolution: {data['atlas_resolution']}um")

    if args.slices:
        # Generate individual slice QC images
        slice_indices = None
        if args.slice_indices:
            slice_indices = [int(x.strip()) for x in args.slice_indices.split(',')]
            print(f"Slice indices: {slice_indices}")

        saved_paths = generate_slice_qc_images(data, slice_indices)

        print(f"\n{'='*60}")
        print(f"[OK] Generated {len(saved_paths)} QC slice images!")
        print(f"{'='*60}")
    else:
        # Create multi-panel QC visualization
        print(f"Levels: {args.levels}")
        output_path = create_qc_visualization(data, args.levels, args.output)

        print(f"\n{'='*60}")
        print("[OK] QC visualization complete!")
        print(f"{'='*60}")
        print(f"\nOpen the image to check registration quality:")
        print(f"  {output_path}")

    print()


if __name__ == '__main__':
    main()
