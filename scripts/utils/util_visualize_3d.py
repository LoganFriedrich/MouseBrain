#!/usr/bin/env python3
"""
util_visualize_3d.py

3D visualization of detected cells and brain regions using brainrender.

Shows detected cells as 3D point clouds in atlas space, colored by eLife
functional group with hemisphere differentiation (left=bright, right=dark).
Also supports region heatmaps via brainglobe-heatmap.

================================================================================
USAGE
================================================================================
    # Show all detected cells (default mode)
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4

    # Heatmap of cell density by region
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --mode heatmap

    # Highlight top 10 regions by cell count
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --mode regions --top-n 10

    # Combined: points + top region meshes
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --mode combined --top-n 5

    # Left hemisphere only, save screenshot
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --hemisphere left --screenshot

================================================================================
REQUIREMENTS
================================================================================
    - brainrender (pip install brainrender)
    - brainglobe-heatmap (pip install brainglobe-heatmap)  [for heatmap mode]
    - mousebrain package installed
    - Pipeline step 6 (region analysis) must have been run on the brain
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

from mousebrain.config import BRAINS_ROOT
from mousebrain.region_mapping import get_elife_group, ELIFE_GROUPS

FOLDER_REGISTERED = "3_Registered_Atlas"
FOLDER_ANALYSIS = "6_Region_Analysis"


# =============================================================================
# BRAIN DISCOVERY
# =============================================================================

def find_brain(brain_name, root=BRAINS_ROOT):
    """Find a brain's pipeline folder using 2-level search."""
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
# DATA LOADING
# =============================================================================

def load_atlas_name(pipeline_folder):
    """Auto-detect atlas name from brainreg.json. Fallback: allen_mouse_25um."""
    brainreg_json = pipeline_folder / FOLDER_REGISTERED / "brainreg.json"
    if brainreg_json.exists():
        with open(brainreg_json, 'r') as f:
            meta = json.load(f)
        return meta.get('atlas', 'allen_mouse_25um')
    return 'allen_mouse_25um'


def load_cell_points(pipeline_folder):
    """Load per-cell atlas coordinates from all_points_information.csv.

    Returns dict with:
        coords_voxel: Nx3 ndarray of atlas voxel coordinates
        structure_names: list of region names
        hemispheres: list of 'left'/'right'
        count: total cells
    """
    csv_path = pipeline_folder / FOLDER_ANALYSIS / "all_points_information.csv"
    coords = []
    structure_names = []
    hemispheres = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append([
                float(row['coordinate_atlas_axis_0']),
                float(row['coordinate_atlas_axis_1']),
                float(row['coordinate_atlas_axis_2']),
            ])
            structure_names.append(row.get('structure_name', ''))
            hemispheres.append(row.get('hemisphere', ''))

    return {
        'coords_voxel': np.array(coords) if coords else np.empty((0, 3)),
        'structure_names': structure_names,
        'hemispheres': hemispheres,
        'count': len(coords),
    }


def load_region_counts(pipeline_folder):
    """Load region cell counts from cell_counts_by_region.csv.

    Returns dict: {acronym: {'total': N, 'left': N, 'right': N, 'name': str}}
    """
    csv_path = pipeline_folder / FOLDER_ANALYSIS / "cell_counts_by_region.csv"
    counts = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            acr = row['region_acronym']
            counts[acr] = {
                'total': int(row['cell_count']),
                'left': int(row['left_count']),
                'right': int(row['right_count']),
                'name': row['region_name'],
            }

    return counts


def coords_to_microns(coords_voxel, atlas_name):
    """Convert atlas voxel indices to micron coordinates for brainrender.

    Atlas coordinates in CSV are voxel indices; brainrender expects microns.
    Simply multiply by atlas resolution (e.g. 10um for allen_mouse_10um).
    """
    from brainglobe_atlasapi import BrainGlobeAtlas
    atlas = BrainGlobeAtlas(atlas_name)
    resolution = np.array(atlas.resolution)
    microns = coords_voxel * resolution
    return microns


# =============================================================================
# COLORING
# =============================================================================

def build_elife_colormap():
    """Assign a distinct hex color to each eLife group.

    Uses tab20 + tab20b colormaps to get 25+ distinct colors.
    Returns dict with both full-brightness and dark variants:
        {group_name: {'bright': '#RRGGBB', 'dark': '#RRGGBB'}}
    """
    from matplotlib import colormaps

    tab20 = colormaps['tab20']
    tab20b = colormaps['tab20b']

    sorted_groups = sorted(ELIFE_GROUPS.items(), key=lambda x: x[1]['id'])

    group_colors = {}
    for i, (group_name, _) in enumerate(sorted_groups):
        if i < 20:
            rgba = tab20(i / 20)
        else:
            rgba = tab20b((i - 20) / 20)
        r, g, b = rgba[0], rgba[1], rgba[2]
        bright = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        dark = f'#{int(r*178):02x}{int(g*178):02x}{int(b*178):02x}'
        group_colors[group_name] = {'bright': bright, 'dark': dark}

    return group_colors


def _build_name_to_group():
    """Build reverse lookup: structure full name or acronym -> eLife group."""
    name_to_group = {}
    for group_name, group_data in ELIFE_GROUPS.items():
        for full_name in group_data.get('full_names', []):
            name_to_group[full_name] = group_name
        for acronym in group_data.get('acronyms', []):
            name_to_group[acronym] = group_name
    return name_to_group


def _get_group_for_name(name, name_to_group):
    """Get eLife group for a structure name (full name or acronym)."""
    group = name_to_group.get(name)
    if group is None:
        group = get_elife_group(name)
    return group


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_points(scene, cell_data, atlas_name, args):
    """Add cell points to the brainrender scene.

    Uses a single Points actor with a per-sphere color list.
    Left hemisphere gets bright colors, right gets darker variant.
    Colors are assigned by eLife functional group.
    """
    from brainrender.actors import Points

    coords = coords_to_microns(cell_data['coords_voxel'], atlas_name)
    structure_names = cell_data['structure_names']
    hemispheres = cell_data['hemispheres']

    # Filter by hemisphere
    if args.hemisphere != 'both':
        mask = [h == args.hemisphere for h in hemispheres]
        coords = coords[mask]
        structure_names = [s for s, m in zip(structure_names, mask) if m]
        hemispheres = [h for h, m in zip(hemispheres, mask) if m]

    if len(coords) == 0:
        print("  Warning: No cells to display after filtering.")
        return

    # Subsample if too many points
    max_pts = args.max_points
    if max_pts and len(coords) > max_pts:
        print(f"  Subsampling {len(coords):,} cells to {max_pts:,} for performance")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), max_pts, replace=False)
        idx.sort()
        coords = coords[idx]
        structure_names = [structure_names[i] for i in idx]
        hemispheres = [hemispheres[i] for i in idx]

    # Build per-cell color list by eLife group + hemisphere
    group_colors = build_elife_colormap()
    name_to_group = _build_name_to_group()
    gray_bright, gray_dark = '#888888', '#5f5f5f'

    colors = []
    group_counts = {}
    for name, hemi in zip(structure_names, hemispheres):
        group = _get_group_for_name(name, name_to_group) or '[Unmapped]'
        group_counts[group] = group_counts.get(group, 0) + 1

        if group == '[Unmapped]':
            color = gray_bright if hemi == 'left' else gray_dark
        elif group in group_colors:
            color = group_colors[group]['bright'] if hemi == 'left' else group_colors[group]['dark']
        else:
            color = gray_bright if hemi == 'left' else gray_dark
        colors.append(color)

    # Single actor with per-sphere colors
    pts = Points(coords, colors=colors, radius=args.point_radius,
                 alpha=args.point_alpha, name='Detected Cells')
    scene.add(pts)

    n_mapped = sum(v for k, v in group_counts.items() if k != '[Unmapped]')
    n_unmapped = group_counts.get('[Unmapped]', 0)
    print(f"  Added {len(coords):,} cells ({n_mapped:,} mapped to {len(group_counts)-1} eLife groups, {n_unmapped:,} unmapped)")


def render_heatmap(pipeline_folder, region_counts, atlas_name, args):
    """Create a brainglobe-heatmap visualization.

    Returns the Heatmap object (it creates its own Scene internally).
    """
    try:
        from brainglobe_heatmap import Heatmap
    except ImportError:
        print("Error: brainglobe-heatmap not installed.")
        print("Install with: pip install brainglobe-heatmap")
        sys.exit(1)

    # Build values dict
    values = {}
    for acr, data in region_counts.items():
        if args.hemisphere == 'both':
            values[acr] = data['total']
        elif args.hemisphere == 'left':
            values[acr] = data['left']
        else:
            values[acr] = data['right']

    # Filter to top-n if requested
    if args.top_n:
        top = get_top_regions(region_counts, args.top_n)
        values = {k: v for k, v in values.items() if k in top}

    # Remove zero-count regions
    values = {k: v for k, v in values.items() if v > 0}

    if not values:
        print("  Warning: No region counts to display.")
        return None

    print(f"  Heatmap with {len(values)} regions")

    hm = Heatmap(
        values=values,
        position=args.heatmap_position,
        orientation=args.heatmap_orientation,
        format="3D",
        cmap=args.cmap,
        atlas_name=atlas_name,
        title=f"Cell Density: {pipeline_folder.name}",
    )

    return hm


def render_regions(scene, region_counts, args):
    """Add highlighted brain regions to the scene, colored by eLife group."""
    group_colors = build_elife_colormap()

    # Determine which regions to show
    if args.regions:
        regions_to_show = args.regions
    elif args.top_n:
        regions_to_show = get_top_regions(region_counts, args.top_n)
    else:
        regions_to_show = get_top_regions(region_counts, 15)

    added = 0
    for region_acr in regions_to_show:
        # Color by eLife group
        group = get_elife_group(region_acr)
        if group and group in group_colors:
            color = group_colors[group]['bright']
        else:
            color = '#888888'

        try:
            scene.add_brain_region(
                region_acr,
                alpha=args.region_alpha,
                color=color,
                hemisphere=args.hemisphere if args.hemisphere != 'both' else 'both',
            )
            added += 1
        except Exception as e:
            print(f"  Warning: Could not add region '{region_acr}': {e}")

    print(f"  Added {added}/{len(regions_to_show)} brain regions")


def render_combined(scene, cell_data, region_counts, atlas_name, args):
    """Combined mode: show cell points with highlighted top regions."""
    import copy
    modified = copy.copy(args)
    # Keep point_alpha at 1.0 (transparency breaks Spheres in offscreen mode)
    modified.region_alpha = min(args.region_alpha, 0.2)
    if not modified.top_n:
        modified.top_n = 10

    render_points(scene, cell_data, atlas_name, modified)
    render_regions(scene, region_counts, modified)


# =============================================================================
# HELPERS
# =============================================================================

def get_top_regions(region_counts, n):
    """Return top N region acronyms by total cell count."""
    sorted_regions = sorted(region_counts.items(), key=lambda x: x[1]['total'], reverse=True)
    return [acr for acr, _ in sorted_regions[:n]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='3D brain visualization of detected cells using brainrender',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive 3D view of all detected cells (colored by eLife group)
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4

    # Heatmap of cell density by region
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --mode heatmap

    # Show top 10 regions as transparent meshes
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --mode regions --top-n 10

    # Combined: cell points + top region meshes
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --mode combined --top-n 5

    # Left hemisphere only, save screenshot
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --hemisphere left --screenshot

    # Specific regions highlighted
    python util_visualize_3d.py --brain 349_CNT_01_02_1p625x_z4 --mode regions --regions GRN RN PAG MOp5
        """
    )

    # Required
    parser.add_argument('--brain', '-b', required=True,
                        help='Brain name or partial match (e.g. 349_CNT_01_02_1p625x_z4)')

    # Mode
    parser.add_argument('--mode', '-m', default='points',
                        choices=['points', 'heatmap', 'regions', 'combined'],
                        help='Visualization mode (default: points)')

    # Root path
    parser.add_argument('--root', type=Path, default=BRAINS_ROOT,
                        help='Root folder for brain search')

    # Region selection
    parser.add_argument('--regions', '-r', nargs='+', default=None,
                        help='Specific region acronyms to highlight')
    parser.add_argument('--top-n', '-n', type=int, default=None,
                        help='Show top N regions by cell count')

    # Hemisphere
    parser.add_argument('--hemisphere', default='both',
                        choices=['both', 'left', 'right'],
                        help='Which hemisphere to display (default: both)')

    # Visual customization
    parser.add_argument('--cmap', default='Reds',
                        help='Colormap for heatmap mode (default: Reds)')
    parser.add_argument('--camera', default='three_quarters',
                        help='Camera angle (default: three_quarters)')
    parser.add_argument('--point-radius', type=float, default=100,
                        help='Cell point radius in microns (default: 100)')
    parser.add_argument('--point-alpha', type=float, default=1.0,
                        help='Cell point opacity 0-1 (default: 1.0; values <1 may not render in screenshot mode)')
    parser.add_argument('--region-alpha', type=float, default=0.4,
                        help='Region mesh opacity 0-1 (default: 0.4)')
    parser.add_argument('--no-root', action='store_true',
                        help='Hide the transparent brain outline')
    parser.add_argument('--max-points', type=int, default=None,
                        help='Max cells to render (random subsample if exceeded)')

    # Output
    parser.add_argument('--screenshot', '-s', action='store_true',
                        help='Save screenshot instead of interactive window')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Screenshot output path')
    parser.add_argument('--scale', type=int, default=2,
                        help='Screenshot resolution scale (default: 2)')

    # Atlas override
    parser.add_argument('--atlas', default=None,
                        help='Atlas name override (auto-detected from brainreg.json)')

    # Heatmap-specific
    parser.add_argument('--heatmap-orientation', default='frontal',
                        choices=['frontal', 'sagittal', 'horizontal'],
                        help='Slice orientation for heatmap (default: frontal)')
    parser.add_argument('--heatmap-position', type=float, default=None,
                        help='Slice position in microns for heatmap (default: brain center)')

    args = parser.parse_args()

    # --- Find brain ---
    pipeline_folder = find_brain(args.brain, args.root)
    if not pipeline_folder:
        print(f"Error: Brain not found: {args.brain}")
        print(f"  Searched under: {args.root}")
        sys.exit(1)

    # --- Header ---
    print(f"\n{'=' * 60}")
    print(f"3D Brain Visualization")
    print(f"{'=' * 60}")
    print(f"  Brain: {pipeline_folder.name}")
    print(f"  Mode:  {args.mode}")

    # --- Validate data exists ---
    analysis_dir = pipeline_folder / FOLDER_ANALYSIS
    if not analysis_dir.exists():
        print(f"\nError: No region analysis found. Run pipeline step 6 first.")
        print(f"  Expected: {analysis_dir}")
        sys.exit(1)

    # --- Auto-detect atlas ---
    atlas_name = args.atlas or load_atlas_name(pipeline_folder)
    print(f"  Atlas: {atlas_name}")

    # --- Load data based on mode ---
    cell_data = None
    region_counts = None

    if args.mode in ('points', 'combined'):
        points_csv = analysis_dir / "all_points_information.csv"
        if not points_csv.exists():
            print(f"\nError: {points_csv.name} not found in {analysis_dir}")
            print("  Run pipeline step 6 first.")
            sys.exit(1)
        cell_data = load_cell_points(pipeline_folder)
        print(f"  Cells: {cell_data['count']:,}")

    if args.mode in ('heatmap', 'regions', 'combined'):
        counts_csv = analysis_dir / "cell_counts_by_region.csv"
        if not counts_csv.exists():
            print(f"\nError: {counts_csv.name} not found in {analysis_dir}")
            print("  Run pipeline step 6 first.")
            sys.exit(1)
        region_counts = load_region_counts(pipeline_folder)
        print(f"  Regions: {len(region_counts)}")

    # --- Import brainrender ---
    try:
        import brainrender
        from brainrender import Scene
    except ImportError:
        print("\nError: brainrender not installed.")
        print("Install with: pip install brainrender")
        sys.exit(1)

    # --- Configure rendering ---
    if args.screenshot:
        brainrender.settings.OFFSCREEN = True

    print(f"\n  Loading atlas and building scene...")

    # --- Heatmap mode (standalone — creates its own Scene) ---
    if args.mode == 'heatmap':
        hm = render_heatmap(pipeline_folder, region_counts, atlas_name, args)
        if hm is None:
            sys.exit(1)
        if args.screenshot:
            output_path = args.output or str(analysis_dir / "brain3d_heatmap.png")
            hm.render()
            hm.scene.screenshot(name=output_path, scale=args.scale)
            print(f"\n  Screenshot saved: {output_path}")
        else:
            print(f"\n  Rendering interactive heatmap...")
            print(f"  Controls: rotate=drag, zoom=scroll, q=quit")
            hm.show()

        print(f"\n{'=' * 60}")
        print("[OK] Visualization complete!")
        print(f"{'=' * 60}\n")
        return

    # --- Points / Regions / Combined (shared Scene) ---
    scene = Scene(
        atlas_name=atlas_name,
        root=not args.no_root,
        title=f"{pipeline_folder.name} — {args.mode}",
        screenshots_folder=str(analysis_dir) if args.screenshot else None,
    )

    if args.mode == 'points':
        render_points(scene, cell_data, atlas_name, args)
    elif args.mode == 'regions':
        render_regions(scene, region_counts, args)
    elif args.mode == 'combined':
        render_combined(scene, cell_data, region_counts, atlas_name, args)

    # --- Render ---
    if args.screenshot:
        output_name = args.output or f"brain3d_{args.mode}.png"
        scene.render(interactive=False, camera=args.camera)
        savepath = scene.screenshot(name=output_name, scale=args.scale)
        print(f"\n  Screenshot saved: {savepath}")
        scene.close()
    else:
        print(f"\n  Rendering interactive 3D view...")
        print(f"  Camera: {args.camera}")
        print(f"  Controls: rotate=drag, zoom=scroll, q=quit")
        scene.render(interactive=True, camera=args.camera)

    print(f"\n{'=' * 60}")
    print("[OK] Visualization complete!")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
