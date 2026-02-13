#!/usr/bin/env python3
"""
util_create_training_cubes.py - Extract training cubes from curated XML coordinates.

Creates BrainGlobe/cellfinder training data from curated cell/non-cell coordinates.

Usage:
    python util_create_training_cubes.py --brain 349_CNT_01_02_1p625x_z4 \
        --cells curated_cells_20260109_232332.xml \
        --non-cells curated_non_cells_20260109_232247.xml

Output format (BrainGlobe standard):
    training_data/
    ├── cells/
    │   ├── pCellz{z}y{y}x{x}Ch0.tif  (signal)
    │   ├── pCellz{z}y{y}x{x}Ch1.tif  (background)
    │   └── ...
    ├── non_cells/
    │   ├── pCellz{z}y{y}x{x}Ch0.tif
    │   ├── pCellz{z}y{y}x{x}Ch1.tif
    │   └── ...
    └── training.yml
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import numpy as np
import zarr
import tifffile
from tqdm import tqdm

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from mousebrain.config import BRAINS_ROOT, MODELS_DIR


# BrainGlobe standard cube size
CUBE_XY = 50
CUBE_Z = 20


def parse_xml_coordinates(xml_path: Path) -> List[Tuple[int, int, int]]:
    """
    Parse CellCounter XML format and return coordinates as (z, y, x) tuples.

    The XML stores coordinates as MarkerX, MarkerY, MarkerZ.
    We return (z, y, x) to match array indexing order.
    """
    if not xml_path.exists():
        print(f"  [WARNING] XML file not found: {xml_path}")
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    coordinates = []
    for marker in root.findall('.//Marker'):
        x = int(marker.find('MarkerX').text)
        y = int(marker.find('MarkerY').text)
        z = int(marker.find('MarkerZ').text)
        coordinates.append((z, y, x))  # Return as (z, y, x) for array indexing

    return coordinates


def load_zarr_data(brain_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load signal and background channel data from zarr.

    Returns: (signal_data, background_data, metadata)
    """
    cropped_dir = brain_path / "2_Cropped_For_Registration"

    # Load metadata to determine channel roles
    metadata_path = cropped_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Determine channel mapping
    # Default: ch0 = signal, ch1 = background
    # But check metadata for swapped channels
    channels = metadata.get("channels", {})
    signal_ch = channels.get("signal_channel", 0)
    bg_ch = channels.get("background_channel", 1)

    # Check if channels were swapped
    if metadata.get("channels_swapped", False):
        # After swapping: ch0 = background, ch1 = signal
        signal_ch = 1
        bg_ch = 0

    print(f"  Channel mapping: signal=ch{signal_ch}, background=ch{bg_ch}")

    # Load zarr arrays
    signal_zarr_path = cropped_dir / f"ch{signal_ch}.zarr"
    bg_zarr_path = cropped_dir / f"ch{bg_ch}.zarr"

    if not signal_zarr_path.exists():
        raise FileNotFoundError(f"Signal zarr not found: {signal_zarr_path}")
    if not bg_zarr_path.exists():
        raise FileNotFoundError(f"Background zarr not found: {bg_zarr_path}")

    print(f"  Loading zarr data...")
    signal_store = zarr.open(str(signal_zarr_path), mode='r')
    bg_store = zarr.open(str(bg_zarr_path), mode='r')

    # Navigate to the data array (might be nested)
    if hasattr(signal_store, 'shape'):
        signal_data = signal_store
    elif 'c' in signal_store:
        signal_data = signal_store['c']
    else:
        # Try to find the data array
        for key in signal_store.keys():
            arr = signal_store[key]
            if hasattr(arr, 'shape') and len(arr.shape) >= 3:
                signal_data = arr
                break
        else:
            raise ValueError(f"Could not find data array in {signal_zarr_path}")

    if hasattr(bg_store, 'shape'):
        bg_data = bg_store
    elif 'c' in bg_store:
        bg_data = bg_store['c']
    else:
        for key in bg_store.keys():
            arr = bg_store[key]
            if hasattr(arr, 'shape') and len(arr.shape) >= 3:
                bg_data = arr
                break
        else:
            raise ValueError(f"Could not find data array in {bg_zarr_path}")

    print(f"  Signal shape: {signal_data.shape}")
    print(f"  Background shape: {bg_data.shape}")

    return signal_data, bg_data, metadata


def extract_cube(data: zarr.Array, z: int, y: int, x: int,
                 half_xy: int = CUBE_XY // 2, half_z: int = CUBE_Z // 2) -> Optional[np.ndarray]:
    """
    Extract a cube centered on (z, y, x).

    Returns None if the cube would be out of bounds.
    """
    shape = data.shape

    # Check bounds
    z_start = z - half_z
    z_end = z + half_z
    y_start = y - half_xy
    y_end = y + half_xy
    x_start = x - half_xy
    x_end = x + half_xy

    if z_start < 0 or z_end > shape[0]:
        return None
    if y_start < 0 or y_end > shape[1]:
        return None
    if x_start < 0 or x_end > shape[2]:
        return None

    # Extract cube
    cube = np.asarray(data[z_start:z_end, y_start:y_end, x_start:x_end])
    return cube


def export_training_cubes(
    brain_path: Path,
    cells_xml: Path,
    non_cells_xml: Path,
    output_dir: Path,
    signal_data: zarr.Array,
    bg_data: zarr.Array
) -> Tuple[int, int, int, int]:
    """
    Export training cubes in BrainGlobe format.

    Returns: (cells_exported, cells_skipped, non_cells_exported, non_cells_skipped)
    """
    cells_dir = output_dir / "cells"
    non_cells_dir = output_dir / "non_cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    non_cells_dir.mkdir(parents=True, exist_ok=True)

    half_xy = CUBE_XY // 2
    half_z = CUBE_Z // 2

    cells_exported = 0
    cells_skipped = 0
    non_cells_exported = 0
    non_cells_skipped = 0

    # Process cells
    print(f"\nExporting cells from {cells_xml.name}...")
    cells_coords = parse_xml_coordinates(cells_xml)
    for z, y, x in tqdm(cells_coords, desc="  Cells"):
        signal_cube = extract_cube(signal_data, z, y, x, half_xy, half_z)
        bg_cube = extract_cube(bg_data, z, y, x, half_xy, half_z)

        if signal_cube is None or bg_cube is None:
            cells_skipped += 1
            continue

        # Save both channels with BrainGlobe naming convention
        # Ch0 = signal, Ch1 = background (BrainGlobe convention)
        base_name = f"pCellz{z}y{y}x{x}"
        tifffile.imwrite(str(cells_dir / f"{base_name}Ch0.tif"), signal_cube.astype(np.uint16))
        tifffile.imwrite(str(cells_dir / f"{base_name}Ch1.tif"), bg_cube.astype(np.uint16))
        cells_exported += 1

    # Process non-cells
    print(f"\nExporting non-cells from {non_cells_xml.name}...")
    non_cells_coords = parse_xml_coordinates(non_cells_xml)
    for z, y, x in tqdm(non_cells_coords, desc="  Non-cells"):
        signal_cube = extract_cube(signal_data, z, y, x, half_xy, half_z)
        bg_cube = extract_cube(bg_data, z, y, x, half_xy, half_z)

        if signal_cube is None or bg_cube is None:
            non_cells_skipped += 1
            continue

        base_name = f"pCellz{z}y{y}x{x}"
        tifffile.imwrite(str(non_cells_dir / f"{base_name}Ch0.tif"), signal_cube.astype(np.uint16))
        tifffile.imwrite(str(non_cells_dir / f"{base_name}Ch1.tif"), bg_cube.astype(np.uint16))
        non_cells_exported += 1

    return cells_exported, cells_skipped, non_cells_exported, non_cells_skipped


def create_training_yaml(output_dir: Path):
    """Create the training.yml config file required by BrainGlobe."""
    yaml_content = {
        'data': [
            {
                'bg_channel': 1,
                'cell_def': '',
                'cube_dir': str(output_dir / 'cells'),
                'signal_channel': 0,
                'type': 'cell'
            },
            {
                'bg_channel': 1,
                'cell_def': '',
                'cube_dir': str(output_dir / 'non_cells'),
                'signal_channel': 0,
                'type': 'no_cell'
            }
        ]
    }

    yaml_path = output_dir / 'training.yml'

    if HAS_YAML:
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
    else:
        # Write YAML manually (simple format)
        with open(yaml_path, 'w') as f:
            f.write("data:\n")
            f.write(f"- bg_channel: 1\n")
            f.write(f"  cell_def: ''\n")
            f.write(f"  cube_dir: {output_dir / 'cells'}\n")
            f.write(f"  signal_channel: 0\n")
            f.write(f"  type: cell\n")
            f.write(f"- bg_channel: 1\n")
            f.write(f"  cell_def: ''\n")
            f.write(f"  cube_dir: {output_dir / 'non_cells'}\n")
            f.write(f"  signal_channel: 0\n")
            f.write(f"  type: no_cell\n")

    print(f"\nCreated {yaml_path}")


def find_brain_path(brain_name: str) -> Optional[Path]:
    """Find the full path to a brain folder."""
    # Try direct path first
    direct_path = BRAINS_ROOT / brain_name
    if direct_path.exists():
        return direct_path

    # Try searching subdirectories (mouse folders)
    for mouse_dir in BRAINS_ROOT.iterdir():
        if mouse_dir.is_dir():
            brain_path = mouse_dir / brain_name
            if brain_path.exists():
                return brain_path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Create BrainGlobe training cubes from curated XML coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default paths (looks for curated XMLs in 5_Classified_Cells)
  python util_create_training_cubes.py --brain 349_CNT_01_02_1p625x_z4

  # With explicit XML paths
  python util_create_training_cubes.py --brain 349_CNT_01_02_1p625x_z4 \\
      --cells curated_cells_20260109_232332.xml \\
      --non-cells curated_non_cells_20260109_232247.xml
"""
    )
    parser.add_argument("--brain", required=True, help="Brain folder name")
    parser.add_argument("--cells", help="Path to cells XML (relative or absolute)")
    parser.add_argument("--non-cells", help="Path to non-cells XML (relative or absolute)")
    parser.add_argument("--output", help="Output directory (default: brain/training_data)")

    args = parser.parse_args()

    print("=" * 60)
    print("BrainGlobe Training Data Generator")
    print("=" * 60)

    # Find brain path
    brain_path = find_brain_path(args.brain)
    if brain_path is None:
        print(f"ERROR: Brain not found: {args.brain}")
        print(f"Searched in: {BRAINS_ROOT}")
        sys.exit(1)
    print(f"\nBrain path: {brain_path}")

    # Find XML files
    classified_dir = brain_path / "5_Classified_Cells"

    if args.cells:
        cells_xml = Path(args.cells)
        if not cells_xml.is_absolute():
            cells_xml = classified_dir / cells_xml
    else:
        # Find most recent curated_cells*.xml
        cells_files = sorted(classified_dir.glob("curated_cells*.xml"), reverse=True)
        if cells_files:
            cells_xml = cells_files[0]
        else:
            print("ERROR: No curated_cells*.xml found. Specify --cells explicitly.")
            sys.exit(1)

    if args.non_cells:
        non_cells_xml = Path(args.non_cells)
        if not non_cells_xml.is_absolute():
            non_cells_xml = classified_dir / non_cells_xml
    else:
        # Find most recent curated_non_cells*.xml
        non_cells_files = sorted(classified_dir.glob("curated_non_cells*.xml"), reverse=True)
        if non_cells_files:
            non_cells_xml = non_cells_files[0]
        else:
            print("ERROR: No curated_non_cells*.xml found. Specify --non-cells explicitly.")
            sys.exit(1)

    print(f"Cells XML: {cells_xml}")
    print(f"Non-cells XML: {non_cells_xml}")

    if not cells_xml.exists():
        print(f"ERROR: Cells XML not found: {cells_xml}")
        sys.exit(1)
    if not non_cells_xml.exists():
        print(f"ERROR: Non-cells XML not found: {non_cells_xml}")
        sys.exit(1)

    # Count coordinates
    cells_coords = parse_xml_coordinates(cells_xml)
    non_cells_coords = parse_xml_coordinates(non_cells_xml)
    print(f"\nFound {len(cells_coords)} cells, {len(non_cells_coords)} non-cells")

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = brain_path / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load zarr data
    print("\nLoading brain data...")
    signal_data, bg_data, metadata = load_zarr_data(brain_path)

    # Export cubes
    cells_exp, cells_skip, non_cells_exp, non_cells_skip = export_training_cubes(
        brain_path, cells_xml, non_cells_xml, output_dir, signal_data, bg_data
    )

    # Create training.yml
    create_training_yaml(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cells exported:      {cells_exp:4d}")
    print(f"Cells skipped:       {cells_skip:4d} (out of bounds)")
    print(f"Non-cells exported:  {non_cells_exp:4d}")
    print(f"Non-cells skipped:   {non_cells_skip:4d} (out of bounds)")
    print(f"\nTotal cubes: {cells_exp + non_cells_exp}")
    print(f"Output: {output_dir}")
    print(f"\nTraining command:")
    print(f"  python util_train_model.py --yaml {output_dir / 'training.yml'}")


if __name__ == "__main__":
    main()
