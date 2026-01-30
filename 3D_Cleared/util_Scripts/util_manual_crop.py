#!/usr/bin/env python3
"""
util_manual_crop.py

CLI launcher for the manual crop tool - loads brain into napari and opens plugin.

Usage:
    python util_manual_crop.py --brain 349_CNT_01_02_1p625x_z4

This will:
    1. Load the full brain volume into napari
    2. Open the manual crop plugin
    3. Add the crop line automatically
    4. Let you adjust and apply the crop
"""

import argparse
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT
FOLDER_EXTRACTED = "1_Extracted_Full"


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
            if brain_name == pipeline_folder.name:
                return pipeline_folder
            if brain_name in pipeline_folder.name:
                return pipeline_folder

    return None


def load_brain_channel(pipeline_folder: Path, channel: int = 0):
    """Load a full brain channel as a numpy array."""
    import tifffile
    import numpy as np

    ch_folder = pipeline_folder / FOLDER_EXTRACTED / f"ch{channel}"
    if not ch_folder.exists():
        return None, None

    tiff_files = sorted(ch_folder.glob("Z*.tif"))
    if not tiff_files:
        return None, None

    print(f"Loading {len(tiff_files)} slices from {ch_folder.name}...")

    images = []
    for i, f in enumerate(tiff_files):
        images.append(tifffile.imread(str(f)))
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(tiff_files)} slices...")

    volume = np.stack(images)
    print(f"  Volume shape: {volume.shape}")

    return volume, ch_folder


def get_voxel_sizes(pipeline_folder: Path):
    """Get voxel sizes from metadata."""
    import json

    meta_path = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            voxel = metadata.get('voxel_size_um', {})
            vz = voxel.get('z', 4.0)
            vy = voxel.get('y', 4.0)
            vx = voxel.get('x', 4.0)
            return (vz, vy, vx)

    return (4.0, 4.0, 4.0)


def get_signal_channel(pipeline_folder: Path):
    """Determine which channel is the signal channel from metadata."""
    import json

    meta_path = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            channels = metadata.get('channels', {})
            signal = channels.get('signal_channel')
            if signal is not None:
                return signal

    return 0  # Default to channel 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Load brain into napari with manual crop plugin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
  1. Loads the full brain volume into napari (from 1_Extracted_Full/)
  2. Opens the manual crop plugin widget
  3. Sets up everything ready for cropping

Examples:
  python util_manual_crop.py --brain 349_CNT_01_02_1p625x_z4
  python util_manual_crop.py --brain 349_CNT_01_02_1p625x_z4 --channel 1
        """
    )

    parser.add_argument('--brain', '-b', required=True,
                        help='Brain/pipeline to crop')
    parser.add_argument('--channel', '-c', type=int, default=None,
                        help='Channel to load (default: auto-detect signal channel)')
    parser.add_argument('--root', type=Path, default=DEFAULT_BRAINGLOBE_ROOT,
                        help='Root folder')

    args = parser.parse_args()

    # Find brain
    pipeline_folder = find_brain(args.brain, args.root)
    if not pipeline_folder:
        print(f"Error: Brain not found: {args.brain}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Manual Crop Tool - CLI Launcher")
    print(f"{'='*60}")
    print(f"Brain: {pipeline_folder.name}")
    print(f"Path: {pipeline_folder}")

    # Determine channel
    if args.channel is not None:
        channel = args.channel
        print(f"Channel: {channel} (user specified)")
    else:
        channel = get_signal_channel(pipeline_folder)
        print(f"Channel: {channel} (auto-detected signal channel)")

    # Load brain
    volume, ch_folder = load_brain_channel(pipeline_folder, channel)

    if volume is None:
        print(f"Error: No extracted data found for channel {channel}")
        print(f"  Looked in: {pipeline_folder / FOLDER_EXTRACTED}")
        sys.exit(1)

    # Get voxel sizes
    scale = get_voxel_sizes(pipeline_folder)
    print(f"Voxel size (Z,Y,X): {scale} um")

    # Launch napari
    try:
        import napari
    except ImportError:
        print("\nError: napari not installed")
        print("  pip install napari[all]")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Launching napari...")
    print(f"{'='*60}\n")

    # Create viewer
    viewer = napari.Viewer(title=f"Manual Crop - {pipeline_folder.name}")

    # Add brain volume
    viewer.add_image(
        volume,
        name=f"ch{channel}",
        scale=scale,
        colormap='green' if channel == 0 else 'magenta'
    )

    # Try to open the manual crop plugin
    try:
        # Import the widget from the installed package
        from napari_manual_crop.manual_crop_widget import ManualCropWidget

        # Load metadata
        import json
        meta_path = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
        metadata = None
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        # Add it to the viewer with pipeline info
        widget = ManualCropWidget(viewer, pipeline_folder=pipeline_folder, metadata=metadata)
        viewer.window.add_dock_widget(widget, name="Manual Crop", area='right')

        print("✓ Manual crop plugin loaded")
        print("\nInstructions:")
        print("  1. Click 'Add Crop Line' in the plugin panel")
        print("  2. Adjust the Y position with +/- buttons")
        print("  3. Click 'Apply Crop' when ready")
        print()

    except Exception as e:
        print(f"⚠️  Could not load manual crop plugin")
        print(f"  Error: {e}")
        print("\nYou can still crop manually using napari's built-in tools.")

    # Run napari
    napari.run()

    # After napari closes, check if user selected a crop position
    crop_file = pipeline_folder / ".crop_position.json"
    if crop_file.exists():
        print(f"\n{'='*60}")
        print("Applying crop...")
        print(f"{'='*60}\n")

        with open(crop_file, 'r') as f:
            crop_data = json.load(f)

        y_crop = crop_data['y_crop']
        rows_kept = crop_data['rows_kept']
        total_rows = crop_data['total_rows']

        print(f"Crop position: Y={y_crop}")
        print(f"Keeping: {rows_kept}/{total_rows} rows ({100*rows_kept/total_rows:.1f}%)\n")

        source_folder = pipeline_folder / FOLDER_EXTRACTED
        output_folder = pipeline_folder / "2_Cropped_For_Registration"

        # Load metadata
        meta_path = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
        metadata = None
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        # Perform the crop
        import tifffile
        output_folder.mkdir(parents=True, exist_ok=True)

        # Find all channels
        channels = sorted([d for d in source_folder.iterdir()
                         if d.is_dir() and d.name.startswith("ch")])

        total_files = sum(len(list(ch.glob("Z*.tif"))) for ch in channels)
        processed = 0

        for ch_folder in channels:
            ch_name = ch_folder.name
            print(f"Processing {ch_name}...")

            out_ch = output_folder / ch_name
            out_ch.mkdir(exist_ok=True)

            tiff_files = sorted(ch_folder.glob("Z*.tif"))

            for tiff_path in tiff_files:
                img = tifffile.imread(str(tiff_path))
                cropped = img[:y_crop, :]  # Keep everything BEFORE y_crop (top portion)

                out_path = out_ch / tiff_path.name
                tifffile.imwrite(str(out_path), cropped)

                processed += 1
                if processed % 50 == 0:
                    print(f"  {processed}/{total_files} files processed ({100*processed/total_files:.1f}%)")

        print(f"  {processed}/{total_files} files processed (100.0%)")

        # Save metadata
        print("\nSaving metadata...")
        first_ch = channels[0]
        first_tif = sorted(first_ch.glob("Z*.tif"))[0]
        first_img = tifffile.imread(str(first_tif))
        orig_y, x = first_img.shape
        z = len(list(first_ch.glob("Z*.tif")))

        crop_meta = metadata.copy() if metadata else {}
        crop_meta['crop_y_end'] = y_crop  # Crop line position (everything before this is kept)
        crop_meta['crop_y_original'] = orig_y
        crop_meta['crop_y_new'] = y_crop  # New height = y_crop (kept top portion)
        crop_meta['crop_method'] = 'napari_plugin_manual'
        crop_meta['dimensions_cropped'] = {
            'z': z,
            'y': y_crop,  # New Y dimension
            'x': x,
        }

        with open(output_folder / "metadata.json", 'w') as f:
            json.dump(crop_meta, f, indent=2)

        # Clean up temp file
        crop_file.unlink()

        print(f"\n{'='*60}")
        print("✓ Crop complete!")
        print(f"{'='*60}")
        print(f"Output: {output_folder}")
        print(f"\nYou can now run the next pipeline step (registration).")
    else:
        print("\n(Crop position not saved - napari closed without applying crop)")


if __name__ == '__main__':
    main()
