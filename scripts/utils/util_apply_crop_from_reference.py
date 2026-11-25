#!/usr/bin/env python3
"""
util_apply_crop_from_reference.py

Apply crop settings from a reference brain to other brains.

Uses either:
1. Proportional cropping (same % of Y kept) - default
2. Absolute Y value (if brains are same size)

Usage:
    # Apply 349's crop to new brains (proportional)
    python util_apply_crop_from_reference.py --reference 001_CNT_01_01_1p625x_z4 --target 006_CNT_03_01_1p625x_z4

    # Apply to multiple brains
    python util_apply_crop_from_reference.py --reference 001_CNT_01_01_1p625x_z4 --target 006_CNT_03_01_1p625x_z4 007_CNT_03_02_1p625x_z4

    # Use absolute Y value instead of proportional
    python util_apply_crop_from_reference.py --reference 001_CNT_01_01_1p625x_z4 --target 006_CNT_03_01_1p625x_z4 --absolute

    # Dry run (show what would happen)
    python util_apply_crop_from_reference.py --reference 001_CNT_01_01_1p625x_z4 --target 006_CNT_03_01_1p625x_z4 --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile

from mousebrain.config import BRAINS_ROOT

FOLDER_EXTRACTED = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"


def find_brain(brain_name: str) -> Path:
    """Find a brain's pipeline folder."""
    from mousebrain.config import find_brain_folder
    # One rule for which folders are brains (config.find_brain_folder): an
    # archive beside a brain is never picked, and a part of a name that fits
    # several brains is refused instead of taking the first one listed.
    found, matches = find_brain_folder(brain_name, BRAINS_ROOT)
    if found is None and len(matches) > 1:
        print(f"Error: '{brain_name}' matches more than one brain; give the full name:")
        for p in matches:
            print(f"  {p.name}")
    return found


def get_crop_info(pipeline_folder: Path) -> dict:
    """Get crop information from a brain's metadata."""
    # Check cropped folder first
    crop_meta = pipeline_folder / FOLDER_CROPPED / "metadata.json"
    if crop_meta.exists():
        with open(crop_meta, 'r') as f:
            meta = json.load(f)
            return {
                'crop_y_end': meta.get('crop_y_new') or meta.get('crop_y_end'),
                'original_y': meta.get('crop_y_original') or meta.get('crop_analysis', {}).get('original_height'),
                'method': meta.get('crop_method', 'unknown'),
                'source': 'cropped_metadata'
            }

    # Fall back to extracted folder metadata
    full_meta = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
    if full_meta.exists():
        with open(full_meta, 'r') as f:
            meta = json.load(f)
            crop_analysis = meta.get('crop_analysis', {})
            return {
                'crop_y_end': meta.get('crop_y_new') or meta.get('crop_y_end') or crop_analysis.get('cropped_height'),
                'original_y': crop_analysis.get('original_height'),
                'method': meta.get('crop_method', 'auto'),
                'source': 'full_metadata'
            }

    return None


def get_full_dimensions(pipeline_folder: Path) -> dict:
    """Get the full (uncropped) dimensions of a brain."""
    # Read from metadata
    full_meta = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
    if full_meta.exists():
        with open(full_meta, 'r') as f:
            meta = json.load(f)
            crop_analysis = meta.get('crop_analysis', {})
            if 'original_height' in crop_analysis:
                return {'y': crop_analysis['original_height']}

    # Fall back to counting TIFFs and reading dimensions
    ch_folder = pipeline_folder / FOLDER_EXTRACTED / "ch0"
    if not ch_folder.exists():
        ch_folder = pipeline_folder / FOLDER_EXTRACTED / "ch1"

    if ch_folder.exists():
        tiff_files = sorted(ch_folder.glob("Z*.tif"))
        if tiff_files:
            sample = tifffile.imread(str(tiff_files[0]))
            return {
                'z': len(tiff_files),
                'y': sample.shape[0],
                'x': sample.shape[1]
            }

    return None


def apply_crop(source_folder: Path, output_folder: Path, y_crop: int,
               metadata: dict, dry_run: bool = False):
    """Apply crop to a brain's extracted data.

    The new crop is built in ``<output_folder>.partial`` and swapped in only
    once every plane, cache and metadata.json are written; the crop that was
    there goes into ``<output_folder>/_archive/<time>/`` (kept, not deleted)
    -- crop_subprocess.new_partial / swap_in, the rule every crop tool
    follows. WHY: writing over the existing crop plane by plane
    (with --force) would let a run stopped part-way leave old and new planes
    mixed under the old metadata, which the batches would register unattended.
    """
    if dry_run:
        print(f"  [DRY RUN] Would crop to Y={y_crop}")
        return True

    from mousebrain.plugin_3d.crop_subprocess import new_partial, swap_in
    partial = new_partial(output_folder)
    written = 0

    # Get channel folders
    for ch_name in ['ch0', 'ch1']:
        ch_source = source_folder / ch_name
        ch_output = partial / ch_name

        if not ch_source.exists():
            continue

        ch_output.mkdir(parents=True, exist_ok=True)

        # Also create zarr output folder (inside the new crop: it moves in
        # with it, and its completion record still matches the planes)
        zarr_output = partial / f"{ch_name}.zarr"

        tiff_files = sorted(ch_source.glob("Z*.tif"))
        print(f"  Cropping {ch_name}: {len(tiff_files)} slices to Y={y_crop}...")

        cropped_slices = []
        for i, tiff_path in enumerate(tiff_files):
            img = tifffile.imread(str(tiff_path))
            cropped = img[:y_crop, :]  # Crop Y dimension

            # Save TIFF
            out_path = ch_output / tiff_path.name
            tifffile.imwrite(str(out_path), cropped)
            cropped_slices.append(cropped)
            written += 1

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(tiff_files)} slices...")

        # Create zarr for fast loading -- under a partial name, completion
        # record written last (mousebrain.zarr_cache), so a failed save can no
        # longer leave an empty cache that the widget loads as the brain.
        try:
            from mousebrain.zarr_cache import write_array
            volume = np.stack(cropped_slices)
            write_array(volume, ch_output, zarr_output)
            print(f"  Created {ch_name}.zarr")
        except Exception as e:
            print(f"  Warning: Could not create zarr (the TIFFs are complete; the widget "
                  f"will build the cache itself): {e}")

    if not written:
        # Nothing to crop: the crop already there is left as it is.
        shutil.rmtree(partial)
        print(f"  ERROR: no extracted planes found in {source_folder} (ch0/ch1); nothing cropped")
        return False

    # Save metadata
    metadata['crop_y_end'] = y_crop
    metadata['crop_y_new'] = y_crop
    metadata['crop_method'] = 'reference_copy'
    metadata['crop_applied_at'] = datetime.now().isoformat()

    meta_path = partial / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Every plane and the metadata are written: swap the new crop in.
    previous = swap_in(output_folder, partial)
    print(f"  Saved metadata to {meta_path.name}")
    if previous:
        print(f"  The previous crop was moved to: {previous}")
        print(f"  A registration made on the previous crop is now OUT OF DATE: "
              f"register the brain again.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply crop settings from a reference brain to other brains"
    )
    parser.add_argument('--reference', '-r', required=True,
                        help="Reference brain name (e.g., 001_CNT_01_01_1p625x_z4)")
    parser.add_argument('--target', '-t', nargs='+', required=True,
                        help="Target brain name(s) to apply crop to")
    parser.add_argument('--absolute', '-a', action='store_true',
                        help="Use absolute Y value instead of proportional")
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help="Show what would happen without making changes")
    parser.add_argument('--force', '-f', action='store_true',
                        help="Replace existing crops (each previous crop is kept in "
                             "2_Cropped_For_Registration/_archive/ once the new one is complete)")

    args = parser.parse_args()

    print("=" * 60)
    print("Apply Crop from Reference Brain")
    print("=" * 60)

    # Find reference brain
    ref_folder = find_brain(args.reference)
    if not ref_folder:
        print(f"ERROR: Reference brain '{args.reference}' not found")
        sys.exit(1)

    print(f"\nReference: {ref_folder.name}")

    # Get reference crop info
    ref_crop = get_crop_info(ref_folder)
    if not ref_crop or not ref_crop.get('crop_y_end'):
        print(f"ERROR: No crop info found for reference brain")
        sys.exit(1)

    ref_y = ref_crop['crop_y_end']
    ref_original = ref_crop.get('original_y')

    if ref_original:
        ref_ratio = ref_y / ref_original
        print(f"  Crop Y: {ref_y} / {ref_original} ({ref_ratio*100:.1f}% kept)")
    else:
        ref_ratio = None
        print(f"  Crop Y: {ref_y} (original unknown)")

    print(f"  Method: {ref_crop.get('method', 'unknown')}")

    # Process each target brain
    print(f"\nTargets: {len(args.target)} brain(s)")
    print("-" * 60)

    for target_name in args.target:
        print(f"\n{target_name}:")

        target_folder = find_brain(target_name)
        if not target_folder:
            print(f"  ERROR: Brain not found")
            continue

        # Check if already cropped
        crop_folder = target_folder / FOLDER_CROPPED
        if crop_folder.exists() and list(crop_folder.glob("ch*/Z*.tif")):
            if not args.force:
                print(f"  SKIP: Already cropped (use --force to overwrite)")
                continue
            else:
                print(f"  Replacing the existing crop (it is kept in "
                      f"{FOLDER_CROPPED}/_archive/ once the new one is complete)")

        # Get target dimensions
        target_dims = get_full_dimensions(target_folder)
        if not target_dims or not target_dims.get('y'):
            print(f"  ERROR: Could not determine target dimensions")
            continue

        target_original_y = target_dims['y']

        # Calculate crop position
        if args.absolute:
            target_crop_y = ref_y
            method = "absolute"
        else:
            if ref_ratio:
                target_crop_y = int(target_original_y * ref_ratio)
                method = f"proportional ({ref_ratio*100:.1f}%)"
            else:
                target_crop_y = ref_y
                method = "absolute (no ratio available)"

        # Clamp to valid range
        target_crop_y = min(target_crop_y, target_original_y)

        print(f"  Original Y: {target_original_y}")
        print(f"  Crop to Y: {target_crop_y} ({method})")
        print(f"  Keeping: {target_crop_y/target_original_y*100:.1f}%")

        # Load metadata from source
        source_folder = target_folder / FOLDER_EXTRACTED
        meta_path = source_folder / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata['crop_y_original'] = target_original_y
        metadata['reference_brain'] = args.reference
        metadata['reference_crop_y'] = ref_y

        # Apply crop
        success = apply_crop(
            source_folder=source_folder,
            output_folder=crop_folder,
            y_crop=target_crop_y,
            metadata=metadata,
            dry_run=args.dry_run
        )

        if success:
            print(f"  {'[DRY RUN] Would be done' if args.dry_run else 'Done!'}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
