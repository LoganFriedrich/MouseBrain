#!/usr/bin/env python3
"""
Subprocess script for performing the actual crop operation.
Called by the napari plugin after crop position is selected.
"""

import sys
import json
from pathlib import Path

FOLDER_EXTRACTED = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"


def do_crop(pipeline_folder: Path):
    """Perform the crop operation based on saved crop position."""
    print(f"[DEBUG] Pipeline folder received: {pipeline_folder}")

    # Validate pipeline folder structure
    source_folder = pipeline_folder / FOLDER_EXTRACTED
    print(f"[DEBUG] Source folder: {source_folder}")
    print(f"[DEBUG] Source folder exists: {source_folder.exists()}")

    if not source_folder.exists():
        print(f"\nError: Source folder not found: {source_folder}")
        print(f"\nExpected structure:")
        print(f"  {pipeline_folder}/")
        print(f"    {FOLDER_EXTRACTED}/")
        print(f"      ch0/")
        print(f"      ch1/")
        print(f"\nPlease check the folder path.")
        return False

    # Read crop position
    crop_file = pipeline_folder / ".crop_position.json"
    if not crop_file.exists():
        print("Error: No crop position file found")
        return False

    with open(crop_file, 'r') as f:
        crop_data = json.load(f)

    y_crop = crop_data['y_crop']
    rows_kept = crop_data['rows_kept']
    total_rows = crop_data['total_rows']

    print(f"\n{'='*60}")
    print("Applying crop...")
    print(f"{'='*60}\n")
    print(f"Crop position: Y={y_crop}")
    print(f"Keeping: {rows_kept}/{total_rows} rows ({100*rows_kept/total_rows:.1f}%)\n")

    source_folder = pipeline_folder / FOLDER_EXTRACTED
    output_folder = pipeline_folder / FOLDER_CROPPED

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
    crop_meta['crop_y_end'] = y_crop
    crop_meta['crop_y_original'] = orig_y
    crop_meta['crop_y_new'] = y_crop
    crop_meta['crop_method'] = 'napari_plugin_manual'
    crop_meta['dimensions_cropped'] = {
        'z': z,
        'y': y_crop,
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

    return True


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: crop_subprocess.py <pipeline_folder>")
        sys.exit(1)

    pipeline_folder = Path(sys.argv[1])
    success = do_crop(pipeline_folder)
    sys.exit(0 if success else 1)
