"""
Test script to verify folder loading works correctly.
Run this from the command line to debug folder loading issues.

Usage:
    python test_folder_load.py "path/to/folder/with/images"
"""

import sys
from pathlib import Path

# Add brainslice to path
sys.path.insert(0, str(Path(__file__).parent))

def test_folder_load(folder_path: str):
    """Test loading a folder of images."""
    from brainslice.core.io import find_images_in_folder, load_folder, load_image

    folder = Path(folder_path)
    print(f"\n=== Testing folder loading ===")
    print(f"Folder: {folder}")
    print(f"Exists: {folder.exists()}")
    print(f"Is directory: {folder.is_dir()}")

    # Step 1: Find images
    print(f"\n--- Finding images ---")
    try:
        files = find_images_in_folder(folder)
        print(f"Found {len(files)} images:")
        for i, f in enumerate(files[:5]):  # Show first 5
            print(f"  {i}: {f.name}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    except Exception as e:
        print(f"ERROR finding images: {e}")
        import traceback
        traceback.print_exc()
        return

    if not files:
        print("No images found! Check that the folder contains .nd2, .tif, or .tiff files")
        return

    # Step 2: Load first image individually
    print(f"\n--- Loading first image individually ---")
    try:
        data, metadata = load_image(files[0], z_projection='max')
        print(f"Loaded: {files[0].name}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Channels: {metadata.get('channels', 'unknown')}")
        print(f"  Z-projection: {metadata.get('z_projection', 'none')}")
    except Exception as e:
        print(f"ERROR loading first image: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Load folder as stack
    print(f"\n--- Loading folder as stack ---")
    try:
        def progress(current, total, filename):
            print(f"  Loading {current}/{total}: {filename}")

        stack, stack_meta = load_folder(
            folder,
            z_projection='max',
            progress_callback=progress,
        )

        print(f"\nStack loaded successfully!")
        print(f"  Shape: {stack.shape}")
        print(f"  Dtype: {stack.dtype}")
        print(f"  Memory: {stack.nbytes / 1024 / 1024:.1f} MB")
        print(f"  N slices: {stack_meta.get('n_slices', 'unknown')}")
        print(f"  N channels: {stack_meta.get('n_channels', 'unknown')}")

    except Exception as e:
        print(f"ERROR loading folder: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Test channel extraction
    print(f"\n--- Testing channel extraction ---")
    try:
        red_idx = 1  # 561nm
        green_idx = 0  # 488nm

        if stack.ndim == 4 and stack.shape[1] > max(red_idx, green_idx):
            red_stack = stack[:, red_idx, :, :]
            green_stack = stack[:, green_idx, :, :]
            print(f"Red channel (idx {red_idx}): {red_stack.shape}")
            print(f"Green channel (idx {green_idx}): {green_stack.shape}")
            print(f"Red min/max: {red_stack.min()}/{red_stack.max()}")
            print(f"Green min/max: {green_stack.min()}/{green_stack.max()}")
        else:
            print(f"Stack shape {stack.shape} doesn't support channel extraction with indices {red_idx}, {green_idx}")

    except Exception as e:
        print(f"ERROR extracting channels: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== Test complete ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_folder_load.py <folder_path>")
        print("\nExample:")
        print('  python test_folder_load.py "Y:\\path\\to\\images"')
        sys.exit(1)

    test_folder_load(sys.argv[1])
