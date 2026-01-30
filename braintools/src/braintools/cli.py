#!/usr/bin/env python3
"""
BrainTools CLI - Launch napari with all plugins loaded.

Usage:
    braintool              # Launch napari
    braintool --crop BRAIN # Launch manual crop for specific brain
    braintool --check      # Check dependencies
    braintool --help       # Show help
"""

import sys
import argparse


def main():
    """Main entry point for braintool command."""
    parser = argparse.ArgumentParser(
        prog="braintool",
        description="Launch napari with SCI-Connectome pipeline tools"
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check dependencies and exit"
    )
    parser.add_argument(
        "--paths", "-p",
        action="store_true",
        help="Show configured paths and exit"
    )
    parser.add_argument(
        "--crop",
        metavar="BRAIN",
        help="Launch manual crop tool for specified brain (e.g., 349_CNT_01_02_1p625x_z4)"
    )

    args = parser.parse_args()

    if args.version:
        from braintools import __version__
        print(f"braintool {__version__}")
        return 0

    if args.paths:
        return show_paths()

    if args.check:
        return check_dependencies()

    if args.crop:
        return launch_manual_crop(args.crop)

    return launch_napari()


def show_paths():
    """Show configured paths."""
    from braintools import (
        TISSUE_ROOT, CLEARED_3D_ROOT, SLICES_2D_ROOT,
        INJURY_ROOT, BRAINS_ROOT, DATA_SUMMARY_DIR, SCRIPTS_DIR
    )

    print("BrainTools Configured Paths")
    print("=" * 50)
    print(f"Tissue Root:     {TISSUE_ROOT}")
    print(f"3D Cleared:      {CLEARED_3D_ROOT}")
    print(f"2D Slices:       {SLICES_2D_ROOT}")
    print(f"Injury:          {INJURY_ROOT}")
    print(f"Brains Data:     {BRAINS_ROOT}")
    print(f"Data Summary:    {DATA_SUMMARY_DIR}")
    print(f"Scripts:         {SCRIPTS_DIR}")
    print("=" * 50)

    # Check existence
    paths = [
        ("Tissue Root", TISSUE_ROOT),
        ("3D Cleared", CLEARED_3D_ROOT),
        ("2D Slices", SLICES_2D_ROOT),
        ("Brains", BRAINS_ROOT),
    ]

    all_ok = True
    for name, path in paths:
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        if not exists:
            all_ok = False
            print(f"  {status} {name}: {path}")

    return 0 if all_ok else 1


def check_dependencies():
    """Check that all required dependencies are available."""
    print("Checking BrainTools dependencies...")
    print("-" * 40)

    deps = [
        ("napari", "napari"),
        ("numpy", "numpy"),
        ("brainreg", "brainreg"),
        ("cellfinder", "cellfinder"),
        ("torch", "torch"),
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("sci_connectome_napari", "sci_connectome_napari"),
    ]

    all_ok = True
    for name, module in deps:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [X]  {name}: {e}")
            all_ok = False

    print("-" * 40)
    if all_ok:
        print("All dependencies OK!")
        return 0
    else:
        print("Some dependencies missing.")
        print("Run: pip install -e .[cuda] --extra-index-url https://download.pytorch.org/whl/cu118")
        return 1


def launch_napari():
    """Launch napari with plugins."""
    try:
        import napari
        print("Launching napari with BrainTools...")
        viewer = napari.Viewer()
        napari.run()
        return 0
    except Exception as e:
        print(f"Error launching napari: {e}")
        return 1


def launch_manual_crop(brain_name):
    """Launch napari with manual crop widget for a specific brain."""
    import time
    t0 = time.time()

    def debug(msg):
        elapsed = time.time() - t0
        print(f"[{elapsed:6.2f}s] {msg}")

    try:
        debug("Starting launch_manual_crop...")
        debug("Importing napari...")
        import napari
        from pathlib import Path
        debug("napari imported")

        # CRITICAL: Use Y: drive directly to avoid UNC path slowness
        BRAINS_ROOT = Path(r"Y:\2_Connectome\Tissue\3D_Cleared\1_Brains")
        debug(f"Checking Y: drive path exists: {BRAINS_ROOT}")
        if not BRAINS_ROOT.exists():
            debug("Y: drive not found, falling back to braintools import...")
            from braintools import BRAINS_ROOT as _fallback
            BRAINS_ROOT = _fallback
            debug(f"Fallback BRAINS_ROOT: {BRAINS_ROOT}")

        debug(f"Searching for brain '{brain_name}' in: {BRAINS_ROOT}")

        # Find the brain using glob (faster than nested iteration over network)
        brain_folder = None
        debug("Searching for brain with glob pattern...")

        # Use glob to find matching brain folders (much faster over network)
        pattern = f"*/*{brain_name}*"
        debug(f"  Pattern: {pattern}")
        matches = list(BRAINS_ROOT.glob(pattern))
        debug(f"  Found {len(matches)} matches")

        if matches:
            brain_folder = matches[0]
            debug(f"  Using: {brain_folder.name}")

        if not brain_folder:
            print(f"Error: Brain '{brain_name}' not found in {BRAINS_ROOT}")
            print("\nAvailable brains:")
            # Use glob for faster listing
            for folder in sorted(BRAINS_ROOT.glob("*/*")):
                if folder.is_dir() and not folder.name.startswith('.'):
                    print(f"  - {folder.name}")
            return 1

        # Check for extracted data
        debug("Checking for extracted data...")
        extracted_folder = brain_folder / "1_Extracted_Full"
        if not extracted_folder.exists():
            print(f"Error: No extracted data found at {extracted_folder}")
            print("Run extraction first: python 2_extract_and_analyze.py")
            return 1

        debug(f"Brain found: {brain_folder.name}")
        debug(f"  Path: {brain_folder}")

        # Launch napari
        debug("Creating napari.Viewer()...")
        viewer = napari.Viewer()
        debug("Viewer created")

        # Try to open the manual crop widget with this brain
        try:
            debug("Importing ManualCropWidget...")
            from sci_connectome_napari.manual_crop_widget import ManualCropWidget
            debug("ManualCropWidget imported")

            debug("Creating ManualCropWidget instance...")
            widget = ManualCropWidget(viewer)
            debug("Widget created")

            debug("Adding widget to dock...")
            viewer.window.add_dock_widget(widget, name="Manual Crop", area="right")
            debug("Widget added to dock")

            # Pre-select the brain if possible
            if hasattr(widget, 'brain_combo'):
                debug("Looking for brain in combo box...")
                for i in range(widget.brain_combo.count()):
                    if brain_name in widget.brain_combo.itemText(i):
                        widget.brain_combo.setCurrentIndex(i)
                        debug(f"Pre-selected brain at index {i}")
                        break
        except Exception as e:
            print(f"Warning: Could not auto-open crop widget: {e}")
            print("Use Plugins → SCI-Connectome Pipeline → Manual Crop")
            import traceback
            traceback.print_exc()

        debug("Calling napari.run()...")
        napari.run()
        debug("napari.run() returned")
        return 0

    except Exception as e:
        print(f"Error launching manual crop: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
