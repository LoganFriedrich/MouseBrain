#!/usr/bin/env python3
"""
MouseBrain CLI - Tissue analysis tools for 3D cleared brains and 2D slices.

Usage:
    mousebrain                      # Launch napari with all plugins
    mousebrain coloc IMAGE          # Run colocalization on a single image
    mousebrain coloc FOLDER         # Batch colocalization on all ND2s in folder
    mousebrain coloc IMAGE --open   # Run and open result image
    mousebrain --crop BRAIN         # Launch manual crop for specific brain
    mousebrain --check              # Check dependencies
    mousebrain --paths              # Show configured paths
"""

import sys
import argparse


def main():
    """Main entry point for mousebrain command."""
    parser = argparse.ArgumentParser(
        prog="mousebrain",
        description="MouseBrain - tissue analysis tools for 3D cleared brains and 2D slices",
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

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    # ── mousebrain coloc ──
    coloc_parser = subparsers.add_parser(
        "coloc",
        help="Nuclear signal colocalization analysis",
        description=(
            "Detect nuclei in the red channel, classify each as green-positive "
            "or negative using background_mean + sigma threshold, and generate "
            "a result figure. Works on single ND2/TIFF files or batch-processes "
            "all ND2 files in a folder."
        ),
    )
    coloc_parser.add_argument(
        "input",
        help="ND2/TIFF file or folder of ND2 files",
    )
    coloc_parser.add_argument(
        "--open", action="store_true", dest="open_result",
        help="Open result image(s) after processing",
    )
    coloc_parser.add_argument(
        "--sigma", type=float, default=1.5,
        help="Sigma threshold for positive classification (default: 1.5)",
    )
    coloc_parser.add_argument(
        "--edge-exclusion", type=int, default=50,
        help="Exclude detections within N pixels of tissue edge (default: 50)",
    )
    coloc_parser.add_argument(
        "--method", default="zscore",
        choices=["zscore", "otsu", "percentile", "manual"],
        help="Detection threshold method (default: zscore)",
    )
    coloc_parser.add_argument(
        "--min-diameter", type=float, default=8.0,
        help="Minimum nucleus diameter in um (default: 8.0)",
    )

    args = parser.parse_args()

    if args.version:
        from mousebrain import __version__
        print(f"mousebrain {__version__}")
        return 0

    if args.paths:
        return show_paths()

    if args.check:
        return check_dependencies()

    if args.command == "coloc":
        return run_coloc_command(args)

    if args.crop:
        return launch_manual_crop(args.crop)

    return launch_napari()


def run_coloc_command(args):
    """Run colocalization on a file or folder."""
    from pathlib import Path
    import time

    target = Path(args.input)
    if not target.exists():
        print(f"Error: {target} does not exist")
        return 1

    # Collect files to process
    if target.is_file():
        files = [target]
    else:
        files = sorted(target.glob("*.nd2"))
        tiffs = sorted(target.glob("*.tif")) + sorted(target.glob("*.tiff"))
        files.extend(tiffs)
        if not files:
            print(f"No ND2 or TIFF files found in {target}")
            return 1
        print(f"Found {len(files)} images in {target.name}/")

    # Build CLI args for run_coloc
    coloc_argv = [
        "--method", args.method,
        "--min-diameter-um", str(args.min_diameter),
        "--sigma-threshold", str(args.sigma),
        "--edge-exclusion", str(args.edge_exclusion),
    ]

    # Process each file
    results = []
    t_total = time.time()

    for i, f in enumerate(files):
        if len(files) > 1:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(files)}] {f.name}")
            print(f"{'='*60}")

        from mousebrain.plugin_2d.sliceatlas.cli.run_coloc import main as coloc_main
        # Temporarily replace sys.argv for the coloc CLI
        import sys as _sys
        saved_argv = _sys.argv
        _sys.argv = ["coloc", str(f)] + coloc_argv
        try:
            coloc_main()
            results.append((f.name, True))
        except SystemExit:
            results.append((f.name, True))  # argparse exits on success
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((f.name, False))
        finally:
            _sys.argv = saved_argv

        # Open result if requested
        if args.open_result:
            result_png = f.with_suffix('.coloc_result.png')
            if result_png.exists():
                import os
                os.startfile(str(result_png))

    # Batch summary
    if len(files) > 1:
        elapsed = time.time() - t_total
        n_ok = sum(1 for _, ok in results if ok)
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {n_ok}/{len(files)} succeeded in {elapsed:.1f}s")
        for name, ok in results:
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {name}")
        print(f"{'='*60}")

    return 0


def show_paths():
    """Show configured paths."""
    from mousebrain import (
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
    """Check that all required dependencies are available, grouped by install extra."""
    from mousebrain import __version__
    print(f"BrainTools v{__version__} -- Dependency Check")
    print("=" * 55)

    # Core (always required)
    core_deps = [
        ("napari", "napari"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("tifffile", "tifffile"),
        ("h5py", "h5py"),
        ("matplotlib", "matplotlib"),
        ("natsort", "natsort"),
        ("qtpy", "qtpy"),
    ]

    # 3D pipeline extras
    deps_3d = [
        ("brainreg", "brainreg"),
        ("brainglobe-atlasapi", "brainglobe_atlasapi"),
        ("brainglobe-utils", "brainglobe_utils"),
        ("cellfinder", "cellfinder"),
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("zarr", "zarr"),
        ("dask", "dask"),
        ("imaris-ims-file-reader", "imaris_ims_file_reader"),
        ("torch (manual install)", "torch"),
    ]

    # 2D pipeline extras
    deps_2d = [
        ("scikit-image", "skimage"),
        ("stardist", "stardist"),
        ("brainglobe-atlasapi", "brainglobe_atlasapi"),
        ("pillow", "PIL"),
        ("nd2", "nd2"),
    ]

    # 2D alignment extras
    deps_2d_align = [
        ("DeepSlice", "DeepSlice"),
        ("SimpleITK", "SimpleITK"),
    ]

    # napari plugins
    plugins = [
        ("3D plugin (plugin_3d)", "mousebrain.plugin_3d"),
        ("2D plugin (plugin_2d)", "mousebrain.plugin_2d"),
    ]

    all_ok = True

    sections = [
        ("Core (always required)", core_deps),
        ("3D Pipeline — pip install mousebrain[3d]", deps_3d),
        ("2D Pipeline — pip install mousebrain[2d]", deps_2d),
        ("2D Alignment — pip install mousebrain[2d-alignment]", deps_2d_align),
        ("napari Plugins", plugins),
    ]

    for section_name, deps in sections:
        print(f"\n{section_name}:")
        section_ok = True
        for name, module in deps:
            try:
                __import__(module)
                print(f"  [OK] {name}")
            except ImportError:
                print(f"  [--] {name}")
                section_ok = False
        if not section_ok and section_name.startswith("Core"):
            all_ok = False

    print("\n" + "=" * 55)
    if all_ok:
        print("Core dependencies OK. Optional extras shown above.")
    else:
        print("Some CORE dependencies missing.")
        print("Run: pip install -e .")
    print("\nInstall extras:")
    print("  pip install -e \".[3d]\"           # 3D cleared brain pipeline")
    print("  pip install -e \".[2d]\"           # 2D slice analysis")
    print("  pip install -e \".[all]\"          # Everything")

    return 0 if all_ok else 1


def launch_napari():
    """Launch napari with plugins."""
    try:
        import napari
        print("Launching napari with MouseBrain...")
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

        # Use config-detected BRAINS_ROOT (handles Y: drive preference internally)
        from mousebrain import BRAINS_ROOT
        debug(f"BRAINS_ROOT from config: {BRAINS_ROOT}")

        debug(f"Searching for brain '{brain_name}' in: {BRAINS_ROOT}")

        # Find the brain using glob (faster than nested iteration over network)
        brain_folder = None
        debug("Searching for brain with glob pattern...")

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
            from mousebrain.plugin_3d.manual_crop_widget import ManualCropWidget
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
            print("Use Plugins > Connectome Pipeline > Manual Crop")
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
