"""
check_dependencies.py - Check SliceAtlas dependencies

Run this first to see what's installed and what's missing.
"""

import sys

def check_package(name, import_name=None, optional=False):
    """Check if a package is installed."""
    if import_name is None:
        import_name = name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        status = f"OK (v{version})"
        return True, status
    except ImportError as e:
        status = f"MISSING - pip install {name}"
        if optional:
            status = f"OPTIONAL - pip install {name}"
        return False, status


def main():
    print("=" * 60)
    print("SliceAtlas Dependency Check")
    print("=" * 60)
    print()

    # Core dependencies (required)
    print("REQUIRED PACKAGES:")
    print("-" * 40)

    required = [
        ("napari", "napari", False),
        ("numpy", "numpy", False),
        ("qtpy", "qtpy", False),
        ("scikit-image", "skimage", False),
        ("scipy", "scipy", False),
        ("brainglobe-atlasapi", "brainglobe_atlasapi", False),
        ("pandas", "pandas", False),
    ]

    all_required_ok = True
    for pkg_name, import_name, _ in required:
        ok, status = check_package(pkg_name, import_name)
        symbol = "[OK]" if ok else "[X] "
        print(f"  {symbol} {pkg_name:25s} {status}")
        if not ok:
            all_required_ok = False

    print()
    print("OPTIONAL PACKAGES (advanced features):")
    print("-" * 40)

    optional = [
        ("DeepSlice", "DeepSlice", True),
        ("SimpleITK", "SimpleITK", True),
        ("stardist", "stardist", True),
        ("tensorflow", "tensorflow", True),
    ]

    for pkg_name, import_name, _ in optional:
        ok, status = check_package(pkg_name, import_name, optional=True)
        symbol = "[OK]" if ok else "[ ] "
        print(f"  {symbol} {pkg_name:25s} {status}")

    print()
    print("=" * 60)

    # Check atlas availability
    print()
    print("ATLAS CHECK:")
    print("-" * 40)

    try:
        from brainglobe_atlasapi import BrainGlobeAtlas

        # Try to load 25um atlas (smaller, faster)
        print("  Checking Allen Mouse Brain Atlas...")
        try:
            atlas = BrainGlobeAtlas("allen_mouse_25um")
            print(f"  [OK] allen_mouse_25um loaded")
            print(f"       Shape: {atlas.shape}")
            print(f"       Resolution: {atlas.resolution} um")
        except Exception as e:
            print(f"  [X]  allen_mouse_25um not downloaded")
            print(f"       Will download automatically on first use (~200MB)")

    except ImportError:
        print("  [X]  Cannot check atlas - brainglobe-atlasapi not installed")

    print()
    print("=" * 60)

    # Summary
    print()
    if all_required_ok:
        print("All required packages installed!")
        print()
        print("Next step: Run test_alignment_widget.py to test the widget")
    else:
        print("Some required packages are missing.")
        print()
        print("Install all required packages with:")
        print("  pip install napari numpy qtpy scikit-image scipy brainglobe-atlasapi pandas")

    print()


if __name__ == "__main__":
    main()
