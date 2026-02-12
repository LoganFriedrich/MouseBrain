"""
test_alignment_widget.py - Test the BrainSlice alignment widget

This script:
1. Creates a synthetic brain-like test image
2. Launches napari with the SliceAtlas widget
3. Prints step-by-step instructions in the console

Run with: python test_alignment_widget.py
"""

import numpy as np
import napari


def create_test_image(shape=(1000, 1200)):
    """Create a synthetic brain-like test image."""
    y, x = np.ogrid[:shape[0], :shape[1]]

    # Center of image
    cy, cx = shape[0] // 2, shape[1] // 2

    # Create elliptical "brain" shape
    a, b = shape[0] * 0.4, shape[1] * 0.45  # semi-axes
    ellipse = ((y - cy) / a) ** 2 + ((x - cx) / b) ** 2
    brain_mask = ellipse < 1

    # Add some internal structure
    image = np.zeros(shape, dtype=np.float32)
    image[brain_mask] = 0.3

    # Add "ventricles" (dark regions in center)
    ventricle_mask = ellipse < 0.15
    image[ventricle_mask] = 0.1

    # Add some "cortex" (brighter outer ring)
    cortex_mask = (ellipse > 0.6) & (ellipse < 1)
    image[cortex_mask] = 0.6

    # Add noise
    noise = np.random.normal(0, 0.05, shape).astype(np.float32)
    image = np.clip(image + noise, 0, 1)

    # Scale to 16-bit range
    image = (image * 65535).astype(np.uint16)

    return image


def print_instructions():
    """Print step-by-step instructions."""
    print()
    print("=" * 70)
    print("  BRAINSLICE ALIGNMENT WIDGET TEST")
    print("=" * 70)
    print()
    print("A synthetic test image has been loaded. Follow these steps:")
    print()
    print("STEP 1: LOAD ATLAS")
    print("-" * 50)
    print("  1. Click on the '2. Align' tab in the widget")
    print("  2. In the 'Atlas' section, select 'Allen Mouse Brain (25um)'")
    print("     (25um is faster to load than 10um)")
    print("  3. Click 'Load Atlas'")
    print("  4. Wait for status to turn green: 'Loaded: allen_mouse_25um'")
    print()
    print("STEP 2: FIND MATCHING POSITION")
    print("-" * 50)
    print("  1. Use the 'Position' slider to browse atlas slices")
    print("  2. Try moving it to ~50% (middle of brain)")
    print("  3. Check 'Show atlas reference' to see the overlay")
    print("  4. Adjust 'Opacity' slider to see both images")
    print()
    print("STEP 3: MANUAL ALIGNMENT")
    print("-" * 50)
    print("  1. Use 'X offset' and 'Y offset' to shift the atlas")
    print("  2. Use 'Rotation' to rotate (try small values like -5 to +5)")
    print("  3. Use 'Scale' to resize (try 0.8 to 1.2)")
    print("  4. Click 'Reset Transform' to start over")
    print()
    print("STEP 4: AUTOMATIC DETECTION (optional)")
    print("-" * 50)
    print("  1. Click 'Detect Position (Single Slice)'")
    print("  2. Wait for progress bar to complete")
    print("  3. The position slider will jump to the best match")
    print()
    print("OVERLAY OPTIONS:")
    print("-" * 50)
    print("  - 'Show atlas reference': grayscale atlas image")
    print("  - 'Show region labels': colored region fill")
    print("  - 'Show region boundaries': outline only")
    print()
    print("=" * 70)
    print("  Close the napari window when done testing")
    print("=" * 70)
    print()


def main():
    print("Creating test image...")

    # Create synthetic images (red = nuclear, green = signal)
    red_image = create_test_image((1000, 1200))
    green_image = create_test_image((1000, 1200)) // 2  # Dimmer green channel

    print("Launching napari...")

    # Create viewer
    viewer = napari.Viewer()

    # Add test images
    viewer.add_image(
        red_image,
        name="Nuclear (red) - TEST",
        colormap='red',
        blending='additive',
    )
    viewer.add_image(
        green_image,
        name="Signal (green) - TEST",
        colormap='green',
        blending='additive',
    )

    # Try to add the BrainSlice widget
    try:
        from brainslice.napari_plugin.main_widget import SliceAtlasWidget

        widget = SliceAtlasWidget(viewer)

        # Manually set the channels so alignment widget can access them
        widget.red_channel = red_image
        widget.green_channel = green_image
        widget.current_file = None

        viewer.window.add_dock_widget(widget, name="BrainSlice", area='right')

        print("BrainSlice widget loaded successfully!")

        # Switch to Align tab
        widget.tabs.setCurrentIndex(1)  # Index 1 = "2. Align" tab

    except ImportError as e:
        print(f"ERROR: Could not import BrainSlice widget: {e}")
        print()
        print("Make sure you've installed the package:")
        print("  cd y:\\2_Connectome\\5_Enhancer_Aim\\Script_Tools\\BrainSlice")
        print("  pip install -e .")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print instructions
    print_instructions()

    # Run napari
    napari.run()

    print("Test complete!")


if __name__ == "__main__":
    main()
