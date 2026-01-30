# imports
import sys
import tifffile
import numpy as np
from brainslice.core.io import load_image

# Create a test TIFF file
def create_image(filename):
    test_image = np.random.randint(0, 256, size=(100, 100))
    tifffile.imwrite(filename, test_image)

# Test load_image function
def test_load_image(filename):
    loaded_image = load_image(filename)
    return(loaded_image.shape)

sys.argv[1] = filename

# CLI call load "filename"
if len(sys.argv) > 1:
    # then the user provided a filename
    filename = sys.argv[sys.argv.index("--filename") + 1]
else:
    print("Specify filename to test by adding --filename <path_to_file> to the end of the command line or press enter to generate one and use that.")
    input("Press Enter to continue...")
    create_image("Default_test.tif")
    shape = test_load_image("Default_test.tif")
    print(f"Loaded image shape: {shape}")