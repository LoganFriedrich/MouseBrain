# Slice Annotator

A comprehensive napari plugin for working with ND2 microscopy files. Replaces expensive commercial software for vibratome-sliced confocal imaging workflows.

## Features

- **Load ND2 files** as z-stacks with slice names visible
- **Channel management** - color, contrast, gamma, opacity per channel
- **Annotation tools** - shapes, text, callouts (coming soon)
- **Export to TIFF** with visualization settings flattened
- **Scale bars and metadata overlays**
- **Project save/load** for annotations (coming soon)

## Installation

### Option 1: New Conda Environment (Recommended)

```bash
cd y:\2_Connectome\5_Enhancer_Aim\Script_Tools\SliceExport\slice_annotator

# Create environment with all dependencies
conda env create -f environment.yml

# Activate environment
conda activate slice-annotator

# Install the plugin in development mode
pip install -e .
```

### Option 2: Existing Environment

If you already have a napari environment:

```bash
conda activate your-napari-env

cd y:\2_Connectome\5_Enhancer_Aim\Script_Tools\SliceExport\slice_annotator

# Install dependencies
pip install nd2 dask natsort pillow

# Install the plugin
pip install -e .
```

## Usage

1. Activate the environment: (or activate the env you put it in)
   ```bash
   conda activate slice-annotator
   ```

2. Launch napari:
   ```bash
   napari
   ```

3. Open the plugin: **Plugins -> Slice Annotator**

4. Use the tabs:
   - **Load** - Select ND2 file or folder
   - **Channels** - Adjust colors, contrast, gamma
   - **Annotate** - Draw shapes and add labels
   - **Export** - Save as TIFF with current settings

## Keyboard Shortcuts

(Coming in future update)

## File Format Support

- **Input**: ND2 (Nikon), TIFF
- **Output**: TIFF (8-bit RGB composite or 16-bit per channel)

## Dependencies

- Python 3.9-3.11
- napari >= 0.4.17
- nd2 >= 0.5.0
- numpy, dask, tifffile, pillow, natsort

## Troubleshooting

### Plugin not appearing in napari

Make sure you installed with `pip install -e .` (note the dot) while in the `slice_annotator` directory that contains `setup.py`.

### ND2 files not loading

Install the nd2 library: `pip install nd2`

### Qt/display issues

Try: `pip install pyqt5` or use the conda environment which handles Qt dependencies.
