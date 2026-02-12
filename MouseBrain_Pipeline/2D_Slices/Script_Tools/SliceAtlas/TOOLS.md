# Tool Reference for 2D Brain Slice Analysis

## Primary Tools

### ABBA (Aligning Big Brains & Atlases)
**Purpose**: 2D slice-to-atlas registration
**Why it's best for slices**: Specifically designed for registering individual 2D sections to 3D reference atlases. Handles the inherent challenge that a 2D slice can match multiple positions in a 3D atlas.

**Key Features**:
- Interactive slice positioning in napari
- Non-linear (spline) deformation for precise alignment
- Multi-channel support
- Export transformations for downstream analysis

**Installation**:
```bash
# Requires Java (for BigDataViewer backend)
pip install napari-abba
```

**napari Plugin**: `napari-abba`
**Documentation**: https://biop.github.io/ijp-imagetoatlas/

---

### DeepSlice
**Purpose**: AI-based initial pose estimation
**Why use it**: Dramatically speeds up ABBA workflow by automatically estimating slice position/angle in the atlas before manual refinement.

**Key Features**:
- Deep learning model trained on Allen CCF
- Estimates AP, DV, ML coordinates and angles
- Web interface or Python API

**Installation**:
```bash
pip install DeepSlice
```

**Usage**:
```python
from DeepSlice import DSModel
model = DSModel("mouse")
model.predict("path/to/images/", ensemble=True)
model.propagate_angles()  # Smooth angles across series
results = model.export()
```

**Documentation**: https://github.com/PolarBean/DeepSlice

---

### brainglobe-atlasapi
**Purpose**: Unified access to brain atlases
**Why use it**: Single API to access Allen CCF, Waxholm, and many other atlases with consistent structure.

**Key Features**:
- Download and cache atlases locally
- Access reference images, annotations, structures
- Consistent coordinate systems

**Installation**:
```bash
pip install brainglobe-atlasapi
```

**Usage**:
```python
from brainglobe_atlasapi import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_25um")

# Access data
reference = atlas.reference  # 3D reference image
annotation = atlas.annotation  # 3D annotation labels
structures = atlas.structures  # Region hierarchy
```

**Available Atlases**:
- `allen_mouse_10um`, `allen_mouse_25um`, `allen_mouse_50um`
- `waxholm_rat_39um`
- `kim_unified_25um` (developmental)
- Many more: https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html

---

## Cell Detection Options

### StarDist
**Best for**: Round/convex nuclei (DAPI staining)
**Pros**: Fast, accurate, well-tested
**Cons**: Struggles with elongated or irregular shapes

```bash
pip install stardist
pip install napari-stardist  # napari plugin
```

### Cellpose
**Best for**: Diverse cell morphologies
**Pros**: Handles varied shapes, good generalization
**Cons**: Slower than StarDist

```bash
pip install cellpose
pip install cellpose-napari  # napari plugin
```

### cellfinder (BrainGlobe)
**Best for**: Integration with BrainGlobe ecosystem
**Pros**: Designed for whole-brain analysis, good for sparse cells
**Cons**: More complex setup

```bash
pip install cellfinder
```

---

## Supporting Tools

### napari
**Version**: Use latest (0.4.x+)
```bash
pip install "napari[all]"
```

### scikit-image
**For**: General image processing
```bash
pip install scikit-image
```

### pandas
**For**: Data management and export
```bash
pip install pandas
```

---

## Tool Compatibility Matrix

| Tool | napari Plugin | Python API | Batch Mode |
|------|---------------|------------|------------|
| ABBA | napari-abba | Limited | Via Fiji |
| DeepSlice | No (planned) | Yes | Yes |
| StarDist | napari-stardist | Yes | Yes |
| Cellpose | cellpose-napari | Yes | Yes |
| brainglobe-atlasapi | Via brainrender | Yes | Yes |

---

## Typical Installation Order

```bash
# 1. Create environment
conda create -n sliceatlas python=3.10
conda activate sliceatlas

# 2. Install napari
pip install "napari[all]"

# 3. Install Java (required for ABBA)
conda install -c conda-forge openjdk

# 4. Install ABBA
pip install napari-abba

# 5. Install cell detection
pip install stardist napari-stardist
# OR
pip install cellpose cellpose-napari

# 6. Install atlas API
pip install brainglobe-atlasapi

# 7. Install DeepSlice
pip install DeepSlice

# 8. Other utilities
pip install pandas scikit-image tifffile
```

---

## Alternatives Considered

### QuPath
**What it is**: Powerful bioimage analysis platform
**Why not primary**: Less Python-native, though has excellent cell detection. Can complement ABBA (they integrate well).

### brainreg
**What it is**: 3D atlas registration from BrainGlobe
**Why not primary**: Designed for 3D volumes, not 2D slices. Use if you have serial sections reconstructed into 3D.

### WholeBrain (R)
**What it is**: R package for slice registration
**Why not primary**: R-based, less integration with napari ecosystem.
